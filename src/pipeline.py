"""
Orquestador del pipeline AIOps de 3 capas.

Modos:
  - replay:  procesa eventos historicos del cluster (snapshot)
  - live:    stream continuo via Watch API
"""

import threading
import time

from rich.console import Console

from config.settings import PipelineConfig
from src.collector.k8s_collector import K8sCollector
from src.detector.isolation_forest import AnomalyDetector, ScoredWindow
from src.detector.window import WindowBuilder
from src.diagnostics.hybrid_react_agent import HybridReActAgent
from src.diagnostics.ollama_rca import DiagnosisResult, OllamaRCA
from src.diagnostics.react_agent import ReActAgent
from src.parser.log_parser import LogParser
from src.remediation.auto_remediation import AutoRemediation
from src.remediation.notifier import Notifier
from src.tracking.mlflow_tracker import MLflowTracker, RetrainEvent

console = Console()


class AIOPsPipeline:
    def __init__(self, cfg: PipelineConfig, event_bus=None):
        self.cfg = cfg
        self._bus = event_bus  # opcional — None cuando se usa sin web

        self.parser = LogParser()
        self.window_builder = WindowBuilder(cfg.collector.window_size_seconds)
        self.detector = AnomalyDetector(
            bootstrap_windows=cfg.collector.bootstrap_windows,
            rolling_window_size=cfg.collector.rolling_window_size,
            retrain_every_n=cfg.collector.retrain_every_n_windows,
            threshold=cfg.detector.anomaly_threshold,
            n_estimators=cfg.detector.n_estimators,
            contamination=cfg.detector.contamination,
            random_state=cfg.detector.random_state,
        )
        if cfg.diagnostics.enabled:
            mode = cfg.diagnostics.react_mode
            if mode == "hybrid":
                self.rca = HybridReActAgent(
                    host=cfg.diagnostics.host,
                    base_model=cfg.diagnostics.react_base_model,
                    expert_model=cfg.diagnostics.model,
                    max_logs=cfg.diagnostics.max_logs_in_prompt,
                    timeout=cfg.diagnostics.timeout_seconds,
                    max_steps=cfg.diagnostics.react_max_steps,
                    dry_run=cfg.diagnostics.react_dry_run,
                )
            elif mode == "react":
                self.rca = ReActAgent(
                    host=cfg.diagnostics.host,
                    model=cfg.diagnostics.model,
                    max_logs=cfg.diagnostics.max_logs_in_prompt,
                    timeout=cfg.diagnostics.timeout_seconds,
                    max_steps=cfg.diagnostics.react_max_steps,
                    dry_run=cfg.diagnostics.react_dry_run,
                )
            else:  # single_shot (default)
                self.rca = OllamaRCA(
                    host=cfg.diagnostics.host,
                    model=cfg.diagnostics.model,
                    max_logs=cfg.diagnostics.max_logs_in_prompt,
                    timeout=cfg.diagnostics.timeout_seconds,
                )
        else:
            self.rca = None

        # Auto-remediación (opcional)
        self.remediation: AutoRemediation | None = None
        if cfg.remediation.enabled:
            notifier = None
            if cfg.remediation.smtp_user and cfg.remediation.notify_email:
                notifier = Notifier(
                    smtp_host=cfg.remediation.smtp_host,
                    smtp_port=cfg.remediation.smtp_port,
                    smtp_user=cfg.remediation.smtp_user,
                    smtp_pass=cfg.remediation.smtp_pass,
                    from_addr=cfg.remediation.smtp_from or cfg.remediation.smtp_user,
                    to_addr=cfg.remediation.notify_email,
                    webhook_base_url=cfg.remediation.webhook_base_url,
                )
            self.remediation = AutoRemediation(
                notifier=notifier,
                max_auto_level=cfg.remediation.max_auto_level,
                verify_wait=cfg.remediation.verify_wait_seconds,
            )

        self.collector = K8sCollector(namespaces=cfg.collector.namespaces)
        self._scored_windows: list[ScoredWindow] = []
        self._diagnoses: list[DiagnosisResult] = []

        self._tracker = MLflowTracker(
            uri=cfg.mlflow.tracking_uri,
            experiment=cfg.mlflow.experiment,
        )
        if not cfg.mlflow.enabled:
            self._tracker._enabled = False

    # ------------------------------------------------------------------
    # Modos de ejecucion
    # ------------------------------------------------------------------

    def run_replay(self) -> None:
        console.print("\n[bold cyan]Modo REPLAY — cargando eventos historicos...[/]")
        entries = self.collector.fetch_events_snapshot()

        if not entries:
            console.print("[yellow]No se encontraron eventos.[/]")
            return

        console.print(f"[cyan]{len(entries)} eventos | "
                      f"rango: {entries[-1].timestamp - entries[0].timestamp:.0f}s[/]\n")

        self._emit("status", {"mode": "replay", "total_events": len(entries)})

        with self._tracker.start_run(self.cfg):
            for entry in entries:
                self._ingest(entry)

            last = self.window_builder.flush()
            if last and last.log_count > 0:
                self._evaluate_window(last)

            self._print_summary()
            self._tracker.log_summary(
                total_windows=len(self._scored_windows),
                total_anomalies=sum(1 for s in self._scored_windows if s.is_anomaly),
                total_rca=len(self._diagnoses),
            )

    def run_live(self) -> None:
        console.print("\n[bold green]Modo LIVE — Watch API[/]")
        self._emit("status", {
            "mode": "live",
            "bootstrap": self.cfg.collector.bootstrap_windows,
            "threshold": self.cfg.detector.anomaly_threshold,
            "window_size": self.cfg.collector.window_size_seconds,
        })

        with self._tracker.start_run(self.cfg):
            # 1. Pre-cargar snapshot historico para el bootstrap
            console.print("[dim]Cargando snapshot historico para bootstrap...[/]")
            entries = self.collector.fetch_events_snapshot()
            console.print(f"[dim]{len(entries)} eventos historicos cargados[/]")
            for entry in entries:
                self._ingest(entry)

            # 2. Timer de cierre de ventanas: cierra la ventana abierta cada window_size segundos
            #    aunque no lleguen eventos nuevos
            self._start_window_flush_timer()

            # 3. Watch API para eventos nuevos
            try:
                for entry in self.collector.stream_events():
                    self._ingest(entry)
            except KeyboardInterrupt:
                console.print("\n[yellow]Detenido.[/]")
                self._print_summary()
                self._tracker.log_summary(
                    total_windows=len(self._scored_windows),
                    total_anomalies=sum(1 for s in self._scored_windows if s.is_anomaly),
                    total_rca=len(self._diagnoses),
                )

    def _start_window_flush_timer(self) -> None:
        """Cierra la ventana actual periodicamente aunque no lleguen eventos."""
        def _flush_loop():
            while True:
                time.sleep(self.cfg.collector.window_size_seconds)
                window = self.window_builder.flush()
                if window and window.log_count > 0:
                    self._evaluate_window(window)
                # Emitir heartbeat para mantener viva la conexion WS
                self._emit("heartbeat", {"ts": time.time()})

        t = threading.Thread(target=_flush_loop, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Procesamiento
    # ------------------------------------------------------------------

    def _ingest(self, entry) -> None:
        parsed = self.parser.parse(
            raw=entry.raw,
            namespace=entry.namespace,
            timestamp=entry.timestamp,
        )

        # Emitir evento de log parseado a la web
        self._emit("log_parsed", {
            "ts": entry.timestamp,
            "namespace": entry.namespace,
            "reason": entry.reason,
            "source": entry.source,
            "raw": entry.raw[:120],
            "template": parsed.template[:120],
            "cluster_id": parsed.cluster_id,
            "template_count": self.parser.cluster_count,
        })

        closed_window = self.window_builder.feed(parsed, entry.timestamp)
        if closed_window is not None and closed_window.log_count > 0:
            self._evaluate_window(closed_window)

    def _evaluate_window(self, window) -> None:
        if not self.detector.is_ready:
            progress = len(self.detector._bootstrap_buffer)
            console.print(
                f"[dim]  [bootstrap] {self.detector.bootstrap_progress} — "
                f"ventana {window.index} ({window.log_count} eventos)[/]"
            )
            self._emit("bootstrap", {
                "progress": progress,
                "total": self.cfg.collector.bootstrap_windows,
                "window_index": window.index,
                "log_count": window.log_count,
            })
            self.detector.process(window)  # retorna (None, False) en bootstrap
            return

        result = self.detector.process(window)
        if result is None:
            return
        scored, retrained = result

        self._scored_windows.append(scored)
        self._print_window_line(scored)
        self._tracker.log_window(scored)

        self._emit("window_scored", {
            "index": window.index,
            "score": round(scored.score, 4),
            "is_anomaly": scored.is_anomaly,
            "log_count": window.log_count,
            "template_count": window.template_count,
            "namespaces": sorted(window.namespaces),
            "threshold": self.cfg.detector.anomaly_threshold,
            "model_version": scored.model_version,
            "start_time": window.start_time,
            "pca_x": round(scored.pca_x, 4),
            "pca_y": round(scored.pca_y, 4),
        })

        # Si reentrenó, emitir el scatter completo del nuevo training set
        if retrained:
            training_windows = list(self.detector._history)
            training_coords = self.detector.get_training_scatter(training_windows)
            self._emit("retrain", {
                "model_version": scored.model_version,
                "training_size": len(training_windows),
                "training_points": [
                    {
                        "x": round(c[0], 4),
                        "y": round(c[1], 4),
                        "log_count": w.log_count,
                        "index": w.index,
                    }
                    for c, w in zip(training_coords, training_windows)
                ],
            })
            self._tracker.log_retrain(RetrainEvent(
                model_version=scored.model_version,
                training_size=len(training_windows),
                n_features=len(self.detector._trained_cluster_ids),
                window_index=window.index,
            ))

        if scored.is_anomaly:
            self._trigger_rca(scored)

    def _trigger_rca(self, scored: ScoredWindow) -> None:
        console.print(f"\n  [bold red][ALERTA] Score={scored.score:.3f}[/]")
        self._emit("anomaly", {
            "window_index": scored.window.index,
            "score": round(scored.score, 4),
            "namespaces": sorted(scored.window.namespaces),
            "log_count": scored.window.log_count,
            "logs_sample": scored.window.raw_logs[-10:],
        })

        if self.rca is None:
            return

        if not self.rca.health_check():
            console.print("  [red]Ollama no disponible[/]")
            self._emit("rca", {"error": "Ollama no disponible", "window_index": scored.window.index})
            return

        try:
            t0 = time.time()
            result = self.rca.diagnose(scored)
            latency = time.time() - t0
            self._diagnoses.append(result)
            mode_tag = f"[dim][{result.mode} · {result.steps_taken} paso(s) · confianza={result.confidence}][/]"
            console.print(f"  [bold]Causa:[/] {result.root_cause}")
            console.print(f"  [bold]kubectl:[/] [cyan]{result.kubectl_command}[/]")
            console.print(f"  {mode_tag}\n")
            self._tracker.log_rca(result, latency_s=latency)
            rca_event: dict = {
                "window_index": result.window_index,
                "score": round(result.anomaly_score, 4),
                "namespaces": sorted(result.namespaces),
                "root_cause": result.root_cause,
                "kubectl": result.kubectl_command,
                "model_version": result.model_version,
                "mode": result.mode,
                "confidence": result.confidence,
                "steps_taken": result.steps_taken,
            }
            if result.react_trace:
                rca_event["trace"] = [
                    {
                        "step": s.step,
                        "thought": s.thought,
                        "action": s.action,
                        "observation": (s.observation or "")[:300],
                        "is_final": s.is_final,
                    }
                    for s in result.react_trace
                ]
            self._emit("rca", rca_event)

            # Auto-remediación en hilo de fondo (no bloquea el pipeline)
            if self.remediation:
                self.remediation.handle_async(scored, result)

        except Exception as e:
            console.print(f"  [red]Error RCA: {e}[/]")
            self._emit("rca", {"error": str(e), "window_index": scored.window.index})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, data: dict) -> None:
        if self._bus:
            from web.event_bus import PipelineEvent
            self._bus.publish(PipelineEvent(type=event_type, data=data))

    @staticmethod
    def _print_window_line(sw: ScoredWindow) -> None:
        w = sw.window
        status = "[bold red]*** ANOMALIA ***[/]" if sw.is_anomaly else "[green]normal[/]"
        ns_str = ", ".join(sorted(w.namespaces)[:3])
        if len(w.namespaces) > 3:
            ns_str += f" +{len(w.namespaces)-3}"
        console.print(
            f"  W{w.index:03d} | score=[bold]{sw.score:.3f}[/] | "
            f"events={w.log_count:3d} | templates={w.template_count:2d} | "
            f"ns=[dim]{ns_str}[/] | {status}"
        )

    def _print_summary(self) -> None:
        total = len(self._scored_windows)
        anomalies = sum(1 for s in self._scored_windows if s.is_anomaly)
        console.print(f"\n{'─'*70}")
        console.print(f"[bold]Resumen:[/] {total} ventanas | [red]{anomalies} anomalias[/] | "
                      f"{len(self._diagnoses)} diagnosticos")
        console.print(f"Templates: {self.parser.cluster_count} | IF versiones: {self.detector._model_version}")
        console.print(f"{'─'*70}\n")
