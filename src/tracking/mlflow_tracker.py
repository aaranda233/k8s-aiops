"""
MLflow tracker para el pipeline AIOps.

Registra tres tipos de eventos:
  - window_scored : métrica por ventana (score, anomalía, logs, templates)
  - retrain       : cada reentrenamiento del Isolation Forest
  - rca           : cada diagnóstico generado por el SLM

Uso:
    tracker = MLflowTracker.from_env()   # lee MLFLOW_TRACKING_URI
    tracker = MLflowTracker(uri="http://192.168.2.204:30803")

    with tracker.start_run(config):
        tracker.log_window(scored)
        tracker.log_retrain(model_version, training_size, n_features)
        tracker.log_rca(result)

Si MLflow no está disponible o MLFLOW_ENABLED=false, todas las llamadas
son no-op — el pipeline no falla.
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrainEvent:
    model_version: int
    training_size: int
    n_features: int
    window_index: int


class MLflowTracker:
    def __init__(self, uri: str, experiment: str = "k8s-aiops"):
        self._uri = uri
        self._experiment = experiment
        self._run = None
        self._enabled = False
        self._step = 0

        try:
            import mlflow
            self._mlflow = mlflow
            self._enabled = True
        except ImportError:
            self._mlflow = None

    @classmethod
    def from_env(cls) -> "MLflowTracker":
        uri = os.getenv("MLFLOW_TRACKING_URI", "http://192.168.2.204:30803")
        experiment = os.getenv("MLFLOW_EXPERIMENT", "k8s-aiops")
        return cls(uri=uri, experiment=experiment)

    # ------------------------------------------------------------------
    # Ciclo de vida del run
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(self, config):
        if not self._enabled:
            yield self
            return

        mlflow = self._mlflow
        mlflow.set_tracking_uri(self._uri)
        mlflow.set_experiment(self._experiment)

        run_name = f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            self._run = run
            # Loguear config como params
            mlflow.log_params({
                "window_size_s":      config.collector.window_size_seconds,
                "bootstrap_windows":  config.collector.bootstrap_windows,
                "rolling_size":       config.collector.rolling_window_size,
                "retrain_every_n":    config.collector.retrain_every_n_windows,
                "anomaly_threshold":  config.detector.anomaly_threshold,
                "if_n_estimators":    config.detector.n_estimators,
                "if_contamination":   config.detector.contamination,
                "ollama_model":       config.diagnostics.model,
                "ollama_host":        config.diagnostics.host,
            })
            try:
                yield self
            finally:
                self._run = None

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------

    def log_window(self, scored) -> None:
        """Registra una ventana puntuada."""
        if not self._enabled or self._run is None:
            return

        self._step += 1
        self._mlflow.log_metrics(
            {
                "window_score":      scored.score,
                "is_anomaly":        float(scored.is_anomaly),
                "log_count":         scored.window.log_count,
                "template_count":    scored.window.template_count,
                "namespace_count":   len(scored.window.namespaces),
                "model_version":     float(scored.model_version),
                "pca_x":             scored.pca_x,
                "pca_y":             scored.pca_y,
            },
            step=self._step,
        )

    def log_retrain(self, event: RetrainEvent) -> None:
        """Registra un reentrenamiento del Isolation Forest."""
        if not self._enabled or self._run is None:
            return

        self._mlflow.log_metrics(
            {
                "retrain_model_version": float(event.model_version),
                "retrain_training_size": float(event.training_size),
                "retrain_n_features":    float(event.n_features),
            },
            step=event.window_index,
        )

    def log_rca(self, result, latency_s: Optional[float] = None) -> None:
        """Registra un diagnóstico RCA."""
        if not self._enabled or self._run is None:
            return

        metrics = {
            "rca_anomaly_score":   result.anomaly_score,
            "rca_window_index":    float(result.window_index),
        }
        if latency_s is not None:
            metrics["rca_latency_s"] = latency_s

        self._mlflow.log_metrics(metrics, step=self._step)

        # Root cause y kubectl como tag del run (útil para búsqueda en UI)
        try:
            self._mlflow.set_tags({
                f"rca_w{result.window_index}_cause":  result.root_cause[:250],
                f"rca_w{result.window_index}_kubectl": result.kubectl_command[:250],
            })
        except Exception:
            pass

    def log_summary(self, total_windows: int, total_anomalies: int, total_rca: int) -> None:
        """Resumen final del run."""
        if not self._enabled or self._run is None:
            return

        self._mlflow.log_metrics({
            "summary_total_windows":   float(total_windows),
            "summary_total_anomalies": float(total_anomalies),
            "summary_total_rca":       float(total_rca),
        })
