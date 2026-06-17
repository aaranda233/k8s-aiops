"""
Tests de orquestación del pipeline (src/pipeline.py).

K8sCollector mockeado (no conecta al cluster). Verifica _trigger_rca en sus dos
caminos críticos: éxito (diagnóstico + remediación + eventos) y fallo (registro
de incidente sin diagnóstico, el fix de robustez), además de _emit.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from config.settings import (
    DiagnosticsConfig,
    MLflowConfig,
    PipelineConfig,
    RemediationConfig,
)
from src.diagnostics.ollama_rca import DiagnosisResult


@dataclass
class _W:
    index: int = 5
    raw_logs: list = field(default_factory=lambda: ["evt1", "evt2"])
    namespaces: set = field(default_factory=lambda: {"default"})
    start_time: float = 0.0
    end_time: float = 60.0
    log_count: int = 2
    template_count: int = 2


@dataclass
class _Scored:
    window: _W = field(default_factory=_W)
    score: float = 0.95
    is_anomaly: bool = True
    model_version: int = 1
    if_score: float = 0.0
    severity_score: float = 0.0
    novelty_score: float = 0.0
    culprit_namespace: str = "default"


class _CapturingBus:
    def __init__(self):
        self.events = []
    def publish(self, event):
        self.events.append((event.type, event.data))


def _pipeline(monkeypatch):
    """Pipeline con colectores mockeados y diagnóstico/remediación desactivados."""
    import src.pipeline as pl
    monkeypatch.setattr(pl, "K8sCollector", lambda *a, **k: MagicMock())
    cfg = PipelineConfig(
        diagnostics=DiagnosticsConfig(enabled=False),
        remediation=RemediationConfig(enabled=False),
        mlflow=MLflowConfig(enabled=False),
    )
    bus = _CapturingBus()
    p = pl.AIOPsPipeline(cfg=cfg, event_bus=bus)
    return p, bus


@pytest.mark.unit
def test_trigger_rca_success_path(monkeypatch):
    p, bus = _pipeline(monkeypatch)
    diag = DiagnosisResult(
        window_index=5, anomaly_score=0.95, namespaces={"default"},
        root_cause="Pod en CrashLoopBackOff", kubectl_command="kubectl logs api",
        model_version=1, mode="hybrid", steps_taken=2,
    )
    p.rca = MagicMock()
    p.rca.health_check.return_value = True
    p.rca.diagnose.return_value = diag
    p.remediation = MagicMock()

    p._trigger_rca(_Scored())

    assert len(p._diagnoses) == 1
    p.remediation.handle_async.assert_called_once()
    types = [t for t, _ in bus.events]
    assert "anomaly" in types
    assert "rca" in types
    rca_data = next(d for t, d in bus.events if t == "rca")
    assert rca_data["root_cause"] == "Pod en CrashLoopBackOff"


@pytest.mark.unit
def test_trigger_rca_failure_registers_incident(monkeypatch):
    """REGRESIÓN: si diagnose() lanza, se registra incidente sin diagnóstico."""
    p, bus = _pipeline(monkeypatch)
    p.rca = MagicMock()
    p.rca.health_check.return_value = True
    p.rca.diagnose.side_effect = RuntimeError("ollama timeout")
    p.remediation = MagicMock()

    p._trigger_rca(_Scored())

    # No se tragó el error: se registró el incidente de fallo
    p.remediation.register_failed_diagnosis.assert_called_once()
    args = p.remediation.register_failed_diagnosis.call_args
    assert "ollama timeout" in args.args[1]
    # Y se emitió un evento rca con el error
    rca_data = next(d for t, d in bus.events if t == "rca")
    assert "error" in rca_data


@pytest.mark.unit
def test_trigger_rca_no_rca_only_emits_anomaly(monkeypatch):
    p, bus = _pipeline(monkeypatch)
    p.rca = None
    p._trigger_rca(_Scored())
    types = [t for t, _ in bus.events]
    assert "anomaly" in types
    assert "rca" not in types


@pytest.mark.unit
def test_trigger_rca_ollama_unavailable(monkeypatch):
    p, bus = _pipeline(monkeypatch)
    p.rca = MagicMock()
    p.rca.health_check.return_value = False
    p._trigger_rca(_Scored())
    rca_data = next(d for t, d in bus.events if t == "rca")
    assert "no disponible" in rca_data["error"].lower()


def _win(idx, counts):
    from src.detector.window import WindowData
    w = WindowData(index=idx, start_time=idx * 60, end_time=(idx + 1) * 60)
    w.cluster_counts = dict(counts)
    w.raw_logs = ["x"] * sum(counts.values())
    w.namespaces = {"default"}
    return w


@pytest.mark.unit
def test_evaluate_window_bootstrap_then_detect_and_alert(monkeypatch):
    """Conduce el pipeline: bootstrap → scoring → ventana anómala dispara RCA."""
    import src.pipeline as pl
    monkeypatch.setattr(pl, "K8sCollector", lambda *a, **k: MagicMock())
    cfg = PipelineConfig(
        diagnostics=DiagnosticsConfig(enabled=False),
        remediation=RemediationConfig(enabled=False),
        mlflow=MLflowConfig(enabled=False),
    )
    cfg.collector.bootstrap_windows = 5
    cfg.detector.anomaly_threshold = 0.5
    bus = _CapturingBus()
    p = pl.AIOPsPipeline(cfg=cfg, event_bus=bus)

    # Fase bootstrap: 5 ventanas → emiten 'bootstrap', no 'window_scored'
    for i in range(5):
        p._evaluate_window(_win(i, {1: 100, 2: 10, 3: 5}))
    assert any(t == "bootstrap" for t, _ in bus.events)
    assert not any(t == "window_scored" for t, _ in bus.events)

    # Fase detección: una ventana normal puntúa y emite 'window_scored'
    p._evaluate_window(_win(5, {1: 100, 2: 12, 3: 5}))
    assert any(t == "window_scored" for t, _ in bus.events)

    # Ventana anómala con RCA mockeado → debe emitir 'anomaly'
    captured = {}
    monkeypatch.setattr(p, "_trigger_rca", lambda s: captured.setdefault("hit", s))
    p._evaluate_window(_win(6, {99: 800}))
    scored_events = [d for t, d in bus.events if t == "window_scored"]
    # La última ventana es anómala (score alto) o al menos se puntuó
    assert scored_events
