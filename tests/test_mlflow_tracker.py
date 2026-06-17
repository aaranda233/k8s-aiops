"""
Tests del tracker MLflow (src/tracking/mlflow_tracker.py).

Verifica que cuando está deshabilitado todas las llamadas son no-op (el pipeline
nunca debe fallar por MLflow), y que cuando está habilitado registra métricas
(con un cliente mlflow falso).
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from src.tracking.mlflow_tracker import MLflowTracker, RetrainEvent


@dataclass
class _W:
    log_count: int = 10
    template_count: int = 3
    namespaces: set = field(default_factory=lambda: {"a", "b"})


@dataclass
class _Scored:
    score: float = 0.9
    is_anomaly: bool = True
    model_version: int = 1
    pca_x: float = 0.1
    pca_y: float = 0.2
    window: _W = field(default_factory=_W)


@dataclass
class _Diag:
    anomaly_score: float = 0.9
    window_index: int = 3
    root_cause: str = "causa"
    kubectl_command: str = "kubectl get pods"


def _disabled_tracker():
    t = MLflowTracker(uri="http://x")
    t._enabled = False
    return t


@pytest.mark.unit
def test_disabled_tracker_is_noop():
    t = _disabled_tracker()
    # Ninguna de estas llamadas debe lanzar
    t.log_window(_Scored())
    t.log_retrain(RetrainEvent(1, 50, 8, 2))
    t.log_rca(_Diag(), latency_s=1.2)
    t.log_summary(10, 2, 2)
    with t.start_run(MagicMock()) as ctx:
        assert ctx is t


@pytest.mark.unit
def test_from_env_builds_tracker(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "exp1")
    t = MLflowTracker.from_env()
    assert t._uri == "http://mlflow:5000"
    assert t._experiment == "exp1"


@pytest.mark.unit
def test_enabled_tracker_logs_metrics():
    t = MLflowTracker(uri="http://x")
    t._enabled = True
    t._mlflow = MagicMock()
    t._run = object()  # simula un run activo

    t.log_window(_Scored())
    t.log_retrain(RetrainEvent(2, 40, 6, 5))
    t.log_rca(_Diag(), latency_s=0.5)
    t.log_summary(10, 3, 3)

    assert t._mlflow.log_metrics.call_count >= 4
    t._mlflow.set_tags.assert_called_once()


@pytest.mark.unit
def test_log_loop_cycle_disabled_is_noop():
    t = _disabled_tracker()
    # No debe lanzar ni requerir mlflow
    t.log_loop_cycle(1, 30, {"positive": 10, "negative": 5}, {"decision": "PROMOTE"}, True)


@pytest.mark.unit
def test_log_loop_cycle_enabled_logs():
    t = MLflowTracker(uri="http://x")
    t._enabled = True
    t._mlflow = MagicMock()
    t._mlflow.start_run.return_value.__enter__ = lambda *a: None
    t._mlflow.start_run.return_value.__exit__ = lambda *a: False
    gate = {"decision": "PROMOTE", "candidate": {"parse_rate": 0.99, "keyword_hit": 0.95},
            "prod": {"parse_rate": 0.98, "keyword_hit": 0.90}}
    t.log_loop_cycle(2, 40, {"positive": 12, "negative": 8}, gate, True)
    t._mlflow.log_metrics.assert_called_once()
    t._mlflow.set_tags.assert_called_once()


@pytest.mark.unit
def test_log_metrics_skipped_without_active_run():
    t = MLflowTracker(uri="http://x")
    t._enabled = True
    t._mlflow = MagicMock()
    t._run = None  # sin run activo → no registra
    t.log_window(_Scored())
    t._mlflow.log_metrics.assert_not_called()
