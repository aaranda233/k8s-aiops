"""
Tests del detector de anomalías Isolation Forest (src/detector/isolation_forest.py).

Verifica el ciclo bootstrap → detección → reentrenamiento, el scoring normalizado,
y que una ventana con distribución de plantillas anómala puntúa más alto que el
baseline normal. Determinista (random_state fijo), sin cluster.
"""

import pytest

from src.detector.isolation_forest import AnomalyDetector, ScoredWindow
from src.detector.window import WindowData


def _window(idx: int, counts: dict[int, int]) -> WindowData:
    w = WindowData(index=idx, start_time=idx * 60, end_time=(idx + 1) * 60)
    w.cluster_counts = dict(counts)
    w.raw_logs = ["x"] * sum(counts.values())
    return w


@pytest.mark.unit
def test_bootstrap_returns_none_until_ready():
    det = AnomalyDetector(bootstrap_windows=3)
    assert not det.is_ready
    for i in range(2):
        scored, retrained = det.process(_window(i, {1: 50, 2: 5}))
        assert scored is None
        assert not det.is_ready
    # La tercera ventana completa el bootstrap y entrena el modelo
    det.process(_window(2, {1: 50, 2: 5}))
    assert det.is_ready


@pytest.mark.unit
def test_bootstrap_progress_string():
    det = AnomalyDetector(bootstrap_windows=5)
    det.process(_window(0, {1: 10}))
    det.process(_window(1, {1: 10}))
    assert det.bootstrap_progress == "2/5 ventanas"


@pytest.mark.unit
def test_scores_window_after_bootstrap():
    det = AnomalyDetector(bootstrap_windows=3, threshold=0.8)
    for i in range(3):
        det.process(_window(i, {1: 50, 2: 5}))
    scored, _ = det.process(_window(3, {1: 50, 2: 5}))
    assert isinstance(scored, ScoredWindow)
    assert 0.0 <= scored.score <= 1.0
    assert scored.model_version >= 1


@pytest.mark.unit
def test_anomalous_window_scores_higher_than_normal():
    """Una ventana con distribución de plantillas anómala puntúa más que el baseline."""
    det = AnomalyDetector(bootstrap_windows=8, retrain_every_n=999, threshold=0.8)
    # Baseline normal CON variación realista (si no, el IF no aprende densidad)
    profiles = [
        {1: 100, 2: 10, 3: 5}, {1: 90, 2: 20, 3: 5},
        {1: 110, 2: 8, 3: 6}, {1: 95, 2: 15, 3: 4},
    ]
    for i in range(8):
        det.process(_window(i, profiles[i % len(profiles)]))
    # Ventana normal parecida al baseline
    normal, _ = det.process(_window(8, {1: 100, 2: 12, 3: 5}))
    # Ventana anómala: la plantilla 3 explota (proporción muy fuera de lo normal)
    anomaly, _ = det.process(_window(9, {1: 5, 2: 5, 3: 600}))
    assert anomaly.score > normal.score


@pytest.mark.unit
def test_threshold_flags_anomaly():
    det = AnomalyDetector(bootstrap_windows=5, retrain_every_n=999, threshold=0.5)
    for i in range(5):
        det.process(_window(i, {1: 100, 2: 5}))
    anomaly, _ = det.process(_window(5, {77: 400}))
    assert anomaly.is_anomaly == (anomaly.score >= 0.5)


@pytest.mark.unit
def test_retrain_increments_model_version():
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=2, threshold=0.8)
    for i in range(3):
        det.process(_window(i, {1: 50, 2: 5}))
    v0 = det._model_version
    det.process(_window(3, {1: 50, 2: 5}))           # 1 desde retrain
    _, retrained = det.process(_window(4, {1: 50, 2: 5}))  # 2 → reentrena
    assert retrained is True
    assert det._model_version > v0


@pytest.mark.unit
def test_vectorize_ignores_unknown_clusters():
    det = AnomalyDetector(bootstrap_windows=2)
    det.process(_window(0, {1: 10}))
    det.process(_window(1, {1: 10}))  # entrena con feature set {1}
    # Una ventana con cluster nuevo (5) se puntúa sin romper (columna ignorada)
    scored, _ = det.process(_window(2, {5: 100}))
    assert isinstance(scored, ScoredWindow)
    assert 0.0 <= scored.score <= 1.0
