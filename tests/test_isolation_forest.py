"""
Tests del detector de anomalías Isolation Forest (src/detector/isolation_forest.py).

Verifica el ciclo bootstrap → detección → reentrenamiento, el scoring normalizado,
y que una ventana con distribución de plantillas anómala puntúa más alto que el
baseline normal. Determinista (random_state fijo), sin cluster.
"""

import pytest

from src.detector.isolation_forest import (
    AnomalyDetector,
    ScoredWindow,
    novelty_score,
    severity_score,
)
from src.detector.window import WindowData


def _window(idx: int, counts: dict[int, int], error_count: int = 0,
            namespace: str = "default") -> WindowData:
    w = WindowData(index=idx, start_time=idx * 60, end_time=(idx + 1) * 60)
    w.cluster_counts = dict(counts)
    total = sum(counts.values())
    w.raw_logs = ["x"] * total
    w.error_count = error_count
    w.ns_log_counts = {namespace: total}
    if error_count:
        w.error_namespaces = {namespace}
        w.ns_error_counts = {namespace: error_count}
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


# ── Señal de severidad (logs de error) ─────────────────────────────────────

@pytest.mark.unit
def test_severity_score_zero_below_min_errors():
    # Menos de 3 errores en el namespace → no puntúa aunque el ratio sea alto
    w = _window(0, {1: 8}, error_count=2)
    assert severity_score(w) == 0.0


@pytest.mark.unit
def test_severity_score_rises_with_local_ratio():
    # 100 logs en un namespace, 70 de error (ratio 0.70) → satura a 1.0
    assert severity_score(_window(0, {1: 100}, error_count=70)) == 1.0
    # ratio bajo (0.25) → 0
    assert severity_score(_window(0, {1: 100}, error_count=25)) == 0.0
    # ratio intermedio → entre 0 y 1
    assert 0.0 < severity_score(_window(0, {1: 100}, error_count=45)) < 1.0


@pytest.mark.unit
def test_severity_per_namespace_catches_quiet_culprit():
    """CLAVE: un namespace 'callado' pero 100% errores dispara, aunque sea poco
    volumen del cluster (antes se diluía y se escapaba)."""
    w = WindowData(index=0, start_time=0, end_time=60)
    # Cluster ruidoso pero sano: 300 logs normales en 'default'
    w.ns_log_counts = {"default": 300, "postgresql": 6}
    w.ns_error_counts = {"postgresql": 6}   # postgresql: 6/6 = 100% errores
    w.error_count = 6
    w.raw_logs = ["x"] * 306
    # Ratio GLOBAL = 6/306 = 0.02 (se diluiría); LOCAL postgresql = 1.0 → dispara
    assert severity_score(w) == 1.0


@pytest.mark.unit
def test_error_log_spike_triggers_anomaly():
    """REGRESIÓN: un pod 'sano' escupiendo errores dispara la detección por severidad."""
    det = AnomalyDetector(bootstrap_windows=5, retrain_every_n=999, threshold=0.80)
    for i in range(5):
        det.process(_window(i, {1: 100, 2: 10}, error_count=0))
    # Namespace con 80% de errores → severidad alta → anomalía
    scored, _ = det.process(_window(5, {1: 100}, error_count=80))
    assert scored.severity_score >= 0.8
    assert scored.is_anomaly is True
    assert scored.score >= 0.80


# ── Señal de novedad (plantillas nunca vistas) ──────────────────────────────

@pytest.mark.unit
def test_novelty_score_zero_when_all_known():
    w = _window(0, {1: 50, 2: 30})
    assert novelty_score(w, trained_ids={1, 2, 3}) == 0.0


@pytest.mark.unit
def test_novelty_score_zero_below_min_novel_logs():
    # Solo 3 logs de una plantilla nueva → por debajo del mínimo
    w = _window(0, {1: 100, 99: 3})
    assert novelty_score(w, trained_ids={1}) == 0.0


@pytest.mark.unit
def test_novelty_score_rises_with_novel_ratio():
    # 100 logs, 50 de plantillas nuevas (99, 98) → ratio 0.50 → satura a 1.0
    w = _window(0, {1: 50, 99: 30, 98: 20})
    assert novelty_score(w, trained_ids={1}) == 1.0
    # ratio bajo (0.15) → 0
    w2 = _window(0, {1: 85, 99: 15})
    assert novelty_score(w2, trained_ids={1}) == 0.0
    # intermedio → entre 0 y 1
    w3 = _window(0, {1: 70, 99: 30})
    assert 0.0 < novelty_score(w3, trained_ids={1}) < 1.0


@pytest.mark.unit
def test_novel_template_burst_triggers_anomaly():
    """REGRESIÓN: una explosión de plantillas nunca vistas dispara la detección."""
    det = AnomalyDetector(bootstrap_windows=5, retrain_every_n=999, threshold=0.80)
    # Baseline con plantillas 1,2,3
    for i in range(5):
        det.process(_window(i, {1: 100, 2: 10, 3: 5}))
    # Ventana dominada por plantillas NUEVAS (nunca vistas en el entrenamiento)
    scored, _ = det.process(_window(5, {1: 10, 777: 60, 888: 40}))
    assert scored.novelty_score >= 0.8
    assert scored.is_anomaly is True
    assert scored.score >= 0.80


@pytest.mark.unit
def test_novelty_transient_after_retrain():
    """Tras reentrenar incorporando la plantilla, deja de ser novedosa."""
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=1, threshold=0.80)
    for i in range(3):
        det.process(_window(i, {1: 100, 2: 10}))
    # Primera aparición de la plantilla 555 → novedosa
    s1, _ = det.process(_window(3, {555: 100}))
    assert s1.novelty_score >= 0.8
    # retrain_every_n=1 → ya reentrenó incorporando 555; segunda aparición no es nueva
    s2, _ = det.process(_window(4, {555: 100}))
    assert s2.novelty_score == 0.0


# ── Warm-up: amortiguar la novedad tras (re)arrancar ────────────────────────

@pytest.mark.unit
def test_warmup_damps_novelty_right_after_bootstrap():
    """Con warm-up, una explosión novedosa JUSTO tras arrancar NO dispara por sí sola.

    La parte conocida imita el baseline (IF bajo); solo la novedad es alta. El
    warm-up la amortigua → no es anomalía todavía.
    """
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=999,
                          threshold=0.80, warmup_windows=10)
    for i in range(3):
        det.process(_window(i, {1: 80, 2: 15, 3: 5}))
    scored, _ = det.process(_window(3, {1: 80, 2: 15, 3: 5, 777: 50, 888: 50}))
    assert scored.novelty_score >= 0.8        # la novedad CRUDA medida sigue alta
    assert scored.is_anomaly is False         # pero amortiguada no dispara


@pytest.mark.unit
def test_no_warmup_novelty_fires_immediately():
    """Sin warm-up (default), la misma explosión novedosa dispara de inmediato."""
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=999, threshold=0.80)
    for i in range(3):
        det.process(_window(i, {1: 80, 2: 15, 3: 5}))
    scored, _ = det.process(_window(3, {1: 80, 2: 15, 3: 5, 777: 50, 888: 50}))
    assert scored.is_anomaly is True


@pytest.mark.unit
def test_warmup_ramps_novelty_back_in():
    """Tras pasar el warm-up, la novedad vuelve a disparar con normalidad."""
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=999,
                          threshold=0.80, warmup_windows=5)
    for i in range(3):
        det.process(_window(i, {1: 80, 2: 15, 3: 5}))
    # Consumir el warm-up con ventanas normales (sin novedad)
    for i in range(3, 3 + 5):
        det.process(_window(i, {1: 80, 2: 15, 3: 5}))
    scored, _ = det.process(_window(20, {1: 80, 2: 15, 3: 5, 777: 50, 888: 50}))
    assert scored.is_anomaly is True          # warm-up superado → novedad activa


@pytest.mark.unit
def test_warmup_does_not_damp_severity():
    """La severidad (errores reales) NO se amortigua: dispara aunque sea el arranque."""
    det = AnomalyDetector(bootstrap_windows=3, retrain_every_n=999,
                          threshold=0.80, warmup_windows=10)
    for i in range(3):
        det.process(_window(i, {1: 100, 2: 10}, error_count=0))
    scored, _ = det.process(_window(3, {1: 100}, error_count=80))
    assert scored.severity_score >= 0.8
    assert scored.is_anomaly is True          # severidad pasa por encima del warm-up


@pytest.mark.unit
def test_vectorize_ignores_unknown_clusters():
    det = AnomalyDetector(bootstrap_windows=2)
    det.process(_window(0, {1: 10}))
    det.process(_window(1, {1: 10}))  # entrena con feature set {1}
    # Una ventana con cluster nuevo (5) se puntúa sin romper (columna ignorada)
    scored, _ = det.process(_window(2, {5: 100}))
    assert isinstance(scored, ScoredWindow)
    assert 0.0 <= scored.score <= 1.0
