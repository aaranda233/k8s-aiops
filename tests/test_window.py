"""
Tests del ventaneo temporal (src/detector/window.py).

Verifica la agrupación de logs parseados en ventanas de tiempo fijas, el cierre
de ventanas al cruzar el límite, y los conteos derivados.
"""

from dataclasses import dataclass

import pytest

from src.detector.window import WindowBuilder, WindowData


@dataclass
class FakeParsed:
    raw: str
    namespace: str
    cluster_id: int
    level: str = ""
    template: str = ""


@pytest.mark.unit
def test_window_add_accumulates_counts_and_namespaces():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("log a", "ns1", 1))
    w.add(FakeParsed("log b", "ns1", 1))
    w.add(FakeParsed("log c", "ns2", 2))
    assert w.log_count == 3
    assert w.namespaces == {"ns1", "ns2"}
    assert w.cluster_counts == {1: 2, 2: 1}
    assert w.template_count == 2


@pytest.mark.unit
def test_window_counts_error_level_logs():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("ok", "ns", 1, level="INFO"))
    w.add(FakeParsed("boom", "ns", 2, level="FATAL"))
    w.add(FakeParsed("bad", "ns", 2, level="error"))   # case-insensitive
    w.add(FakeParsed("crit", "ns", 3, level="CRITICAL"))
    w.add(FakeParsed("warn", "ns", 4, level="WARNING"))  # WARNING no cuenta como error
    assert w.error_count == 3
    assert w.error_ratio == 3 / 5


@pytest.mark.unit
def test_window_keeps_structured_error_records():
    """Los logs de error guardan plantilla/cluster/namespace, no solo el raw."""
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed('FATAL: role "alice" does not exist', "pg", 7,
                     level="FATAL", template='FATAL: role "<*>" does not exist'))
    w.add(FakeParsed("INFO ready", "pg", 8, level="INFO"))  # no error, no record
    assert len(w.error_records) == 1
    rec = w.error_records[0]
    assert rec.cluster_id == 7
    assert rec.namespace == "pg"
    assert rec.template == 'FATAL: role "<*>" does not exist'
    assert rec.raw == 'FATAL: role "alice" does not exist'
    # error_logs (raw) se conserva para compatibilidad
    assert w.error_logs == ['FATAL: role "alice" does not exist']


@pytest.mark.unit
def test_ns_cluster_counts_tracks_templates_per_namespace():
    """Cada namespace mantiene su propia distribución de plantillas (para scoring por ns)."""
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("a", "pg", 1))
    w.add(FakeParsed("a", "pg", 1))
    w.add(FakeParsed("b", "pg", 2))
    w.add(FakeParsed("c", "longhorn", 9))
    assert w.ns_cluster_counts == {"pg": {1: 2, 2: 1}, "longhorn": {9: 1}}
    # el agregado global sigue existiendo (compat)
    assert w.cluster_counts == {1: 2, 2: 1, 9: 1}


@pytest.mark.unit
def test_primary_namespace_is_dominant_error_namespace():
    """El namespace con más errores es el culpable principal."""
    w = WindowData(index=0, start_time=0, end_time=60)
    for _ in range(9):
        w.add(FakeParsed("boom", "postgresql", 1, level="FATAL"))
    for _ in range(2):
        w.add(FakeParsed("warn", "longhorn-system", 2, level="ERROR"))
    w.add(FakeParsed("ok", "argocd", 3, level="INFO"))  # sin errores, no cuenta
    assert w.primary_namespace == "postgresql"


@pytest.mark.unit
def test_primary_namespace_none_without_errors():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("ok", "ns", 1, level="INFO"))
    assert w.primary_namespace is None


@pytest.mark.unit
def test_primary_namespace_tie_is_deterministic():
    """Empate en nº de errores → alfabético (determinista entre ejecuciones)."""
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("e", "zeta", 1, level="ERROR"))
    w.add(FakeParsed("e", "alpha", 2, level="ERROR"))
    assert w.primary_namespace == "alpha"


@pytest.mark.unit
def test_window_error_ratio_zero_without_errors():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("a", "ns", 1, level="INFO"))
    assert w.error_count == 0
    assert w.error_ratio == 0.0


@pytest.mark.unit
def test_focus_namespaces_uses_error_namespaces():
    """Atribuye la anomalía a los namespaces de los logs de error, no a todos."""
    w = WindowData(index=0, start_time=0, end_time=60)
    # Muchos namespaces con logs normales, solo 'postgresql' con error
    w.add(FakeParsed("ok", "argocd", 1, level="INFO"))
    w.add(FakeParsed("ok", "default", 2, level="INFO"))
    w.add(FakeParsed("ok", "kube-system", 3, level="INFO"))
    w.add(FakeParsed("boom", "postgresql", 4, level="FATAL"))
    assert set(w.namespaces) == {"argocd", "default", "kube-system", "postgresql"}
    assert w.focus_namespaces == ["postgresql"]  # solo el culpable


@pytest.mark.unit
def test_focus_namespaces_falls_back_to_all_without_errors():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("a", "ns1", 1, level="INFO"))
    w.add(FakeParsed("b", "ns2", 2, level="INFO"))
    assert w.focus_namespaces == ["ns1", "ns2"]  # sin errores -> todos


@pytest.mark.unit
def test_feed_keeps_logs_in_same_window_until_boundary():
    b = WindowBuilder(window_size_seconds=60)
    assert b.feed(FakeParsed("a", "ns", 1), timestamp=1000.0) is None
    assert b.feed(FakeParsed("b", "ns", 1), timestamp=1030.0) is None  # misma ventana
    assert b.feed(FakeParsed("c", "ns", 2), timestamp=1059.0) is None
    # Aún no se ha cerrado ninguna ventana
    assert len(b.all_windows) == 1
    assert b.all_windows[0].log_count == 3


@pytest.mark.unit
def test_feed_closes_window_on_boundary_cross():
    b = WindowBuilder(window_size_seconds=60)
    b.feed(FakeParsed("a", "ns", 1), timestamp=1000.0)
    b.feed(FakeParsed("b", "ns", 1), timestamp=1030.0)
    # Cruza a la siguiente ventana (>= +60s) → devuelve la ventana cerrada
    closed = b.feed(FakeParsed("c", "ns", 2), timestamp=1065.0)
    assert closed is not None
    assert closed.index == 0
    assert closed.log_count == 2
    # La ventana actual es la nueva
    assert b.all_windows[-1].index == 1
    assert b.all_windows[-1].log_count == 1


@pytest.mark.unit
def test_feed_skips_empty_windows_indices():
    """Un salto temporal grande crea una ventana con índice mayor (no se rellenan huecos)."""
    b = WindowBuilder(window_size_seconds=60)
    b.feed(FakeParsed("a", "ns", 1), timestamp=0.0)
    closed = b.feed(FakeParsed("b", "ns", 1), timestamp=200.0)  # 3 ventanas después
    assert closed.index == 0
    assert b.all_windows[-1].index == 3


@pytest.mark.unit
def test_flush_returns_current_open_window():
    b = WindowBuilder(window_size_seconds=60)
    b.feed(FakeParsed("a", "ns", 1), timestamp=0.0)
    flushed = b.flush()
    assert flushed is not None
    assert flushed.log_count == 1
    # Tras flush, no hay ventana actual
    assert b.flush() is None
