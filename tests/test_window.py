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
def test_window_error_ratio_zero_without_errors():
    w = WindowData(index=0, start_time=0, end_time=60)
    w.add(FakeParsed("a", "ns", 1, level="INFO"))
    assert w.error_count == 0
    assert w.error_ratio == 0.0


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
