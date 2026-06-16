"""
Tests de la conversión evento K8s → LogEntry (src/collector/k8s_collector.py).

Solo la lógica de normalización (sin conexión al cluster): la instancia se crea
con __new__ para no ejecutar el __init__ que carga el kubeconfig.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.collector.k8s_collector import K8sCollector, LogEntry


def _collector():
    # Evita __init__ (que cargaría kubeconfig); solo probamos _event_to_entry.
    return K8sCollector.__new__(K8sCollector)


def _event(ns="prod", kind="Pod", name="api-1", reason="CrashLoopBackOff",
           message="Back-off restarting", ts=None):
    return SimpleNamespace(
        last_timestamp=ts or datetime(2026, 6, 16, 7, 0, tzinfo=timezone.utc),
        event_time=None,
        involved_object=SimpleNamespace(kind=kind, name=name),
        reason=reason,
        message=message,
        metadata=SimpleNamespace(namespace=ns),
    )


@pytest.mark.unit
def test_event_to_entry_normalizes_fields():
    c = _collector()
    entry = c._event_to_entry(_event())
    assert isinstance(entry, LogEntry)
    assert entry.namespace == "prod"
    assert entry.source == "Pod/api-1"
    assert entry.reason == "CrashLoopBackOff"
    assert entry.raw == "prod Pod/api-1 CrashLoopBackOff Back-off restarting"
    assert entry.event_type == "ADDED"


@pytest.mark.unit
def test_event_to_entry_uses_timestamp_epoch():
    c = _collector()
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    entry = c._event_to_entry(_event(ts=ts))
    assert entry.timestamp == ts.timestamp()


@pytest.mark.unit
def test_event_to_entry_handles_missing_fields():
    c = _collector()
    ev = SimpleNamespace(
        last_timestamp=None, event_time=None, involved_object=None,
        reason=None, message=None, metadata=SimpleNamespace(namespace=None),
    )
    entry = c._event_to_entry(ev)
    assert entry.source == "unknown"
    assert entry.reason == "Unknown"
    assert entry.message == ""
    assert entry.namespace == "default"


@pytest.mark.unit
def test_event_to_entry_respects_event_type():
    c = _collector()
    entry = c._event_to_entry(_event(), event_type="MODIFIED")
    assert entry.event_type == "MODIFIED"
