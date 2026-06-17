"""
Tests de deduplicación de incidentes: un problema recurrente no crea un incidente
por ventana, sino que incrementa el contador del existente.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.remediation.auto_remediation import AutoRemediation
from src.remediation.incident_store import Incident, IncidentStore


def _inc(store, iid, namespaces, created_at=None):
    inc = Incident(
        id=iid, created_at=created_at or time.time(), namespaces=namespaces, score=0.9,
        root_cause="x", kubectl_cmd="kubectl get pods", risk_level=0, risk_label="lectura",
    )
    store.add(inc)
    return inc


# ── IncidentStore: find_recent_duplicate / bump ─────────────────────────────

@pytest.mark.unit
def test_find_recent_duplicate_same_namespaces():
    s = IncidentStore()
    _inc(s, "INC-1", ["postgresql"])
    dup = s.find_recent_duplicate({"postgresql"}, ttl_seconds=1800)
    assert dup is not None and dup.id == "INC-1"


@pytest.mark.unit
def test_no_duplicate_different_namespaces():
    s = IncidentStore()
    _inc(s, "INC-1", ["postgresql"])
    assert s.find_recent_duplicate({"longhorn-system"}, ttl_seconds=1800) is None


@pytest.mark.unit
def test_duplicate_expires_after_ttl():
    s = IncidentStore()
    _inc(s, "INC-1", ["postgresql"], created_at=time.time() - 4000)
    assert s.find_recent_duplicate({"postgresql"}, ttl_seconds=1800) is None


@pytest.mark.unit
def test_bump_increments_and_refreshes():
    s = IncidentStore()
    inc = _inc(s, "INC-1", ["postgresql"], created_at=time.time() - 1000)
    assert s.bump("INC-1") is True
    assert inc.occurrence_count == 2
    # last_seen se actualiza -> sigue siendo "reciente" (sliding window)
    assert s.find_recent_duplicate({"postgresql"}, ttl_seconds=1800).id == "INC-1"


# ── AutoRemediation: no crea duplicados ─────────────────────────────────────

@pytest.fixture
def scored_window():
    from types import SimpleNamespace
    w = SimpleNamespace(index=1, namespaces={"postgresql"}, log_count=10,
                        template_count=2, start_time=0, end_time=60, raw_logs=["e"],
                        focus_namespaces=["postgresql"])
    return SimpleNamespace(window=w, score=0.9, model_version=1)


def _diag(ns=("postgresql",)):
    from types import SimpleNamespace
    return SimpleNamespace(root_cause="rol inexistente", kubectl_command="kubectl get pods -n postgresql",
                           react_trace=[], namespaces=set(ns))


@pytest.mark.unit
def test_recurring_problem_deduped(scored_window):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), dedup_window=1800)
    r1 = rem._handle(scored_window, _diag())
    r2 = rem._handle(scored_window, _diag())
    r3 = rem._handle(scored_window, _diag())
    assert r2.action_taken == "deduped"
    assert r3.action_taken == "deduped"
    # Un solo incidente, con contador 3
    assert len(rem.incidents.list()) == 1
    inc = rem.incidents.get(r1.incident_id)
    assert inc.occurrence_count == 3


@pytest.mark.unit
def test_dedup_disabled_creates_each(scored_window):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), dedup_window=0)
    rem._handle(scored_window, _diag())
    rem._handle(scored_window, _diag())
    assert len(rem.incidents.list()) == 2  # sin dedup, uno por ventana


@pytest.mark.unit
def test_different_namespaces_not_deduped(scored_window):
    from types import SimpleNamespace
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), dedup_window=1800)
    rem._handle(scored_window, _diag(("postgresql",)))
    w2 = SimpleNamespace(index=2, namespaces={"longhorn-system"}, log_count=10, template_count=2,
                         start_time=0, end_time=60, raw_logs=["e"], focus_namespaces=["longhorn-system"])
    sw2 = SimpleNamespace(window=w2, score=0.9, model_version=1)
    rem._handle(sw2, _diag(("longhorn-system",)))
    assert len(rem.incidents.list()) == 2  # problemas distintos -> incidentes distintos
