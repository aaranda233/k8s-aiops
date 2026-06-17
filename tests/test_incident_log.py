"""
Tests del log durable de incidentes (append-only) y su enganche en IncidentStore.
"""

import time

import pytest

from src.remediation.incident_log import IncidentLog
from src.remediation.incident_store import (
    STATUS_PENDING,
    STATUS_RESOLVED,
    Incident,
    IncidentStore,
)


def _incident(iid="INC-1", status=STATUS_PENDING):
    return Incident(
        id=iid, created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="memoria", kubectl_cmd="kubectl rollout restart deployment/x -n prod",
        risk_level=1, risk_label="reversible", investigation=[], status=status,
    )


@pytest.mark.unit
def test_append_and_read(tmp_path):
    log = IncidentLog(str(tmp_path / "incidents.jsonl"))
    log.append_event({"id": "INC-1", "created_at": 1.0}, "created")
    log.append_event({"id": "INC-1", "created_at": 1.0}, "terminal")
    recs = log.read_all()
    assert len(recs) == 2
    assert recs[0]["event_type"] == "created"
    assert recs[1]["event_type"] == "terminal"
    assert recs[0]["incident"]["id"] == "INC-1"


@pytest.mark.unit
def test_read_all_tolerates_corrupt_lines(tmp_path):
    p = tmp_path / "incidents.jsonl"
    p.write_text('{"event_type":"created","incident":{"id":"A"}}\nNOT JSON\n{"event_type":"x","incident":{"id":"B"}}\n')
    log = IncidentLog(str(p))
    recs = log.read_all()
    assert len(recs) == 2  # la línea corrupta se salta


@pytest.mark.unit
def test_latest_incidents_dedups_by_id(tmp_path):
    log = IncidentLog(str(tmp_path / "i.jsonl"))
    log.append_event({"id": "INC-1", "created_at": 1.0, "status": "pending_approval"}, "created")
    log.append_event({"id": "INC-1", "created_at": 1.0, "status": "resolved"}, "terminal")
    latest = log.latest_incidents()
    assert len(latest) == 1
    assert latest[0]["status"] == "resolved"  # último snapshot gana


@pytest.mark.unit
def test_store_persists_created_response_and_terminal(tmp_path):
    log = IncidentLog(str(tmp_path / "i.jsonl"))
    store = IncidentStore(incident_log=log)
    store.add(_incident("INC-1"))
    store.set_response("INC-1", "approved")
    store.update("INC-1", status=STATUS_RESOLVED)

    events = [r["event_type"] for r in log.read_all()]
    assert events == ["created", "response", "terminal"]


@pytest.mark.unit
def test_store_terminal_only_logged_once(tmp_path):
    log = IncidentLog(str(tmp_path / "i.jsonl"))
    store = IncidentStore(incident_log=log)
    store.add(_incident("INC-1"))
    store.update("INC-1", status=STATUS_RESOLVED)
    store.update("INC-1", execution_output="x")  # ya terminal, mismo estado -> no re-loguea
    terminal = [r for r in log.read_all() if r["event_type"] == "terminal"]
    assert len(terminal) == 1


@pytest.mark.unit
def test_feedback_hook_fires_on_terminal(tmp_path):
    store = IncidentStore(incident_log=IncidentLog(str(tmp_path / "i.jsonl")))
    captured = []
    store.set_feedback_hook(lambda inc: captured.append(inc))
    store.add(_incident("INC-1"))
    assert captured == []                      # creación no dispara feedback
    store.update("INC-1", status=STATUS_RESOLVED)
    assert len(captured) == 1
    assert captured[0]["status"] == STATUS_RESOLVED


@pytest.mark.unit
def test_store_without_log_still_works(tmp_path):
    # Backward-compat: sin log, todo funciona igual (no persiste)
    store = IncidentStore()
    store.add(_incident("INC-1"))
    store.update("INC-1", status=STATUS_RESOLVED)
    assert store.get("INC-1").status == STATUS_RESOLVED
