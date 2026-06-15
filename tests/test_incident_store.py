"""Tests del registro de incidentes."""

import time

import pytest

from src.remediation.incident_store import (
    STATUS_PENDING,
    STATUS_RESOLVED,
    Incident,
    IncidentStore,
)


def _inc(id="INC-1", created=None):
    return Incident(
        id=id, created_at=time.time() if created is None else created, namespaces=["prod"], score=0.9,
        root_cause="OOMKilled", kubectl_cmd="kubectl rollout restart deploy/x -n prod",
        risk_level=1, risk_label="reversible",
    )


@pytest.mark.unit
def test_add_and_get():
    s = IncidentStore()
    s.add(_inc("INC-1"))
    assert s.get("INC-1").id == "INC-1"
    assert s.get("nope") is None


@pytest.mark.unit
def test_list_most_recent_first():
    s = IncidentStore()
    s.add(_inc("INC-OLD", created=100))
    s.add(_inc("INC-NEW", created=200))
    ids = [i.id for i in s.list()]
    assert ids == ["INC-NEW", "INC-OLD"]


@pytest.mark.unit
def test_set_response():
    s = IncidentStore()
    s.add(_inc("INC-1"))
    assert s.set_response("INC-1", "approved") is True
    assert s.get("INC-1").response == "approved"
    assert s.set_response("missing", "approved") is False


@pytest.mark.unit
def test_update_fields():
    s = IncidentStore()
    s.add(_inc("INC-1"))
    s.update("INC-1", status=STATUS_RESOLVED, verified=True)
    inc = s.get("INC-1")
    assert inc.status == STATUS_RESOLVED
    assert inc.verified is True


@pytest.mark.unit
def test_to_dict_serializable():
    import json
    inc = _inc("INC-1")
    json.dumps(inc.to_dict())  # no debe lanzar


@pytest.mark.unit
def test_eviction_keeps_newest():
    s = IncidentStore(max_incidents=3)
    for i in range(5):
        s.add(_inc(f"INC-{i}", created=float(i)))
    ids = {i.id for i in s.list()}
    assert len(ids) == 3
    assert "INC-4" in ids and "INC-0" not in ids
