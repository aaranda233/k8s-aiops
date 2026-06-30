"""Tests del audit log durable de remediaciones."""

import pytest

from src.remediation.audit_log import RemediationAudit


@pytest.mark.unit
def test_audit_record_and_read(tmp_path):
    a = RemediationAudit(path=str(tmp_path / "audit.jsonl"))
    a.record(incident_id="INC-1", namespace="argocd",
             command="kubectl rollout restart deployment/x -n argocd",
             status="done", output="restarted", source="console", root_cause="proxy mal")
    a.record(incident_id="INC-2", namespace="db", command="kubectl rollout restart deployment/y -n db",
             status="manual", source="auto")
    rows = a.read_all()
    assert len(rows) == 2
    assert rows[0]["incident_id"] == "INC-1" and rows[0]["status"] == "done"
    assert rows[1]["status"] == "manual"
    assert all("ts" in r for r in rows)


@pytest.mark.unit
def test_audit_truncates_output(tmp_path):
    a = RemediationAudit(path=str(tmp_path / "audit.jsonl"))
    a.record(incident_id="INC-3", namespace="x", command="kubectl get pods",
             status="done", output="z" * 2000)
    assert len(a.read_all()[0]["output"]) <= 600


@pytest.mark.unit
def test_audit_empty_when_no_file(tmp_path):
    assert RemediationAudit(path=str(tmp_path / "none.jsonl")).read_all() == []


@pytest.mark.unit
def test_audit_prune_drops_old(tmp_path, monkeypatch):
    import time
    p = tmp_path / "audit.jsonl"
    a = RemediationAudit(path=str(p))
    a.record(incident_id="OLD", namespace="x", command="c", status="done")
    # envejecer ese registro 11 días reescribiendo su ts
    import json
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0]); rec["ts"] = time.time() - 11 * 86400
    p.write_text(json.dumps(rec) + "\n")
    a.record(incident_id="NEW", namespace="x", command="c", status="done")
    dropped = a.prune(max_age_days=10)
    rows = a.read_all()
    ids = [r["incident_id"] for r in rows]
    assert "OLD" not in ids and "NEW" in ids
    assert dropped >= 1


@pytest.mark.unit
def test_incident_log_prune_and_retention(tmp_path):
    import json, time
    from src.remediation.incident_log import IncidentLog
    log = IncidentLog(path=str(tmp_path / "inc.jsonl"))
    log.append_event({"id": "INC-OLD", "created_at": 0}, "created")
    log.append_event({"id": "INC-NEW", "created_at": 0}, "created")
    # envejecer la primera línea 11 días
    p = tmp_path / "inc.jsonl"
    lines = p.read_text().splitlines()
    r0 = json.loads(lines[0]); r0["logged_at"] = time.time() - 11 * 86400
    p.write_text(json.dumps(r0, ensure_ascii=False) + "\n" + lines[1] + "\n")
    dropped = log.prune(max_age_days=10)
    ids = [rec["incident"]["id"] for rec in log.read_all()]
    assert dropped == 1 and ids == ["INC-NEW"]
