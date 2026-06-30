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
