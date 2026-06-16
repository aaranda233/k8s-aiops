"""
Tests del flujo de ejecución y aprobación de la remediación.

Cubre _execute_level1 (ejecución + verificación) y _handle_level2 (polling de
decisión humana: aprobado / rechazado / timeout) sin polling real ni kubectl.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.remediation import auto_remediation as ar
from src.remediation.auto_remediation import AutoRemediation
from src.remediation.base_notifier import (
    KIND_EXECUTED,
    KIND_FAILED,
    KIND_RESOLVED,
)
from src.remediation.executor import ExecutionResult
from src.remediation.incident_store import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_RESOLVED,
    STATUS_TIMEOUT,
    Incident,
    IncidentStore,
)


def _ok_exec(cmd="kubectl rollout restart deployment/x -n prod"):
    return ExecutionResult(command=cmd, dry_run_ok=True, dry_run_output="dry ok",
                           executed=True, real_output="restarted", success=True)


def _fail_exec(cmd="kubectl rollout restart deployment/x -n prod"):
    return ExecutionResult(command=cmd, dry_run_ok=False, dry_run_output="err",
                           executed=False, real_output="", success=False, error="dry-run falló")


def _incident(store, iid="INC-TEST0001"):
    inc = Incident(
        id=iid, created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="memoria", kubectl_cmd="kubectl rollout restart deployment/x -n prod",
        risk_level=1, risk_label="reinicio", investigation=[], status=STATUS_PENDING,
    )
    store.add(inc)
    return inc


def _kinds(notifier):
    return [c.args[1] for c in notifier.notify.call_args_list]


# ── _execute_level1 ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_execute_level1_success_and_verified(monkeypatch):
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_exec(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    res = rem._execute_level1(inc, "fp1")
    assert res.action_taken == "executed"
    assert res.verified is True
    assert rem.incidents.get(inc.id).status == STATUS_RESOLVED
    assert KIND_EXECUTED in _kinds(notifier)
    assert KIND_RESOLVED in _kinds(notifier)


@pytest.mark.unit
def test_execute_level1_success_but_not_verified(monkeypatch):
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_exec(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: False)

    res = rem._execute_level1(inc, "fp1")
    assert res.verified is False
    assert rem.incidents.get(inc.id).status == STATUS_FAILED
    assert KIND_FAILED in _kinds(notifier)


@pytest.mark.unit
def test_execute_level1_dryrun_failure_records_circuit(monkeypatch):
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _fail_exec(cmd))

    res = rem._execute_level1(inc, "fp1")
    assert res.verified is False
    assert rem.incidents.get(inc.id).status == STATUS_FAILED
    assert KIND_FAILED in _kinds(notifier)


# ── _handle_level2 (decisión humana) ────────────────────────────────────────

@pytest.mark.unit
def test_level2_approved_executes_and_resolves(monkeypatch):
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    rem.incidents.set_response(inc.id, "approved")  # decisión ya tomada
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_exec(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    res = rem._handle_level2(inc, "fp2")
    assert res.action_taken == "approved"
    assert res.verified is True
    assert rem.incidents.get(inc.id).status == STATUS_RESOLVED


@pytest.mark.unit
def test_level2_approved_but_verify_fails(monkeypatch):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    rem.incidents.set_response(inc.id, "approved")
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_exec(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: False)

    res = rem._handle_level2(inc, "fp2")
    assert res.action_taken == "approved"
    assert rem.incidents.get(inc.id).status == STATUS_FAILED


@pytest.mark.unit
def test_level2_rejected(monkeypatch):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore())
    inc = _incident(rem.incidents)
    rem.incidents.set_response(inc.id, "rejected")
    spy = MagicMock(side_effect=AssertionError("rechazado NO debe ejecutar"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    res = rem._handle_level2(inc, "fp2")
    assert res.action_taken == "rejected"
    assert rem.incidents.get(inc.id).status == STATUS_REJECTED
    spy.assert_not_called()


@pytest.mark.unit
def test_level2_timeout_when_no_decision():
    # approval_timeout=0 → el bucle no entra y devuelve timeout sin dormir
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), approval_timeout=0)
    inc = _incident(rem.incidents)
    res = rem._handle_level2(inc, "fp2")
    assert res.action_taken == "timeout"
    assert rem.incidents.get(inc.id).status == STATUS_TIMEOUT


# ── _verify (comprobación post-fix) ─────────────────────────────────────────

class _FakeProc:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


@pytest.mark.unit
def test_verify_specific_resource_healthy(monkeypatch):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), verify_wait=0)
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeProc(stdout="3/3", returncode=0))
    assert rem._verify("kubectl rollout restart deployment/api -n prod", {"prod"}) is True


@pytest.mark.unit
def test_verify_specific_resource_unhealthy(monkeypatch):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), verify_wait=0)
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeProc(stdout="1/3", returncode=0))
    assert rem._verify("kubectl rollout restart deployment/api -n prod", {"prod"}) is False


@pytest.mark.unit
def test_verify_namespace_events_few_warnings(monkeypatch):
    rem = AutoRemediation(notifier=MagicMock(), incident_store=IncidentStore(), verify_wait=0)
    monkeypatch.setattr(ar.time, "sleep", lambda s: None)
    # Sin recurso específico → cuenta Warnings; pocos = sano
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _FakeProc(stdout="LAST SEEN TYPE\n", returncode=0))
    assert rem._verify("kubectl get events -n prod", {"prod"}) is True
