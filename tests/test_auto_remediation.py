"""
Tests del orquestador de auto-remediación (interfaz basada en incidentes).

Verifica el enrutamiento por nivel de riesgo, el estado del incidente en el
store y las protecciones anti-bucle, sin tocar un cluster real.
"""

from unittest.mock import MagicMock

import pytest

from src.remediation import auto_remediation as ar
from src.remediation.auto_remediation import AutoRemediation
from src.remediation.base_notifier import (
    KIND_APPROVAL,
    KIND_CIRCUIT,
    KIND_EXECUTED,
    KIND_MANUAL,
    KIND_RESOLVED,
)
from src.remediation.executor import ExecutionResult
from src.remediation.incident_store import (
    STATUS_BLOCKED,
    STATUS_ESCALATED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RESOLVED,
    IncidentStore,
)


def _ok_execution(cmd="kubectl rollout restart deployment/x"):
    return ExecutionResult(
        command=cmd, dry_run_ok=True, dry_run_output="dry ok",
        executed=True, real_output="restarted", success=True,
    )


def _kinds(notifier_mock):
    """Tipos de aviso enviados (segundo argumento posicional de notify)."""
    return [c.args[1] for c in notifier_mock.notify.call_args_list]


def _rem(notifier, max_level=1):
    return AutoRemediation(notifier=notifier, max_auto_level=max_level, incident_store=IncidentStore())


@pytest.mark.unit
def test_level0_resolved(scored_window, diagnosis):
    diagnosis.kubectl_command = "kubectl get pods -n producción"
    rem = _rem(MagicMock())
    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 0
    inc = rem.incidents.get(result.incident_id)
    assert inc.status == STATUS_RESOLVED


@pytest.mark.unit
def test_level1_executed_and_verified(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/scheduler -n producción"
    notifier = MagicMock()
    rem = _rem(notifier)
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_execution(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 1
    assert result.verified is True
    assert rem.incidents.get(result.incident_id).status == STATUS_RESOLVED
    # Avisa de ejecución y de resolución
    assert KIND_EXECUTED in _kinds(notifier)
    assert KIND_RESOLVED in _kinds(notifier)


@pytest.mark.unit
def test_level1_failed_execution(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    notifier = MagicMock()
    rem = _rem(notifier)
    failed = ExecutionResult(command="x", dry_run_ok=False, dry_run_output="err",
                             executed=False, real_output="", success=False, error="dry-run falló")
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: failed)

    result = rem._handle(scored_window, diagnosis)
    assert result.verified is False
    assert rem.incidents.get(result.incident_id).status == STATUS_FAILED


@pytest.mark.unit
def test_level3_never_executed(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl delete pod scheduler -n producción"
    notifier = MagicMock()
    rem = _rem(notifier)
    spy = MagicMock(side_effect=AssertionError("¡Level 3 NO debe ejecutarse!"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 3
    spy.assert_not_called()
    assert rem.incidents.get(result.incident_id).status == STATUS_ESCALATED
    assert KIND_MANUAL in _kinds(notifier)


@pytest.mark.unit
def test_level2_escalates_to_console_not_executed(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl set resources deployment/x --limits=memory=512Mi"
    notifier = MagicMock()
    rem = _rem(notifier, max_level=1)  # max 1 → Level 2 no auto
    spy = MagicMock(side_effect=AssertionError("Level 2 no debe auto-ejecutarse con max_level=1"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 2
    spy.assert_not_called()
    assert rem.incidents.get(result.incident_id).status == STATUS_PENDING
    assert KIND_APPROVAL in _kinds(notifier)


@pytest.mark.unit
def test_circuit_breaker_blocks(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    notifier = MagicMock()
    rem = _rem(notifier)
    fp = rem._circuit.fingerprint(scored_window.window.namespaces, diagnosis.root_cause)
    for _ in range(3):
        rem._circuit.record(fp, "kubectl rollout restart", success=False)
    spy = MagicMock(side_effect=AssertionError("No debe ejecutar con circuit breaker activo"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.action_taken == "blocked"
    spy.assert_not_called()
    assert rem.incidents.get(result.incident_id).status == STATUS_BLOCKED
    assert KIND_CIRCUIT in _kinds(notifier)


@pytest.mark.unit
def test_successful_fix_resets_circuit_breaker(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    rem = _rem(MagicMock())
    fp = rem._circuit.fingerprint(scored_window.window.namespaces, diagnosis.root_cause)
    rem._circuit.record(fp, "cmd", success=False)
    rem._circuit.record(fp, "cmd", success=False)
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_execution(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    rem._handle(scored_window, diagnosis)
    assert rem._circuit.is_blocked(fp) == (False, 0)


@pytest.mark.unit
def test_shadow_mode_level1_routes_to_approval(scored_window, diagnosis, monkeypatch):
    """En modo sombra, un Level 1 NO se auto-ejecuta: se enruta a aprobación humana."""
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    rem = AutoRemediation(notifier=MagicMock(), max_auto_level=1,
                          incident_store=IncidentStore(), shadow_mode=True)
    # No ejecutamos el bucle real de polling; verificamos el enrutamiento
    routed = {}
    monkeypatch.setattr(rem, "_handle_level2",
                        lambda inc, fp: routed.update(id=inc.id) or
                        RemediationResultStub(inc.id))
    exec_spy = MagicMock(side_effect=AssertionError("Modo sombra NO debe auto-ejecutar"))
    monkeypatch.setattr(ar, "execute_with_dryrun", exec_spy)

    rem._handle(scored_window, diagnosis)
    assert "id" in routed              # un Level 1 acabó en la vía de aprobación
    exec_spy.assert_not_called()       # nada se ejecutó automáticamente


class RemediationResultStub:
    def __init__(self, incident_id):
        self.incident_id = incident_id; self.action_taken = "pending"; self.risk_level = 1


@pytest.mark.unit
def test_incident_registered_in_store(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl get pods -n producción"
    rem = _rem(MagicMock())
    result = rem._handle(scored_window, diagnosis)
    # El incidente queda en el store para que la consola lo liste
    listed = rem.incidents.list()
    assert any(i.id == result.incident_id for i in listed)
    inc = rem.incidents.get(result.incident_id)
    assert inc.root_cause == diagnosis.root_cause
    assert inc.investigation  # trae los pasos de investigación
