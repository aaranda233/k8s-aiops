"""
Tests del orquestador de auto-remediación.

Verifica el enrutamiento por nivel de riesgo y las protecciones anti-bucle,
sin tocar un cluster real (executor y verificación mockeados).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.remediation import auto_remediation as ar
from src.remediation.auto_remediation import AutoRemediation
from src.remediation.executor import ExecutionResult


def _ok_execution(cmd="kubectl rollout restart deployment/x"):
    return ExecutionResult(
        command=cmd, dry_run_ok=True, dry_run_output="dry ok",
        executed=True, real_output="restarted", success=True,
    )


@pytest.mark.unit
def test_level0_skipped(scored_window, diagnosis):
    diagnosis.kubectl_command = "kubectl get pods -n producción"
    rem = AutoRemediation(notifier=None, max_auto_level=1)
    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 0
    assert result.action_taken == "skipped"


@pytest.mark.unit
def test_level1_executed_and_verified(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/scheduler -n producción"
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, max_auto_level=1)

    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_execution(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 1
    assert result.action_taken == "executed"
    assert result.verified is True
    notifier.notify_level1_executed.assert_called_once()


@pytest.mark.unit
def test_level1_failed_execution_escalates(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, max_auto_level=1)

    failed = ExecutionResult(
        command="x", dry_run_ok=False, dry_run_output="err",
        executed=False, real_output="", success=False, error="dry-run falló",
    )
    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: failed)

    result = rem._handle(scored_window, diagnosis)
    assert result.verified is False
    notifier.notify_level3.assert_called_once()  # escala al humano


@pytest.mark.unit
def test_level3_never_executed(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl delete pod scheduler -n producción"
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, max_auto_level=1)

    # Si se ejecutara algo, este mock lo detectaría
    spy = MagicMock(side_effect=AssertionError("¡Level 3 NO debe ejecutarse!"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 3
    assert result.action_taken == "skipped"
    spy.assert_not_called()
    notifier.notify_level3.assert_called_once()


@pytest.mark.unit
def test_level2_requires_approval_not_executed_when_max_level_1(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl set resources deployment/x --limits=memory=512Mi"
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, max_auto_level=1)  # max 1 → Level 2 no auto

    spy = MagicMock(side_effect=AssertionError("Level 2 no debe auto-ejecutarse con max_level=1"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.risk_level == 2
    spy.assert_not_called()
    # Con max_level=1, Level 2 escala vía email de aprobación
    notifier.notify_level2_pending.assert_called_once()


@pytest.mark.unit
def test_circuit_breaker_blocks_repeated_anomaly(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    notifier = MagicMock()
    rem = AutoRemediation(notifier=notifier, max_auto_level=1)

    # Pre-llenar el circuit breaker con 3 intentos
    fp = rem._circuit.fingerprint(scored_window.window.namespaces, diagnosis.root_cause)
    for _ in range(3):
        rem._circuit.record(fp, "kubectl rollout restart", success=False)

    spy = MagicMock(side_effect=AssertionError("No debe ejecutar con circuit breaker activo"))
    monkeypatch.setattr(ar, "execute_with_dryrun", spy)

    result = rem._handle(scored_window, diagnosis)
    assert result.action_taken == "blocked"
    spy.assert_not_called()
    notifier.notify_circuit_breaker.assert_called_once()


@pytest.mark.unit
def test_successful_fix_resets_circuit_breaker(scored_window, diagnosis, monkeypatch):
    diagnosis.kubectl_command = "kubectl rollout restart deployment/x -n producción"
    rem = AutoRemediation(notifier=MagicMock(), max_auto_level=1)

    fp = rem._circuit.fingerprint(scored_window.window.namespaces, diagnosis.root_cause)
    rem._circuit.record(fp, "cmd", success=False)
    rem._circuit.record(fp, "cmd", success=False)

    monkeypatch.setattr(ar, "execute_with_dryrun", lambda cmd: _ok_execution(cmd))
    monkeypatch.setattr(rem, "_verify", lambda cmd, ns: True)

    rem._handle(scored_window, diagnosis)
    # Tras verificación exitosa, el circuit breaker se resetea
    assert rem._circuit.is_blocked(fp) == (False, 0)
