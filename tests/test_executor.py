"""
Tests del executor seguro.

CRÍTICO: el dry-run debe ejecutarse SIEMPRE antes del comando real,
y si el dry-run falla, el comando real NUNCA debe ejecutarse.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.remediation import executor


def _proc(returncode=0, stdout="ok", stderr=""):
    m = MagicMock()
    m.returncode = returncode
    m.stdout = stdout
    m.stderr = stderr
    return m


@pytest.mark.unit
def test_dryrun_runs_before_real():
    with patch.object(executor.subprocess, "run") as mock_run:
        mock_run.side_effect = [_proc(0, "dry ok"), _proc(0, "real ok")]
        result = executor.execute_with_dryrun("kubectl rollout restart deployment/x -n prod")

    # Primera llamada debe contener --dry-run=client
    first_call_args = mock_run.call_args_list[0].args[0]
    assert "--dry-run=client" in first_call_args
    # Segunda llamada NO debe tener dry-run
    second_call_args = mock_run.call_args_list[1].args[0]
    assert "--dry-run=client" not in second_call_args
    assert result.executed is True
    assert result.success is True


@pytest.mark.unit
def test_real_command_not_executed_if_dryrun_fails():
    """El test de seguridad clave: dry-run fallido → no ejecuta real."""
    with patch.object(executor.subprocess, "run") as mock_run:
        mock_run.return_value = _proc(returncode=1, stderr="error de validación")
        result = executor.execute_with_dryrun("kubectl rollout restart deployment/x")

    # Solo se llamó UNA vez (el dry-run), nunca el real
    assert mock_run.call_count == 1
    assert result.dry_run_ok is False
    assert result.executed is False
    assert result.success is False
    assert result.error is not None


@pytest.mark.unit
def test_real_failure_reported():
    with patch.object(executor.subprocess, "run") as mock_run:
        mock_run.side_effect = [_proc(0, "dry ok"), _proc(1, stderr="conflicto")]
        result = executor.execute_with_dryrun("kubectl scale deployment/x --replicas=3")

    assert result.dry_run_ok is True
    assert result.executed is True
    assert result.success is False


@pytest.mark.unit
def test_timeout_handled_gracefully():
    import subprocess as sp
    with patch.object(executor.subprocess, "run") as mock_run:
        mock_run.side_effect = sp.TimeoutExpired(cmd="kubectl", timeout=30)
        result = executor.execute_with_dryrun("kubectl rollout restart deployment/x")
    assert result.success is False
    assert "Timeout" in (result.dry_run_output + (result.error or ""))


@pytest.mark.unit
def test_kubectl_not_found_handled():
    with patch.object(executor.subprocess, "run") as mock_run:
        mock_run.side_effect = FileNotFoundError()
        result = executor.execute_with_dryrun("kubectl rollout restart deployment/x")
    assert result.success is False
    assert result.executed is False
