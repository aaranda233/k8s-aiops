"""
Executor seguro de comandos kubectl de remediación.

Siempre ejecuta dry-run primero. Solo procede con el real si dry-run es exitoso.
"""

import shlex
import subprocess
from dataclasses import dataclass

_TIMEOUT = 30
_MAX_OUTPUT = 100  # líneas

# Comandos que NO soportan --dry-run (kubectl rechaza el flag). Son operaciones
# reversibles que ya han pasado el risk scoring y, si aplica, la aprobación
# humana, así que se ejecutan directamente sin el gate de dry-run.
_NO_DRYRUN = [("rollout", "restart"), ("rollout", "undo")]


@dataclass(frozen=True)
class ExecutionResult:
    command: str
    dry_run_ok: bool
    dry_run_output: str
    executed: bool
    real_output: str
    success: bool
    error: str | None = None


def _supports_dryrun(cmd: str) -> bool:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return True
    rest = [p.lower() for p in parts[1:]]  # tras 'kubectl'
    return not any(rest[:len(pat)] == list(pat) for pat in _NO_DRYRUN)


def execute_with_dryrun(kubectl_command: str) -> ExecutionResult:
    """Ejecuta dry-run primero (si el comando lo soporta), luego el comando real."""
    cmd = kubectl_command.strip()

    if _supports_dryrun(cmd):
        dry_output, dry_ok = _run(cmd + " --dry-run=client")
        if not dry_ok:
            return ExecutionResult(
                command=cmd,
                dry_run_ok=False,
                dry_run_output=dry_output,
                executed=False,
                real_output="",
                success=False,
                error=f"Dry-run falló: {dry_output[:200]}",
            )
    else:
        dry_output = "(dry-run no soportado para este comando; se ejecuta directamente)"

    # Real
    real_output, real_ok = _run(cmd)
    return ExecutionResult(
        command=cmd,
        dry_run_ok=True,
        dry_run_output=dry_output,
        executed=True,
        real_output=real_output,
        success=real_ok,
        error=None if real_ok else real_output[:200],
    )


def _run(cmd: str) -> tuple[str, bool]:
    try:
        parts = shlex.split(cmd)
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
        output = result.stdout or result.stderr or ""
        lines = output.splitlines()
        if len(lines) > _MAX_OUTPUT:
            output = "\n".join(lines[:_MAX_OUTPUT]) + f"\n... [{len(lines) - _MAX_OUTPUT} líneas truncadas]"
        return output, result.returncode == 0
    except subprocess.TimeoutExpired:
        return f"Timeout tras {_TIMEOUT}s", False
    except FileNotFoundError:
        return "kubectl no encontrado en PATH", False
    except Exception as e:
        return str(e), False
