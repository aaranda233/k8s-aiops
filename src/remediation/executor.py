"""
Executor seguro de comandos kubectl de remediación.

Siempre ejecuta dry-run primero. Solo procede con el real si dry-run es exitoso.
"""

import shlex
import subprocess
from dataclasses import dataclass

_TIMEOUT = 30
_MAX_OUTPUT = 100  # líneas


@dataclass(frozen=True)
class ExecutionResult:
    command: str
    dry_run_ok: bool
    dry_run_output: str
    executed: bool
    real_output: str
    success: bool
    error: str | None = None


def execute_with_dryrun(kubectl_command: str) -> ExecutionResult:
    """Ejecuta primero dry-run, luego el comando real si todo va bien."""
    cmd = kubectl_command.strip()

    # Dry-run
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
