"""
Toolbox de kubectl de solo lectura para el agente ReAct.
Rechaza cualquier operación de escritura antes de ejecutar.
"""

import shlex
import subprocess
from dataclasses import dataclass

_ALLOWED_VERBS = {"describe", "get", "logs", "top"}
_FORBIDDEN_VERBS = {
    "apply", "create", "delete", "patch", "replace",
    "edit", "scale", "rollout", "exec", "port-forward",
    "drain", "cordon", "uncordon", "taint", "label", "annotate",
}
_FORBIDDEN_FLAGS = {"-f", "--filename", "--force", "--dry-run=none", "--overwrite"}
_MAX_OUTPUT_LINES = 150
_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class ToolResult:
    command: str
    stdout: str
    returncode: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def execute(raw_command: str, max_lines: int = _MAX_OUTPUT_LINES) -> ToolResult:
    raw_command = raw_command.strip()

    try:
        parts = shlex.split(raw_command)
    except ValueError as e:
        return ToolResult(command=raw_command, stdout="", returncode=1, error=f"Parse error: {e}")

    if not parts or parts[0] != "kubectl":
        return ToolResult(command=raw_command, stdout="", returncode=1, error="Solo se permiten comandos kubectl")

    if len(parts) < 2:
        return ToolResult(command=raw_command, stdout="", returncode=1, error="Falta el verbo kubectl")

    verb = parts[1].lower()

    if verb in _FORBIDDEN_VERBS:
        return ToolResult(
            command=raw_command, stdout="", returncode=1,
            error=f"Verbo '{verb}' prohibido — modo solo lectura"
        )

    for part in parts:
        if part.lower() in _FORBIDDEN_FLAGS:
            return ToolResult(
                command=raw_command, stdout="", returncode=1,
                error=f"Flag '{part}' no permitida"
            )

    try:
        proc = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        stdout = proc.stdout or ""
        lines = stdout.splitlines()
        if len(lines) > max_lines:
            stdout = "\n".join(lines[:max_lines])
            stdout += f"\n... [truncados {len(lines) - max_lines} líneas]"

        return ToolResult(
            command=raw_command,
            stdout=stdout if stdout else proc.stderr[:500],
            returncode=proc.returncode,
            error=proc.stderr[:200] if proc.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            command=raw_command, stdout="", returncode=1,
            error=f"Timeout tras {_TIMEOUT_SECONDS}s"
        )
    except FileNotFoundError:
        return ToolResult(
            command=raw_command, stdout="", returncode=1,
            error="kubectl no encontrado en PATH"
        )
    except Exception as e:
        return ToolResult(command=raw_command, stdout="", returncode=1, error=str(e))
