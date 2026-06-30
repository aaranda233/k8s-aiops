"""
Executor seguro de comandos kubectl de remediación.

Siempre ejecuta dry-run primero. Solo procede con el real si dry-run es exitoso.
"""

import json
import shlex
import subprocess
from dataclasses import dataclass

_UNHEALTHY_WAITING = {
    "CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull",
    "CreateContainerConfigError", "RunContainerError", "CreateContainerError",
}

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


def execute_if_reversible(kubectl_command: str) -> "ExecutionResult | None":
    """Ejecuta solo si el comando es L0 (lectura) o L1 (reversible). Devuelve None
    si es L2/L3 (config/destructivo) — esos nunca se ejecutan automáticamente, ni
    siquiera tras aprobación. Es la barrera de seguridad del modo B."""
    from src.remediation.risk_scorer import score
    if score(kubectl_command).level >= 2:
        return None
    return execute_with_dryrun(kubectl_command)


def resolve_restart_target(namespace: str) -> str | None:
    """Encuentra el controlador real (deployment/statefulset/daemonset) del pod que
    está fallando en el namespace, para que el `rollout restart` apunte al workload
    que existe de verdad — y no a un nombre adivinado del texto del diagnóstico.

    Devuelve 'kind/name' (p. ej. 'deployment/inventory-api') o None si no hay un
    pod fallando con un controlador reiniciable.
    """
    out, ok = _run(f"kubectl get pods -n {namespace} -o json")
    if not ok:
        return None
    try:
        data = json.loads(out)
    except (ValueError, TypeError):
        return None

    worst = None
    worst_restarts = -1
    for pod in data.get("items", []):
        st = pod.get("status", {}) or {}
        css = st.get("containerStatuses", []) or []
        restarts = sum(cs.get("restartCount", 0) for cs in css)
        unhealthy = st.get("phase") not in ("Running", "Succeeded")
        for cs in css:
            waiting = (cs.get("state", {}) or {}).get("waiting") or {}
            if waiting.get("reason") in _UNHEALTHY_WAITING:
                unhealthy = True
            if not cs.get("ready", False) and cs.get("restartCount", 0) > 0:
                unhealthy = True
        if unhealthy and restarts >= worst_restarts:
            worst_restarts = restarts
            worst = pod

    if worst is None:
        return None
    owners = (worst.get("metadata", {}) or {}).get("ownerReferences", []) or []
    if not owners:
        return None
    kind, name = owners[0].get("kind"), owners[0].get("name")
    if kind == "ReplicaSet":
        # El owner real es el Deployment dueño del ReplicaSet.
        rout, rok = _run(
            f"kubectl get rs {name} -n {namespace} "
            "-o jsonpath={.metadata.ownerReferences[0].kind}/{.metadata.ownerReferences[0].name}"
        )
        if rok and "/" in rout:
            dkind, dname = rout.split("/", 1)
            return f"{dkind.lower()}/{dname}" if dname else None
        return None
    if kind in ("Deployment", "StatefulSet", "DaemonSet"):
        return f"{kind.lower()}/{name}"
    return None


def workload_exists(target: str, namespace: str) -> bool:
    """True si el workload `kind/name` (p. ej. 'deployment/oauth2-proxy') existe en
    el namespace. Lectura pura — usado para preferir el target del plan sobre el
    resolver heurístico cuando el plan ya nombró un workload real."""
    if not target or "/" not in target:
        return False
    out, ok = _run(f"kubectl get {target} -n {namespace} -o name")
    return ok and bool(out.strip())


def run_readonly(kubectl_command: str) -> ExecutionResult:
    """Ejecuta un comando de SOLO LECTURA directamente (sin dry-run; get/describe
    no aceptan --dry-run). Para los pasos de investigación del log de ejecución."""
    cmd = kubectl_command.strip()
    out, ok = _run(cmd)
    return ExecutionResult(
        command=cmd, dry_run_ok=True, dry_run_output="", executed=True,
        real_output=out, success=ok, error=None if ok else out[:200],
    )


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
