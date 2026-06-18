"""
Orquestador de auto-remediación.

Flujo completo:
  1. Crear incidente en el IncidentStore (fuente de verdad de la consola)
  2. Circuit breaker — ¿demasiados intentos para esta anomalía?
  3. Risk scoring — clasifica el kubectl propuesto (0-3)
  4. Level 0 → resuelto (solo lectura)
     Level 1 → ejecutar automáticamente + verificar
     Level 2 → avisar y esperar decisión humana en la consola web
     Level 3 → avisar, no ejecutar (acción manual)
  5. Verificación post-fix — ¿desapareció la anomalía?

Las notificaciones (Teams/email) solo AVISAN con un link a /incidents/{id};
la decisión humana (aprobar/rechazar) ocurre en la consola web, que fija
incident.response — sobre el que este orquestador hace polling.

Se ejecuta en hilo de fondo para no bloquear el pipeline principal.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass

from rich.console import Console

from src.diagnostics.ollama_rca import DiagnosisResult
from src.remediation.base_notifier import (
    KIND_APPROVAL,
    KIND_CIRCUIT,
    KIND_EXECUTED,
    KIND_FAILED,
    KIND_MANUAL,
    KIND_RESOLVED,
    BaseNotifier,
)
from src.remediation.circuit_breaker import CircuitBreaker
from src.remediation.executor import ExecutionResult, execute_with_dryrun
from src.remediation.incident_store import (
    STATUS_APPROVED,
    STATUS_BLOCKED,
    STATUS_ESCALATED,
    STATUS_EXECUTED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_REJECTED,
    STATUS_RESOLVED,
    STATUS_TIMEOUT,
    Incident,
    IncidentStore,
)
from src.remediation.risk_scorer import score as risk_score

console = Console()
log = logging.getLogger("aiops.remediation")

_VERIFY_WAIT_SECONDS = 90
_APPROVAL_TIMEOUT_SECONDS = 1800  # 30 minutos
_APPROVAL_POLL_INTERVAL = 10

_EN_MARKERS = (" the ", " is ", " are ", " will ", " command ", " indicates", " issue")
_ES_MARKERS = (" el ", " la ", " de ", " que ", " los ", " un ", " está", " espacio", " memoria")


def _looks_spanish(text: str) -> bool:
    """Heurística simple: más indicios de español que de inglés (para la UI)."""
    t = f" {text.lower()} "
    return sum(m in t for m in _ES_MARKERS) >= sum(m in t for m in _EN_MARKERS)


@dataclass
class RemediationResult:
    incident_id: str
    fingerprint: str
    risk_level: int
    action_taken: str  # "executed" | "approved" | "rejected" | "timeout" | "blocked" | "skipped"
    kubectl_cmd: str
    execution: ExecutionResult | None
    verified: bool | None  # None = no verificable, True = resuelto, False = persiste


class AutoRemediation:
    def __init__(
        self,
        notifier: BaseNotifier | None,
        max_auto_level: int = 1,
        verify_wait: int = _VERIFY_WAIT_SECONDS,
        approval_timeout: int = _APPROVAL_TIMEOUT_SECONDS,
        incident_store: IncidentStore | None = None,
        shadow_mode: bool = False,
        dedup_window: int = 1800,
    ):
        self.notifier = notifier
        self.max_auto_level = max_auto_level
        # Modo sombra: NADA se ejecuta automáticamente; todo (incluido Level 1)
        # pasa por aprobación humana en la consola. Para validar en producción.
        self.shadow_mode = shadow_mode
        self.verify_wait = verify_wait
        self.approval_timeout = approval_timeout
        # Ventana de deduplicación: un mismo problema recurrente no crea un
        # incidente por ventana; incrementa el contador del existente.
        self.dedup_window = dedup_window
        self.incidents = incident_store or IncidentStore()
        self._circuit = CircuitBreaker()

    def _notify(self, incident: Incident, kind: str) -> None:
        if self.notifier:
            try:
                self.notifier.notify(incident, kind)
            except Exception as e:
                console.print(f"  [dim]notificación falló: {e}[/]")

    def handle_async(self, scored_window, diagnosis: DiagnosisResult) -> None:
        """Lanza el loop de remediación en un hilo de fondo."""
        t = threading.Thread(
            target=self._handle_safe,
            args=(scored_window, diagnosis),
            daemon=True,
        )
        t.start()

    def _handle_safe(self, scored_window, diagnosis: DiagnosisResult) -> None:
        # Garantiza que un fallo en el hilo de remediación no desaparezca en silencio.
        try:
            self._handle(scored_window, diagnosis)
        except Exception as e:
            log.exception("Fallo en el hilo de remediación")
            console.print(f"  [red]Fallo en remediación: {e}[/]")

    def register_failed_diagnosis(self, scored_window, error: str) -> str:
        """Crea un incidente aunque el diagnóstico haya fallado, para que la
        consola NUNCA se quede vacía ante una anomalía detectada."""
        import uuid as _uuid
        w = scored_window.window
        ns = sorted(getattr(w, "focus_namespaces", None) or w.namespaces)
        # Deduplicación también para fallos de diagnóstico recurrentes.
        if self.dedup_window > 0:
            dup = self.incidents.find_recent_duplicate(ns, self.dedup_window)
            if dup is not None:
                self.incidents.bump(dup.id)
                return dup.id
        incident_id = f"INC-{_uuid.uuid4().hex[:8].upper()}"
        inc = Incident(
            id=incident_id,
            created_at=time.time(),
            namespaces=ns,
            score=scored_window.score,
            root_cause=f"No se pudo diagnosticar automáticamente (error: {error}). "
                       f"Anomalía real detectada; revisa la ventana manualmente.",
            kubectl_cmd="kubectl get events --all-namespaces --sort-by='.lastTimestamp'",
            risk_level=2,
            risk_label="sin diagnóstico",
            investigation=[],
            status=STATUS_ESCALATED,
        )
        self.incidents.add(inc)
        log.warning("Incidente sin diagnóstico registrado: %s (%s)", incident_id, error)
        self._notify(inc, KIND_MANUAL)
        return incident_id

    def _handle(self, scored_window, diagnosis: DiagnosisResult) -> RemediationResult:
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        w = scored_window.window
        # Namespaces realmente implicados (los del diagnóstico, ya enfocados a los
        # logs de error), no todos los que la ventana de 60s agregó del cluster.
        namespaces = set(diagnosis.namespaces) if diagnosis.namespaces else set(w.namespaces)
        root_cause = diagnosis.root_cause
        kubectl_cmd = diagnosis.kubectl_command

        # Deduplicación: si el mismo problema (mismos namespaces) ya tiene un
        # incidente reciente, incrementamos su contador en vez de crear otro.
        if self.dedup_window > 0:
            dup = self.incidents.find_recent_duplicate(namespaces, self.dedup_window)
            if dup is not None:
                self.incidents.bump(dup.id)
                console.print(f"  [dim]Duplicado de {dup.id} (x{dup.occurrence_count}) — no se crea otro[/]")
                return RemediationResult(dup.id, dup.fingerprint, dup.risk_level,
                                         "deduped", dup.kubectl_cmd, None, None)

        # Pasos de investigación del trace si es modo hybrid/react. Las acciones
        # (comandos) se sanean; la prosa THOUGHT solo se muestra si está en español
        # (el modelo base investigador a veces piensa en inglés — ruido para la UI).
        from src.diagnostics.ollama_rca import sanitize_kubectl
        investigation_steps = []
        for step in diagnosis.react_trace:
            if step.thought and _looks_spanish(step.thought):
                investigation_steps.append(f"THOUGHT: {step.thought[:120]}")
            if step.action:
                investigation_steps.append(f"ACTION: {sanitize_kubectl(step.action)}")

        risk = risk_score(kubectl_cmd)
        fp = self._circuit.fingerprint(namespaces, root_cause)

        # Crear el incidente — única fuente de verdad para la consola
        incident = Incident(
            id=incident_id,
            created_at=time.time(),
            namespaces=sorted(namespaces),
            score=scored_window.score,
            root_cause=root_cause,
            kubectl_cmd=kubectl_cmd,
            risk_level=risk.level,
            risk_label=risk.label,
            investigation=investigation_steps,
            status=STATUS_PENDING,
            prompt_user=getattr(diagnosis, "prompt_user", ""),
            remediation_command=getattr(diagnosis, "remediation_command", ""),
            command_explanation=getattr(diagnosis, "command_explanation", ""),
            remediation_explanation=getattr(diagnosis, "remediation_explanation", ""),
        )
        self.incidents.add(incident)

        console.print(f"\n  [bold yellow][REMEDIATION][/] {incident_id} — {', '.join(namespaces)}")
        console.print(f"  Risk: Level {risk.level} ({risk.label}) — {risk.reason}")
        console.print(f"  kubectl: [cyan]{kubectl_cmd}[/]")

        # 1. Circuit breaker
        blocked, attempts = self._circuit.is_blocked(fp)
        if blocked:
            console.print(f"  [red]Circuit breaker activo ({attempts} intentos)[/]")
            self.incidents.update(incident_id, status=STATUS_BLOCKED)
            self._notify(incident, KIND_CIRCUIT)
            return RemediationResult(incident_id, fp, -1, "blocked", kubectl_cmd, None, None)

        # 2. Enrutar según nivel de riesgo
        if risk.level == 0:
            console.print("  [dim]Level 0 — solo lectura, sin acción adicional[/]")
            self.incidents.update(incident_id, status=STATUS_RESOLVED)
            return RemediationResult(incident_id, fp, 0, "skipped", kubectl_cmd, None, None)

        # Modo sombra: nada se auto-ejecuta; todo va a aprobación humana en la consola.
        if self.shadow_mode and risk.level in (1, 2):
            console.print("  [magenta]Modo sombra — esperando aprobación en la consola (no auto-ejecuta)[/]")
            return self._handle_level2(incident, fp)

        if risk.level == 1 and self.max_auto_level >= 1:
            return self._execute_level1(incident, fp)

        if risk.level == 2 and self.max_auto_level >= 2:
            return self._handle_level2(incident, fp)

        # Level 3, o Level 2 sin auto-nivel suficiente → escalar a la consola
        console.print(f"  [red]Level {risk.level} — escalando a la consola[/]")
        if risk.level == 2:
            self.incidents.update(incident_id, status=STATUS_PENDING)
            self._notify(incident, KIND_APPROVAL)
        else:
            self.incidents.update(incident_id, status=STATUS_ESCALATED)
            self._notify(incident, KIND_MANUAL)
        return RemediationResult(incident_id, fp, risk.level, "skipped", kubectl_cmd, None, None)

    def _execute_level1(self, incident: Incident, fp: str) -> RemediationResult:
        incident_id, kubectl_cmd = incident.id, incident.kubectl_cmd
        namespaces = set(incident.namespaces)
        console.print("  [green]Level 1 — ejecutando automáticamente...[/]")
        result = execute_with_dryrun(kubectl_cmd)

        if not result.success:
            console.print(f"  [red]Ejecución fallida: {result.error}[/]")
            self._circuit.record(fp, kubectl_cmd, success=False)
            self.incidents.update(
                incident_id, status=STATUS_FAILED,
                execution_output=result.error or result.dry_run_output, verified=False,
            )
            self._notify(self.incidents.get(incident_id), KIND_FAILED)
            return RemediationResult(incident_id, fp, 1, "executed", kubectl_cmd, result, False)

        console.print(f"  [green]Ejecutado OK. Verificando en {self.verify_wait}s...[/]")
        self._circuit.record(fp, kubectl_cmd, success=True)
        self.incidents.update(incident_id, status=STATUS_EXECUTED, execution_output=result.real_output)
        self._notify(self.incidents.get(incident_id), KIND_EXECUTED)

        verified = self._verify(kubectl_cmd, namespaces)
        if verified:
            console.print("  [bold green]✓ Anomalía resuelta[/]")
            self._circuit.reset(fp)
            self.incidents.update(incident_id, status=STATUS_RESOLVED, verified=True)
            self._notify(self.incidents.get(incident_id), KIND_RESOLVED)
        else:
            console.print("  [yellow]⚠ Anomalía persiste tras el fix[/]")
            self.incidents.update(incident_id, status=STATUS_FAILED, verified=False)
            self._notify(self.incidents.get(incident_id), KIND_FAILED)

        return RemediationResult(incident_id, fp, 1, "executed", kubectl_cmd, result, verified)

    def _handle_level2(self, incident: Incident, fp: str) -> RemediationResult:
        incident_id, kubectl_cmd = incident.id, incident.kubectl_cmd
        namespaces = set(incident.namespaces)
        console.print("  [yellow]Level 2 — esperando decisión humana en la consola...[/]")

        self.incidents.update(incident_id, status=STATUS_PENDING)
        self._notify(incident, KIND_APPROVAL)

        # Polling sobre el incident store: la consola web fija incident.response
        deadline = time.time() + self.approval_timeout
        while time.time() < deadline:
            current = self.incidents.get(incident_id)
            response = current.response if current else None
            if response == "approved":
                console.print("  [green]Aprobado — ejecutando...[/]")
                self.incidents.update(incident_id, status=STATUS_APPROVED)
                result = execute_with_dryrun(kubectl_cmd)
                self._circuit.record(fp, kubectl_cmd, success=result.success)
                verified = self._verify(kubectl_cmd, namespaces) if result.success else False
                self.incidents.update(
                    incident_id,
                    status=STATUS_RESOLVED if verified else STATUS_FAILED,
                    execution_output=result.real_output if result.success else (result.error or ""),
                    verified=verified,
                )
                self._notify(self.incidents.get(incident_id), KIND_RESOLVED if verified else KIND_FAILED)
                return RemediationResult(incident_id, fp, 2, "approved", kubectl_cmd, result, verified)
            if response == "rejected":
                console.print("  [yellow]Rechazado por el operador[/]")
                self.incidents.update(incident_id, status=STATUS_REJECTED)
                return RemediationResult(incident_id, fp, 2, "rejected", kubectl_cmd, None, None)
            time.sleep(_APPROVAL_POLL_INTERVAL)

        console.print("  [dim]Timeout de decisión (30min)[/]")
        self.incidents.update(incident_id, status=STATUS_TIMEOUT)
        return RemediationResult(incident_id, fp, 2, "timeout", kubectl_cmd, None, None)

    def _verify(self, kubectl_cmd: str, namespaces: set[str]) -> bool:
        """Espera y comprueba si el recurso afectado está sano."""
        time.sleep(self.verify_wait)
        ns = next(iter(namespaces), "default")

        # Extraer el tipo de recurso y nombre del comando ejecutado
        import shlex
        try:
            parts = shlex.split(kubectl_cmd)
        except ValueError:
            return False

        # Buscar patrones como "deployment/X" o "pod/X" en el comando
        resource = None
        for p in parts:
            if "/" in p and any(p.startswith(r) for r in
                                ["deployment/", "pod/", "daemonset/", "statefulset/"]):
                resource = p
                break

        if not resource:
            # Sin recurso específico — verificar eventos del namespace
            import subprocess
            result = subprocess.run(
                ["kubectl", "get", "events", "-n", ns,
                 "--field-selector=type=Warning",
                 "--sort-by=.lastTimestamp"],
                capture_output=True, text=True, timeout=15
            )
            # Si hay muchos Warning recientes, asumimos que persiste
            warning_lines = [l for l in result.stdout.splitlines() if l.strip()]
            return len(warning_lines) <= 3

        # Verificar el recurso específico
        import subprocess
        rtype, rname = resource.split("/", 1)
        result = subprocess.run(
            ["kubectl", "get", rtype, rname, "-n", ns, "-o",
             "jsonpath={.status.readyReplicas}/{.status.replicas}"],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        if "/" in output:
            ready, total = output.split("/")
            try:
                return int(ready) == int(total) and int(total) > 0
            except ValueError:
                pass
        return result.returncode == 0
