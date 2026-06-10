"""
Orquestador de auto-remediación.

Flujo completo:
  1. Circuit breaker — ¿demasiados intentos para esta anomalía?
  2. Risk scoring — clasifica el kubectl propuesto (0-3)
  3. Level 0 → nada (solo lectura, ya ejecutado en investigación)
     Level 1 → ejecutar automáticamente + verificar + notificar
     Level 2 → email con aprobación, esperar, ejecutar si aprobado
     Level 3 → email de alerta, no ejecutar
  4. Verificación post-fix — ¿desapareció la anomalía?
  5. Registro en MLflow

Se ejecuta en hilo de fondo para no bloquear el pipeline principal.
"""

import threading
import time
import uuid
from dataclasses import dataclass

from rich.console import Console

from src.diagnostics.ollama_rca import DiagnosisResult
from src.remediation.circuit_breaker import CircuitBreaker
from src.remediation.executor import ExecutionResult, execute_with_dryrun
from src.remediation.notifier import Notifier
from src.remediation.risk_scorer import score as risk_score

console = Console()

_VERIFY_WAIT_SECONDS = 90
_APPROVAL_TIMEOUT_SECONDS = 1800  # 30 minutos
_APPROVAL_POLL_INTERVAL = 10


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
        notifier: Notifier | None,
        max_auto_level: int = 1,
        verify_wait: int = _VERIFY_WAIT_SECONDS,
        approval_timeout: int = _APPROVAL_TIMEOUT_SECONDS,
    ):
        self.notifier = notifier
        self.max_auto_level = max_auto_level
        self.verify_wait = verify_wait
        self.approval_timeout = approval_timeout
        self._circuit = CircuitBreaker()

    def handle_async(self, scored_window, diagnosis: DiagnosisResult) -> None:
        """Lanza el loop de remediación en un hilo de fondo."""
        t = threading.Thread(
            target=self._handle,
            args=(scored_window, diagnosis),
            daemon=True,
        )
        t.start()

    def _handle(self, scored_window, diagnosis: DiagnosisResult) -> RemediationResult:
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        w = scored_window.window
        namespaces = w.namespaces
        root_cause = diagnosis.root_cause
        kubectl_cmd = diagnosis.kubectl_command

        # Pasos de investigación del trace si es modo hybrid/react
        investigation_steps = []
        for step in diagnosis.react_trace:
            if step.thought:
                investigation_steps.append(f"THOUGHT: {step.thought[:120]}")
            if step.action:
                investigation_steps.append(f"ACTION: {step.action}")

        fp = self._circuit.fingerprint(namespaces, root_cause)

        console.print(f"\n  [bold yellow][REMEDIATION][/] {incident_id} — {', '.join(namespaces)}")

        # 1. Circuit breaker
        blocked, attempts = self._circuit.is_blocked(fp)
        if blocked:
            console.print(f"  [red]Circuit breaker activo ({attempts} intentos)[/]")
            if self.notifier:
                self.notifier.notify_circuit_breaker(incident_id, namespaces, attempts, root_cause)
            return RemediationResult(incident_id, fp, -1, "blocked", kubectl_cmd, None, None)

        # 2. Risk scoring
        risk = risk_score(kubectl_cmd)
        console.print(f"  Risk: Level {risk.level} ({risk.label}) — {risk.reason}")
        console.print(f"  kubectl: [cyan]{kubectl_cmd}[/]")

        # 3. Acción según nivel de riesgo
        if risk.level == 0:
            console.print("  [dim]Level 0 — solo lectura, sin acción adicional[/]")
            return RemediationResult(incident_id, fp, 0, "skipped", kubectl_cmd, None, None)

        if risk.level == 1 and self.max_auto_level >= 1:
            return self._execute_level1(
                incident_id, fp, namespaces, root_cause, kubectl_cmd, investigation_steps
            )

        if risk.level == 2 and self.max_auto_level >= 2:
            return self._handle_level2(
                incident_id, fp, namespaces, root_cause, kubectl_cmd, risk.reason, investigation_steps
            )

        # Level 3 o nivel por encima del máximo configurado
        console.print(f"  [red]Level {risk.level} — escalando al humano[/]")
        if self.notifier:
            if risk.level == 3:
                self.notifier.notify_level3(incident_id, namespaces, root_cause, kubectl_cmd, investigation_steps)
            elif risk.level == 2:
                self.notifier.notify_level2_pending(
                    incident_id, namespaces, root_cause, kubectl_cmd, risk.reason, investigation_steps
                )
        return RemediationResult(incident_id, fp, risk.level, "skipped", kubectl_cmd, None, None)

    def _execute_level1(
        self,
        incident_id: str,
        fp: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        investigation_steps: list[str],
    ) -> RemediationResult:
        console.print("  [green]Level 1 — ejecutando automáticamente...[/]")
        result = execute_with_dryrun(kubectl_cmd)

        if not result.success:
            console.print(f"  [red]Ejecución fallida: {result.error}[/]")
            self._circuit.record(fp, kubectl_cmd, success=False)
            if self.notifier:
                self.notifier.notify_level3(
                    incident_id, namespaces,
                    f"EJECUCIÓN FALLIDA: {root_cause}",
                    kubectl_cmd, investigation_steps
                )
            return RemediationResult(incident_id, fp, 1, "executed", kubectl_cmd, result, False)

        console.print(f"  [green]Ejecutado OK. Verificando en {self.verify_wait}s...[/]")
        self._circuit.record(fp, kubectl_cmd, success=True)

        if self.notifier:
            self.notifier.notify_level1_executed(
                incident_id, namespaces, root_cause, kubectl_cmd,
                result.real_output, investigation_steps
            )

        verified = self._verify(kubectl_cmd, namespaces)
        if verified:
            console.print("  [bold green]✓ Anomalía resuelta[/]")
            self._circuit.reset(fp)
        else:
            console.print("  [yellow]⚠ Anomalía persiste tras el fix[/]")

        return RemediationResult(incident_id, fp, 1, "executed", kubectl_cmd, result, verified)

    def _handle_level2(
        self,
        incident_id: str,
        fp: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        risk_reason: str,
        investigation_steps: list[str],
    ) -> RemediationResult:
        console.print("  [yellow]Level 2 — esperando aprobación por email...[/]")

        if not self.notifier:
            console.print("  [dim]Sin notifier configurado — acción omitida[/]")
            return RemediationResult(incident_id, fp, 2, "skipped", kubectl_cmd, None, None)

        token = self.notifier.notify_level2_pending(
            incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps
        )

        # Polling con timeout
        deadline = time.time() + self.approval_timeout
        while time.time() < deadline:
            response = self.notifier.get_response(token)
            if response == "approved":
                console.print("  [green]Aprobado — ejecutando...[/]")
                result = execute_with_dryrun(kubectl_cmd)
                self._circuit.record(fp, kubectl_cmd, success=result.success)
                verified = self._verify(kubectl_cmd, namespaces) if result.success else False
                return RemediationResult(incident_id, fp, 2, "approved", kubectl_cmd, result, verified)
            if response == "rejected":
                console.print("  [yellow]Rechazado por el operador[/]")
                return RemediationResult(incident_id, fp, 2, "rejected", kubectl_cmd, None, None)
            time.sleep(_APPROVAL_POLL_INTERVAL)

        console.print("  [dim]Timeout de aprobación (30min)[/]")
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
