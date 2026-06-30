"""
Registro de incidentes en memoria — única fuente de verdad compartida
entre el orquestador de remediación y la consola web.

El orquestador crea incidentes y actualiza su estado; la web los lista,
muestra el detalle y registra la decisión humana (approve/reject).
La aprobación se identifica por incident_id (no por token aparte).
"""

import time
from dataclasses import asdict, dataclass, field

from src.remediation.incident_log import TERMINAL_STATUSES

# Estados del ciclo de vida de un incidente
STATUS_PENDING = "pending_approval"   # Level 2 esperando decisión humana
STATUS_APPROVED = "approved"          # humano aprobó, en ejecución
STATUS_REJECTED = "rejected"          # humano rechazó
STATUS_EXECUTED = "executed"          # Level 1 ejecutado automáticamente
STATUS_RESOLVED = "resolved"          # verificado: anomalía desapareció
STATUS_FAILED = "failed"              # ejecución o verificación falló
STATUS_ESCALATED = "escalated"        # Level 3 o requiere acción manual
STATUS_BLOCKED = "blocked"            # circuit breaker activo
STATUS_TIMEOUT = "timeout"            # sin respuesta humana a tiempo


@dataclass
class Incident:
    id: str
    created_at: float
    namespaces: list[str]
    score: float
    root_cause: str
    kubectl_cmd: str
    risk_level: int
    risk_label: str
    investigation: list[str] = field(default_factory=list)  # THOUGHT/ACTION del trace
    status: str = STATUS_PENDING
    execution_output: str = ""
    verified: bool | None = None
    response: str | None = None        # "approved" | "rejected" (decisión humana)
    updated_at: float = 0.0
    prompt_user: str = ""              # prompt/eventos de entrada del SLM (para reentrenar)
    human_correction: str = ""         # corrección humana opcional (root_cause + kubectl)
    remediation_command: str = ""      # acción reversible propuesta (shadow); "" si manual
    command_explanation: str = ""      # qué hace el comando de investigación (lenguaje natural)
    remediation_explanation: str = ""  # qué hace el comando de remediación
    remediation_guidance: str = ""     # guía de solución (texto) cuando no hay comando seguro
    remediation_plan: list = field(default_factory=list)  # plan multi-paso del grafo (si hit)
    solution_source: str = "catalog"   # 'graph' | 'catalog' | 'escalated'
    solution_key: str = ""             # clave del nodo del grafo (para verificación)
    category: str = "app"              # 'app' (código/config) | 'platform' (infra)
    occurrence_count: int = 1          # nº de veces que se ha repetido (deduplicación)
    last_seen: float = 0.0             # última vez que se observó el mismo problema
    execution_log: list = field(default_factory=list)  # pasos ejecutados en vivo: {order,type,command,output,status}
    manual_confirmed: bool = False     # el operador confirmó el paso manual previo a la acción

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Huella para deduplicar: el conjunto de namespaces implicados."""
        return ",".join(sorted(self.namespaces))


def plan_command(obj) -> str:
    """Comando de escritura ejecutable de una remediación: el paso 'command' del
    plan del grafo, o el remediation_command determinista como fallback. Devuelve
    '' si el plan es puramente manual (solo investigación + guía).

    Acepta cualquier objeto con atributos remediation_plan / remediation_command
    (un Incident o un DiagnosisResult).
    """
    for s in (getattr(obj, "remediation_plan", None) or []):
        if isinstance(s, dict) and s.get("action_type") == "command" and s.get("action"):
            return s["action"]
    return getattr(obj, "remediation_command", "") or ""


class IncidentStore:
    """Almacén thread-safe-enough para el caso de uso (dict + asignaciones atómicas)."""

    def __init__(self, max_incidents: int = 500, incident_log=None):
        self._incidents: dict[str, Incident] = {}
        self._max = max_incidents
        # Log durable opcional (persistencia + dataset de aprendizaje).
        self._log = incident_log
        self._feedback_hook = None  # callback(incident_dict) en estado terminal

    def set_feedback_hook(self, hook) -> None:
        """Registra un callback que se invoca con el incidente al llegar a terminal."""
        self._feedback_hook = hook

    def add(self, incident: Incident) -> None:
        incident.updated_at = incident.created_at
        incident.last_seen = incident.created_at
        self._incidents[incident.id] = incident
        self._evict_if_needed()
        self._record(incident, "created")

    def find_recent_duplicate(self, namespaces, ttl_seconds: float) -> Incident | None:
        """Incidente reciente con la misma huella (mismos namespaces) dentro del TTL.

        Evita crear un incidente nuevo por ventana ante un problema persistente:
        si el mismo conjunto de namespaces sigue fallando, se reutiliza el incidente
        existente (sliding window por last_seen).
        """
        fp = ",".join(sorted(namespaces))
        now = _now()
        candidates = [
            i for i in self._incidents.values()
            if i.fingerprint == fp and (now - max(i.last_seen, i.created_at)) <= ttl_seconds
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda i: max(i.last_seen, i.created_at))

    def bump(self, incident_id: str) -> bool:
        """Registra otra ocurrencia del mismo problema (deduplicación)."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.occurrence_count += 1
        inc.last_seen = _now()
        inc.updated_at = inc.last_seen
        return True

    def get(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    def list(self, limit: int = 100) -> list[Incident]:
        """Más recientes primero."""
        items = sorted(self._incidents.values(), key=lambda i: i.created_at, reverse=True)
        return items[:limit]

    def set_response(self, incident_id: str, response: str) -> bool:
        """Registra la decisión humana. Devuelve True si el incidente existe."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return False
        inc.response = response
        inc.updated_at = _now()
        self._record(inc, "response")
        return True

    def update(self, incident_id: str, **fields) -> None:
        inc = self._incidents.get(incident_id)
        if inc is None:
            return
        prev_status = inc.status
        for k, v in fields.items():
            if hasattr(inc, k):
                setattr(inc, k, v)
        inc.updated_at = _now()
        if inc.status != prev_status:
            # Persistir/feedback al ENTRAR en un estado terminal (señal de outcome).
            if inc.status in TERMINAL_STATUSES:
                self._record(inc, "terminal")
                if self._feedback_hook is not None:
                    try:
                        self._feedback_hook(inc.to_dict())
                    except Exception:
                        pass  # el feedback nunca debe tumbar la remediación
            # 'executed' no es terminal pero sí una transición a persistir: deja el
            # snapshot (con execution_log) en disco para el histórico completo.
            elif inc.status == "executed":
                self._record(inc, "executed")

    def set_execution_log(self, incident_id: str, log_list: list) -> None:
        """Reemplaza el log de ejecución paso a paso (lo consume la consola en vivo)."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return
        inc.execution_log = list(log_list)
        inc.updated_at = _now()

    def mark_executed(self, incident_id: str, output: str, success: bool) -> None:
        """Modo B: tras ejecutar la remediación aprobada. EXECUTED no es terminal
        (esperamos a la verificación por re-detección); si la ejecución falló →
        FAILED (terminal → verificación negativa)."""
        self.update(
            incident_id,
            execution_output=(output or "")[:2000],
            last_seen=_now(),
            status=STATUS_EXECUTED if success else STATUS_FAILED,
            verified=None if success else False,
        )

    def fail_executed(self, incident_id: str) -> None:
        """El problema reapareció tras ejecutar el fix → no funcionó → FAILED."""
        inc = self._incidents.get(incident_id)
        if inc is not None and inc.status == STATUS_EXECUTED:
            self.update(incident_id, status=STATUS_FAILED, verified=False)

    def sweep_resolved(self, grace_seconds: float) -> None:
        """Verificación por re-detección: un incidente EXECUTED que NO reaparece
        tras 'grace' (el detector no lo vuelve a marcar) → RESOLVED + verificado."""
        now = _now()
        for inc in list(self._incidents.values()):
            if inc.status == STATUS_EXECUTED and (now - inc.last_seen) >= grace_seconds:
                self.update(inc.id, status=STATUS_RESOLVED, verified=True)

    def _record(self, incident: Incident, event_type: str) -> None:
        if self._log is not None:
            try:
                self._log.append_event(incident.to_dict(), event_type)
            except Exception:
                pass  # la persistencia nunca debe tumbar la remediación

    def _evict_if_needed(self) -> None:
        if len(self._incidents) <= self._max:
            return
        # Eliminar los más antiguos
        oldest = sorted(self._incidents.values(), key=lambda i: i.created_at)
        for inc in oldest[: len(self._incidents) - self._max]:
            self._incidents.pop(inc.id, None)


def _now() -> float:
    return time.time()
