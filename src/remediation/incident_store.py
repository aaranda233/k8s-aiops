"""
Registro de incidentes en memoria — única fuente de verdad compartida
entre el orquestador de remediación y la consola web.

El orquestador crea incidentes y actualiza su estado; la web los lista,
muestra el detalle y registra la decisión humana (approve/reject).
La aprobación se identifica por incident_id (no por token aparte).
"""

import time
from dataclasses import asdict, dataclass, field

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

    def to_dict(self) -> dict:
        return asdict(self)


class IncidentStore:
    """Almacén thread-safe-enough para el caso de uso (dict + asignaciones atómicas)."""

    def __init__(self, max_incidents: int = 500):
        self._incidents: dict[str, Incident] = {}
        self._max = max_incidents

    def add(self, incident: Incident) -> None:
        incident.updated_at = incident.created_at
        self._incidents[incident.id] = incident
        self._evict_if_needed()

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
        return True

    def update(self, incident_id: str, **fields) -> None:
        inc = self._incidents.get(incident_id)
        if inc is None:
            return
        for k, v in fields.items():
            if hasattr(inc, k):
                setattr(inc, k, v)
        inc.updated_at = _now()

    def _evict_if_needed(self) -> None:
        if len(self._incidents) <= self._max:
            return
        # Eliminar los más antiguos
        oldest = sorted(self._incidents.values(), key=lambda i: i.created_at)
        for inc in oldest[: len(self._incidents) - self._max]:
            self._incidents.pop(inc.id, None)


def _now() -> float:
    return time.time()
