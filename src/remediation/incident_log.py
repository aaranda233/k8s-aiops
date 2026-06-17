"""
Log durable de incidentes (append-only, JSONL).

El IncidentStore es solo en memoria: al reiniciar el proceso se pierden los
incidentes y, con ellos, las decisiones humanas y los resultados. Este log
persiste cada transición relevante (creación, decisión, estado terminal) a
disco de forma append-only e inmutable, para:
  - sobrevivir reinicios y dar trazabilidad temporal,
  - alimentar el bucle de aprendizaje (dataset de feedback).

Append-only: nunca se reescribe una línea, solo se añaden. Evita corrupción y
da un historial auditable.
"""

import json
import threading
import time
from pathlib import Path

# Estados que cierran el ciclo de vida de un incidente (señal de outcome).
TERMINAL_STATUSES = {
    "resolved", "failed", "rejected", "timeout", "escalated", "blocked",
}


class IncidentLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append_event(self, incident: dict, event_type: str) -> None:
        """Añade una línea con el snapshot del incidente y el tipo de evento."""
        record = {
            "event_type": event_type,        # created | response | terminal
            "logged_at": time.time(),
            "incident": incident,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def read_all(self) -> list[dict]:
        """Lee todos los registros (tolerante a líneas corruptas)."""
        if not self.path.exists():
            return []
        records: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def latest_incidents(self, limit: int = 200) -> list[dict]:
        """Último snapshot de cada incidente (por id), para rehidratar la consola."""
        by_id: dict[str, dict] = {}
        for rec in self.read_all():
            inc = rec.get("incident") or {}
            iid = inc.get("id")
            if iid:
                by_id[iid] = inc  # el último gana (append-only -> orden temporal)
        items = sorted(by_id.values(), key=lambda i: i.get("created_at", 0), reverse=True)
        return items[:limit]
