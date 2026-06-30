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
import os
import threading
import time
from pathlib import Path

# Estados que cierran el ciclo de vida de un incidente (señal de outcome).
TERMINAL_STATUSES = {
    "resolved", "failed", "rejected", "timeout", "escalated", "blocked",
}

# Retención: se conservan los últimos N días; lo anterior se purga (configurable).
RETENTION_DAYS = float(os.getenv("AIOPS_HISTORY_RETENTION_DAYS", "10"))
_PRUNE_INTERVAL_S = 3600  # purga como mucho una vez por hora


class IncidentLog:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_prune = 0.0
        self.prune()  # limpia lo viejo al arrancar

    def append_event(self, incident: dict, event_type: str) -> None:
        """Añade una línea con el snapshot del incidente y el tipo de evento."""
        self._maybe_prune()
        record = {
            "event_type": event_type,        # created | response | executed | terminal
            "logged_at": time.time(),
            "incident": incident,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _maybe_prune(self) -> None:
        if time.time() - self._last_prune > _PRUNE_INTERVAL_S:
            self.prune()

    def prune(self, max_age_days: float = RETENTION_DAYS) -> int:
        """Reescribe el log conservando solo registros de los últimos `max_age_days`.
        Devuelve cuántos se descartaron. Atómico (escribe a .tmp y reemplaza)."""
        if not self.path.exists():
            self._last_prune = time.time()
            return 0
        cutoff = time.time() - max_age_days * 86400
        with self._lock:
            kept, dropped = [], 0
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)  # preserva líneas corruptas (no perder datos)
                    continue
                ts = rec.get("logged_at")
                if ts is not None and ts < cutoff:
                    dropped += 1            # solo descarta lo confirmadamente viejo
                else:
                    kept.append(line)
            if dropped:
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
                tmp.replace(self.path)
            self._last_prune = time.time()
            return dropped

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
