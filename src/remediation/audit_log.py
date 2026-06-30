"""Audit log durable de remediaciones ejecutadas.

Histórico append-only (sobrevive a reinicios) de cada ACCIÓN de remediación que se
ejecuta — desde la consola (play paso a paso) o desde el orquestador automático. A
diferencia de `incident.execution_log` (que vive en memoria y solo se persiste de
forma parcial), esto deja una línea JSON por acción en `data/remediation/audit.jsonl`.

Cada registro: cuándo, qué incidente/namespace, el comando real ejecutado, su
resultado (done/failed/manual) y un extracto del output. Permite reconstruir qué se
hizo y verificarlo contra el cluster a posteriori (ver `scripts/remediation_audit.py`).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_DEFAULT_PATH = os.getenv("AIOPS_AUDIT_FILE", "data/remediation/audit.jsonl")
_RETENTION_DAYS = float(os.getenv("AIOPS_HISTORY_RETENTION_DAYS", "10"))
_PRUNE_INTERVAL_S = 3600


class RemediationAudit:
    def __init__(self, path: str = _DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_prune = 0.0
        self.prune()  # limpia lo viejo al arrancar

    def record(self, *, incident_id: str, namespace: str, command: str, status: str,
               output: str = "", source: str = "", root_cause: str = "") -> None:
        self._maybe_prune()
        rec = {
            "ts": time.time(),
            "incident_id": incident_id,
            "namespace": namespace,
            "command": command,
            "status": status,            # done | failed | manual
            "output": (output or "")[:600],
            "source": source,            # 'console' | 'auto'
            "root_cause": (root_cause or "")[:300],
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _maybe_prune(self) -> None:
        if time.time() - self._last_prune > _PRUNE_INTERVAL_S:
            self.prune()

    def prune(self, max_age_days: float = _RETENTION_DAYS) -> int:
        """Conserva solo registros de los últimos `max_age_days`; descarta lo anterior."""
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
                    kept.append(line)  # preserva líneas corruptas
                    continue
                ts = rec.get("ts")
                if ts is not None and ts < cutoff:
                    dropped += 1
                else:
                    kept.append(line)
            if dropped:
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
                tmp.replace(self.path)
            self._last_prune = time.time()
            return dropped

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out: list[dict] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


_AUDIT: RemediationAudit | None = None


def get_audit() -> RemediationAudit:
    global _AUDIT
    if _AUDIT is None:
        _AUDIT = RemediationAudit()
    return _AUDIT


def audit_action(**kwargs) -> None:
    """Atajo tolerante a fallos (nunca rompe la ejecución de remediación)."""
    try:
        get_audit().record(**kwargs)
    except Exception:
        pass
