"""
Circuit breaker para prevenir bucles de remediación.

Si el mismo tipo de anomalía aparece N veces en una ventana de tiempo,
bloquea cualquier acción automática y fuerza escalación al humano.
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class _Attempt:
    timestamp: float
    kubectl_cmd: str
    success: bool


class CircuitBreaker:
    def __init__(
        self,
        max_attempts: int = 3,
        window_seconds: int = 600,  # 10 minutos
    ):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._history: dict[str, list[_Attempt]] = defaultdict(list)

    def fingerprint(self, namespaces: set[str], root_cause: str) -> str:
        """Genera una firma única para el tipo de anomalía."""
        key = "|".join(sorted(namespaces)) + "|" + root_cause[:60]
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def is_blocked(self, fp: str) -> tuple[bool, int]:
        """Devuelve (bloqueado, intentos_recientes)."""
        self._purge_old(fp)
        attempts = len(self._history[fp])
        return attempts >= self.max_attempts, attempts

    def record(self, fp: str, kubectl_cmd: str, success: bool) -> None:
        self._history[fp].append(
            _Attempt(timestamp=time.time(), kubectl_cmd=kubectl_cmd, success=success)
        )

    def reset(self, fp: str) -> None:
        """Resetear tras resolución confirmada."""
        self._history.pop(fp, None)

    def _purge_old(self, fp: str) -> None:
        cutoff = time.time() - self.window_seconds
        self._history[fp] = [a for a in self._history[fp] if a.timestamp > cutoff]
