"""
Gestion de ventanas temporales.

Agrupa ParsedLogs en WindowData por franja de tiempo.
"""

from dataclasses import dataclass, field

# Niveles considerados de error para la señal de severidad del detector.
_ERROR_LEVELS = {"ERROR", "FATAL", "CRITICAL"}
_MAX_ERROR_LOGS = 80  # tope de líneas de error que se guardan para el RCA


@dataclass
class WindowData:
    index: int
    start_time: float
    end_time: float
    raw_logs: list[str] = field(default_factory=list)
    namespaces: set[str] = field(default_factory=set)
    # frecuencia de cada cluster_id en esta ventana
    cluster_counts: dict[int, int] = field(default_factory=dict)
    # logs de nivel error/fatal/critical en esta ventana (señal de severidad)
    error_count: int = 0
    # muestra de las líneas de error (alta señal para el RCA), acotada
    error_logs: list[str] = field(default_factory=list)
    # namespaces que produjeron logs de error (los realmente implicados)
    error_namespaces: set[str] = field(default_factory=set)
    # conteos POR namespace: total y errores (para severidad local, no global)
    ns_log_counts: dict[str, int] = field(default_factory=dict)
    ns_error_counts: dict[str, int] = field(default_factory=dict)
    anomaly_score: float = 0.0
    is_anomaly: bool = False

    def add(self, parsed) -> None:
        self.raw_logs.append(parsed.raw)
        ns = parsed.namespace
        self.namespaces.add(ns)
        self.ns_log_counts[ns] = self.ns_log_counts.get(ns, 0) + 1
        cid = parsed.cluster_id
        self.cluster_counts[cid] = self.cluster_counts.get(cid, 0) + 1
        if getattr(parsed, "level", "").upper() in _ERROR_LEVELS:
            self.error_count += 1
            if ns:
                self.error_namespaces.add(ns)
                self.ns_error_counts[ns] = self.ns_error_counts.get(ns, 0) + 1
            if len(self.error_logs) < _MAX_ERROR_LOGS:
                self.error_logs.append(parsed.raw)

    @property
    def focus_namespaces(self) -> list[str]:
        """Namespaces realmente implicados en la anomalía.

        Si hay logs de error, son sus namespaces (el culpable); si no, todos los
        de la ventana. Evita atribuir la anomalía a todo el cluster cuando la
        ventana de 60s agrega logs de muchos namespaces.
        """
        src = self.error_namespaces if self.error_namespaces else self.namespaces
        return sorted(src)

    @property
    def log_count(self) -> int:
        return len(self.raw_logs)

    @property
    def template_count(self) -> int:
        return len(self.cluster_counts)

    @property
    def error_ratio(self) -> float:
        """Fracción de logs de nivel error en la ventana (0-1)."""
        n = self.log_count
        return self.error_count / n if n else 0.0


class WindowBuilder:
    """Acumula ParsedLogs y los agrupa en ventanas temporales fijas."""

    def __init__(self, window_size_seconds: float = 60.0):
        self.window_size = window_size_seconds
        self._windows: list[WindowData] = []
        self._t_start: float | None = None
        self._current: WindowData | None = None

    def feed(self, parsed, timestamp: float) -> WindowData | None:
        """
        Alimenta un log parseado.

        Retorna la ventana cerrada si el timestamp ha superado el limite,
        None si el log fue a la ventana actual (aun abierta).
        """
        if self._t_start is None:
            self._t_start = timestamp

        window_idx = int((timestamp - self._t_start) / self.window_size)

        if self._current is None or self._current.index != window_idx:
            closed = self._current
            self._current = WindowData(
                index=window_idx,
                start_time=self._t_start + window_idx * self.window_size,
                end_time=self._t_start + (window_idx + 1) * self.window_size,
            )
            self._windows.append(self._current)
            self._current.add(parsed)
            return closed  # devuelve la ventana que se cerro (si habia)

        self._current.add(parsed)
        return None

    def flush(self) -> WindowData | None:
        """Cierra y devuelve la ventana actual aunque no haya expirado."""
        w = self._current
        self._current = None
        return w

    @property
    def all_windows(self) -> list[WindowData]:
        return self._windows
