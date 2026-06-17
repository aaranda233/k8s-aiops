"""
Colector de LOGS de aplicación — fuente de señal rica para la detección.

A diferencia de K8sCollector (eventos del plano de control), este lee los logs
de stdout/stderr de los pods. Da señal continua: errores, trazas, picos de 500s,
deadlocks... cosas que no siempre generan un Evento de Kubernetes.

SEGURIDAD (diseño defensivo, no toca el cluster):
  - Solo lectura: read_namespaced_pod_log (un GET, nunca modifica nada)
  - Acotado: namespaces explícitos (NUNCA todo el cluster por defecto),
    tail_lines limitado, since_seconds (solo logs recientes), max_pods tope
  - Polling suave: lista pods + lee logs recientes cada poll_interval,
    en vez de abrir N streams persistentes que saturarían el API server
  - Opt-in: deshabilitado por defecto

Produce LogEntry (mismo formato que K8sCollector) → reutiliza Drain3 + IF sin cambios.
"""

import time
from collections.abc import Iterator

from kubernetes import client
from kubernetes import config as k8s_config
from kubernetes.client.rest import ApiException

from src.collector.k8s_collector import LogEntry

# Niveles de log detectables en el texto para clasificar el "reason"
_LEVELS = ("ERROR", "FATAL", "CRITICAL", "WARN", "WARNING", "INFO", "DEBUG")


class LogCollector:
    def __init__(
        self,
        namespaces: list[str] | None = None,
        poll_interval: float = 30.0,
        tail_lines: int = 50,
        since_seconds: int = 35,
        max_pods: int = 300,
        use_incluster: bool = False,
    ):
        # namespaces vacío/None = TODO el cluster (list_pod_for_all_namespaces).
        # Lista explícita = solo esos namespaces.
        self.namespaces = namespaces or []
        self.all_namespaces = not self.namespaces
        self.poll_interval = poll_interval
        self.tail_lines = tail_lines
        self.since_seconds = since_seconds
        self.max_pods = max_pods

        if use_incluster:
            k8s_config.load_incluster_config()
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
        self._v1 = client.CoreV1Api()

    def stream_log_entries(self) -> Iterator[LogEntry]:
        """Generador infinito: cada poll_interval lee los logs recientes de los pods."""
        while True:
            try:
                yield from self._poll_once()
            except Exception:
                # Nunca propagar: un fallo de lectura no debe tumbar el pipeline
                pass
            time.sleep(self.poll_interval)

    def _poll_once(self) -> Iterator[LogEntry]:
        pods = self._list_target_pods()
        for ns, pod_name, containers in pods[: self.max_pods]:
            # Pods multi-contenedor: leer cada contenedor (si no, la API rechaza
            # la lectura sin -c y el pod se saltaría entero).
            if len(containers) <= 1:
                yield from self._read_pod_logs(ns, pod_name, containers[0] if containers else None)
            else:
                for c in containers:
                    yield from self._read_pod_logs(ns, pod_name, c)

    def _list_target_pods(self) -> list[tuple[str, str, list[str]]]:
        result: list[tuple[str, str, list[str]]] = []

        def _containers(p):
            return [c.name for c in (p.spec.containers or [])] if p.spec else []

        if self.all_namespaces:
            # Todo el cluster en una sola llamada (read-only)
            try:
                pods = self._v1.list_pod_for_all_namespaces(limit=self.max_pods)
                for p in pods.items:
                    result.append((p.metadata.namespace, p.metadata.name, _containers(p)))
            except ApiException:
                pass
            return result
        for ns in self.namespaces:
            try:
                pods = self._v1.list_namespaced_pod(ns, limit=self.max_pods)
                for p in pods.items:
                    result.append((ns, p.metadata.name, _containers(p)))
            except ApiException:
                continue
        return result

    def _read_pod_logs(self, namespace: str, pod_name: str, container: str | None = None) -> Iterator[LogEntry]:
        try:
            # _preload_content=False evita un bug del cliente k8s que devuelve el
            # repr de bytes ("b'...\\n...'") en vez del texto. Leemos los bytes
            # reales de la respuesta y los decodificamos nosotros.
            kwargs = {} if container is None else {"container": container}
            resp = self._v1.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                since_seconds=self.since_seconds,
                tail_lines=self.tail_lines,
                timestamps=False,
                _preload_content=False,
                **kwargs,
            )
            raw = resp.data.decode("utf-8", errors="replace")
        except ApiException:
            # Pod sin logs, terminando, etc. → saltar
            return

        if not raw:
            return

        now = time.time()
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            yield LogEntry(
                timestamp=now,
                namespace=namespace,
                source=f"Pod/{pod_name}",
                reason=_detect_level(line),
                message=line,
                raw=f"{namespace} Pod/{pod_name} {line}",
                event_type="LOG",
            )

    def health_check(self) -> bool:
        try:
            if self.all_namespaces:
                self._v1.list_pod_for_all_namespaces(limit=1)
            else:
                self._v1.list_namespaced_pod(self.namespaces[0], limit=1)
            return True
        except Exception:
            return False


def _detect_level(line: str) -> str:
    """Clasifica la línea por nivel de log si es detectable, si no 'LOG'."""
    upper = line.upper()
    for lvl in _LEVELS:
        if lvl in upper:
            return lvl
    return "LOG"
