"""
Colector de eventos de Kubernetes via Watch API.

Watch abre una conexion HTTP larga (chunked transfer) contra el API server.
K8s empuja cada evento en cuanto ocurre — latencia ~0ms vs polling cada 10s.

El protocolo interno:
  1. GET /api/v1/events?watch=true&resourceVersion=<rv>
  2. K8s responde con un stream de objetos JSON (ADDED / MODIFIED / DELETED)
  3. Cada objeto incluye un nuevo resourceVersion
  4. Si la conexion se corta, reconectamos desde el ultimo resourceVersion
     para no perder eventos (garantia at-least-once del API server)

Produce LogEntry normalizadas para Capa 1.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from kubernetes import client, config as k8s_config, watch
from kubernetes.client.rest import ApiException


@dataclass
class LogEntry:
    timestamp: float      # epoch seconds
    namespace: str
    source: str           # "Pod/payment-service-7d4f"
    reason: str           # "CrashLoopBackOff", "Pulled", "OOMKilling"...
    message: str          # mensaje en bruto del evento
    raw: str              # linea normalizada para Drain3
    event_type: str = "ADDED"   # ADDED | MODIFIED | DELETED


class K8sCollector:
    def __init__(self, namespaces: list[str] | None = None, use_incluster: bool = False):
        """
        Args:
            namespaces:    lista de namespaces a monitorizar. None = todos.
            use_incluster: True si el pipeline corre dentro del propio cluster.
        """
        # Auto-detección: in-cluster (pod con ServiceAccount) o kubeconfig local.
        # use_incluster=True fuerza in-cluster; por defecto se intenta primero
        # in-cluster y se cae a kubeconfig si no hay token de ServiceAccount.
        if use_incluster:
            k8s_config.load_incluster_config()
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

        self._v1 = client.CoreV1Api()
        self.namespaces = namespaces

    # ------------------------------------------------------------------
    # Conversion de evento K8s → LogEntry
    # ------------------------------------------------------------------

    def _event_to_entry(self, event, event_type: str = "ADDED") -> LogEntry:
        ts = event.last_timestamp or event.event_time or datetime.now(timezone.utc)
        epoch = ts.timestamp() if hasattr(ts, "timestamp") else time.time()

        obj = event.involved_object
        source = f"{obj.kind}/{obj.name}" if obj else "unknown"
        reason = event.reason or "Unknown"
        message = event.message or ""
        ns = event.metadata.namespace or "default"

        return LogEntry(
            timestamp=epoch,
            namespace=ns,
            source=source,
            reason=reason,
            message=message,
            raw=f"{ns} {source} {reason} {message}",
            event_type=event_type,
        )

    # ------------------------------------------------------------------
    # Snapshot (modo replay)
    # ------------------------------------------------------------------

    def fetch_events_snapshot(self) -> list[LogEntry]:
        """
        Snapshot puntual de todos los eventos actuales.
        Usado en modo replay y para el bootstrap inicial del detector.
        """
        entries: list[LogEntry] = []
        namespaces = self._resolve_namespaces()

        for ns in namespaces:
            try:
                resp = self._v1.list_namespaced_event(namespace=ns, limit=500)
                for ev in resp.items:
                    entries.append(self._event_to_entry(ev))
            except ApiException:
                continue

        entries.sort(key=lambda e: e.timestamp)
        return entries

    # ------------------------------------------------------------------
    # Watch API (modo live)
    # ------------------------------------------------------------------

    def stream_events(self) -> Iterator[LogEntry]:
        """
        Stream continuo de eventos via Watch API.

        Emite un LogEntry por evento en cuanto K8s lo genera.
        Reconecta automaticamente si la conexion se interrumpe,
        retomando desde el ultimo resourceVersion visto.

        Uso:
            for entry in collector.stream_events():
                pipeline.ingest(entry)
        """
        namespaces = self._resolve_namespaces()

        # Un Watch por namespace en paralelo no es posible en un solo hilo,
        # pero si hay namespaces filtrados es mas eficiente que watch global.
        # Para "todos los namespaces" usamos list_event_for_all_namespaces.
        if self.namespaces is None:
            yield from self._watch_all_namespaces()
        else:
            yield from self._watch_namespaces(namespaces)

    def _watch_all_namespaces(self) -> Iterator[LogEntry]:
        """Watch global — un solo stream para todos los namespaces."""
        w = watch.Watch()
        resource_version = self._get_latest_resource_version()

        while True:
            try:
                stream = w.stream(
                    self._v1.list_event_for_all_namespaces,
                    resource_version=resource_version,
                    timeout_seconds=300,   # reconectar cada 5 min como maximo
                )
                for event in stream:
                    ev_type = event["type"]          # ADDED / MODIFIED / DELETED
                    ev_obj = event["object"]

                    # Actualizar resourceVersion para reconexion limpia
                    rv = ev_obj.metadata.resource_version
                    if rv:
                        resource_version = rv

                    # Solo ADDED y MODIFIED son relevantes para deteccion
                    if ev_type in ("ADDED", "MODIFIED"):
                        yield self._event_to_entry(ev_obj, ev_type)

            except ApiException as e:
                if e.status == 410:
                    # resourceVersion expirado del cache del API server → reset
                    resource_version = self._get_latest_resource_version()
                else:
                    time.sleep(2)
            except Exception:
                time.sleep(2)

    def _watch_namespaces(self, namespaces: list[str]) -> Iterator[LogEntry]:
        """
        Watch por namespace individual.
        Util cuando solo monitorizamos un subconjunto del cluster.
        Itera los namespaces de forma round-robin con timeout corto por cada uno.
        """
        resource_versions: dict[str, str] = {}
        watches: dict[str, watch.Watch] = {ns: watch.Watch() for ns in namespaces}

        while True:
            for ns in namespaces:
                rv = resource_versions.get(ns, "")
                w = watches[ns]
                try:
                    stream = w.stream(
                        self._v1.list_namespaced_event,
                        namespace=ns,
                        resource_version=rv,
                        timeout_seconds=5,   # timeout corto para rotar entre ns
                    )
                    for event in stream:
                        ev_type = event["type"]
                        ev_obj = event["object"]
                        new_rv = ev_obj.metadata.resource_version
                        if new_rv:
                            resource_versions[ns] = new_rv
                        if ev_type in ("ADDED", "MODIFIED"):
                            yield self._event_to_entry(ev_obj, ev_type)

                except ApiException as e:
                    if e.status == 410:
                        resource_versions.pop(ns, None)
                    else:
                        time.sleep(1)
                except Exception:
                    time.sleep(1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_namespaces(self) -> list[str]:
        if self.namespaces:
            return self.namespaces
        try:
            ns_list = self._v1.list_namespace()
            return [ns.metadata.name for ns in ns_list.items]
        except ApiException:
            return ["default"]

    def _get_latest_resource_version(self) -> str:
        """Obtiene el resourceVersion actual para arrancar el Watch desde 'ahora'."""
        try:
            resp = self._v1.list_event_for_all_namespaces(limit=1)
            return resp.metadata.resource_version or ""
        except ApiException:
            return ""
