"""
Constructor de la topología del cluster — el "cuadro eléctrico".

Lee la estructura del cluster en SOLO LECTURA y la convierte en un grafo
(nodos + enlaces) para visualizar cómo está montado todo:

    Internet → Ingress → Service → Pod → Node

Tipos de nodo: ingress · service · pod · node
Tipos de enlace: routes (ingress→service) · serves (service→pod) · runs-on (pod→node)

Cada nodo lleva un estado de salud (ok/warn/error) para colorear el mapa
y resaltar dónde hay un problema.

Solo lectura: 5 list_* (ingress, service, endpoints, pod, node). No modifica nada.
"""

import time

from kubernetes import client
from kubernetes import config as k8s_config

HEALTH_OK = "ok"
HEALTH_WARN = "warn"
HEALTH_ERROR = "error"
HEALTH_UNKNOWN = "unknown"

_ERROR_WAITING = {"CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerError", "Error"}


class TopologyCollector:
    def __init__(self, use_incluster: bool = False, cache_ttl: float = 15.0):
        if use_incluster:
            k8s_config.load_incluster_config()
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
        self._v1 = client.CoreV1Api()
        self._net = client.NetworkingV1Api()
        self._cache_ttl = cache_ttl
        self._cache: dict | None = None
        self._cache_ts = 0.0

    def build_graph(self) -> dict:
        """Devuelve {nodes, links, stats}. Cacheado cache_ttl segundos."""
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache
        graph = self._build()
        self._cache = graph
        self._cache_ts = now
        return graph

    def _build(self) -> dict:
        nodes: dict[str, dict] = {}
        links: list[dict] = []

        # --- Nodos del cluster (k8s nodes) ---
        try:
            for n in self._v1.list_node().items:
                nid = f"node/{n.metadata.name}"
                nodes[nid] = {
                    "id": nid, "type": "node", "label": n.metadata.name,
                    "namespace": "", "status": _node_status(n), "health": _node_health(n),
                }
        except Exception:
            pass

        # --- Pods (+ enlace pod→node) ---
        pod_index: dict[tuple[str, str], str] = {}  # (ns, pod_name) → node_id
        try:
            for p in self._v1.list_pod_for_all_namespaces().items:
                ns = p.metadata.namespace
                name = p.metadata.name
                pid = f"pod/{ns}/{name}"
                nodes[pid] = {
                    "id": pid, "type": "pod", "label": name, "namespace": ns,
                    "status": _pod_status(p), "health": _pod_health(p),
                }
                pod_index[(ns, name)] = pid
                node_name = p.spec.node_name if p.spec else None
                if node_name:
                    tgt = f"node/{node_name}"
                    if tgt in nodes:
                        links.append({"source": pid, "target": tgt, "kind": "runs-on"})
        except Exception:
            pass

        # --- Services (+ enlace service→pod vía endpoints) ---
        try:
            for s in self._v1.list_service_for_all_namespaces().items:
                ns = s.metadata.namespace
                name = s.metadata.name
                sid = f"svc/{ns}/{name}"
                svc_type = s.spec.type if s.spec else "ClusterIP"
                nodes[sid] = {
                    "id": sid, "type": "service", "label": name, "namespace": ns,
                    "status": svc_type, "health": HEALTH_OK,
                    "is_lb": svc_type == "LoadBalancer",
                }
        except Exception:
            pass

        # endpoints: qué pods respaldan cada service
        try:
            for ep in self._v1.list_endpoints_for_all_namespaces().items:
                ns = ep.metadata.namespace
                sid = f"svc/{ns}/{ep.metadata.name}"
                if sid not in nodes:
                    continue
                backed = False
                for subset in (ep.subsets or []):
                    for addr in (subset.addresses or []):
                        ref = addr.target_ref
                        if ref and ref.kind == "Pod":
                            pid = pod_index.get((ns, ref.name))
                            if pid:
                                links.append({"source": sid, "target": pid, "kind": "serves"})
                                backed = True
                # service sin endpoints → warn (no tiene pods detrás)
                if not backed:
                    nodes[sid]["health"] = HEALTH_WARN
                    nodes[sid]["status"] = nodes[sid]["status"] + " (sin endpoints)"
        except Exception:
            pass

        # --- Ingress (+ enlace ingress→service) ---
        try:
            for ing in self._net.list_ingress_for_all_namespaces().items:
                ns = ing.metadata.namespace
                name = ing.metadata.name
                iid = f"ing/{ns}/{name}"
                hosts = []
                for rule in (ing.spec.rules or []) if ing.spec else []:
                    if rule.host:
                        hosts.append(rule.host)
                    http = rule.http
                    for path in (http.paths or []) if http else []:
                        svc = path.backend.service if path.backend else None
                        if svc:
                            sid = f"svc/{ns}/{svc.name}"
                            if sid in nodes:
                                links.append({"source": iid, "target": sid, "kind": "routes"})
                nodes[iid] = {
                    "id": iid, "type": "ingress", "label": name, "namespace": ns,
                    "status": ", ".join(hosts[:3]) or "ingress", "health": HEALTH_OK,
                }
        except Exception:
            pass

        node_list = list(nodes.values())
        stats = {
            "nodes": sum(1 for n in node_list if n["type"] == "node"),
            "pods": sum(1 for n in node_list if n["type"] == "pod"),
            "services": sum(1 for n in node_list if n["type"] == "service"),
            "ingresses": sum(1 for n in node_list if n["type"] == "ingress"),
            "unhealthy": sum(1 for n in node_list if n.get("health") == HEALTH_ERROR),
        }
        return {"nodes": node_list, "links": links, "stats": stats}


def _node_status(n) -> str:
    for c in (n.status.conditions or []) if n.status else []:
        if c.type == "Ready":
            return "Ready" if c.status == "True" else "NotReady"
    return "Unknown"


def _node_health(n) -> str:
    return HEALTH_OK if _node_status(n) == "Ready" else HEALTH_ERROR


def _pod_status(p) -> str:
    phase = p.status.phase if p.status else "Unknown"
    for cs in (p.status.container_statuses or []) if p.status else []:
        waiting = cs.state.waiting if cs.state else None
        if waiting and waiting.reason:
            return waiting.reason
    return phase or "Unknown"


def _pod_health(p) -> str:
    if not p.status:
        return HEALTH_UNKNOWN
    phase = p.status.phase
    if phase == "Succeeded":
        return HEALTH_OK
    if phase == "Failed":
        return HEALTH_ERROR
    for cs in (p.status.container_statuses or []):
        waiting = cs.state.waiting if cs.state else None
        if waiting and waiting.reason in _ERROR_WAITING:
            return HEALTH_ERROR
        if cs.restart_count and cs.restart_count > 5:
            return HEALTH_WARN
    if phase == "Pending":
        return HEALTH_WARN
    if phase == "Running":
        # ¿todos los contenedores ready?
        statuses = p.status.container_statuses or []
        if statuses and all(cs.ready for cs in statuses):
            return HEALTH_OK
        return HEALTH_WARN
    return HEALTH_UNKNOWN
