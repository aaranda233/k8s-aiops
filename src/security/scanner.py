"""
Escáner de postura de seguridad del cluster — read-only.

Recorre los recursos via API de Kubernetes (solo lectura, igual que
TopologyCollector) y aplica un conjunto de checks de seguridad de alto valor,
produciendo hallazgos con severidad. No instala nada ni modifica el cluster.

Checks (todos desde la API, sin agentes ni binarios externos):
  - Contenedores privilegiados
  - Ejecución como root (runAsUser 0)
  - hostNetwork / hostPID / hostIPC
  - Volúmenes hostPath
  - Capabilities peligrosas (SYS_ADMIN, NET_ADMIN, ALL...)
  - Imágenes con tag mutable (:latest o sin tag)
  - Secretos hardcodeados en variables de entorno
  - Sin límites de recursos (riesgo de noisy-neighbor / DoS)
  - ClusterRoleBindings a cluster-admin para sujetos no-sistema
  - Namespaces con cargas pero sin NetworkPolicy
"""

import time
from dataclasses import asdict, dataclass

from kubernetes import client
from kubernetes import config as k8s_config

SEV_CRITICAL = "critical"
SEV_HIGH = "high"
SEV_MEDIUM = "medium"
SEV_LOW = "low"
_SEV_ORDER = {SEV_CRITICAL: 0, SEV_HIGH: 1, SEV_MEDIUM: 2, SEV_LOW: 3}

_DANGEROUS_CAPS = {"SYS_ADMIN", "NET_ADMIN", "ALL", "SYS_PTRACE", "SYS_MODULE"}
_SECRET_HINTS = ("PASSWORD", "PASSWD", "SECRET", "TOKEN", "APIKEY", "API_KEY", "PRIVATE_KEY", "ACCESS_KEY")
_SYSTEM_NS = {"kube-system", "kube-public", "kube-node-lease"}


@dataclass
class Finding:
    severity: str
    category: str
    resource: str       # "pod/ns/name"
    kind: str
    namespace: str
    title: str
    detail: str
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


class SecurityScanner:
    def __init__(self, use_incluster: bool = False, cache_ttl: float = 30.0):
        if use_incluster:
            k8s_config.load_incluster_config()
        else:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
        self._v1 = client.CoreV1Api()
        self._rbac = client.RbacAuthorizationV1Api()
        self._net = client.NetworkingV1Api()
        self._cache_ttl = cache_ttl
        self._cache: dict | None = None
        self._cache_ts = 0.0

    def scan(self) -> dict:
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache
        result = self._scan()
        self._cache = result
        self._cache_ts = now
        return result

    def _scan(self) -> dict:
        findings: list[Finding] = []
        try:
            findings += self._scan_pods()
        except Exception:
            pass
        try:
            findings += self._scan_rbac()
        except Exception:
            pass
        try:
            findings += self._scan_netpol()
        except Exception:
            pass

        findings.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 9), f.namespace, f.resource))
        summary = {
            "critical": sum(1 for f in findings if f.severity == SEV_CRITICAL),
            "high": sum(1 for f in findings if f.severity == SEV_HIGH),
            "medium": sum(1 for f in findings if f.severity == SEV_MEDIUM),
            "low": sum(1 for f in findings if f.severity == SEV_LOW),
            "total": len(findings),
        }
        return {"findings": [f.to_dict() for f in findings], "summary": summary}

    # ------------------------------------------------------------------
    def _scan_pods(self) -> list[Finding]:
        out: list[Finding] = []
        pods = self._v1.list_pod_for_all_namespaces()
        for p in pods.items:
            ns = p.metadata.namespace
            name = p.metadata.name
            res = f"pod/{ns}/{name}"
            spec = p.spec
            if not spec:
                continue

            # host namespaces
            if spec.host_network:
                out.append(Finding(SEV_HIGH, "Aislamiento", res, "Pod", ns,
                    "hostNetwork habilitado",
                    "El pod comparte la red del nodo, saltándose el aislamiento de red.",
                    "Quita hostNetwork:true salvo que sea imprescindible (CNI, monitorización)."))
            if spec.host_pid:
                out.append(Finding(SEV_HIGH, "Aislamiento", res, "Pod", ns,
                    "hostPID habilitado", "El pod ve los procesos del nodo.",
                    "Quita hostPID:true."))
            if spec.host_ipc:
                out.append(Finding(SEV_MEDIUM, "Aislamiento", res, "Pod", ns,
                    "hostIPC habilitado", "El pod comparte IPC con el nodo.", "Quita hostIPC:true."))

            # hostPath volumes
            for v in (spec.volumes or []):
                if v.host_path:
                    out.append(Finding(SEV_HIGH, "Volúmenes", res, "Pod", ns,
                        f"Volumen hostPath: {v.host_path.path}",
                        "Monta una ruta del nodo; permite acceso/escape al host.",
                        "Sustituye hostPath por PVC, configMap o emptyDir."))
                    break

            containers = list(spec.containers or []) + list(spec.init_containers or [])
            for c in containers:
                sc = c.security_context
                # privileged
                if sc and sc.privileged:
                    out.append(Finding(SEV_CRITICAL, "Privilegios", res, "Pod", ns,
                        f"Contenedor privilegiado: {c.name}",
                        "Un contenedor privilegiado tiene acceso total al nodo.",
                        "Pon securityContext.privileged:false."))
                # runAsUser 0
                uid = (sc.run_as_user if sc else None)
                if uid is None and spec.security_context:
                    uid = spec.security_context.run_as_user
                if uid == 0:
                    out.append(Finding(SEV_HIGH, "Privilegios", res, "Pod", ns,
                        f"Ejecuta como root (uid 0): {c.name}",
                        "El contenedor corre como root; amplía el impacto de un compromiso.",
                        "Define runAsNonRoot:true y un runAsUser no-cero."))
                # capabilities peligrosas
                caps = (sc.capabilities.add if sc and sc.capabilities and sc.capabilities.add else [])
                bad = [cap for cap in caps if str(cap).upper() in _DANGEROUS_CAPS]
                if bad:
                    out.append(Finding(SEV_HIGH, "Privilegios", res, "Pod", ns,
                        f"Capabilities peligrosas en {c.name}: {', '.join(bad)}",
                        "Capabilities elevadas permiten operaciones a nivel de kernel/host.",
                        "Elimina las capabilities añadidas o usa drop:[ALL]."))
                # imagen :latest o sin tag
                img = c.image or ""
                tag = img.rsplit(":", 1)[-1] if ":" in img.rsplit("/", 1)[-1] else ""
                if not tag or tag == "latest":
                    out.append(Finding(SEV_MEDIUM, "Imágenes", res, "Pod", ns,
                        f"Imagen con tag mutable: {img}",
                        "Sin tag fijo no hay reproducibilidad ni control de versión desplegada.",
                        "Fija un tag inmutable o usa el digest (@sha256:...)."))
                # secretos hardcodeados en env
                for e in (c.env or []):
                    if e.value and any(h in (e.name or "").upper() for h in _SECRET_HINTS):
                        out.append(Finding(SEV_HIGH, "Secretos", res, "Pod", ns,
                            f"Posible secreto en variable de entorno: {e.name}",
                            "Un valor sensible está en claro en el manifiesto en vez de en un Secret.",
                            "Usa valueFrom.secretKeyRef en lugar de un valor literal."))
                        break
                # sin límites de recursos
                lim = c.resources.limits if c.resources else None
                if not lim or "memory" not in (lim or {}):
                    out.append(Finding(SEV_MEDIUM, "Recursos", res, "Pod", ns,
                        f"Sin límite de memoria: {c.name}",
                        "Sin límites, un contenedor puede agotar la memoria del nodo (noisy neighbor / DoS).",
                        "Define resources.limits.memory (y cpu)."))
        return out

    def _scan_rbac(self) -> list[Finding]:
        out: list[Finding] = []
        bindings = self._rbac.list_cluster_role_binding()
        for b in bindings.items:
            if not b.role_ref or b.role_ref.name != "cluster-admin":
                continue
            for s in (b.subjects or []):
                subj = f"{s.kind}/{s.namespace or ''}/{s.name}".replace("//", "/")
                # ignorar los del sistema
                if (s.name or "").startswith("system:") or (s.namespace in _SYSTEM_NS):
                    continue
                out.append(Finding(SEV_CRITICAL, "RBAC", f"clusterrolebinding/{b.metadata.name}",
                    "ClusterRoleBinding", s.namespace or "",
                    f"cluster-admin concedido a {subj}",
                    f"El binding '{b.metadata.name}' da control total del cluster a un sujeto no-sistema.",
                    "Sustituye cluster-admin por un Role con permisos mínimos necesarios."))
        return out

    def _scan_netpol(self) -> list[Finding]:
        out: list[Finding] = []
        pods = self._v1.list_pod_for_all_namespaces()
        ns_with_pods = {p.metadata.namespace for p in pods.items} - _SYSTEM_NS
        netpols = self._net.list_network_policy_for_all_namespaces()
        ns_with_np = {np.metadata.namespace for np in netpols.items}
        for ns in sorted(ns_with_pods - ns_with_np):
            out.append(Finding(SEV_LOW, "Red", f"namespace/{ns}", "Namespace", ns,
                f"Namespace sin NetworkPolicy: {ns}",
                "Sin NetworkPolicy, todo el tráfico entre pods está permitido por defecto.",
                "Aplica una NetworkPolicy default-deny y abre solo lo necesario."))
        return out
