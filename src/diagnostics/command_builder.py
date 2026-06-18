"""
Constructor determinista de comandos kubectl para el RCA.

Problema: el SLM de 1.5B propone comandos con namespace equivocado, substituciones
frágiles ($(...)), placeholders o irrelevantes a la causa. Igual que con la causa
raíz, garantizamos la calidad por post-proceso determinista en vez de confiar en
la varianza del modelo:

  Fase 1 — guardarraíles: extraer el recurso real de la evidencia, forzar el
           namespace al culpable, rechazar comandos frágiles.
  Fase 2 — catálogo intención→comando: mapear el patrón de fallo al comando de
           investigación correcto (verbo alineado con el escenario).
  Fase 3 — remediación opcional: una acción reversible (rollout restart) etiquetada
           por riesgo, en shadow.

El builder trabaja sobre el TEXTO de evidencia (la muestra de eventos), así que
funciona tanto en producción (ventana) como en evaluación (mensaje de usuario).
"""

import re

_DEFAULT = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"

# ── Extracción de recursos desde la evidencia ───────────────────────────────

_POD_RE = re.compile(r"Pod/([a-z0-9][a-z0-9.\-]*)", re.IGNORECASE)
_NODE_RE = re.compile(r"\bNode\s+([a-z0-9][a-z0-9.\-]*)", re.IGNORECASE)
_NODE_FALLBACK_RE = re.compile(r"\b(node-?\d+)\b", re.IGNORECASE)
_PVC_RE = re.compile(
    r"\b(pvc-[a-z0-9][a-z0-9.\-]*|[a-z0-9][a-z0-9.\-]*-pvc)\b", re.IGNORECASE
)
_SVC_RE = re.compile(r"\bService[/ ]([a-z0-9][a-z0-9.\-]*)", re.IGNORECASE)


def extract_pod(evidence: str) -> str | None:
    m = _POD_RE.search(evidence or "")
    return m.group(1) if m else None


def extract_node(evidence: str) -> str | None:
    # Primero el patrón claro de nombre de nodo (node-1, node-12...).
    m = _NODE_FALLBACK_RE.search(evidence or "")
    if m:
        return m.group(1)
    # "Node <name>" solo si el nombre parece un host (tiene dígito o punto), para
    # no capturar frases como "the node was low".
    m = _NODE_RE.search(evidence or "")
    if m and re.search(r"[\d.]", m.group(1)):
        return m.group(1)
    return None


def extract_pvc(evidence: str) -> str | None:
    m = _PVC_RE.search(evidence or "")
    return m.group(1) if m else None


def extract_service(evidence: str) -> str | None:
    m = _SVC_RE.search(evidence or "")
    return m.group(1) if m else None


def extract_workload(evidence: str) -> str | None:
    """Deduce el deployment/statefulset desde el nombre del pod (quita sufijos)."""
    pod = extract_pod(evidence)
    if not pod:
        return None
    # Deployment: <name>-<replicaset(10)>-<pod(5)>
    m = re.match(r"^(.*)-[a-z0-9]{6,10}-[a-z0-9]{5}$", pod)
    if m:
        return m.group(1)
    # StatefulSet/DaemonSet: <name>-<ordinal>
    m = re.match(r"^(.*)-\d+$", pod)
    if m:
        return m.group(1)
    return pod


# ── Catálogo de intenciones (orden: específico → general) ───────────────────
# Cada intención: keywords, verbo (alineado con eval/metrics SCENARIO_KUBECTL_VERB),
# constructor del comando de investigación, y remediación opcional (reversible).

def _ns_flag(ns: str) -> str:
    return f" -n {ns}" if ns else ""


def _pod_or_pods(evidence: str, ns: str) -> str:
    pod = extract_pod(evidence)
    target = f"pod {pod}" if pod else "pods"
    return f"kubectl describe {target}{_ns_flag(ns)}"


def _restart(evidence: str, ns: str) -> str:
    wl = extract_workload(evidence)
    return f"kubectl rollout restart deployment/{wl}{_ns_flag(ns)}" if wl else ""


_INTENTS: list[dict] = [
    {
        "name": "pvc",
        "kw": ["persistentvolumeclaim", "pvc", "failedbinding", "unbound", "no volume",
               "provision", "waitforfirstconsumer", "volume claim"],
        "verb": "describe",
        "investigate": lambda ev, ns: (
            f"kubectl describe pvc {extract_pvc(ev)}{_ns_flag(ns)}" if extract_pvc(ev)
            else f"kubectl describe pvc{_ns_flag(ns)}"
        ),
        "remediate": lambda ev, ns: "",  # storage → manual
    },
    {
        "name": "node_pressure",
        "kw": ["node was low", "diskpressure", "disk pressure", "memorypressure",
               "memory pressure", "evicted"],
        "verb": "describe",
        "investigate": lambda ev, ns: (
            f"kubectl describe node {extract_node(ev)}" if extract_node(ev)
            else "kubectl describe nodes"
        ),
        "remediate": lambda ev, ns: "",  # nodo → manual / escalado
    },
    {
        "name": "network",
        "kw": ["networkpolicy", "network policy", "connection refused", "denied",
               "i/o timeout", "no route to host"],
        "verb": "get",
        "investigate": lambda ev, ns: f"kubectl get networkpolicy{_ns_flag(ns)}",
        "remediate": lambda ev, ns: "",
    },
    {
        "name": "image_auth",
        "kw": ["unauthorized", "403", "pull access denied", "authentication required",
               "imagepullsecret", "credential"],
        "verb": "get",
        "investigate": lambda ev, ns: f"kubectl get secret{_ns_flag(ns)}",
        "remediate": lambda ev, ns: "",
    },
    {
        "name": "image",
        "kw": ["imagepullbackoff", "errimagepull", "errimage", "manifest",
               "no such image", "back-off pulling image", "failed to pull image",
               "image can't be pulled"],
        "verb": "describe",
        "investigate": _pod_or_pods,
        "remediate": lambda ev, ns: "",
    },
    {
        "name": "oom",
        "kw": ["oomkill", "oom ", "out of memory", "memory cgroup", "oomkilled"],
        "verb": "describe",
        "investigate": _pod_or_pods,
        "remediate": _restart,
    },
    {
        "name": "crash_secret",
        "kw": ["secret", "role", "does not exist", "password", "authentication failed",
               "couldn't find key", "missing key"],
        "verb": "get",
        "investigate": lambda ev, ns: f"kubectl get secret{_ns_flag(ns)}",
        "remediate": lambda ev, ns: "",
    },
    {
        "name": "crash_config",
        "kw": ["crashloopbackoff", "crashloop", "back-off restarting", "configmap",
               "exit code", "exit status", "containerd task", "invalid configuration",
               "env var"],
        "verb": "logs",
        "investigate": lambda ev, ns: (
            f"kubectl logs {extract_pod(ev)}{_ns_flag(ns)} --previous" if extract_pod(ev)
            else f"kubectl logs{_ns_flag(ns)} --previous"
        ),
        "remediate": _restart,
    },
    {
        "name": "probe",
        "kw": ["liveness probe", "readiness probe", "probe failed", "unhealthy",
               "health check"],
        "verb": "describe",
        "investigate": _pod_or_pods,
        "remediate": _restart,
    },
    {
        "name": "endpoints",
        "kw": ["no endpoints", "endpoints", "selector", "notready", "no healthy upstream"],
        "verb": "get",
        "investigate": lambda ev, ns: (
            f"kubectl get endpoints {extract_service(ev)}{_ns_flag(ns)}" if extract_service(ev)
            else f"kubectl get endpoints{_ns_flag(ns)}"
        ),
        "remediate": lambda ev, ns: "",
    },
    {
        "name": "pending_cpu",
        "kw": ["insufficient cpu", "unschedulable", "failedscheduling", "pending",
               "didn't match", "no nodes available"],
        "verb": "describe",
        "investigate": _pod_or_pods,
        "remediate": lambda ev, ns: "",
    },
]


# Clasificación por dueño/origen: plataforma (infra) vs app (código/config).
_PLATFORM_INTENTS = {"node_pressure", "oom", "image", "image_auth", "pvc",
                     "network", "pending_cpu"}
_APP_INTENTS = {"crash_secret", "crash_config", "probe", "endpoints"}


def classify_category(evidence: str, root_cause: str = "") -> str:
    """Clasifica el incidente en 'platform' (infra: nodo/recursos/storage/red/imagen)
    o 'app' (código/config/credenciales/salud de la app). Default 'app'."""
    intent = detect_intent(f"{root_cause}\n{evidence}")
    if intent is not None and intent["name"] in _PLATFORM_INTENTS:
        return "platform"
    return "app"


def detect_intent(text: str) -> dict | None:
    low = (text or "").lower()
    for intent in _INTENTS:
        if any(kw in low for kw in intent["kw"]):
            return intent
    return None


# ── Validación del comando del modelo ───────────────────────────────────────

_VALID_VERBS = {"get", "describe", "logs", "top", "explain", "events"}
_NON_NAMESPACED = re.compile(r"\b(node|nodes|pv|persistentvolume|persistentvolumes)\b")


def _is_fragile(cmd: str) -> bool:
    return ("$(" in cmd or "`" in cmd or "|" in cmd or ">" in cmd
            or "<" in cmd or ";" in cmd or "&&" in cmd)


def _force_namespace(cmd: str, ns: str) -> str:
    """Corrige/inserta el -n al namespace culpable (salvo recursos cluster-scoped)."""
    if not ns or _NON_NAMESPACED.search(cmd):
        return re.sub(r"\s+-n\s+\S+", "", cmd).strip()
    if re.search(r"\s-n\s+\S+", cmd):
        return re.sub(r"(\s-n\s+)\S+", rf"\1{ns}", cmd).strip()
    return f"{cmd} -n {ns}".strip()


def _model_command_ok(cmd: str, intent: dict | None) -> bool:
    """True si el comando del modelo es seguro y coherente con la intención."""
    parts = cmd.split()
    if len(parts) < 2 or parts[0] != "kubectl":
        return False
    verb = parts[1].lower()
    if verb not in _VALID_VERBS:
        return False
    if _is_fragile(cmd):
        return False
    # Coherente con la intención detectada (mismo verbo) — si hay intención.
    if intent is not None and verb != intent["verb"]:
        return False
    return True


def build_command(evidence: str, namespace: str, root_cause: str = "",
                  model_cmd: str = "") -> str:
    """Comando kubectl de INVESTIGACIÓN: dirigido, válido y con el namespace correcto.

    Estrategia: si el comando del modelo es seguro y coherente con la intención
    detectada, se conserva (corrigiendo el namespace); si no, se usa el comando
    determinista del catálogo. Nunca devuelve placeholders ni comandos frágiles.
    """
    ns = (namespace or "").strip()
    text = f"{root_cause}\n{evidence}"
    intent = detect_intent(text)

    model_cmd = (model_cmd or "").strip()
    if model_cmd and _model_command_ok(model_cmd, intent):
        return _force_namespace(model_cmd, ns)

    if intent is not None:
        cmd = intent["investigate"](evidence, ns).strip()
        if cmd and "<" not in cmd and ">" not in cmd:
            return re.sub(r"\s+", " ", cmd)

    # Sin intención clara: dirigir al pod/namespace si se conoce, si no, fallback.
    pod = extract_pod(evidence)
    if pod and ns:
        return f"kubectl describe pod {pod} -n {ns}"
    if ns:
        return f"kubectl get pods -n {ns}"
    return _DEFAULT


_NAME_AFTER_RE = re.compile(
    r"\b(?:pods?|pvc|persistentvolumeclaim|nodes?|endpoints|svc|service)\s+(\S+)",
    re.IGNORECASE,
)
_TARGET_RE = re.compile(r"\b((?:deployment|statefulset|daemonset)/\S+)", re.IGNORECASE)

_GENERIC_BY_VERB = {
    "describe": "Muestra el detalle y los eventos del recurso indicado.",
    "get": "Lista el recurso indicado y su estado.",
    "logs": "Muestra los logs del recurso indicado.",
    "top": "Muestra el consumo de CPU/memoria del recurso indicado.",
}


def _ns_of(cmd: str) -> str | None:
    m = re.search(r"-n\s+(\S+)", cmd)
    return m.group(1) if m else None


def _name_after_kind(cmd: str) -> str | None:
    m = _NAME_AFTER_RE.search(cmd)
    if not m:
        return None
    name = m.group(1)
    return None if name.startswith("-") else name


def explain_command(cmd: str) -> str:
    """Explica en español qué hace un comando kubectl (determinista, sin modelo).

    Parsea verbo/recurso/nombre/namespace y mapea a una frase que dice qué muestra
    y qué buscar. Vacío si no es un comando kubectl.
    """
    cmd = (cmd or "").strip()
    low = cmd.lower()
    if not low.startswith("kubectl"):
        return ""

    ns = _ns_of(cmd)
    nsx = f" en «{ns}»" if ns else ""
    name = _name_after_kind(cmd)

    if "rollout restart" in low:
        m = _TARGET_RE.search(cmd)
        target = m.group(1) if m else "el workload"
        return (f"Reinicia de forma controlada {target}{nsx} "
                f"(rolling restart: recrea los pods uno a uno, es reversible).")
    if "describe pod" in low:
        if name:
            return (f"Muestra el estado y los últimos eventos del pod «{name}»{nsx}: "
                    f"reinicios, OOMKilled y motivos de fallo.")
        return f"Muestra el estado y los eventos de los pods{nsx} para ver cuál falla y por qué."
    if "describe pvc" in low or "describe persistentvolumeclaim" in low:
        if name:
            return (f"Muestra el detalle del PersistentVolumeClaim «{name}»{nsx} "
                    f"y por qué no se vincula a un volumen.")
        return f"Muestra los PersistentVolumeClaims{nsx} y su estado de vinculación."
    if "describe node" in low:
        if name:
            return (f"Muestra los recursos, la presión (CPU/memoria/disco) y las "
                    f"condiciones del nodo «{name}».")
        return "Muestra el estado y la presión de recursos de los nodos del cluster."
    if "get secret" in low:
        return (f"Lista los secrets{nsx} para comprobar si falta el secret con las "
                f"credenciales o claves que el pod no encuentra.")
    if "get networkpolicy" in low:
        return f"Lista las NetworkPolicies{nsx} que podrían estar bloqueando el tráfico de red."
    if "get endpoints" in low:
        if name:
            return f"Comprueba si el service «{name}»{nsx} tiene endpoints (pods listos detrás)."
        return f"Comprueba qué services{nsx} se han quedado sin endpoints (sin pods listos)."
    if low.startswith("kubectl logs"):
        prev = " de la instancia anterior (la que crasheó)" if "--previous" in low else ""
        # 'kubectl logs <pod>' lleva el nombre directo (sin keyword de tipo).
        m = re.match(r"kubectl\s+logs\s+(?:pod/)?([^\s-]\S*)", cmd, re.IGNORECASE)
        pod = m.group(1) if m else None
        if pod:
            return f"Muestra los logs{prev} del pod «{pod}»{nsx}."
        return f"Muestra los logs{prev} de los pods{nsx}."
    if "get events" in low:
        return "Lista los eventos recientes del cluster ordenados por fecha para ver qué ha pasado."
    if "get pods" in low:
        return f"Lista los pods{nsx} con su estado (Running/CrashLoop/Pending…)."

    parts = cmd.split()
    verb = parts[1].lower() if len(parts) > 1 else ""
    return _GENERIC_BY_VERB.get(verb, "Ejecuta el comando de diagnóstico indicado.")


def build_remediation(evidence: str, namespace: str, root_cause: str = "") -> str:
    """Acción de remediación reversible (rollout restart) si aplica; si no, ''.

    Shadow: nunca se ejecuta sin aprobación. Storage/red/nodo → remediación manual.
    """
    ns = (namespace or "").strip()
    intent = detect_intent(f"{root_cause}\n{evidence}")
    if intent is None:
        return ""
    cmd = intent["remediate"](evidence, ns).strip()
    return re.sub(r"\s+", " ", cmd) if cmd else ""
