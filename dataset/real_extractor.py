"""
Extractor de muestras de entrenamiento desde el cluster real.

Tres fuentes:
  1. Events API  — eventos K8s actuales
  2. Pod logs    — logs de todos los pods buscando patrones de error
  3. Synthetic++ — sintético con nombres y namespaces reales del cluster

Produce muestras etiquetadas automáticamente mediante reglas.
"""

import json
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ── Nombres reales del cluster (extraidos del cluster) ────────────────────────

REAL_NAMESPACES = [
    "aeat-retenciones", "argocd", "banca-conection", "banca-dashboard",
    "default", "firmas", "haproxy-ingress", "hortideploy", "intranet",
    "kube-system", "longhorn-system", "mazafresh", "mcp", "ml-abonos-api",
    "ml-cobros-api", "ml-prevision-api", "mlflow", "n8n", "openwebui",
    "pageindex", "pepino-produccion", "postgresql", "previsiones-entrada",
    "qdrant", "redis", "rrhh", "session-manager", "tecnicosnetfront",
    "tesoreria-app",
]

REAL_DEPLOYMENTS = [
    "edeka-parser", "sigpac-explorer", "outlook-backend", "mapping-backend",
    "dashboard", "mcp-fullbd", "mcp-contabilidad", "mcp-facturas", "mcp-pgc",
    "mcp-mazafresh", "sigpac-mcp", "ml-prevision-api", "ml-cobros-api",
    "ml-abonos-api", "mlflow", "pepino-api", "pageindex-web", "pageindex-mcp",
    "hortideploy-backend", "hortideploy-frontend", "mazafresh-backend",
    "mazafresh-frontend", "rrhh-backend", "rrhh-frontend", "previsiones-api",
    "previsiones-frontend", "tesoreria-app", "session-manager-backend",
    "session-manager-frontend", "tecnicosnetfront", "firmas", "banca-dashboard",
    "banca-conection", "n8n", "openwebui", "postgresql", "redis", "qdrant",
    "hortinet", "oauth2-proxy-intranet", "oauth2-proxy-mapping",
    "anecoop-parser", "eurogroup-parser", "agricultores-bot",
    "aeat-retenciones", "outlook-anecoop", "outlook-eurogroup",
]

# ── Patrones de error en logs de aplicacion ───────────────────────────────────
# Cada patron: regex que matchea en el log → metadata del fallo

LOG_PATTERNS = [
    {
        "id": "connection_refused",
        "regex": r"Connection refused|ECONNREFUSED|Failed to establish a new connection",
        "reason": "ConnectionRefused",
        "root_cause_tpl": "El pod {pod} no puede conectar con el servicio dependiente en {host}. El servicio está caído, el puerto es incorrecto o una NetworkPolicy bloquea el tráfico.",
        "kubectl_tpl": "kubectl get svc -n {namespace} && kubectl get networkpolicy -n {namespace}",
    },
    {
        "id": "connection_reset",
        "regex": r"ECONNRESET|Connection reset by peer|read ECONNRESET",
        "reason": "ConnectionReset",
        "root_cause_tpl": "Reseteos de conexión TCP en {pod}. El servicio upstream cierra conexiones abruptamente — posible timeout, reinicio del pod destino o límite de conexiones alcanzado.",
        "kubectl_tpl": "kubectl logs {pod} -n {namespace} --previous --tail=50",
    },
    {
        "id": "timeout",
        "regex": r"timeout|timed out|deadline exceeded|context deadline",
        "reason": "Timeout",
        "root_cause_tpl": "Timeouts persistentes en {pod}. El servicio no responde dentro del tiempo límite configurado. Posible sobrecarga, latencia de red o base de datos lenta.",
        "kubectl_tpl": "kubectl top pod -n {namespace}",
    },
    {
        "id": "oom_app",
        "regex": r"Out of memory|OOM|MemoryError|Cannot allocate memory|killed.*memory",
        "reason": "OOMApp",
        "root_cause_tpl": "El proceso en {pod} agota la memoria disponible a nivel de aplicación. El límite de memoria del contenedor puede ser insuficiente para la carga actual.",
        "kubectl_tpl": "kubectl set resources deployment/{deployment} --limits=memory=1Gi -n {namespace}",
    },
    {
        "id": "db_connection",
        "regex": r"could not connect to server|connection to server|FATAL.*database|pg_connect|psycopg2",
        "reason": "DatabaseConnectionError",
        "root_cause_tpl": "Error de conexión a base de datos desde {pod}. El pod de PostgreSQL puede estar reiniciándose, las credenciales son incorrectas o se ha alcanzado el límite de conexiones.",
        "kubectl_tpl": "kubectl get pods -n postgresql && kubectl logs -n postgresql -l app=postgresql --tail=20",
    },
    {
        "id": "http_5xx",
        "regex": r"HTTP [5][0-9][0-9]|status 5[0-9][0-9]|500 Internal|502 Bad|503 Service|504 Gateway",
        "reason": "HTTP5xx",
        "root_cause_tpl": "Errores HTTP 5xx en {pod}. El servicio upstream devuelve errores del servidor. Puede indicar un crash en el backend, base de datos no disponible o configuración incorrecta.",
        "kubectl_tpl": "kubectl logs {pod} -n {namespace} --tail=100 | grep -i error",
    },
    {
        "id": "disk_full",
        "regex": r"No space left on device|disk full|ENOSPC|no space",
        "reason": "DiskFull",
        "root_cause_tpl": "Sin espacio en disco en {pod}. El volumen efímero o PVC está lleno. Puede estar acumulando logs, archivos temporales o datos sin límite.",
        "kubectl_tpl": "kubectl exec {pod} -n {namespace} -- df -h",
    },
    {
        "id": "permission_denied",
        "regex": r"Permission denied|EACCES|403 Forbidden|Forbidden|unauthorized",
        "reason": "PermissionDenied",
        "root_cause_tpl": "Error de permisos en {pod}. El ServiceAccount no tiene los permisos RBAC necesarios o el sistema de archivos tiene permisos incorrectos.",
        "kubectl_tpl": "kubectl auth can-i --list --as=system:serviceaccount:{namespace}:default",
    },
    {
        "id": "ssl_cert",
        "regex": r"SSL|certificate|x509|TLS|CERTIFICATE_VERIFY_FAILED",
        "reason": "SSLError",
        "root_cause_tpl": "Error de certificado TLS en {pod}. El certificado ha expirado, es autofirmado sin la CA correcta o el hostname no coincide.",
        "kubectl_tpl": "kubectl get certificate -n {namespace} && kubectl get secret -n {namespace} | grep tls",
    },
    {
        "id": "crash_exit",
        "regex": r"exit code [1-9]|segmentation fault|panic:|fatal error|SIGKILL|SIGTERM",
        "reason": "ProcessCrash",
        "root_cause_tpl": "El proceso en {pod} termina inesperadamente. Un exit code no-cero o señal indica crash de la aplicación, excepción no capturada o kill externo.",
        "kubectl_tpl": "kubectl logs {pod} -n {namespace} --previous -n {namespace}",
    },
]

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You will receive a set of raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task is to:
1. Identify the root cause of the anomaly in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate the issue.

Output format (strict):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>

Be concise. Focus on actionable diagnosis."""


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=10).decode()
    except Exception:
        return ""


def _rand_pod(deployment: str) -> str:
    s1 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
    s2 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
    return f"{deployment}-{s1}-{s2}"


def _rand_host() -> str:
    return random.choice([
        "192.168.2.215", "192.168.2.213", "192.168.2.203",
        "10.96.0.1", "postgresql.postgresql.svc.cluster.local",
        "redis.redis.svc.cluster.local", "qdrant.qdrant.svc.cluster.local",
    ])


def _build_sample(logs: list[str], namespace: str, deployment: str,
                  root_cause: str, kubectl: str, score: float,
                  source: str, pattern_id: str) -> dict:
    sample_logs = logs[:20]

    user_msg = (
        f"Anomaly Score: {score:.3f}\n"
        f"Namespace: {namespace}\n"
        f"Deployment: {deployment}\n"
        f"Total events in window: {len(logs)}\n"
        f"Event sample:\n" +
        "\n".join(f"  {l}" for l in sample_logs)
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": f"ROOT CAUSE: {root_cause}\nKUBECTL: {kubectl}"},
        ],
        "metadata": {
            "source": source, "pattern_id": pattern_id,
            "namespace": namespace, "deployment": deployment,
        }
    }


# ── Fuente 1: Events API del cluster ─────────────────────────────────────────

def extract_from_events() -> list[dict]:
    print("  [1/3] Extrayendo Events API...")
    raw = _run("kubectl get events --all-namespaces -o json")
    if not raw:
        return []

    data = json.loads(raw)
    samples = []

    # Agrupar por (namespace, reason)
    groups = defaultdict(list)
    for ev in data["items"]:
        ns     = ev["metadata"]["namespace"]
        reason = ev.get("reason", "Unknown")
        obj    = ev.get("involvedObject", {})
        name   = obj.get("name", "unknown")
        msg    = ev.get("message", "")
        count  = ev.get("count", 1)
        groups[(ns, reason)].append((name, msg, count))

    for (ns, reason), events in groups.items():
        # Simular una ventana con estos eventos repetidos por su count
        logs = []
        for name, msg, count in events:
            for _ in range(min(count, 20)):
                logs.append(f"{ns} {name} {reason} {msg}")

        if len(logs) < 3:
            continue

        # Buscar patron que aplique
        all_text = " ".join(logs).lower()
        matched = None
        for pat in LOG_PATTERNS:
            if re.search(pat["regex"], all_text, re.IGNORECASE):
                matched = pat
                break

        if not matched:
            continue

        # Deployment desde el nombre del objeto
        dep = events[0][0].rsplit("-", 2)[0] if events else random.choice(REAL_DEPLOYMENTS)
        pod = _rand_pod(dep)
        host = _rand_host()
        score = round(random.uniform(0.82, 0.97), 3)

        root = matched["root_cause_tpl"].format(
            pod=pod, namespace=ns, deployment=dep, host=host)
        kctl = matched["kubectl_tpl"].format(
            pod=pod, namespace=ns, deployment=dep)

        samples.append(_build_sample(logs, ns, dep, root, kctl, score,
                                     "events_api", matched["id"]))

    print(f"     → {len(samples)} samples de Events API")
    return samples


# ── Fuente 2: Pod logs ────────────────────────────────────────────────────────

def extract_from_pod_logs(max_pods: int = 60) -> list[dict]:
    print("  [2/3] Extrayendo logs de pods...")
    pods_raw = _run("kubectl get pods --all-namespaces -o json")
    if not pods_raw:
        return []

    pods = json.loads(pods_raw)["items"]
    running = [p for p in pods if p.get("status", {}).get("phase") == "Running"]
    random.shuffle(running)
    running = running[:max_pods]

    samples = []
    for pod_obj in running:
        ns       = pod_obj["metadata"]["namespace"]
        pod_name = pod_obj["metadata"]["name"]
        dep      = pod_name.rsplit("-", 2)[0]

        # Saltar pods de infra que no nos interesan para RCA
        if any(skip in dep for skip in ["longhorn", "calico", "coredns", "etcd",
                                         "kube-", "metallb", "tigera", "csi-"]):
            continue

        logs_raw = _run(f"kubectl logs {pod_name} -n {ns} --tail=80 2>/dev/null")
        if not logs_raw.strip():
            continue

        log_lines = logs_raw.strip().splitlines()

        for pat in LOG_PATTERNS:
            matching = [l for l in log_lines if re.search(pat["regex"], l, re.IGNORECASE)]
            if len(matching) < 2:
                continue

            # Construir ventana: mezcla de logs de error + contexto
            context = [l for l in log_lines if l not in matching][-5:]
            window_logs = []
            for line in matching[:15]:
                window_logs.append(f"{ns} Pod/{pod_name} {pat['reason']} {line.strip()[:120]}")
            for line in context:
                window_logs.append(f"{ns} Pod/{pod_name} Normal {line.strip()[:120]}")

            random.shuffle(window_logs)
            score = round(random.uniform(0.80, 0.96), 3)
            host  = _rand_host()

            root = pat["root_cause_tpl"].format(
                pod=pod_name, namespace=ns, deployment=dep, host=host)
            kctl = pat["kubectl_tpl"].format(
                pod=pod_name, namespace=ns, deployment=dep)

            samples.append(_build_sample(
                window_logs, ns, dep, root, kctl, score,
                "pod_logs", pat["id"]))

            break  # un patron por pod, para no sesgar

    print(f"     → {len(samples)} samples de pod logs")
    return samples


# ── Fuente 3: Sintético con nombres reales ────────────────────────────────────

def generate_synthetic_real_names(n_per_pattern: int = 25) -> list[dict]:
    print("  [3/3] Generando sintético con nombres reales del cluster...")

    # Importar escenarios YAML
    sys.path.insert(0, str(Path(__file__).parent))
    from generator import load_scenarios

    base = Path(__file__).parent
    scenarios = load_scenarios(base / "scenarios")

    # Parchar el generador para que use nombres reales
    import generator as gen
    gen.NAMESPACES   = REAL_NAMESPACES
    gen.DEPLOYMENTS  = REAL_DEPLOYMENTS

    samples = []
    for scenario in scenarios:
        for _ in range(n_per_pattern):
            ctx    = gen._build_context(scenario)
            logs   = gen._generate_events(scenario, ctx)
            sample = gen._build_sample(scenario, ctx, logs)
            sample["metadata"]["source"] = "synthetic_real_names"
            samples.append(sample)

    print(f"     → {len(samples)} samples sintéticos con nombres reales")
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main(output_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    print("\nExtrayendo muestras del cluster real...\n")

    all_samples = []
    all_samples += extract_from_events()
    all_samples += extract_from_pod_logs(max_pods=80)
    all_samples += generate_synthetic_real_names(n_per_pattern=40)

    # Deduplicar por hash del user_msg
    seen = set()
    unique = []
    for s in all_samples:
        key = hash(s["messages"][1]["content"][:300])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    random.shuffle(unique)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in unique:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Resumen por fuente
    from collections import Counter
    sources = Counter(s["metadata"]["source"] for s in unique)
    patterns = Counter(s["metadata"].get("pattern_id", s["metadata"].get("scenario_id", "?"))
                       for s in unique)

    print(f"\n{'═'*55}")
    print(f"  Total samples únicos: {len(unique)}")
    print("\n  Por fuente:")
    for src, n in sources.most_common():
        print(f"    {src:<30} {n}")
    print("\n  Por patrón (top 10):")
    for pid, n in patterns.most_common(10):
        print(f"    {pid:<30} {n}")
    print(f"\n  Guardado en: {output_path}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main(Path(__file__).parent / "output" / "real_cluster.jsonl")
