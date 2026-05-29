"""
Generador de dataset sintético para fine-tuning del SLM K8s-RCA.

Toma los escenarios YAML y genera N variaciones por escenario
sustituyendo variables con valores realistas aleatorios.

Salida: JSONL en formato chat (system/user/assistant) listo para unsloth/trl.
"""

import json
import random
import time
from pathlib import Path

import yaml

# ── Valores realistas para sustitucion de variables ──────────────────────────

NAMESPACES = [
    "default", "production", "staging", "mcp", "ml-cobros", "ml-prevision",
    "banca-conection", "pageindex", "intranet", "monitoring", "argocd",
]

DEPLOYMENTS = [
    "api-gateway", "payment-service", "user-service", "auth-service",
    "inventory-api", "notification-svc", "report-generator", "data-pipeline",
    "ml-predictor", "webhook-handler", "sync-worker", "scheduler",
    "edeka-parser", "outlook-backend", "sigpac-explorer", "pepino-api",
]

CONTAINERS = ["app", "sidecar", "proxy", "worker", "init-db", "api", "server"]
NODES = ["ubuntumaster", "worker", "workerpedidos", "node-1", "node-2", "node-3"]
SECRETS = ["db-credentials", "api-keys", "tls-cert", "registry-auth", "jwt-secret", "s3-config"]
STORAGE_CLASSES = ["longhorn", "standard", "fast-ssd", "nfs-storage"]
MEM_LIMITS = [128, 256, 512]
MEM_FIXES  = {128: 256, 256: 512, 512: 1024}
HTTP_CODES = [500, 502, 503, 504]

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You will receive a set of raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task is to:
1. Identify the root cause of the anomaly in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate the issue.

Output format (strict):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>

Be concise. Focus on actionable diagnosis."""


def _random_pod(deployment: str) -> str:
    suffix1 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
    suffix2 = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=5))
    return f"{deployment}-{suffix1}-{suffix2}"


def _random_ip() -> str:
    return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(2,254)}"


def _fill(template: str, ctx: dict) -> str:
    """Sustituye {variable} por valores del contexto."""
    result = template
    for k, v in ctx.items():
        result = result.replace("{" + k + "}", str(v))
    return result


def _build_context(scenario: dict) -> dict:
    deployment = random.choice(DEPLOYMENTS)
    namespace  = random.choice(NAMESPACES)
    mem        = random.choice(MEM_LIMITS)
    return {
        "deployment": deployment,
        "namespace":  namespace,
        "pod":        _random_pod(deployment),
        "container":  random.choice(CONTAINERS),
        "node":       random.choice(NODES),
        "secret":     random.choice(SECRETS),
        "pvc":        f"pvc-{deployment}-data",
        "sc":         random.choice(STORAGE_CLASSES),
        "svc":        f"{deployment}-svc",
        "image":      f"ghcr.io/hortichuelas/{deployment}:{random.randint(1,9)}.{random.randint(0,9)}.{random.randint(0,9)}",
        "image_fix":  f"ghcr.io/hortichuelas/{deployment}:latest",
        "mem":        str(mem),
        "mem_fix":    str(MEM_FIXES[mem]),
        "threshold":  str(mem - 32),
        "code":       str(random.choice(HTTP_CODES)),
        "port":       str(random.choice([8080, 8443, 3000, 9090, 5000])),
        "ip":         _random_ip(),
        "user":       "aaranda233",
        "token":      "ghp_xxxxxxxxxxxx",
    }


def _generate_events(scenario: dict, ctx: dict) -> list[str]:
    """Genera una lista de strings de log a partir del escenario."""
    logs = []
    for ev_def in scenario["events"]:
        lo, hi = ev_def["count"]
        count = random.randint(lo, hi)
        message = _fill(ev_def["message"], ctx)
        reason  = ev_def["reason"]
        source  = f"Pod/{ctx['pod']}"

        # Generar `count` variaciones del evento (timestamps distintos)
        for _ in range(count):
            logs.append(f"{ctx['namespace']} {source} {reason} {message}")

    random.shuffle(logs)
    return logs


def _build_sample(scenario: dict, ctx: dict, logs: list[str]) -> dict:
    """Construye un sample en formato chat para SFT."""
    score = round(random.uniform(0.81, 0.99), 3)
    n_logs = len(logs)
    sample_logs = logs[:20]  # max 20 para mantener muestras dentro del contexto

    user_msg = (
        f"Anomaly Score: {score}\n"
        f"Namespace: {ctx['namespace']}\n"
        f"Deployment: {ctx['deployment']}\n"
        f"Total events in window: {n_logs}\n"
        f"Event sample:\n" +
        "\n".join(f"  {l}" for l in sample_logs)
    )

    assistant_msg = (
        f"ROOT CAUSE: {_fill(scenario['root_cause'], ctx)}\n"
        f"KUBECTL: {_fill(scenario['kubectl'], ctx)}"
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "metadata": {
            "scenario_id": scenario["id"],
            "failure":     scenario["failure"],
            "namespace":   ctx["namespace"],
            "score":       score,
        }
    }


def load_scenarios(scenarios_dir: Path) -> list[dict]:
    scenarios = []
    for yaml_file in sorted(scenarios_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text())
        for s in data.get("scenarios", []):
            scenarios.append(s)
    return scenarios


def generate(
    scenarios_dir: Path,
    output_path: Path,
    samples_per_scenario: int = 30,
    seed: int = 42,
) -> int:
    random.seed(seed)
    scenarios = load_scenarios(scenarios_dir)

    if not scenarios:
        print("No se encontraron escenarios.")
        return 0

    samples = []
    for scenario in scenarios:
        for _ in range(samples_per_scenario):
            ctx   = _build_context(scenario)
            logs  = _generate_events(scenario, ctx)
            sample = _build_sample(scenario, ctx, logs)
            samples.append(sample)

    # Mezclar para que el entrenamiento no vea todos los crashloops juntos
    random.shuffle(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return len(samples)


if __name__ == "__main__":
    base = Path(__file__).parent
    n = generate(
        scenarios_dir=base / "scenarios",
        output_path=base / "output" / "synthetic.jsonl",
        samples_per_scenario=30,
    )
    print(f"Generados {n} samples → dataset/output/synthetic.jsonl")
