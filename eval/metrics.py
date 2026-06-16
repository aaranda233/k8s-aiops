"""
Métricas de evaluación para el SLM k8s-RCA.

- parse_rate     : el modelo produjo ROOT CAUSE + KUBECTL en el formato esperado
- keyword_hit    : ROOT CAUSE menciona al menos una keyword del escenario real
- rouge_l        : similitud ROUGE-L entre ROOT CAUSE generado y referencia
- kubectl_ns_ok  : el comando kubectl menciona el namespace correcto
- kubectl_verb   : el verbo kubectl (get/describe/logs/…) es el esperado
"""

from __future__ import annotations

# ── Keyword oracle por escenario ──────────────────────────────────────────────

SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "crash_config":             ["crashloop", "config", "configmap", "variable", "env", "exit"],
    "crash_oom":                ["oom", "memory", "limit", "killed", "oomkill"],
    "crash_probe":              ["probe", "liveness", "readiness", "health", "check"],
    "crash_secret":             ["secret", "missing", "mount", "volume", "credential"],
    "image_auth":               ["auth", "registry", "credential", "pull", "unauthorized", "403"],
    "image_not_found":          ["image", "not found", "tag", "pull", "404", "imagepull"],
    "image_registry_down":      ["registry", "unavailable", "timeout", "pull", "unreachable"],
    "network_policy_block":     ["network", "policy", "block", "connection", "refused", "denied"],
    "node_disk_pressure":       ["disk", "pressure", "storage", "evict", "space"],
    "node_pressure_memory":     ["memory", "pressure", "node", "evict", "resource"],
    "pending_insufficient_cpu": ["cpu", "insufficient", "resource", "pending", "schedule", "unschedul"],
    "pvc_pending":              ["pvc", "volume", "storage", "pending", "bound", "claim"],
    "readiness_failing":        ["readiness", "probe", "unhealthy", "endpoint", "failing"],
    "service_no_endpoints":     ["endpoint", "service", "selector", "no endpoint", "notready"],
}

SCENARIO_KUBECTL_VERB: dict[str, str] = {
    "crash_config":             "logs",
    "crash_oom":                "describe",
    "crash_probe":              "describe",
    "crash_secret":             "get",
    "image_auth":               "get",
    "image_not_found":          "describe",
    "image_registry_down":      "describe",
    "network_policy_block":     "get",
    "node_disk_pressure":       "describe",
    "node_pressure_memory":     "describe",
    "pending_insufficient_cpu": "describe",
    "pvc_pending":              "describe",
    "readiness_failing":        "describe",
    "service_no_endpoints":     "get",
}


# ── ROUGE-L (implementación propia, sin dependencias externas) ────────────────

def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longitud de la subsecuencia común más larga."""
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if a[i - 1] == b[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 a nivel de tokens."""
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall    = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Funciones de métricas individuales ───────────────────────────────────────

def parse_rate(root_cause: str, kubectl: str) -> bool:
    """True si el modelo produjo ambos campos en el formato esperado."""
    return (
        bool(root_cause)
        and root_cause != "Could not parse root cause."
        and bool(kubectl)
        and kubectl != "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
    )


def keyword_hit(root_cause: str, scenario_id: str) -> bool:
    """True si ROOT CAUSE menciona al menos una keyword del escenario."""
    keywords = SCENARIO_KEYWORDS.get(scenario_id, [])
    if not keywords:
        return False
    rc_lower = root_cause.lower()
    return any(kw in rc_lower for kw in keywords)


def kubectl_ns_ok(kubectl: str, namespace: str) -> bool:
    """True si el comando menciona el namespace correcto."""
    return namespace.lower() in kubectl.lower()


def kubectl_verb_ok(kubectl: str, scenario_id: str) -> bool:
    """True si el verbo kubectl es el esperado para el escenario."""
    expected = SCENARIO_KUBECTL_VERB.get(scenario_id, "")
    if not expected:
        return False
    parts = kubectl.strip().split()
    # kubectl <verb>  o  kubectl <plugin> <verb>
    verbs_in_cmd = {p for p in parts if not p.startswith("-")}
    return expected in verbs_in_cmd


# ── Agregación ────────────────────────────────────────────────────────────────

def aggregate(results: list[dict]) -> dict:
    """Calcula métricas agregadas sobre una lista de resultados individuales."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "n":              n,
        "parse_rate":     round(sum(r["parsed"]        for r in results) / n, 3),
        "keyword_hit":    round(sum(r["keyword_hit"]   for r in results) / n, 3),
        "rouge_l":        round(sum(r["rouge_l"]       for r in results) / n, 3),
        "kubectl_ns_ok":  round(sum(r["kubectl_ns_ok"] for r in results) / n, 3),
        "kubectl_verb_ok":round(sum(r["kubectl_verb_ok"]for r in results)/ n, 3),
        "latency_mean":   round(sum(r["latency_s"]     for r in results) / n, 3),
        "latency_p95":    round(sorted(r["latency_s"]  for r in results)[int(n * 0.95)], 3),
    }
