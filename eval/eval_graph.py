"""Evaluación del grafo de conocimiento de remediación (Fase 5).

Mide, sobre los 14 escenarios de fallo del proyecto, si el grafo:
  - cobertura: devuelve un PLAN (hit) para el escenario,
  - intent-ok: detecta la intención correcta,
  - multi-paso: el plan tiene >1 paso (no un comando suelto),
  - ns-ok: los comandos llevan el namespace correcto,
  - reversible: el último recurso es L1 (rollout restart), no destructivo.

Es la métrica de "cobertura/corrección" del grafo para el paper. Usa un grafo
en memoria sembrado del catálogo (determinista, sin tocar el store real).

Uso:  python eval/eval_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.remediation.remediation_graph import COMMAND, RemediationGraph  # noqa: E402

# (scenario_id, evidencia representativa, namespace, root_cause, intent esperado)
SCENARIOS = [
    ("crash_config", "default Pod/web-abc12345de-x4k2p CrashLoopBackOff exit code 1 configmap",
     "default", "crashloop por configmap mal configurado", "crash_config"),
    ("crash_oom", "default Pod/api-abc12345de-x4k2p OOMKilled out of memory",
     "default", "el contenedor se quedó sin memoria", "oom"),
    ("crash_probe", "default Pod/web-abc12345de-x4k2p liveness probe failed unhealthy",
     "default", "la sonda de liveness falla", "probe"),
    ("crash_secret", "pg Pod/postgresql-0 FATAL: role does not exist",
     "pg", "el rol/secret no existe", "crash_secret"),
    ("image_auth", "ns Pod/x-abc12345de-x4k2p Failed to pull image: unauthorized authentication required",
     "ns", "fallo de autenticación con el registry", "image_auth"),
    ("image_not_found", "ns Pod/x-abc12345de-x4k2p ErrImagePull manifest no such image",
     "ns", "la imagen no existe / tag incorrecto", "image"),
    ("image_registry_down", "ns Pod/x-abc12345de-x4k2p back-off pulling image registry timeout",
     "ns", "no se pudo descargar la imagen del registry", "image"),
    ("network_policy_block", "ing Pod/haproxy-controller-abc12345de-x4k2p connection refused denied",
     "ing", "interrupción de red en el ingress", "network"),
    ("node_disk_pressure", "ns Pod/x Evicted The node was low on resource: disk. Node node-1",
     "ns", "presión de disco en el nodo", "node_pressure"),
    ("node_pressure_memory", "ns Pod/x Evicted The node was low on resource: memory. Node node-1",
     "ns", "presión de memoria en el nodo", "node_pressure"),
    ("pending_insufficient_cpu", "ns Pod/x-abc12345de-x4k2p FailedScheduling Insufficient cpu",
     "ns", "no hay cpu suficiente para planificar", "pending_cpu"),
    ("pvc_pending", "ns Pod/x FailedBinding PVC data-pvc pending no volume",
     "ns", "el pvc no se vincula a un volumen", "pvc"),
    ("readiness_failing", "ns Pod/x-abc12345de-x4k2p readiness probe failed unhealthy endpoint",
     "ns", "la sonda de readiness falla", "probe"),
    ("service_no_endpoints", "ns Service/web no endpoints selector notready",
     "ns", "el service se quedó sin endpoints", "endpoints"),
]


def evaluate(graph: RemediationGraph) -> list[dict]:
    rows = []
    for sid, ev, ns, rc, expected in SCENARIOS:
        plan = graph.resolve(ev, ns, rc)
        hit = plan is not None
        intent_ok = bool(plan and plan.intent == expected)
        n_steps = len(plan.steps) if plan else 0
        cmds = [s for s in (plan.steps if plan else []) if s.action_type != "guidance"]
        ns_ok = all((f"-n {ns}" in s.action) or s.action.startswith("kubectl describe node")
                    for s in cmds) if cmds else False
        has_reversible = any(s.action_type == COMMAND and s.risk_level <= 1
                             for s in (plan.steps if plan else []))
        rows.append({"scenario": sid, "hit": hit, "intent_ok": intent_ok,
                     "steps": n_steps, "multi": n_steps > 1, "ns_ok": ns_ok,
                     "reversible": has_reversible, "intent": plan.intent if plan else "-"})
    return rows


def main() -> None:
    g = RemediationGraph(db_path=":memory:")
    g.seed_from_catalog()
    rows = evaluate(g)
    n = len(rows)
    print(f"{'escenario':<26} hit intent multi ns-ok rev  pasos  intent")
    for r in rows:
        print(f"{r['scenario']:<26} "
              f"{'✓' if r['hit'] else '✗'}   "
              f"{'✓' if r['intent_ok'] else '✗'}     "
              f"{'✓' if r['multi'] else '·'}    "
              f"{'✓' if r['ns_ok'] else '·'}    "
              f"{'✓' if r['reversible'] else '·'}   "
              f"{r['steps']}     {r['intent']}")

    def pct(key):
        return 100.0 * sum(1 for r in rows if r[key]) / n
    print("\n=== Agregado (cobertura/corrección del grafo) ===")
    print(f"  Cobertura (hit):     {pct('hit'):.1f}%")
    print(f"  Intent correcto:     {pct('intent_ok'):.1f}%")
    print(f"  Plan multi-paso:     {pct('multi'):.1f}%")
    print(f"  NS-ok (binding):     {pct('ns_ok'):.1f}%")
    print(f"  Acción reversible:   {pct('reversible'):.1f}%")
    print(f"  Pasos por plan:      {sum(r['steps'] for r in rows)/n:.1f} (media)")


if __name__ == "__main__":
    main()
