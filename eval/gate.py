"""
Gate de no-regresión: decide si un modelo candidato puede promocionarse a
producción comparando sus métricas con el modelo activo.

Regla: el candidato solo se promociona si NO rompe el formato (parse_rate no
baja) y NO regresiona en contenido (keyword_hit no baja más de epsilon). Es la
barrera anti loop-degenerativo: impide que un feedback ruidoso degrade el modelo.

La decisión (evaluate_gate) es pura y testeable; run_gate la conecta con
eval/runner para producir las métricas en vivo.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

KEYWORD_EPSILON = 0.02   # tolerancia de regresión de keyword_hit
PROMOTE = "PROMOTE"
REJECT = "REJECT"


def evaluate_gate(candidate: dict, prod: dict, keyword_epsilon: float = KEYWORD_EPSILON) -> dict:
    """Decide PROMOTE/REJECT a partir de dos dicts de métricas (de eval.metrics.aggregate)."""
    reasons: list[str] = []

    cand_parse = candidate.get("parse_rate", 0.0)
    prod_parse = prod.get("parse_rate", 0.0)
    cand_kw = candidate.get("keyword_hit", 0.0)
    prod_kw = prod.get("keyword_hit", 0.0)

    ok_parse = cand_parse >= prod_parse
    ok_kw = cand_kw >= prod_kw - keyword_epsilon

    if not ok_parse:
        reasons.append(f"parse_rate regresiona: {cand_parse:.3f} < {prod_parse:.3f}")
    if not ok_kw:
        reasons.append(f"keyword_hit regresiona: {cand_kw:.3f} < {prod_kw:.3f}-{keyword_epsilon}")

    decision = PROMOTE if (ok_parse and ok_kw) else REJECT
    if decision == PROMOTE and not reasons:
        reasons.append("sin regresión; mejora o iguala al modelo activo")

    return {
        "decision": decision,
        "reasons": reasons,
        "candidate": {"parse_rate": cand_parse, "keyword_hit": cand_kw},
        "prod": {"parse_rate": prod_parse, "keyword_hit": prod_kw},
    }


def run_gate(candidate_model: str, prod_model: str, test_set: str = "eval/test_set.jsonl",
             host: str = "http://localhost:11434") -> dict:
    """Ejecuta eval/runner sobre ambos modelos y aplica el gate. Requiere Ollama."""
    from eval.metrics import aggregate
    from eval.runner import ModelConfig, evaluate_model

    samples = _load_test_set(test_set)
    cand = aggregate(evaluate_model(ModelConfig(name=candidate_model, host=host), samples))
    prod = aggregate(evaluate_model(ModelConfig(name=prod_model, host=host), samples))
    result = evaluate_gate(cand, prod)
    result["candidate_full"] = cand
    result["prod_full"] = prod
    return result


def _load_test_set(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
