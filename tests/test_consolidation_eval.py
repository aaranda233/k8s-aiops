"""Tests de Fase 4 (consolidación) y Fase 5 (eval del grafo)."""

import pytest

from eval.eval_graph import SCENARIOS, evaluate
from finetune.graph_to_orpo import graph_verified_positives
from src.remediation.remediation_graph import RemediationGraph

# ── Fase 4: export de consolidación ─────────────────────────────────────────

@pytest.mark.unit
def test_graph_verified_positives_filter():
    ex = [
        {"label": "positive", "solution_source": "graph", "verified": True},      # keep
        {"label": "positive", "solution_source": "escalated", "verified": True},  # keep
        {"label": "positive", "solution_source": "catalog", "verified": True},    # drop (no grafo)
        {"label": "positive", "solution_source": "graph", "verified": None},      # drop (no verif.)
        {"label": "negative", "solution_source": "graph", "verified": True},      # drop (no positivo)
    ]
    out = graph_verified_positives(ex)
    assert len(out) == 2
    assert all(e["solution_source"] in ("graph", "escalated") for e in out)


# ── Fase 5: eval del grafo ──────────────────────────────────────────────────

@pytest.mark.unit
def test_eval_graph_coverage_and_intent():
    g = RemediationGraph(db_path=":memory:")
    g.seed_from_catalog()
    rows = evaluate(g)
    assert len(rows) == len(SCENARIOS)
    assert all(r["hit"] for r in rows), "cobertura debe ser 100%"
    assert all(r["intent_ok"] for r in rows), "intent correcto en todos los escenarios"
    assert all(r["multi"] for r in rows), "todos los planes deben ser multi-paso"
    assert all(r["ns_ok"] for r in rows), "binding de namespace correcto"
