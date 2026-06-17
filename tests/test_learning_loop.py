"""
Tests de la lógica del bucle de aprendizaje (Fases 3-5): gate, registro de
modelos y disparo de entrenamiento. El entrenamiento real (GPU) no se ejecuta.
"""

import pytest

from eval.gate import PROMOTE, REJECT, evaluate_gate
from finetune.deploy_model import ModelRegistry
from finetune.loop_train import count_examples, should_train

# ── Gate de no-regresión ────────────────────────────────────────────────────

@pytest.mark.unit
def test_gate_promotes_on_improvement():
    cand = {"parse_rate": 0.99, "keyword_hit": 0.95}
    prod = {"parse_rate": 0.98, "keyword_hit": 0.90}
    assert evaluate_gate(cand, prod)["decision"] == PROMOTE


@pytest.mark.unit
def test_gate_rejects_parse_regression():
    cand = {"parse_rate": 0.90, "keyword_hit": 0.95}
    prod = {"parse_rate": 0.98, "keyword_hit": 0.90}
    r = evaluate_gate(cand, prod)
    assert r["decision"] == REJECT
    assert any("parse_rate" in x for x in r["reasons"])


@pytest.mark.unit
def test_gate_rejects_keyword_regression_beyond_epsilon():
    cand = {"parse_rate": 0.98, "keyword_hit": 0.80}
    prod = {"parse_rate": 0.98, "keyword_hit": 0.90}
    assert evaluate_gate(cand, prod)["decision"] == REJECT


@pytest.mark.unit
def test_gate_tolerates_tiny_keyword_drop():
    cand = {"parse_rate": 0.98, "keyword_hit": 0.89}
    prod = {"parse_rate": 0.98, "keyword_hit": 0.90}
    assert evaluate_gate(cand, prod, keyword_epsilon=0.02)["decision"] == PROMOTE


# ── Registro de modelos + rollback ──────────────────────────────────────────

@pytest.mark.unit
def test_registry_record_and_active(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    assert reg.next_version() == 1
    reg.record(1, "g1.gguf", {"decision": "PROMOTE"}, "ds1", promoted=True)
    assert reg.active() == 1
    assert reg.active_model() == "k8s-rca-orpo-v1"
    assert reg.next_version() == 2


@pytest.mark.unit
def test_registry_rejected_candidate_not_active(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    reg.record(1, "g1", {"decision": "PROMOTE"}, "ds", promoted=True)
    reg.record(2, "g2", {"decision": "REJECT"}, "ds", promoted=False)
    assert reg.active() == 1  # el candidato rechazado no se activa


@pytest.mark.unit
def test_registry_rollback(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    reg.record(1, "g1", {}, "ds", promoted=True)
    reg.record(2, "g2", {}, "ds", promoted=True)
    assert reg.active() == 2
    assert reg.rollback() == 1  # vuelve a la anterior promocionada


@pytest.mark.unit
def test_registry_persists(tmp_path):
    p = str(tmp_path / "reg.json")
    ModelRegistry(path=p).record(1, "g", {}, "ds", promoted=True)
    assert ModelRegistry(path=p).active() == 1  # otra instancia lee del disco


# ── Disparo del entrenamiento ───────────────────────────────────────────────

@pytest.mark.unit
def test_should_train_threshold():
    assert should_train(total_examples=30, last_trained_count=0, min_new=20) is True
    assert should_train(total_examples=15, last_trained_count=0, min_new=20) is False
    assert should_train(total_examples=100, last_trained_count=90, min_new=20) is False


@pytest.mark.unit
def test_count_examples(tmp_path):
    p = tmp_path / "feedback.jsonl"
    p.write_text('{"a":1}\n{"b":2}\n\n{"c":3}\n')
    assert count_examples(str(p)) == 3
    assert count_examples(str(tmp_path / "nope.jsonl")) == 0


# ── Cierre del ciclo: watermark de consolidación (RAG se "vacía") ────────────

@pytest.mark.unit
def test_watermark_zero_without_active(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    assert reg.consolidation_watermark() == 0


@pytest.mark.unit
def test_watermark_follows_active_version(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    reg.record(1, "g1", {}, "ds", promoted=True, feedback_count=10)
    assert reg.consolidation_watermark() == 10
    reg.record(2, "g2", {}, "ds", promoted=True, feedback_count=25)
    assert reg.consolidation_watermark() == 25


@pytest.mark.unit
def test_watermark_ignores_rejected_candidate(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    reg.record(1, "g1", {}, "ds", promoted=True, feedback_count=10)
    reg.record(2, "g2", {}, "ds", promoted=False, feedback_count=30)  # rechazado
    # el watermark sigue al modelo ACTIVO (v1), no al candidato rechazado
    assert reg.consolidation_watermark() == 10


@pytest.mark.unit
def test_watermark_reverts_on_rollback(tmp_path):
    reg = ModelRegistry(path=str(tmp_path / "reg.json"))
    reg.record(1, "g1", {}, "ds", promoted=True, feedback_count=10)
    reg.record(2, "g2", {}, "ds", promoted=True, feedback_count=25)
    assert reg.consolidation_watermark() == 25
    reg.rollback()
    assert reg.consolidation_watermark() == 10  # al revertir, RAG recupera lo de v2


@pytest.mark.unit
def test_retriever_excludes_consolidated(tmp_path):
    """RAG retira los ejemplos ya consolidados (skip_consolidated)."""
    import json

    from src.diagnostics.incident_retriever import IncidentRetriever
    p = tmp_path / "feedback.jsonl"
    rows = [
        {"label": "positive", "prompt": {"user": "caso viejo consolidado"},
         "root_cause": "viejo", "kubectl_cmd": "kubectl a"},
        {"label": "positive", "prompt": {"user": "caso nuevo sin consolidar"},
         "root_cause": "nuevo", "kubectl_cmd": "kubectl b"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    # watermark=1 -> el primer caso (consolidado) se excluye
    r = IncidentRetriever.from_feedback(str(p), skip_consolidated=1)
    texts = [c["text"] for c in r.cases]
    assert "caso viejo consolidado" not in texts
    assert "caso nuevo sin consolidar" in texts
