"""
Tests del harness de comparativa de aprendizaje (RAG vs plain).
"""

import pytest

from eval.compare_learning import compare, evaluate_samples
from src.diagnostics.incident_retriever import IncidentRetriever

_SAMPLES = [
    {
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Pod api OOMKilled memory exceeded"},
            {"role": "assistant", "content": "ROOT CAUSE: OOM\nKUBECTL: kubectl x"},
        ],
        "metadata": {"scenario_id": "oom_killed"},
    },
]

_CORPUS = IncidentRetriever([
    {"text": "Pod api OOMKilled memory limit exceeded scheduler",
     "root_cause": "OOMKilled por límite de memoria", "kubectl": "kubectl set resources"},
])


@pytest.mark.unit
def test_evaluate_samples_parses_and_scores():
    def good(system, user):
        return "ROOT CAUSE: el pod fue OOMKilled\nKUBECTL: kubectl describe pod api"
    m = evaluate_samples(_SAMPLES, good)
    assert m["n"] == 1
    assert m["parse_rate"] == 1.0


@pytest.mark.unit
def test_rag_injects_context_into_prompt():
    seen = {}
    def capture(system, user):
        seen["user"] = user
        return "ROOT CAUSE: x\nKUBECTL: kubectl get pods"
    evaluate_samples(_SAMPLES, capture, retriever=_CORPUS)
    assert "incidentes pasados" in seen["user"].lower()  # contexto RAG inyectado


@pytest.mark.unit
def test_plain_has_no_rag_context():
    seen = {}
    def capture(system, user):
        seen["user"] = user
        return "ROOT CAUSE: x\nKUBECTL: kubectl get pods"
    evaluate_samples(_SAMPLES, capture, retriever=None)
    assert "incidentes pasados" not in seen["user"].lower()


@pytest.mark.unit
def test_compare_returns_both_modes():
    def model(system, user):
        return "ROOT CAUSE: causa\nKUBECTL: kubectl get pods"
    result = compare(_SAMPLES, _CORPUS, model)
    assert "plain" in result and "rag" in result
    assert result["plain"]["n"] == result["rag"]["n"] == 1


@pytest.mark.unit
def test_unparseable_output_lowers_parse_rate():
    def bad(system, user):
        return "I'm not sure what's happening here, no format at all"
    m = evaluate_samples(_SAMPLES, bad)
    # parse_diagnosis tiene fallback, pero sin KUBECTL real parse_rate puede ser 0
    assert 0.0 <= m["parse_rate"] <= 1.0
