"""
Tests del recuperador de incidentes (RAG) — aprendizaje sin reentrenar.
"""

import json

import pytest

from src.diagnostics.incident_retriever import (
    IncidentRetriever,
    RefreshingRetriever,
    rag_context,
)

_CASES = [
    {"text": "PostgreSQL connection refused role postgres does not exist",
     "root_cause": "rol postgres inexistente", "kubectl": "kubectl logs postgres-0 -n db"},
    {"text": "OOMKilled memory limit exceeded on api pod",
     "root_cause": "OOM por límite de memoria bajo", "kubectl": "kubectl set resources deploy/api"},
    {"text": "ImagePullBackOff cannot pull image registry timeout",
     "root_cause": "fallo al descargar imagen", "kubectl": "kubectl describe pod web"},
]


@pytest.mark.unit
def test_retrieve_finds_most_similar():
    r = IncidentRetriever(_CASES)
    hits = r.retrieve("the api pod was OOMKilled, memory exceeded", k=1)
    assert len(hits) == 1
    assert "OOM" in hits[0]["root_cause"]


@pytest.mark.unit
def test_retrieve_respects_k():
    r = IncidentRetriever(_CASES)
    assert len(r.retrieve("postgres connection error", k=2)) <= 2


@pytest.mark.unit
def test_retrieve_filters_low_similarity():
    r = IncidentRetriever(_CASES)
    # query sin nada en común → por debajo de min_score
    hits = r.retrieve("zzz totally unrelated quantum banana", k=3, min_score=0.05)
    assert hits == []


@pytest.mark.unit
def test_empty_corpus_returns_nothing():
    r = IncidentRetriever([])
    assert r.retrieve("anything") == []


@pytest.mark.unit
def test_empty_query_returns_nothing():
    r = IncidentRetriever(_CASES)
    assert r.retrieve("   ") == []


@pytest.mark.unit
def test_rag_context_is_bounded_and_formatted():
    r = IncidentRetriever(_CASES)
    hits = r.retrieve("postgres role does not exist connection refused", k=2)
    ctx = rag_context(hits, max_chars=600)
    assert "incidentes pasados" in ctx.lower()
    assert "CAUSA:" in ctx
    assert len(ctx) <= 600


@pytest.mark.unit
def test_rag_context_empty_when_no_cases():
    assert rag_context([]) == ""


@pytest.mark.unit
def test_refreshing_retriever_picks_up_new_feedback(tmp_path):
    """A) Memoria instantánea: una corrección guardada entra en el RAG sin reiniciar."""
    import json
    import os
    import time

    fb = tmp_path / "feedback.jsonl"
    fb.write_text(json.dumps({
        "label": "positive", "prompt": {"user": "pod api OOMKilled memory exceeded"},
        "root_cause": "OOM caso viejo", "kubectl_cmd": "kubectl a",
    }) + "\n")
    r = RefreshingRetriever(str(fb))
    assert len(r.cases) == 1

    # Llega una corrección NUEVA (append) sobre otro caso
    time.sleep(0.01)
    with open(fb, "a") as f:
        f.write(json.dumps({
            "label": "positive", "prompt": {"user": "ingress 503 upstream timeout gateway"},
            "root_cause": "timeout del upstream", "kubectl_cmd": "kubectl b",
        }) + "\n")
    os.utime(fb, None)  # asegurar cambio de mtime

    hits = r.retrieve("ingress returns 503 upstream timeout", k=1)
    assert len(r.cases) == 2                       # se reconstruyó
    assert hits and "timeout" in hits[0]["root_cause"]  # el caso nuevo es recuperable al instante


@pytest.mark.unit
def test_refreshing_retriever_respects_watermark(tmp_path):
    import json

    fb = tmp_path / "feedback.jsonl"
    fb.write_text("\n".join(json.dumps(r) for r in [
        {"label": "positive", "prompt": {"user": "consolidado viejo"}, "root_cause": "v", "kubectl_cmd": "k"},
        {"label": "positive", "prompt": {"user": "nuevo sin consolidar"}, "root_cause": "n", "kubectl_cmd": "k"},
    ]))
    r = RefreshingRetriever(str(fb), watermark_fn=lambda: 1)
    texts = [c["text"] for c in r.cases]
    assert "consolidado viejo" not in texts
    assert "nuevo sin consolidar" in texts


@pytest.mark.unit
def test_from_sources_combines_feedback_and_corpus(tmp_path):
    fb = tmp_path / "feedback.jsonl"
    fb.write_text(json.dumps({
        "label": "positive", "prompt": {"user": "feedback case events"},
        "root_cause": "causa feedback", "kubectl_cmd": "kubectl a", "namespaces": ["x"],
    }))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "corpus case events"},
            {"role": "assistant", "content": "ROOT CAUSE: causa corpus\nKUBECTL: kubectl b"},
        ]
    }))
    r = IncidentRetriever.from_sources(str(fb), str(corpus))
    texts = [c["text"] for c in r.cases]
    assert "feedback case events" in texts
    assert "corpus case events" in texts


@pytest.mark.unit
def test_from_feedback_loads_positives(tmp_path):
    p = tmp_path / "feedback.jsonl"
    rows = [
        {"label": "positive", "prompt": {"user": "events A"}, "root_cause": "causa A",
         "kubectl_cmd": "kubectl get pods", "namespaces": ["a"]},
        {"label": "negative", "prompt": {"user": "events B"}, "root_cause": "causa B",
         "kubectl_cmd": "kubectl get pods", "namespaces": ["b"]},
        {"label": "ambiguous", "human_correction": "ROOT CAUSE: corregida\nKUBECTL: kubectl logs x",
         "prompt": {"user": "events C"}, "namespaces": ["c"]},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    r = IncidentRetriever.from_feedback(str(p))
    # positives + corregidos (no el negative puro)
    texts = [c["text"] for c in r.cases]
    assert "events A" in texts
    assert "events C" in texts        # corrección humana entra aunque sea ambiguous
    assert "events B" not in texts    # negative sin corrección se excluye
    # la corrección humana se usa como verdad del caso
    caseC = next(c for c in r.cases if c["text"] == "events C")
    assert caseC["root_cause"] == "corregida"
