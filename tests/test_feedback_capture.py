"""
Tests de la captura de feedback del bucle cerrado (dataset/feedback_capture.py).
"""

import json
import time

import pytest

from dataset.feedback_capture import (
    AMBIGUOUS,
    NEGATIVE,
    POSITIVE,
    build_example,
    derive_label,
    record_feedback,
)
from src.remediation.incident_store import (
    STATUS_RESOLVED,
    Incident,
    IncidentStore,
)

# ── Etiquetado ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_label_positive_approved_resolved():
    assert derive_label("approved", "resolved", True) == POSITIVE
    assert derive_label(None, "resolved", True) == POSITIVE  # verificado auto


@pytest.mark.unit
def test_label_negative_rejected_or_failed():
    assert derive_label("rejected", "rejected", None) == NEGATIVE
    assert derive_label("approved", "failed", False) == NEGATIVE


@pytest.mark.unit
def test_label_ambiguous_timeout_or_unverified():
    assert derive_label(None, "timeout", None) == AMBIGUOUS
    assert derive_label(None, "escalated", None) == AMBIGUOUS


# ── build_example ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_build_example_requires_prompt():
    assert build_example({"id": "INC-1", "prompt_user": ""}) is None


@pytest.mark.unit
def test_build_example_structure():
    inc = {
        "id": "INC-1", "prompt_user": "Anomaly Score: 0.9\nEvents: ...",
        "root_cause": "OOMKilled en api", "kubectl_cmd": "kubectl describe pod api",
        "response": "approved", "status": "resolved", "verified": True,
        "risk_level": 1, "namespaces": ["prod"], "score": 0.9,
    }
    ex = build_example(inc)
    assert ex["label"] == POSITIVE
    assert ex["prompt"]["user"].startswith("Anomaly Score")
    assert "ROOT CAUSE: OOMKilled" in ex["model_output"]
    assert "KUBECTL: kubectl describe pod api" in ex["model_output"]
    assert ex["source"] == "closed_loop"


# ── record_feedback (escritura) ─────────────────────────────────────────────

@pytest.mark.unit
def test_record_feedback_writes_line(tmp_path):
    p = tmp_path / "feedback.jsonl"
    inc = {
        "id": "INC-1", "prompt_user": "eventos...", "root_cause": "x",
        "kubectl_cmd": "kubectl get pods", "response": "rejected",
        "status": "rejected", "verified": None, "namespaces": [],
    }
    ex = record_feedback(inc, path=str(p))
    assert ex is not None and ex["label"] == NEGATIVE
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["incident_id"] == "INC-1"


@pytest.mark.unit
def test_record_feedback_skips_without_prompt(tmp_path):
    p = tmp_path / "feedback.jsonl"
    assert record_feedback({"id": "INC-1", "prompt_user": ""}, path=str(p)) is None
    assert not p.exists()


# ── Integración con IncidentStore (hook en terminal) ────────────────────────

@pytest.mark.unit
def test_store_hook_captures_feedback_on_terminal(tmp_path):
    captured = []
    store = IncidentStore()
    store.set_feedback_hook(lambda inc: captured.append(record_feedback(inc, path=str(tmp_path / "f.jsonl"))))
    inc = Incident(
        id="INC-1", created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="memoria", kubectl_cmd="kubectl rollout restart deployment/x -n prod",
        risk_level=1, risk_label="reversible", prompt_user="Anomaly...eventos",
    )
    store.add(inc)
    store.set_response("INC-1", "approved")
    store.update("INC-1", status=STATUS_RESOLVED, verified=True)
    assert len(captured) == 1
    assert captured[0]["label"] == POSITIVE
