"""
Tests del constructor de dataset de preferencias desde el feedback (Fase 2).
"""

import pytest

from finetune.build_loop_dataset import (
    cap_per_namespace,
    dedup,
    feedback_to_pairs,
    mix_with_replay,
)


def _fb(iid, user, label="positive", model_output="ROOT CAUSE: a\nKUBECTL: kubectl get pods",
        correction=None, namespaces=None):
    return {
        "incident_id": iid,
        "prompt": {"system": "SYS", "user": user},
        "model_output": model_output,
        "label": label,
        "human_correction": correction,
        "namespaces": namespaces or ["default"],
    }


@pytest.mark.unit
def test_correction_makes_clean_pair():
    ex = _fb("INC-1", "eventos del pod X",
             model_output="ROOT CAUSE: malo\nKUBECTL: kubectl delete pod x",
             correction="ROOT CAUSE: OOMKilled, subir limite\nKUBECTL: kubectl set resources ...")
    pairs = feedback_to_pairs([ex])
    assert len(pairs) == 1
    assert pairs[0]["chosen"][0]["content"].startswith("ROOT CAUSE: OOMKilled")
    assert "delete pod x" in pairs[0]["rejected"][0]["content"]
    assert pairs[0]["prompt"][1]["content"] == "eventos del pod X"


@pytest.mark.unit
def test_positive_without_correction_skipped_without_generator():
    pairs = feedback_to_pairs([_fb("INC-1", "ev", label="positive", correction=None)])
    assert pairs == []  # sin generador de rejected, se omite


@pytest.mark.unit
def test_positive_with_generator_creates_pair():
    ex = _fb("INC-1", "ev", label="positive",
             model_output="ROOT CAUSE: causa correcta detallada\nKUBECTL: kubectl describe pod x")
    def gen(prompt):
        return "ROOT CAUSE: respuesta vaga del modelo base\nKUBECTL: kubectl get pods"
    pairs = feedback_to_pairs([ex], gen_rejected=gen)
    assert len(pairs) == 1
    assert "correcta" in pairs[0]["chosen"][0]["content"]


@pytest.mark.unit
def test_negative_without_correction_skipped():
    pairs = feedback_to_pairs([_fb("INC-1", "ev", label="negative", correction=None)])
    assert pairs == []


@pytest.mark.unit
def test_rouge_filter_discards_near_identical():
    # chosen y rejected casi idénticos -> filtrados
    ex = _fb("INC-1", "ev",
             model_output="ROOT CAUSE: el pod se cae\nKUBECTL: kubectl get pods",
             correction="ROOT CAUSE: el pod se cae\nKUBECTL: kubectl get pods")
    assert feedback_to_pairs([ex]) == []


@pytest.mark.unit
def test_dedup_by_prompt():
    a = _fb("INC-1", "mismo prompt")
    b = _fb("INC-2", "mismo prompt")
    c = _fb("INC-3", "otro prompt")
    out = dedup([a, b, c])
    assert len(out) == 2


@pytest.mark.unit
def test_cap_per_namespace():
    exs = [_fb(f"INC-{i}", f"ev{i}", namespaces=["prod"]) for i in range(5)]
    assert len(cap_per_namespace(exs, cap=2)) == 2


@pytest.mark.unit
def test_mix_with_replay_ratio():
    loop = [{"prompt": [], "chosen": [], "rejected": [], "_source": "closed_loop"} for _ in range(3)]
    base = [{"prompt": [], "chosen": [], "rejected": [], "_source": "base"} for _ in range(100)]
    mixed = mix_with_replay(loop, base, loop_ratio=0.30)
    # 3 loop ~30% -> ~7 base -> ~10 total
    n_loop = sum(1 for m in mixed if m.get("_source") == "closed_loop")
    assert n_loop == 3
    assert 9 <= len(mixed) <= 11


@pytest.mark.unit
def test_mix_without_loop_returns_base():
    base = [{"_source": "base"} for _ in range(5)]
    assert mix_with_replay([], base) == base
