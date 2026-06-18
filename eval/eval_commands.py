"""
Evaluación determinista del constructor de comandos (sin GPU/modelo).

Mide kubectl_ns_ok / kubectl_verb_ok del command_builder sobre el test set,
usando la evidencia (mensaje de usuario) y el namespace/scenario de cada muestra.
Compara contra el baseline del SLM reportado en el paper.

Uso:  python3 -m eval.eval_commands
"""

import json
from pathlib import Path

from eval.metrics import kubectl_ns_ok, kubectl_verb_ok
from src.diagnostics.command_builder import build_command

_TEST_SET = Path(__file__).parent / "test_set.jsonl"

# Baseline del SLM fine-tuneado (SFT) reportado en RESEARCH.md §9.
_BASELINE = {"ns_ok": 0.330, "verb_ok": 0.410}


def _user_evidence(sample: dict) -> str:
    for m in sample.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def run() -> dict:
    samples = [json.loads(line) for line in _TEST_SET.read_text().splitlines() if line.strip()]
    ns_hits = verb_hits = 0
    for s in samples:
        meta = s.get("metadata", {})
        ns = meta.get("namespace", "")
        scenario = meta.get("scenario_id", "")
        evidence = _user_evidence(s)
        cmd = build_command(evidence, namespace=ns)
        ns_hits += int(kubectl_ns_ok(cmd, ns))
        verb_hits += int(kubectl_verb_ok(cmd, scenario))
    n = len(samples)
    res = {"n": n, "ns_ok": round(ns_hits / n, 3), "verb_ok": round(verb_hits / n, 3)}

    print(f"\n  Command builder — evaluación determinista sobre {n} muestras\n")
    print(f"  {'Métrica':<12}{'Builder':>10}{'SLM (SFT)':>12}{'Δ':>10}")
    print(f"  {'-'*42}")
    for key, label in (("ns_ok", "NS-ok%"), ("verb_ok", "Verb-ok%")):
        b, base = res[key] * 100, _BASELINE[key] * 100
        print(f"  {label:<12}{b:>9.1f}%{base:>11.1f}%{b - base:>+9.1f}")
    print()
    return res


if __name__ == "__main__":
    run()
