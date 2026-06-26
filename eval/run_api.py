#!/usr/bin/env python3
"""Baselines SOTA por API para el diagnóstico (E1).

Evalúa GPT-4o / Claude sobre el MISMO test set y con las MISMAS métricas que los
modelos locales (mismo system prompt del experto, mismo parseo ROOT CAUSE/KUBECTL),
para una comparación apples-to-apples. Guarda los resultados en el formato de
`run_eval.py` (con `per_sample`) → reutilizable por `bootstrap_ci.py`.

Subconjunto por escenario para controlar el gasto (default 3/escenario = 42).

Claves por entorno: OPENAI_API_KEY / ANTHROPIC_API_KEY.

Uso:
    python eval/run_api.py --provider openai    --model gpt-4o
    python eval/run_api.py --provider anthropic --model claude-3-5-sonnet-latest
    python eval/run_api.py --provider openai --per-scenario 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (  # noqa: E402
    aggregate, keyword_hit, kubectl_ns_ok, kubectl_verb_ok, parse_rate, rouge_l,
)
from eval.runner import _SYSTEM_PROMPT  # noqa: E402
from src.diagnostics.ollama_rca import parse_diagnosis  # noqa: E402

import os  # noqa: E402

TEST_SET = PROJECT_ROOT / "eval" / "test_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def select_subset(samples: list[dict], per_scenario: int) -> list[dict]:
    """Toma los primeros `per_scenario` de cada scenario_id (determinista)."""
    by_sc: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_sc[s["metadata"]["scenario_id"]].append(s)
    out = []
    for sid in sorted(by_sc):
        out.extend(by_sc[sid][:per_scenario])
    return out


def call_openai(system: str, user: str, model: str, key: str, timeout: float) -> str:
    r = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "temperature": 0.1, "max_tokens": 220,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_anthropic(system: str, user: str, model: str, key: str, timeout: float) -> str:
    r = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        json={"model": model, "max_tokens": 220, "system": system,
              "messages": [{"role": "user", "content": user}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def call_ollama(system: str, user: str, model: str, key: str, timeout: float) -> str:
    """Config de PRODUCCIÓN local: single-shot experto + digest determinista +
    gramática GBNF vía /api/generate (mismo camino que OllamaRCA)."""
    from src.diagnostics.ollama_rca import _GRAMMAR_GBNF, evidence_digest
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    enriched = evidence_digest(user) + user
    prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
              f"<|im_start|>user\n{enriched}<|im_end|>\n<|im_start|>assistant\n")
    r = httpx.post(
        f"{host}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "grammar": _GRAMMAR_GBNF,
              "options": {"temperature": 0.1, "num_predict": 160, "num_ctx": 2048}},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()["response"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["openai", "anthropic", "ollama"])
    ap.add_argument("--model", default="")
    ap.add_argument("--per-scenario", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    _DEFAULT_MODEL = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-6",
                      "ollama": "k8s-rca-orpo"}
    model = args.model or _DEFAULT_MODEL[args.provider]
    key = ""
    if args.provider != "ollama":
        env_key = "OPENAI_API_KEY" if args.provider == "openai" else "ANTHROPIC_API_KEY"
        key = os.getenv(env_key, "")
        if not key:
            raise SystemExit(f"Falta {env_key} en el entorno.")

    samples = [json.loads(l) for l in TEST_SET.read_text().splitlines() if l.strip()]
    subset = select_subset(samples, args.per_scenario)
    call = {"openai": call_openai, "anthropic": call_anthropic,
            "ollama": call_ollama}[args.provider]
    label = f"api_{args.provider}_{model}".replace("/", "_").replace(":", "_")

    print(f"[E1] {label} · {len(subset)} muestras ({args.per_scenario}/escenario)\n")
    results: list[dict] = []
    for i, s in enumerate(subset):
        meta = s["metadata"]
        sid, ns = meta["scenario_id"], meta.get("namespace", "")
        user = s["messages"][1]["content"]
        ref = s["messages"][2]["content"]
        t0 = time.time()
        try:
            text = call(_SYSTEM_PROMPT, user, model, key, args.timeout)
        except Exception as e:
            print(f"  [{i+1:3d}/{len(subset)}] {sid:<26} ERROR: {str(e)[:80]}")
            text = ""
        latency = time.time() - t0
        rc, kc = parse_diagnosis(text)
        r = {
            "idx": i, "scenario_id": sid, "namespace": ns,
            "gen_root_cause": rc, "gen_kubectl": kc,
            "parsed": parse_rate(rc, kc),
            "keyword_hit": keyword_hit(rc, sid),
            "rouge_l": rouge_l(rc, ref),
            "kubectl_ns_ok": kubectl_ns_ok(kc, ns),
            "kubectl_verb_ok": kubectl_verb_ok(kc, sid),
            "latency_s": round(latency, 3),
        }
        results.append(r)
        kw = "✓" if r["keyword_hit"] else "✗"
        print(f"  [{i+1:3d}/{len(subset)}] {sid:<26} kw={kw} ns={'✓' if r['kubectl_ns_ok'] else '·'} {latency:5.1f}s")

    agg = aggregate(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_api_{args.provider}_{ts}.json"
    out_path.write_text(json.dumps(
        {"timestamp": ts, "aggregate": {label: agg}, "per_sample": {label: results}},
        ensure_ascii=False, indent=2))
    print(f"\n[agg] {label}: " + " · ".join(f"{k}={v}" for k, v in agg.items() if k != "n"))
    print(f"[resultados] {out_path}")


if __name__ == "__main__":
    main()
