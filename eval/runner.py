"""
Ejecuta inferencia en uno o varios modelos Ollama sobre el test set.

Devuelve una lista de resultados con métricas por muestra.

Modo grammar (--grammar):
  Usa GBNF grammar-constrained sampling para forzar el formato
  ROOT CAUSE: / KUBECTL: a nivel de token. Parse% -> ~100% garantizado.
  Esto desacopla la calidad del CONTENIDO del fallo de formato:
  una vez forzado el formato, Keyword%, NS-ok% y Verb-ok% miden
  exclusivamente si el modelo sabe la respuesta, no si sabe escribirla.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (
    aggregate,
    keyword_hit,
    kubectl_ns_ok,
    kubectl_verb_ok,
    parse_rate,
    rouge_l,
)

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You will receive a set of raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task is to:
1. Identify the root cause of the anomaly in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate the issue.

Output format (strict):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>

Be concise. Focus on actionable diagnosis."""

# GBNF grammar que garantiza el formato ROOT CAUSE: ... \n KUBECTL: ...
# Desacopla la calidad del contenido del fallo de formato:
# una vez forzado el formato, Keyword%/NS-ok%/Verb-ok% miden solo si
# el modelo sabe la respuesta, no si sabe escribirla.
_GRAMMAR_GBNF = r"""root   ::= "ROOT CAUSE: " rc-text "\nKUBECTL: " kubectl-text
rc-text      ::= [^\n]+ (" " [^\n]+)*
kubectl-text ::= "kubectl " [^\n]+
"""


@dataclass
class ModelConfig:
    name: str           # nombre legible (para la tabla)
    ollama_model: str   # nombre en Ollama (ej. "k8s-rca-slm", "qwen2.5:1.5b")
    host: str = "http://192.168.2.205:11434"
    temperature: float = 0.0
    num_predict: int = 300
    use_grammar: bool = False   # activar grammar-constrained sampling


@dataclass
class HybridModelConfig:
    name: str
    base_model: str            # investigador vanilla (qwen2.5:1.5b)
    expert_model: str          # diagnosticador fine-tuneado (k8s-rca-orpo)
    host: str = "http://192.168.2.205:11434"
    max_steps: int = 3
    temperature: float = 0.0
    num_predict: int = 300


def _call_ollama(sample: dict, cfg: ModelConfig) -> tuple[str, str, float]:
    """Llama a Ollama y devuelve (root_cause, kubectl, latency_s)."""
    msgs = sample["messages"]
    user_content = msgs[1]["content"]

    if cfg.use_grammar:
        # Grammar-constrained sampling usa /api/generate (no /api/chat)
        # porque el campo "grammar" solo esta disponible en ese endpoint.
        prompt = (
            f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "model":   cfg.ollama_model,
            "prompt":  prompt,
            "stream":  False,
            "grammar": _GRAMMAR_GBNF,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.num_predict,
            },
        }
        t0 = time.time()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{cfg.host}/api/generate", json=payload)
            resp.raise_for_status()
        latency = time.time() - t0
        text = resp.json()["response"].strip()
    else:
        payload = {
            "model": cfg.ollama_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.num_predict,
            },
        }
        t0 = time.time()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{cfg.host}/api/chat", json=payload)
            resp.raise_for_status()
        latency = time.time() - t0
        text = resp.json()["message"]["content"].strip()

    root_cause = "Could not parse root cause."
    kubectl_cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"

    for line in text.splitlines():
        if line.startswith("ROOT CAUSE:"):
            root_cause = line.removeprefix("ROOT CAUSE:").strip()
        elif line.startswith("KUBECTL:"):
            kubectl_cmd = line.removeprefix("KUBECTL:").strip()

    return root_cause, kubectl_cmd, latency


def evaluate_model(
    test_samples: list[dict],
    cfg: ModelConfig,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Evalua un modelo sobre el test set.

    Returns:
        (per_sample_results, aggregate_metrics)
    """
    if cfg.use_grammar and verbose:
        print("  [grammar] GBNF activo — formato ROOT CAUSE/KUBECTL forzado a nivel token")

    results = []
    n = len(test_samples)

    for i, sample in enumerate(test_samples):
        meta = sample.get("metadata", {})
        reference_output = sample["messages"][2]["content"]

        ref_root_cause = ""
        ref_kubectl = ""
        for line in reference_output.splitlines():
            if line.startswith("ROOT CAUSE:"):
                ref_root_cause = line.removeprefix("ROOT CAUSE:").strip()
            elif line.startswith("KUBECTL:"):
                ref_kubectl = line.removeprefix("KUBECTL:").strip()

        try:
            gen_root_cause, gen_kubectl, latency = _call_ollama(sample, cfg)
        except Exception as e:
            if verbose:
                print(f"  [!] error en muestra {i}: {e}")
            gen_root_cause = ""
            gen_kubectl = ""
            latency = 0.0

        scenario_id = meta.get("scenario_id", "")
        namespace   = meta.get("namespace", "")

        result = {
            "idx":             i,
            "scenario_id":     scenario_id,
            "namespace":       namespace,
            "gen_root_cause":  gen_root_cause,
            "gen_kubectl":     gen_kubectl,
            "ref_root_cause":  ref_root_cause,
            "ref_kubectl":     ref_kubectl,
            "grammar_forced":  cfg.use_grammar,
            "parsed":          parse_rate(gen_root_cause, gen_kubectl),
            "keyword_hit":     keyword_hit(gen_root_cause, scenario_id),
            "rouge_l":         rouge_l(gen_root_cause, ref_root_cause),
            "kubectl_ns_ok":   kubectl_ns_ok(gen_kubectl, namespace),
            "kubectl_verb_ok": kubectl_verb_ok(gen_kubectl, scenario_id),
            "latency_s":       latency,
        }
        results.append(result)

        if verbose:
            parsed_ok = "✓" if result["parsed"] else "✗"
            kw_ok     = "✓" if result["keyword_hit"] else "✗"
            print(
                f"  [{i+1:3d}/{n}] {scenario_id:<30} "
                f"parsed={parsed_ok} kw={kw_ok} "
                f"rl={result['rouge_l']:.2f} "
                f"lat={latency:.2f}s"
            )

    agg = aggregate(results)
    return results, agg


# ---------------------------------------------------------------------------
# Modo HYBRID: base model investiga, expert model diagnostica
# ---------------------------------------------------------------------------

_INVESTIGATOR_SYSTEM = """\
You are a Kubernetes SRE assistant. Given anomalous cluster events, plan your investigation.

For each step output:
THOUGHT: <reasoning about what to check>
ACTION: kubectl <read-only command>

When done (or after 3 steps), output:
THOUGHT: <final summary>
DONE"""

_EXPERT_SYSTEM = """\
You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You receive raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task:
1. Identify the root cause of the anomaly in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate the issue.

Output format (strict):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>

Be concise. Focus on actionable diagnosis."""


def _call_hybrid(sample: dict, cfg: HybridModelConfig) -> tuple[str, str, float, int]:
    """Llama al pipeline híbrido y devuelve (root_cause, kubectl, latency_s, steps)."""
    user_content = sample["messages"][1]["content"]
    t0 = time.time()

    # Fase 1: investigador (base model)
    messages = [
        {"role": "system", "content": _INVESTIGATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    plan_lines: list[str] = []
    steps = 0

    with httpx.Client(timeout=120.0) as client:
        for _ in range(cfg.max_steps):
            payload = {
                "model": cfg.base_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": cfg.temperature, "num_predict": 200},
            }
            resp = client.post(f"{cfg.host}/api/chat", json=payload)
            resp.raise_for_status()
            response = resp.json()["message"]["content"].strip()
            steps += 1

            thought, action, is_done = "", None, False
            for line in response.splitlines():
                line = line.strip()
                if line.startswith("THOUGHT:"):
                    thought = line.removeprefix("THOUGHT:").strip()
                elif line.startswith("ACTION:"):
                    action = line.removeprefix("ACTION:").strip()
                elif line == "DONE":
                    is_done = True

            if thought:
                plan_lines.append(f"  Step {steps} thought: {thought}")
            if action:
                plan_lines.append(f"  Step {steps} action (planned): {action}")

            if is_done or not action:
                break

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": "[dry-run: no cluster access] Continue or output DONE.",
            })

        # Fase 2: experto con grammar-constrained sampling → formato garantizado
        plan_section = ""
        if plan_lines:
            plan_section = "\n\n[Investigation notes from first-pass analysis:]\n" + "\n".join(plan_lines)

        expert_prompt = (
            f"<|im_start|>system\n{_EXPERT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content + plan_section}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        expert_payload = {
            "model": cfg.expert_model,
            "prompt": expert_prompt,
            "stream": False,
            "grammar": _GRAMMAR_GBNF,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.num_predict,
                "num_ctx": 2048,
            },
        }
        resp = client.post(f"{cfg.host}/api/generate", json=expert_payload)
        resp.raise_for_status()
        expert_text = resp.json()["response"].strip()

    latency = time.time() - t0
    root_cause = "Could not parse root cause."
    kubectl_cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
    for line in expert_text.splitlines():
        if line.startswith("ROOT CAUSE:"):
            root_cause = line.removeprefix("ROOT CAUSE:").strip()
        elif line.startswith("KUBECTL:"):
            kubectl_cmd = line.removeprefix("KUBECTL:").strip()

    return root_cause, kubectl_cmd, latency, steps


def evaluate_hybrid_model(
    test_samples: list[dict],
    cfg: HybridModelConfig,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Evalúa el pipeline híbrido sobre el test set."""
    if verbose:
        print(f"  [hybrid] base={cfg.base_model} + expert={cfg.expert_model} · max_steps={cfg.max_steps}")

    results = []
    n = len(test_samples)

    for i, sample in enumerate(test_samples):
        meta = sample.get("metadata", {})
        reference_output = sample["messages"][2]["content"]

        ref_root_cause, ref_kubectl = "", ""
        for line in reference_output.splitlines():
            if line.startswith("ROOT CAUSE:"):
                ref_root_cause = line.removeprefix("ROOT CAUSE:").strip()
            elif line.startswith("KUBECTL:"):
                ref_kubectl = line.removeprefix("KUBECTL:").strip()

        try:
            gen_root_cause, gen_kubectl, latency, steps = _call_hybrid(sample, cfg)
        except Exception as e:
            if verbose:
                print(f"  [!] error en muestra {i}: {e}")
            gen_root_cause, gen_kubectl, latency, steps = "", "", 0.0, 0

        scenario_id = meta.get("scenario_id", "")
        namespace   = meta.get("namespace", "")

        result = {
            "idx":             i,
            "scenario_id":     scenario_id,
            "namespace":       namespace,
            "gen_root_cause":  gen_root_cause,
            "gen_kubectl":     gen_kubectl,
            "ref_root_cause":  ref_root_cause,
            "ref_kubectl":     ref_kubectl,
            "grammar_forced":  False,
            "parsed":          parse_rate(gen_root_cause, gen_kubectl),
            "keyword_hit":     keyword_hit(gen_root_cause, scenario_id),
            "rouge_l":         rouge_l(gen_root_cause, ref_root_cause),
            "kubectl_ns_ok":   kubectl_ns_ok(gen_kubectl, namespace),
            "kubectl_verb_ok": kubectl_verb_ok(gen_kubectl, scenario_id),
            "latency_s":       latency,
            "hybrid_steps":    steps,
        }
        results.append(result)

        if verbose:
            parsed_ok = "✓" if result["parsed"] else "✗"
            kw_ok     = "✓" if result["keyword_hit"] else "✗"
            print(
                f"  [{i+1:3d}/{n}] {scenario_id:<30} "
                f"parsed={parsed_ok} kw={kw_ok} "
                f"rl={result['rouge_l']:.2f} "
                f"steps={steps} lat={latency:.2f}s"
            )

    agg = aggregate(results)
    return results, agg
