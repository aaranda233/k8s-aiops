"""Evalúa un modelo Gemma 4 (adaptadores ORPO, vía Unsloth/transformers) sobre el
test set held-out, con las MISMAS métricas que el arnés Ollama (eval/metrics.py).

Necesario porque Gemma 4 (abril 2026, arquitectura Per-Layer Embeddings +
Gemma4ClippableLinear) aún no tiene soporte de conversión GGUF en llama.cpp ni en
Ollama, así que no se puede servir por Ollama. Aquí generamos con transformers y
puntuamos idéntico para una comparación apples-to-apples con el qwen k8s-rca-orpo.

El prompt se construye EXACTAMENTE como en finetune/train_orpo.load_orpo_dataset
(apply_chat_template con [system,user] + add_generation_prompt, con fallback ChatML)
para que el formato de inferencia coincida con el de entrenamiento.

Uso (dentro de la imagen k8s-rca-train, repo montado en /workspace):
  python eval/eval_gemma4_hf.py --adapter finetune/output/k8s-rca-orpo-gemma4
"""
from __future__ import annotations

# ruff: noqa: I001  (unsloth debe importarse antes que transformers)

from unsloth import FastLanguageModel  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import (  # noqa: E402
    aggregate,
    keyword_hit,
    kubectl_ns_ok,
    kubectl_verb_ok,
    parse_rate,
    rouge_l,
)


def build_prompt(tokenizer, system_content: str, user_content: str) -> str:
    """Réplica exacta de finetune/train_orpo.load_orpo_dataset."""
    prompt_msgs = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    try:
        return tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        result = ""
        for msg in prompt_msgs:
            result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        result += "<|im_start|>assistant\n"
        return result


def parse_output(text: str) -> tuple[str, str]:
    rc = "Could not parse root cause."
    cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
    for line in text.splitlines():
        if line.startswith("ROOT CAUSE:"):
            rc = line.removeprefix("ROOT CAUSE:").strip()
        elif line.startswith("KUBECTL:"):
            cmd = line.removeprefix("KUBECTL:").strip()
    return rc, cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="finetune/output/k8s-rca-orpo-gemma4")
    ap.add_argument("--test-set", default="eval/test_set.jsonl")
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-last", type=int, default=0,
                    help="si >0, conserva solo los ultimos N tokens del prompt "
                         "(replica truncation_mode=keep_end del entrenamiento)")
    ap.add_argument("--out", default="eval/results/eval_gemma4_orpo.json")
    args = ap.parse_args()

    adapter_path = str(PROJECT_ROOT / args.adapter)
    print(f"[1/3] Cargando {adapter_path} (Unsloth 4-bit)...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )
    tok = getattr(processor, "tokenizer", processor)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    FastLanguageModel.for_inference(model)

    with open(PROJECT_ROOT / args.test_set) as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        samples = samples[: args.limit]
    print(f"[2/3] {len(samples)} muestras held-out...")

    results = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        system_content = s["messages"][0]["content"]
        user_content = s["messages"][1]["content"]
        reference = s["messages"][2]["content"]

        ref_rc = ""
        for line in reference.splitlines():
            if line.startswith("ROOT CAUSE:"):
                ref_rc = line.removeprefix("ROOT CAUSE:").strip()

        prompt = build_prompt(tok, system_content, user_content)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        if args.keep_last and inputs["input_ids"].shape[1] > args.keep_last:
            inputs = {k: v[:, -args.keep_last:] for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        latency = time.time() - t0
        gen = tok.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        rc, cmd = parse_output(gen)
        sid = meta.get("scenario_id", "")
        ns = meta.get("namespace", "")

        r = {
            "idx": i,
            "scenario_id": sid,
            "namespace": ns,
            "gen_root_cause": rc,
            "gen_kubectl": cmd,
            "raw": gen[:600],
            "parsed": parse_rate(rc, cmd),
            "keyword_hit": keyword_hit(rc, sid),
            "rouge_l": rouge_l(rc, ref_rc),
            "kubectl_ns_ok": kubectl_ns_ok(cmd, ns),
            "kubectl_verb_ok": kubectl_verb_ok(cmd, sid),
            "latency_s": latency,
        }
        results.append(r)
        if i == 0 or (i + 1) % 10 == 0:
            print(
                f"  [{i+1:3d}/{len(samples)}] {sid:<28} "
                f"parsed={int(r['parsed'])} kw={int(r['keyword_hit'])} "
                f"ns={int(r['kubectl_ns_ok'])} verb={int(r['kubectl_verb_ok'])} "
                f"{latency:.1f}s"
            )

    agg = aggregate(results)
    print("\n[3/3] Agregado Gemma4-E2B-ORPO:")
    print(json.dumps(agg, indent=2))

    outp = PROJECT_ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(
            {"model": "gemma4-e2b-orpo", "aggregate": agg, "results": results},
            f,
            indent=2,
        )
    print(f"\nGuardado: {outp}")


if __name__ == "__main__":
    main()
