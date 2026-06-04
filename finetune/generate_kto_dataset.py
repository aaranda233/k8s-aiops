"""
Genera el dataset KTO (Kahneman-Tversky Optimization) a partir de datos existentes.

KTOTrainer espera muestras individuales con label booleano:
  - label=True  → respuesta deseable (ground-truth correcto)
  - label=False → respuesta indeseable (rejected del dataset DPO v2)

No necesita pares — cada muestra es independiente.

Fuentes:
  - Positivos (True):  dataset/output/combined.jsonl  (SFT ground-truth)
  - Negativos (False): dataset/output/dpo_dataset_v2.jsonl → columna rejected

Uso:
  python finetune/generate_kto_dataset.py
  python finetune/generate_kto_dataset.py --ratio 1.0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-dataset",  default="dataset/output/combined.jsonl",
                   help="Dataset SFT con respuestas ground-truth (positivos)")
    p.add_argument("--dpo-dataset",  default="dataset/output/dpo_dataset_v2.jsonl",
                   help="Dataset DPO v2 con pares chosen/rejected")
    p.add_argument("--output",       default="dataset/output/kto_dataset.jsonl")
    p.add_argument("--ratio",        type=float, default=1.0,
                   help="Ratio negativos/positivos (default: 1.0 → 1:1)")
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args()


def build_prompt(messages: list[dict], tokenizer=None) -> str:
    """Construye el prompt en formato ChatML a partir de mensajes [system, user]."""
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


def load_positives(sft_path: str) -> list[dict]:
    """Carga respuestas ground-truth del dataset SFT como muestras deseables."""
    records = []
    with open(sft_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = sample.get("messages", [])
            if len(messages) < 3:
                continue

            # Separar system+user (prompt) del assistant (completion)
            prompt_msgs = [m for m in messages if m["role"] in ("system", "user")]
            assistant = next((m for m in messages if m["role"] == "assistant"), None)
            if not assistant:
                continue

            # Verificar formato básico
            content = assistant["content"]
            if "ROOT CAUSE:" not in content or "KUBECTL:" not in content:
                continue

            records.append({
                "prompt":     build_prompt(prompt_msgs),
                "completion": content,
                "label":      True,
            })
    return records


def load_negatives(dpo_path: str) -> list[dict]:
    """Extrae las respuestas rejected del dataset DPO v2 como muestras indeseables."""
    records = []
    with open(dpo_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            prompt_msgs  = sample.get("prompt", [])    # [system, user]
            rejected_msg = sample.get("rejected", [])  # [assistant_wrong]

            if not prompt_msgs or not rejected_msg:
                continue

            rejected_content = rejected_msg[0]["content"] if isinstance(rejected_msg[0], dict) else rejected_msg[0]

            records.append({
                "prompt":     build_prompt(prompt_msgs),
                "completion": rejected_content,
                "label":      False,
            })
    return records


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print(f"\n{'='*60}")
    print("  K8s-RCA-SLM — Generación Dataset KTO")
    print(f"{'='*60}\n")

    # Cargar positivos
    print(f"[1/3] Cargando positivos desde {args.sft_dataset} ...")
    positives = load_positives(args.sft_dataset)
    print(f"  {len(positives)} muestras deseables (label=True)")

    # Cargar negativos
    print(f"[2/3] Cargando negativos desde {args.dpo_dataset} ...")
    negatives = load_negatives(args.dpo_dataset)
    print(f"  {len(negatives)} muestras indeseables disponibles (label=False)")

    # Balancear según ratio
    n_neg = int(len(positives) * args.ratio)
    if n_neg < len(negatives):
        negatives = random.sample(negatives, n_neg)
    print(f"  {len(negatives)} negativos seleccionados (ratio={args.ratio:.1f})")

    # Mezclar y guardar
    all_samples = positives + negatives
    random.shuffle(all_samples)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_pos = sum(1 for s in all_samples if s["label"])
    n_neg_final = sum(1 for s in all_samples if not s["label"])

    print(f"\n[3/3] Dataset KTO guardado: {out_path}")
    print(f"  Total: {len(all_samples)} muestras")
    print(f"  Deseables  (True):  {n_pos}")
    print(f"  Indeseables (False): {n_neg_final}")
    print(f"  Ratio positivos/negativos: {n_pos/n_neg_final:.2f}")
    print(f"\n  Usar con: python finetune/train_kto.py --dataset {out_path}\n")


if __name__ == "__main__":
    main()
