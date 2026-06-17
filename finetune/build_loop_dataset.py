"""
Construye el dataset de preferencias ORPO desde el feedback del bucle cerrado.

Entrada:  data/feedback/feedback.jsonl  (de dataset/feedback_capture.py)
Salida:   dataset/output/loop_dpo_dataset.jsonl  (pares chosen/rejected)
          dataset/output/orpo_train_v{N}.jsonl   (loop + replay del base, anti-olvido)

Reglas de pareo (de mayor a menor calidad):
  1. Con corrección humana: chosen = corrección, rejected = salida original del
     modelo. MISMO prompt -> par ORPO perfecto y sin llamar a ningún modelo.
  2. positive sin corrección: chosen = salida del modelo (validada por el outcome);
     el rejected se genera opcionalmente con el modelo base vanilla (--gen-rejected).
  3. negative sin corrección: no aporta "chosen" -> se descarta (no se entrena a
     imitar un error; solo se usa como rejected si comparte prompt con un chosen).

Salvaguardas:
  - Dedup por hash del prompt de usuario.
  - Filtro ROUGE: descarta pares casi idénticos (chosen ≈ rejected).
  - Replay del dataset base mezclado (mitiga catastrophic forgetting) — OBLIGATORIO.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from eval.metrics import rouge_l

ROUGE_FILTER = 0.8           # descartar par si chosen ≈ rejected
DEFAULT_LOOP_RATIO = 0.30    # 30% feedback nuevo / 70% base (replay)


def _assistant(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


def _prompt_msgs(example: dict) -> list[dict]:
    p = example["prompt"]
    return [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user"]},
    ]


def _prompt_hash(example: dict) -> str:
    return hashlib.sha1(example["prompt"]["user"].encode("utf-8")).hexdigest()


def feedback_to_pairs(examples: list[dict], gen_rejected=None) -> list[dict]:
    """Convierte ejemplos de feedback en pares ORPO {prompt, chosen, rejected}.

    gen_rejected: callable(prompt_msgs)->str opcional para generar el rejected de
    los positives sin corrección (p.ej. el modelo base vanilla). Si None, esos
    positives se omiten.
    """
    pairs: list[dict] = []
    for ex in examples:
        prompt = _prompt_msgs(ex)
        correction = ex.get("human_correction")
        model_out = ex.get("model_output", "")

        if correction:
            chosen, rejected = correction, model_out
        elif ex.get("label") == "positive":
            if gen_rejected is None:
                continue
            chosen, rejected = model_out, gen_rejected(prompt)
        else:
            continue  # negative sin corrección no aporta chosen

        if not chosen or not rejected:
            continue
        if rouge_l(chosen, rejected) > ROUGE_FILTER:
            continue  # demasiado parecidos -> no enseñan preferencia

        pairs.append({
            "prompt": prompt,
            "chosen": _assistant(chosen),
            "rejected": _assistant(rejected),
            "_source": "closed_loop",
            "_incident": ex.get("incident_id"),
        })
    return pairs


def dedup(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        h = _prompt_hash(ex)
        if h in seen:
            continue
        seen.add(h)
        out.append(ex)
    return out


def cap_per_namespace(examples: list[dict], cap: int) -> list[dict]:
    """Limita ejemplos por namespace para que un incidente recurrente no domine."""
    counts: dict[str, int] = {}
    out = []
    for ex in examples:
        ns = (ex.get("namespaces") or ["?"])[0]
        if counts.get(ns, 0) >= cap:
            continue
        counts[ns] = counts.get(ns, 0) + 1
        out.append(ex)
    return out


def mix_with_replay(loop_pairs: list[dict], base_pairs: list[dict],
                    loop_ratio: float = DEFAULT_LOOP_RATIO, seed: int = 42) -> list[dict]:
    """Mezcla pares del bucle con una muestra del dataset base (anti-olvido)."""
    if not loop_pairs:
        return list(base_pairs)
    rng = random.Random(seed)
    # nº de ejemplos base para que loop sea ~loop_ratio del total
    n_base = int(len(loop_pairs) * (1 - loop_ratio) / max(loop_ratio, 1e-6))
    n_base = min(n_base, len(base_pairs))
    sampled_base = rng.sample(base_pairs, n_base) if n_base else []
    mixed = loop_pairs + sampled_base
    rng.shuffle(mixed)
    return mixed


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", default="data/feedback/feedback.jsonl")
    ap.add_argument("--base", default="dataset/output/dpo_dataset_v2.jsonl",
                    help="dataset base ORPO para replay (anti-olvido)")
    ap.add_argument("--out", default="dataset/output/orpo_train_loop.jsonl")
    ap.add_argument("--loop-ratio", type=float, default=DEFAULT_LOOP_RATIO)
    ap.add_argument("--cap-per-ns", type=int, default=20)
    args = ap.parse_args()

    examples = dedup(_read_jsonl(Path(args.feedback)))
    examples = cap_per_namespace(examples, args.cap_per_ns)
    loop_pairs = feedback_to_pairs(examples)
    base_pairs = _read_jsonl(Path(args.base))
    mixed = mix_with_replay(loop_pairs, base_pairs, args.loop_ratio)

    _write_jsonl(Path("dataset/output/loop_dpo_dataset.jsonl"), loop_pairs)
    _write_jsonl(Path(args.out), mixed)
    print(f"feedback examples: {len(examples)}")
    print(f"loop pairs: {len(loop_pairs)} (con corrección humana o positives)")
    print(f"base replay: {len(mixed) - len(loop_pairs)}")
    print(f"dataset final: {len(mixed)} -> {args.out}")


if __name__ == "__main__":
    main()
