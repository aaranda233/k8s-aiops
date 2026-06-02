"""
Fase 1 del pipeline DPO: genera el dataset de pares (chosen, rejected).

Para cada muestra del training set:
  - chosen  = respuesta correcta ya existente (ground truth)
  - rejected = respuesta generada por el modelo base vanilla (qwen2.5:1.5b)

Filtrado de calidad:
  - Descarta pares con ROUGE-L(chosen, rejected) > 0.8 (demasiado parecidos)
  - Descarta rejected que parseen correctamente Y tengan todas las keywords
    del escenario (sería un "buen" rejected, no sirve para DPO)

Salida: dataset/output/dpo_dataset.jsonl en formato TRL DPOTrainer

Uso:
  python finetune/generate_dpo_dataset.py
  python finetune/generate_dpo_dataset.py --host http://localhost:11434 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import SCENARIO_KEYWORDS, rouge_l

TRAIN_PATH  = PROJECT_ROOT / "dataset" / "output" / "combined.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "output" / "dpo_dataset.jsonl"
STATS_PATH  = PROJECT_ROOT / "dataset" / "output" / "dpo_stats.json"

DRAFT_MODEL   = "qwen2.5:1.5b"
ROUGE_FILTER  = 0.8    # descartar si chosen ≈ rejected
KEYWORD_ALL   = 2      # descartar si rejected tiene ≥ N keywords del escenario


# ── Inferencia ────────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict], host: str, model: str) -> str:
    """Llama a Ollama y devuelve el texto generado."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{host}/api/chat", json=payload)
        resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _extract_root_cause(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("ROOT CAUSE:"):
            return line.removeprefix("ROOT CAUSE:").strip()
    return text[:200]  # fallback: primeros 200 chars si no parsea


def _keyword_hits(text: str, scenario_id: str) -> int:
    keywords = SCENARIO_KEYWORDS.get(scenario_id, [])
    low = text.lower()
    return sum(1 for kw in keywords if kw in low)


# ── Procesamiento de una muestra ──────────────────────────────────────────────

def process_sample(
    sample: dict,
    host: str,
    idx: int,
    total: int,
) -> tuple[dict | None, str]:
    """
    Genera el par DPO para una muestra.
    Devuelve (par_dpo, motivo_descarte) — si descarta, par_dpo=None.
    """
    msgs = sample["messages"]
    meta = sample.get("metadata", {})
    scenario_id = meta.get("scenario_id", "")

    # Prompt = system + user (sin la respuesta)
    prompt_msgs = [msgs[0], msgs[1]]

    # chosen = respuesta ground truth
    chosen_text = msgs[2]["content"]
    chosen_rc   = _extract_root_cause(chosen_text)

    # rejected = respuesta del modelo vanilla
    try:
        rejected_text = _call_ollama(prompt_msgs, host, DRAFT_MODEL)
    except Exception as e:
        return None, f"ollama_error: {e}"

    rejected_rc = _extract_root_cause(rejected_text)

    # ── Filtros de calidad ────────────────────────────────────────────────────

    # 1. Demasiado parecidos (vanilla acertó)
    rl = rouge_l(chosen_rc, rejected_rc)
    if rl > ROUGE_FILTER:
        return None, f"rouge_too_high ({rl:.2f})"

    # 2. Rejected tiene demasiadas keywords correctas (también es buena respuesta)
    kw_hits = _keyword_hits(rejected_text, scenario_id)
    if kw_hits >= KEYWORD_ALL:
        return None, f"rejected_too_good (kw_hits={kw_hits})"

    # ── Construir par DPO ─────────────────────────────────────────────────────
    pair = {
        "prompt": [
            {"role": "system", "content": msgs[0]["content"]},
            {"role": "user",   "content": msgs[1]["content"]},
        ],
        "chosen":   [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "metadata": {
            "scenario_id":  scenario_id,
            "rouge_l":      round(rl, 3),
            "kw_hits_rej":  kw_hits,
        },
    }
    return pair, "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host",    default="http://localhost:11434")
    p.add_argument("--workers", type=int, default=2,
                   help="Llamadas paralelas a Ollama (default: 2)")
    p.add_argument("--limit",   type=int, default=None,
                   help="Limitar a N muestras (para pruebas)")
    return p.parse_args()


def main():
    args = parse_args()

    # Verificar que el modelo está disponible
    try:
        r = httpx.get(f"{args.host}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(DRAFT_MODEL in m for m in models):
            print(f"[!] Modelo '{DRAFT_MODEL}' no encontrado en Ollama.")
            print(f"    Modelos disponibles: {models}")
            sys.exit(1)
        print(f"[ok] Modelo '{DRAFT_MODEL}' disponible.")
    except Exception as e:
        print(f"[!] No se puede conectar a Ollama en {args.host}: {e}")
        sys.exit(1)

    # Cargar training set
    samples = [json.loads(l) for l in TRAIN_PATH.read_text().splitlines() if l.strip()]
    if args.limit:
        samples = samples[:args.limit]
    total = len(samples)
    print(f"[dataset] {total} muestras cargadas de {TRAIN_PATH.name}")
    print(f"[config]  workers={args.workers} · rouge_filter={ROUGE_FILTER} · host={args.host}\n")

    pairs   = []
    stats   = {"ok": 0, "rouge_too_high": 0, "rejected_too_good": 0, "ollama_error": 0}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_sample, s, args.host, i, total): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                pair, reason = future.result()
            except Exception as e:
                reason = f"ollama_error: {e}"
                pair   = None

            # contabilizar
            key = reason.split(" ")[0] if " " in reason else reason
            stats[key] = stats.get(key, 0) + 1

            if pair:
                pairs.append(pair)

            elapsed = time.time() - t_start
            done    = i + 1
            eta     = (elapsed / done) * (total - done) if done > 0 else 0
            print(
                f"  [{done:4d}/{total}] {reason:<30} "
                f"pares_ok={len(pairs):4d} "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    # Guardar dataset DPO
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Guardar estadísticas
    stats["total_input"]  = total
    stats["total_output"] = len(pairs)
    stats["yield_pct"]    = round(len(pairs) / total * 100, 1)
    stats["elapsed_min"]  = round((time.time() - t_start) / 60, 1)
    STATS_PATH.write_text(json.dumps(stats, indent=2))

    print(f"\n{'═'*55}")
    print(f"  Dataset DPO generado")
    print(f"{'═'*55}")
    print(f"  Input:          {total} muestras")
    print(f"  Output:         {len(pairs)} pares válidos ({stats['yield_pct']}%)")
    print(f"  Descartados:")
    print(f"    rouge_too_high:    {stats.get('rouge_too_high', 0)}")
    print(f"    rejected_too_good: {stats.get('rejected_too_good', 0)}")
    print(f"    ollama_error:      {stats.get('ollama_error', 0)}")
    print(f"  Tiempo:         {stats['elapsed_min']} min")
    print(f"  Guardado en:    {OUTPUT_PATH}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
