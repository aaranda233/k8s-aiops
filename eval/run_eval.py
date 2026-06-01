"""
Harness de evaluación — SFT vs Baseline (Qwen2.5-1.5B vanilla).

Genera el test set ciego (seed=99, distinto al de entrenamiento seed=42),
ejecuta inferencia en ambos modelos y muestra la tabla de comparación.

Uso:
  python eval/run_eval.py
  python eval/run_eval.py --samples 10   # rápido, 10 por escenario
  python eval/run_eval.py --host http://192.168.2.205:11434
  python eval/run_eval.py --models sft   # solo el modelo SFT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.generator import generate, load_scenarios
from eval.runner import ModelConfig, evaluate_model

SCENARIOS_DIR = PROJECT_ROOT / "dataset" / "scenarios"
TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.jsonl"
RESULTS_DIR   = PROJECT_ROOT / "eval" / "results"

MODELS = {
    "sft":      "k8s-rca-slm",
    "baseline": "qwen2.5:1.5b",
}


def build_test_set(samples_per_scenario: int, force: bool = False) -> list[dict]:
    """Genera (o carga) el test set ciego."""
    if TEST_SET_PATH.exists() and not force:
        samples = [json.loads(l) for l in TEST_SET_PATH.read_text().splitlines() if l.strip()]
        print(f"[test set] cargado desde caché: {len(samples)} muestras")
        return samples

    print(f"[test set] generando {samples_per_scenario} muestras/escenario con seed=99 ...")
    n = generate(
        scenarios_dir=SCENARIOS_DIR,
        output_path=TEST_SET_PATH,
        samples_per_scenario=samples_per_scenario,
        seed=99,                           # seed distinto al entrenamiento (42)
    )
    samples = [json.loads(l) for l in TEST_SET_PATH.read_text().splitlines() if l.strip()]
    print(f"[test set] generado: {n} muestras → {TEST_SET_PATH}")
    return samples


def print_table(all_results: dict[str, dict]) -> None:
    """Imprime la tabla comparativa de modelos."""
    metrics = ["parse_rate", "keyword_hit", "rouge_l", "kubectl_ns_ok", "kubectl_verb_ok", "latency_mean", "latency_p95"]
    labels  = ["Parse%", "Keyword%", "ROUGE-L", "NS-ok%", "Verb-ok%", "Lat.mean", "Lat.p95"]

    col_w = 12
    header = f"{'Métrica':<18}" + "".join(f"{name:>{col_w}}" for name in all_results)
    sep    = "─" * len(header)

    print(f"\n{'═'*len(header)}")
    print("  RESULTADOS DE EVALUACIÓN — K8s-RCA SLM")
    print(f"{'═'*len(header)}")
    print(header)
    print(sep)

    for metric, label in zip(metrics, labels):
        row = f"{label:<18}"
        for agg in all_results.values():
            val = agg.get(metric, 0)
            if metric in ("latency_mean", "latency_p95"):
                row += f"{val:>{col_w}.2f}s"
            else:
                row += f"{val*100:>{col_w-1}.1f}%"
        print(row)

    print(f"{'─'*len(header)}")
    row = f"{'N muestras':<18}"
    for agg in all_results.values():
        row += f"{agg.get('n', 0):>{col_w}}"
    print(row)
    print(f"{'═'*len(header)}\n")


def print_scenario_breakdown(per_sample: list[dict], model_name: str) -> None:
    """Muestra keyword_hit por escenario para un modelo."""
    from collections import defaultdict
    hits: dict[str, list] = defaultdict(list)
    for r in per_sample:
        hits[r["scenario_id"]].append(r["keyword_hit"])

    print(f"\n  Keyword hit por escenario — {model_name}")
    print(f"  {'Escenario':<35} {'Hit%':>6}  {'N':>4}")
    print(f"  {'─'*48}")
    for sid in sorted(hits):
        lst = hits[sid]
        pct = sum(lst) / len(lst) * 100
        print(f"  {sid:<35} {pct:>5.1f}%  {len(lst):>4}")


def save_results(all_per_sample: dict[str, list], all_agg: dict[str, dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "aggregate": all_agg,
        "per_sample": all_per_sample,
    }
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[resultados] guardados en {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=15,
                   help="Muestras por escenario en el test set (default: 15)")
    p.add_argument("--host", default="http://192.168.2.205:11434",
                   help="URL de Ollama")
    p.add_argument("--models", default="sft,baseline",
                   help="Modelos a evaluar: sft,baseline (o solo uno)")
    p.add_argument("--regen", action="store_true",
                   help="Regenerar test set aunque ya exista")
    return p.parse_args()


def main():
    args = parse_args()
    models_to_run = [m.strip() for m in args.models.split(",")]

    # 1. Test set
    test_samples = build_test_set(args.samples, force=args.regen)
    print(f"\n[eval] {len(test_samples)} muestras · {len(models_to_run)} modelo(s)\n")

    all_per_sample: dict[str, list] = {}
    all_agg:        dict[str, dict] = {}

    # 2. Evaluar cada modelo
    for key in models_to_run:
        if key not in MODELS:
            print(f"[!] modelo desconocido: {key}. Opciones: {list(MODELS)}")
            continue

        ollama_model = MODELS[key]
        cfg = ModelConfig(name=key, ollama_model=ollama_model, host=args.host)

        print(f"{'─'*60}")
        print(f"  Evaluando: {key} ({ollama_model})")
        print(f"{'─'*60}")

        per_sample, agg = evaluate_model(test_samples, cfg, verbose=True)
        all_per_sample[key] = per_sample
        all_agg[key]        = agg

        print_scenario_breakdown(per_sample, key)

    # 3. Tabla comparativa
    if all_agg:
        print_table(all_agg)

    # 4. Guardar
    if all_per_sample:
        save_results(all_per_sample, all_agg)


if __name__ == "__main__":
    main()
