#!/usr/bin/env python3
"""Intervalos de confianza por bootstrap sobre los resultados de eval (E2).

Carga un JSON de `eval/results/eval_*.json` (con `per_sample` por modelo) y, para
cada modelo y cada métrica, reporta la media y el IC 95% por bootstrap percentil
(remuestreo con reemplazo). Da la incertidumbre que un revisor Q1 exige, sin
re-ejecutar el modelo: se bootstrapea sobre las N muestras ya evaluadas.

Uso:
    python eval/bootstrap_ci.py [results.json] [--iters 10000] [--seed 99] [--md salida.md]
Si no se pasa archivo, usa el `eval_*.json` más reciente de eval/results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"

# (clave en per_sample, etiqueta, ¿es porcentaje?)
_METRICS = [
    ("parsed",          "Parse%",   True),
    ("keyword_hit",     "Keyword%", True),
    ("kubectl_ns_ok",   "NS-ok%",   True),
    ("kubectl_verb_ok", "Verb-ok%", True),
    ("rouge_l",         "ROUGE-L",  True),
    ("latency_s",       "Lat. (s)", False),
]


def _latest_results() -> Path:
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    if not files:
        raise SystemExit(
            f"No hay eval_*.json en {RESULTS_DIR}. Corre primero `python eval/run_eval.py` "
            "(o copia un results del servidor)."
        )
    return files[-1]


def bootstrap_ci(values: list[float], iters: int, rng: np.random.Generator,
                 pct: bool) -> tuple[float, float, float]:
    """Devuelve (media, lo, hi) con IC 95% percentil. Vacío → (nan, nan, nan)."""
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    idx = rng.integers(0, x.size, size=(iters, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    scale = 100.0 if pct else 1.0
    return (x.mean() * scale, lo * scale, hi * scale)


def analyse(per_sample: dict[str, list[dict]], iters: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for model, rows in per_sample.items():
        n = len(rows)
        out[model] = {"n": n, "metrics": {}}
        for key, label, pct in _METRICS:
            if not rows or key not in rows[0]:
                continue
            vals = [r.get(key) for r in rows]
            mean, lo, hi = bootstrap_ci(vals, iters, rng, pct)
            out[model]["metrics"][label] = (mean, lo, hi)
    return out


def _fmt(mean: float, lo: float, hi: float, pct: bool) -> str:
    if mean != mean:  # nan
        return "—"
    unit = "%" if pct else "s"
    return f"{mean:.1f}{unit} [{lo:.1f}, {hi:.1f}]"


def to_markdown(analysis: dict, src: str, iters: int, seed: int) -> str:
    labels = [lbl for _, lbl, _ in _METRICS]
    is_pct = {lbl: pct for _, lbl, pct in _METRICS}
    lines = [
        f"# Bootstrap 95% CI — `{src}`",
        "",
        f"Remuestreo percentil, {iters:,} iteraciones, seed={seed}. "
        "Cada celda: media [IC inferior, IC superior].",
        "",
        "| Modelo | n | " + " | ".join(labels) + " |",
        "|---|---:|" + "|".join([":---:"] * len(labels)) + "|",
    ]
    for model, data in analysis.items():
        cells = []
        for lbl in labels:
            if lbl in data["metrics"]:
                cells.append(_fmt(*data["metrics"][lbl], is_pct[lbl]))
            else:
                cells.append("—")
        lines.append(f"| {model} | {data['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", help="ruta a eval_*.json (default: el más reciente)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--md", help="escribe la tabla markdown a este archivo")
    args = ap.parse_args()

    path = Path(args.results) if args.results else _latest_results()
    data = json.loads(path.read_text())
    per_sample = data.get("per_sample")
    if not per_sample:
        raise SystemExit(f"{path} no tiene 'per_sample' (¿es un run completo?).")

    analysis = analyse(per_sample, args.iters, args.seed)
    md = to_markdown(analysis, path.name, args.iters, args.seed)
    print(md)
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
        print(f"[escrito] {args.md}")


if __name__ == "__main__":
    main()
