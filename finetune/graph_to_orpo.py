"""Consolidación (Fase 4): exporta a dataset ORPO los DIAGNÓSTICOS verificados por
el grafo, para entrenar la prosa de diagnóstico del SLM con recompensa verificada
por outcome.

NO reentrena soluciones (eso lo sirve el grafo de forma determinista y auditable);
solo el DIAGNÓSTICO, etiquetado por causas que la remediación confirmó. Lee
feedback.jsonl, conserva los positivos cuya solución vino del grafo
(solution_source ∈ {graph, escalated}) y se verificó por outcome (verified=True),
y reusa build_loop_dataset.feedback_to_pairs para los pares ORPO.

Es un export offline e idempotente: no toca el cluster ni reentrena (eso es
loop_train.py, protegido por el gate de no-regresión). Si no hay positivos
verificados-por-grafo, no genera nada (a prueba de vacío).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from finetune.build_loop_dataset import feedback_to_pairs


def load_feedback(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def graph_verified_positives(examples: list[dict]) -> list[dict]:
    """Positivos cuyo diagnóstico quedó CONFIRMADO por la verificación del grafo."""
    return [
        e for e in examples
        if e.get("label") == "positive"
        and e.get("solution_source") in ("graph", "escalated")
        and e.get("verified") is True
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback",
                    default=os.getenv("AIOPS_FEEDBACK_FILE", "data/feedback/feedback.jsonl"))
    ap.add_argument("--out", default="dataset/output/graph_consolidation.jsonl")
    args = ap.parse_args()

    fb = load_feedback(args.feedback)
    pos = graph_verified_positives(fb)
    pairs = feedback_to_pairs(pos)  # sin gen_rejected: solo pares con señal clara

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"feedback={len(fb)}  verificados-grafo={len(pos)}  pares_orpo={len(pairs)}  -> {outp}")


if __name__ == "__main__":
    main()
