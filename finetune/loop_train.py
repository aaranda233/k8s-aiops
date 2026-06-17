"""
Orquestador del ciclo de aprendizaje (Fase 3): feedback -> dataset -> entrenamiento
-> gate -> deploy. Requiere GPU para el paso de entrenamiento (unsloth).

Flujo:
  1. Cuenta ejemplos nuevos de feedback desde el último ciclo (loop_state.json).
  2. Si supera MIN_NEW_EXAMPLES, construye el dataset (build_loop_dataset).
  3. Entrena con train_orpo.py (subprocess; GPU).
  4. Crea la versión candidata en Ollama y aplica el gate (eval/gate.py).
  5. Si PROMOTE, repunta el alias; registra todo en model_registry.json + MLflow.

La decisión de disparo (should_train) es pura/testeable; el entrenamiento real es
subprocess y solo corre donde hay GPU.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

STATE_PATH = "finetune/output/loop_state.json"
MIN_NEW_EXAMPLES = 20


def count_examples(feedback_path: str) -> int:
    p = Path(feedback_path)
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"last_trained_count": 0, "last_version": 0}


def _save_state(path: str, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(state, indent=2))


def should_train(total_examples: int, last_trained_count: int, min_new: int = MIN_NEW_EXAMPLES) -> bool:
    """True si hay suficientes ejemplos nuevos desde el último entrenamiento."""
    return (total_examples - last_trained_count) >= min_new


def run_cycle(feedback="data/feedback/feedback.jsonl", state_path=STATE_PATH,
              min_new=MIN_NEW_EXAMPLES, base_model_dataset="dataset/output/dpo_dataset_v2.jsonl",
              dry_run=False) -> dict:
    """Ejecuta un ciclo. dry_run=True hace todo menos el entrenamiento (para validar sin GPU)."""
    from eval.gate import run_gate
    from finetune.deploy_model import ModelRegistry, ollama_create, set_alias

    state = _load_state(state_path)
    total = count_examples(feedback)
    if not should_train(total, state["last_trained_count"], min_new):
        return {"trained": False, "reason": f"insuficientes ejemplos nuevos ({total - state['last_trained_count']} < {min_new})"}

    # 1. Construir dataset (loop + replay)
    out_dataset = "dataset/output/orpo_train_loop.jsonl"
    subprocess.run([sys.executable, "finetune/build_loop_dataset.py",
                    "--feedback", feedback, "--base", base_model_dataset, "--out", out_dataset],
                   check=True)

    reg = ModelRegistry()
    version = reg.next_version()
    if dry_run:
        return {"trained": False, "reason": "dry_run", "would_train_version": version,
                "dataset": out_dataset, "new_examples": total - state["last_trained_count"]}

    # 2. Entrenar (GPU)
    out_dir = f"finetune/output/k8s-rca-orpo-v{version}"
    subprocess.run([sys.executable, "finetune/train_orpo.py",
                    "--dataset", out_dataset, "--output", out_dir], check=True)

    # 3. Crear versión candidata en Ollama
    gguf = f"{out_dir}/k8s-rca-slm-q4_k_m.gguf"
    ollama_create(version, gguf)

    # 4. Gate vs producción
    prod = reg.active_model() or "k8s-rca-orpo"
    gate = run_gate(f"k8s-rca-orpo-v{version}", prod)
    promote = gate["decision"] == "PROMOTE"

    # 5. Registrar y, si procede, promover
    reg.record(version, gguf, gate, out_dataset, promoted=promote)
    if promote:
        set_alias(version)

    state["last_trained_count"] = total
    state["last_version"] = version
    _save_state(state_path, state)
    return {"trained": True, "version": version, "gate": gate, "promoted": promote}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", default="data/feedback/feedback.jsonl")
    ap.add_argument("--min-new", type=int, default=MIN_NEW_EXAMPLES)
    ap.add_argument("--dry-run", action="store_true", help="todo menos entrenar (sin GPU)")
    args = ap.parse_args()
    result = run_cycle(feedback=args.feedback, min_new=args.min_new, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
