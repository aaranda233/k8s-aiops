"""
Registra retroactivamente el run de fine-tuning QLoRA en MLflow.

Lee trainer_state.json del checkpoint final y sube:
  - Hiperparámetros del entrenamiento como params
  - Loss y learning_rate por step como métricas
  - Métricas finales del run (loss final, runtime, etc.)

Uso:
  python finetune/log_finetune_to_mlflow.py
  python finetune/log_finetune_to_mlflow.py --checkpoint finetune/output/k8s-rca-slm/checkpoint-186
"""

import argparse
import json
from pathlib import Path

MLFLOW_URI = "http://192.168.2.204:30803"
EXPERIMENT  = "k8s-aiops-finetune"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="finetune/output/k8s-rca-slm/checkpoint-186",
        help="Ruta al checkpoint final con trainer_state.json",
    )
    p.add_argument("--mlflow-uri", default=MLFLOW_URI)
    p.add_argument("--experiment",  default=EXPERIMENT)
    return p.parse_args()


def main():
    args = parse_args()
    state_path = Path(args.checkpoint) / "trainer_state.json"

    if not state_path.exists():
        raise FileNotFoundError(f"No encontrado: {state_path}")

    with open(state_path) as f:
        state = json.load(f)

    import mlflow

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="qwen2.5-1.5b-qlora-k8s-rca"):

        # ── Hiperparámetros ───────────────────────────────────────────
        mlflow.log_params({
            "base_model":        "Qwen/Qwen2.5-1.5B-Instruct",
            "epochs":            state["num_train_epochs"],
            "batch_size":        state["train_batch_size"],
            "grad_accum":        4,
            "effective_batch":   state["train_batch_size"] * 4,
            "learning_rate":     2e-4,
            "lr_scheduler":      "cosine",
            "warmup_steps":      10,
            "lora_r":            16,
            "lora_alpha":        32,
            "lora_dropout":      0.05,
            "lora_targets":      "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            "max_seq_length":    1024,
            "quantization":      "4bit-nf4",
            "optimizer":         "adamw_8bit",
            "bf16":              True,
            "dataset":           "aaranda233/k8s-rca-dataset",
            "dataset_samples":   986,
            "total_steps":       state["global_step"],
            "gpu":               "NVIDIA A30 24GB",
            "framework":         "unsloth + trl SFTTrainer",
        })

        # ── Loss y LR por step ────────────────────────────────────────
        for entry in state["log_history"]:
            step = entry["step"]
            mlflow.log_metrics(
                {
                    "train_loss":    entry["loss"],
                    "learning_rate": entry["learning_rate"],
                    "grad_norm":     entry["grad_norm"],
                    "epoch":         entry["epoch"],
                },
                step=step,
            )

        # ── Métricas finales ──────────────────────────────────────────
        last = state["log_history"][-1]
        mlflow.log_metrics({
            "final_loss":      last["loss"],
            "final_epoch":     last["epoch"],
            "total_flos":      state["total_flos"],
        })

        # ── Tags ──────────────────────────────────────────────────────
        mlflow.set_tags({
            "model_hf":        "aaranda233/k8s-rca-slm",
            "gguf_q8_size_gb": "1.6",
            "inference":       "ollama CPU ~1s/resp warm",
            "task":            "k8s-root-cause-analysis",
        })

        print(f"\n  Run registrado en MLflow.")
        print(f"  URI     : {args.mlflow_uri}")
        print(f"  Experimento: {args.experiment}")
        print(f"  Steps   : {state['global_step']}")
        print(f"  Loss final: {last['loss']:.4f}")


if __name__ == "__main__":
    main()
