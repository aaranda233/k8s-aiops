"""
Fase 2 DPO: fine-tuning con Direct Preference Optimization sobre el SFT.

Parte del checkpoint SFT (k8s-rca-slm) y entrena con pares chosen/rejected
generados por el modelo vanilla qwen2.5:1.5b.

Objetivo: reducir alucinaciones y mejorar keyword accuracy (60% → ≥80%)
manteniendo la estructura de formato del SFT.

Uso:
  python finetune/train_dpo.py
  python finetune/train_dpo.py --beta 0.05 --epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-model",  default="finetune/output/k8s-rca-slm",
                   help="Checkpoint SFT del que partir")
    p.add_argument("--dataset",    default="dataset/output/dpo_dataset.jsonl")
    p.add_argument("--output",     default="finetune/output/k8s-rca-dpo")
    p.add_argument("--epochs",     type=int,   default=2)
    p.add_argument("--beta",       type=float, default=0.05,
                   help="Parámetro β de DPO — controla desviación del ref model. "
                        "Bajo (0.05) = conservador, evita colapso sobre SFT memorizado.")
    p.add_argument("--batch-size", type=int,   default=2)
    p.add_argument("--grad-accum", type=int,   default=8,
                   help="Batch efectivo = batch_size * grad_accum = 16 por defecto")
    p.add_argument("--lr",         type=float, default=5e-5,
                   help="LR más bajo que SFT (2e-4) para estabilidad DPO")
    p.add_argument("--max-seq-len",type=int,   default=1024)
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--no-gguf",    action="store_true")
    return p.parse_args()


# ── Carga y formateo del dataset DPO ─────────────────────────────────────────

def load_dpo_dataset(path: str, tokenizer):
    """
    Carga el JSONL de pares DPO y los formatea para TRL DPOTrainer.

    TRL espera tres campos string:
      prompt   = system + user formateados con chat template (add_generation_prompt=True)
      chosen   = solo el contenido del assistant correcto
      rejected = solo el contenido del assistant incorrecto
    """
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            prompt_msgs  = sample["prompt"]   # [system, user]
            chosen_msgs  = sample["chosen"]   # [assistant_correct]
            rejected_msgs = sample["rejected"] # [assistant_wrong]

            # Formatear prompt hasta el turno del assistant (inclusive el marcador)
            prompt_str = tokenizer.apply_chat_template(
                prompt_msgs,
                tokenize=False,
                add_generation_prompt=True,  # añade <|im_start|>assistant\n
            )

            chosen_str   = chosen_msgs[0]["content"]
            rejected_str = rejected_msgs[0]["content"]

            records.append({
                "prompt":   prompt_str,
                "chosen":   chosen_str,
                "rejected": rejected_str,
            })

    dataset = Dataset.from_list(records)
    print(f"  Dataset DPO cargado: {len(dataset)} pares")
    return dataset


# ── Entrenamiento DPO ─────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  K8s-RCA-SLM — DPO Fine-tuning")
    print(f"  SFT base   : {args.sft_model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  β (beta)   : {args.beta}")
    print(f"  Épocas     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Output     : {args.output}")
    print("═" * 60 + "\n")

    from trl import DPOTrainer, DPOConfig

    # Intentar parche de unsloth (acelera 2x) — ignorar si hay incompatibilidad de versiones
    try:
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()
        print("  [unsloth] PatchDPOTrainer activado.")
    except Exception as e:
        print(f"  [unsloth] PatchDPOTrainer no disponible ({e}) — usando TRL estándar.")
        from unsloth import FastLanguageModel

    # ── 1. Cargar modelo SFT (punto de partida del DPO) ──────────────────────
    print("[1/4] Cargando modelo SFT con QLoRA 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.sft_model,
        max_seq_length = args.max_seq_len,
        dtype          = None,
        load_in_4bit   = True,
    )

    # ── 2. Añadir adaptadores LoRA nuevos para DPO ───────────────────────────
    print("[2/4] Configurando adaptadores LoRA para DPO...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                = args.lora_r,
        lora_alpha       = args.lora_alpha,
        lora_dropout     = 0.05,
        target_modules   = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias             = "none",
        use_gradient_checkpointing = "unsloth",
        random_state     = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("[3/4] Preparando dataset DPO...")
    dataset = load_dpo_dataset(args.dataset, tokenizer)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 10,
        beta                        = args.beta,
        fp16                        = False,
        bf16                        = True,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        max_length                  = args.max_seq_len,
        max_prompt_length           = 768,
        logging_steps               = 10,
        save_strategy               = "epoch",
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = 42,
        remove_unused_columns       = False,
    )

    trainer = DPOTrainer(
        model     = model,
        ref_model = None,   # None = usa el modelo base como referencia (unsloth lo gestiona)
        args      = dpo_config,
        train_dataset = dataset,
        tokenizer = tokenizer,
    )

    # ── 4. Entrenar ───────────────────────────────────────────────────────────
    print("[4/4] Entrenando con DPO...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    print(f"\n  Entrenamiento completado en {runtime_min:.1f} min")
    print(f"  Loss final: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adaptadores LoRA DPO guardados en: {output_dir}")


# ── Cuantización a GGUF Q4_K_M ───────────────────────────────────────────────

def quantize_to_gguf(lora_path: str, output_dir: str) -> None:
    import shutil
    from unsloth import FastLanguageModel

    out_dir  = Path(output_dir)
    tmp_dir  = out_dir / "_merged_tmp"

    print("  Fusionando LoRA con modelo base...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        lora_path,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(tmp_dir), tokenizer, save_method="merged_16bit")
    del model

    print("  Cuantizando a Q4_K_M...")
    model_f16, tok = FastLanguageModel.from_pretrained(str(tmp_dir), load_in_4bit=False)
    model_f16.save_pretrained_gguf(
        str(out_dir / "k8s-rca-dpo"),
        tok,
        quantization_method="q4_k_m",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  GGUF listo: {out_dir}/k8s-rca-dpo-Q4_K_M.gguf")
    print(f"\n  Registrar en Ollama:")
    print(f"    ollama create k8s-rca-dpo -f finetune/Modelfile_dpo")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset DPO no encontrado: {args.dataset}\n"
                                f"Ejecuta primero: python finetune/generate_dpo_dataset.py")

    if not Path(args.sft_model).exists():
        raise FileNotFoundError(f"Modelo SFT no encontrado: {args.sft_model}")

    train(args)

    if not args.no_gguf:
        print("\n" + "─" * 60)
        print("  Cuantizando a GGUF Q4_K_M...")
        print("─" * 60)
        quantize_to_gguf(args.output, str(Path(args.output).parent))

    print("\n" + "═" * 60)
    print("  DPO completado.")
    print("  Siguiente paso:")
    print("    1. ollama create k8s-rca-dpo -f finetune/Modelfile_dpo")
    print("    2. python eval/run_eval.py --models sft,dpo,baseline")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
