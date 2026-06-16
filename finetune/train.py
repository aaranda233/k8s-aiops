"""
Fine-tuning QLoRA del modelo k8s-rca-slm en la A30 (24GB VRAM).

Modelo base : Qwen2.5-1.5B-Instruct  (corre luego en CPU como un tiro)
Dataset     : dataset/output/combined.jsonl  (~986 samples, chat format)
Salida      : finetune/output/k8s-rca-slm   (HuggingFace format)
             + finetune/output/k8s-rca-slm.gguf  (Q4_K_M, para Ollama)

Uso:
  python finetune/train.py
  python finetune/train.py --dataset dataset/output/combined.jsonl --epochs 4
"""

import argparse
import json
from pathlib import Path

# ── Argumentos ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="dataset/output/combined.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Modelo base de HuggingFace. Alternativas: "
                        "Qwen/Qwen2.5-3B-Instruct, microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--output",     default="finetune/output/k8s-rca-slm")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch-size", type=int,   default=4,
                   help="Por GPU. Con A30 24GB y modelo 1.5B caben hasta 8.")
    p.add_argument("--grad-accum", type=int,   default=4,
                   help="Batch efectivo = batch_size * grad_accum = 16 por defecto")
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--max-seq-len",type=int,   default=1024,
                   help="Max tokens por muestra. 986 samples caben bien en 1024.")
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--no-gguf",    action="store_true",
                   help="Saltar la cuantización a GGUF al final")
    return p.parse_args()


# ── Carga y formato del dataset ───────────────────────────────────────────────

def load_dataset_from_jsonl(path: str):
    """
    Carga el JSONL y lo convierte al formato que espera unsloth/trl:
    lista de dicts con clave 'messages' (ya viene así del generator).
    """
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                records.append({"messages": sample["messages"]})

    dataset = Dataset.from_list(records)
    print(f"  Dataset cargado: {len(dataset)} samples")
    return dataset


# ── Fine-tuning principal ─────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  K8s-RCA-SLM — Fine-tuning QLoRA")
    print("  Modelo base :", args.base_model)
    print("  Dataset     :", args.dataset)
    print("  Épocas      :", args.epochs)
    print("  Output      :", args.output)
    print("═" * 60 + "\n")

    # Unsloth acelera el training 2x y reduce VRAM ~40%
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # ── 1. Cargar modelo base con QLoRA 4-bit ─────────────────────────────────
    print("[1/4] Cargando modelo base con QLoRA 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = args.base_model,
        max_seq_length= args.max_seq_len,
        dtype         = None,       # auto: bfloat16 en A30 (Ampere)
        load_in_4bit  = True,       # QLoRA — 4-bit NF4
    )

    # ── 2. Añadir adaptadores LoRA ────────────────────────────────────────────
    print("[2/4] Configurando adaptadores LoRA...")
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
        use_gradient_checkpointing = "unsloth",  # ahorra VRAM adicional
        random_state     = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("[3/4] Preparando dataset...")
    dataset = load_dataset_from_jsonl(args.dataset)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Formatear cada sample al formato chat del tokenizer (Qwen2.5 usa chatml).
    # Unsloth testea con un ejemplo único (dict) y luego llama con batches (dict de listas).
    def formatting_func(examples):
        messages = examples["messages"]
        # Single example: messages = [{role, content}, ...] (lista de dicts)
        # Batch:          messages = [[{...}, ...], [{...}, ...]] (lista de listas)
        if isinstance(messages[0], dict):
            return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages
        ]

    sft_config = SFTConfig(
        output_dir              = str(output_dir),
        num_train_epochs        = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate           = args.lr,
        lr_scheduler_type       = "cosine",
        warmup_steps            = 10,
        fp16                    = False,
        bf16                    = True,         # A30 soporta bfloat16
        optim                   = "adamw_8bit", # bitsandbytes 8-bit — menos VRAM
        weight_decay            = 0.01,
        max_seq_length          = args.max_seq_len,
        logging_steps           = 10,
        save_strategy           = "epoch",
        save_total_limit        = 2,
        load_best_model_at_end  = False,
        report_to               = "none",       # sin wandb
        seed                    = 42,
        dataset_text_field      = "",           # usamos formatting_func
    )

    trainer = SFTTrainer(
        model             = model,
        tokenizer         = tokenizer,
        train_dataset     = dataset,
        formatting_func   = formatting_func,
        args              = sft_config,
    )

    # ── 4. Entrenar ───────────────────────────────────────────────────────────
    print("[4/4] Entrenando...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    print(f"\n  Entrenamiento completado en {runtime_min:.1f} min")
    print(f"  Loss final: {trainer_stats.metrics['train_loss']:.4f}")

    # Guardar adaptadores LoRA (pequeños, ~50MB — backup ligero)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  ✓ Adaptadores LoRA guardados en: {output_dir}")


# ── Cuantización a GGUF Q4_K_M ───────────────────────────────────────────────

def quantize_to_gguf(lora_path: str, base_model: str, output_dir: str) -> None:
    """
    Fusiona los adaptadores LoRA con el modelo base y convierte a GGUF Q4_K_M.
    El merged fp16 temporal se borra tras la cuantización — solo queda el GGUF.

    Qwen2.5-1.5B Q4_K_M ≈ 1.0 GB → 8-15 tok/s en CPU sin GPU.
    """

    from unsloth import FastLanguageModel

    out_dir  = Path(output_dir)
    tmp_dir  = out_dir / "_merged_tmp"

    # Paso 1: fusionar adaptadores + base → safetensors temporal
    print("  Fusionando LoRA con modelo base...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        lora_path,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model.save_pretrained_merged(str(tmp_dir), tokenizer, save_method="merged_16bit")
    del model  # liberar VRAM antes de convertir

    # Paso 2: fusionado → GGUF (unsloth lo hace internamente con llama.cpp)
    print("  Cuantizando a Q4_K_M...")
    model_f16, tok = FastLanguageModel.from_pretrained(str(tmp_dir), load_in_4bit=False)
    model_f16.save_pretrained_gguf(
        str(out_dir / "k8s-rca-slm"),
        tok,
        quantization_method="q4_k_m",
    )

    # Paso 3: borrar el merged temporal (3GB fp16 que ya no sirven)
    import shutil as _shutil
    _shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  Merged temporal eliminado.")

    print(f"  ✓ GGUF listo: {out_dir}/k8s-rca-slm-Q4_K_M.gguf")
    print("\n  Para registrarlo en Ollama:")
    print("    ollama create k8s-rca-slm -f finetune/Modelfile")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset no encontrado: {args.dataset}")

    train(args)

    if not args.no_gguf:
        print("\n" + "─" * 60)
        print("  Cuantizando a GGUF Q4_K_M para Ollama / llama.cpp...")
        print("─" * 60)
        quantize_to_gguf(args.output, args.base_model, str(Path(args.output).parent))

    print("\n" + "═" * 60)
    print("  Pipeline completado.")
    print("  Siguiente paso:")
    print("    1. Copiar el .gguf a la máquina con Ollama")
    print("    2. ollama create k8s-rca-slm -f finetune/Modelfile")
    print("    3. Actualizar OLLAMA_MODEL=k8s-rca-slm en .env")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
