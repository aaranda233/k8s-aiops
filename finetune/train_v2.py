"""
SFT v2 — Fine-tuning QLoRA con dataset ampliado (26 escenarios) y early stopping.

Diferencias clave respecto a train.py (SFT v1):
  - eval_dataset separado (15% del dataset, generado con seed distinto)
  - EarlyStoppingCallback: para si eval_loss no mejora en 3 evaluaciones
  - load_best_model_at_end: guarda el checkpoint de menor eval_loss
  - lora_dropout=0.10: mas regularizacion para evitar memorizar
  - Objetivo: eval_loss > 0.30 al final (generalizacion real, no memorizacion)

El criterio de exito NO es minimizar el train loss — es que el eval loss
converja a un valor razonablemente alto. Si el modelo alcanza eval_loss < 0.15
con el nuevo dataset, hay que anadir mas diversidad antes de intentar PO.

Uso:
  python finetune/train_v2.py
  python finetune/train_v2.py --dataset dataset/output/train_train.jsonl \
                               --eval-dataset dataset/output/train_val.jsonl
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="dataset/output/train_train.jsonl")
    p.add_argument("--eval-dataset", default="dataset/output/train_val.jsonl")
    p.add_argument("--base-model",   default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--output",       default="finetune/output/k8s-rca-slm-v2")
    p.add_argument("--epochs",       type=int,   default=5,
                   help="Maximo de epocas — early stopping puede parar antes")
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max-seq-len",  type=int,   default=1024)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.10,
                   help="Dropout en adaptadores LoRA (0.10 > 0.05 del v1)")
    p.add_argument("--patience",     type=int,   default=3,
                   help="Early stopping: parar si eval_loss no mejora en N evaluaciones")
    p.add_argument("--eval-steps",   type=int,   default=50,
                   help="Evaluar cada N pasos")
    p.add_argument("--no-gguf",      action="store_true")
    return p.parse_args()


def load_jsonl(path: str):
    from datasets import Dataset
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                records.append({"messages": sample["messages"]})
    dataset = Dataset.from_list(records)
    print(f"  Cargado: {len(dataset)} samples  ({path})")
    return dataset


def train(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("  K8s-RCA-SLM v2 — SFT QLoRA con early stopping")
    print(f"  Train      : {args.dataset}")
    print(f"  Eval       : {args.eval_dataset}")
    print(f"  Max epocas : {args.epochs}  (early stopping patience={args.patience})")
    print(f"  LoRA drop  : {args.lora_dropout}  (v1=0.05, v2={args.lora_dropout})")
    print(f"  Objetivo   : eval_loss > 0.30 al finalizar")
    print("=" * 60 + "\n")

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    # 1. Modelo base QLoRA 4-bit
    print("[1/4] Cargando modelo base con QLoRA 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = args.base_model,
        max_seq_length= args.max_seq_len,
        dtype         = None,
        load_in_4bit  = True,
    )

    # 2. Adaptadores LoRA — dropout mas alto para regularizar
    print("[2/4] Configurando adaptadores LoRA (dropout={})...".format(args.lora_dropout))
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parametros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 3. Datasets
    print("[3/4] Preparando datasets...")
    train_ds = load_jsonl(args.dataset)
    eval_ds  = load_jsonl(args.eval_dataset)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def formatting_func(examples):
        messages = examples["messages"]
        if isinstance(messages[0], dict):
            return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages
        ]

    sft_config = SFTConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 20,
        fp16                        = False,
        bf16                        = True,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        max_seq_length              = args.max_seq_len,
        # Evaluacion frecuente para early stopping
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        logging_steps               = 10,
        save_strategy               = "steps",
        save_steps                  = args.eval_steps,
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = "none",
        seed                        = 42,
        dataset_text_field          = "",
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        formatting_func = formatting_func,
        args            = sft_config,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # 4. Entrenar
    print("[4/4] Entrenando...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    train_loss  = trainer_stats.metrics.get("train_loss", -1)

    print(f"\n  Completado en {runtime_min:.1f} min")
    print(f"  Train loss final : {train_loss:.4f}")

    # Evaluar el modelo que quedo cargado (el mejor checkpoint)
    eval_result = trainer.evaluate()
    eval_loss   = eval_result.get("eval_loss", -1)
    print(f"  Eval loss final  : {eval_loss:.4f}")

    if eval_loss < 0.15:
        print("\n  [!] AVISO: eval_loss < 0.15 — el modelo puede seguir memorizando.")
        print("      Considera anadir mas escenarios o reducir epocas antes de PO.")
    elif eval_loss > 0.30:
        print("\n  [OK] eval_loss > 0.30 — generalizacion real. Listo para PO si se desea.")
    else:
        print(f"\n  [~] eval_loss en rango intermedio ({eval_loss:.3f}).")
        print("      Evalua con el harness completo antes de decidir sobre PO.")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adaptadores LoRA guardados en: {output_dir}")

    # Guardar resumen de metricas
    summary = {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "runtime_min": runtime_min,
        "dataset": args.dataset,
        "epochs_max": args.epochs,
        "lora_dropout": args.lora_dropout,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


def quantize_to_gguf(lora_path: str, output_dir: str) -> None:
    import shutil
    from unsloth import FastLanguageModel

    out_dir = Path(output_dir)
    tmp_dir = out_dir / "_merged_tmp_v2"

    print("  Fusionando LoRA con modelo base...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        lora_path, max_seq_length=1024, load_in_4bit=True,
    )
    model.save_pretrained_merged(str(tmp_dir), tokenizer, save_method="merged_16bit")
    del model

    print("  Cuantizando a Q4_K_M...")
    model_f16, tok = FastLanguageModel.from_pretrained(str(tmp_dir), load_in_4bit=False)
    model_f16.save_pretrained_gguf(
        str(out_dir / "k8s-rca-slm-v2"),
        tok,
        quantization_method="q4_k_m",
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  GGUF listo: {out_dir}/k8s-rca-slm-v2-Q4_K_M.gguf")
    print(f"\n  Registrar en Ollama:")
    print(f"    cd finetune && ollama create k8s-rca-slm-v2 -f Modelfile_v2")


def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {args.dataset}\n"
            f"Ejecuta primero:\n"
            f"  python dataset/generator.py --samples 70 --seed 42 --output dataset/output/train.jsonl"
        )
    if not Path(args.eval_dataset).exists():
        raise FileNotFoundError(f"Eval dataset no encontrado: {args.eval_dataset}")

    train(args)

    if not args.no_gguf:
        print("\n" + "-" * 60)
        print("  Cuantizando a GGUF Q4_K_M...")
        print("-" * 60)
        quantize_to_gguf(args.output, str(Path(args.output).parent))

    print("\n" + "=" * 60)
    print("  SFT v2 completado.")
    print("  Siguiente:")
    print("    1. ollama create k8s-rca-slm-v2 -f finetune/Modelfile_v2")
    print("    2. python eval/run_eval.py --models sft_v2,baseline")
    print("    3. Si eval_loss > 0.30 y kw% > 75% -> intentar DPO v2")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
