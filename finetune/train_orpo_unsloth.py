"""
ORPO con Unsloth — para modelos cuya arquitectura PEFT vanilla no sabe envolver
(p. ej. Gemma 4, cuyas proyecciones son Gemma4ClippableLinear sobre Linear4bit).

Unsloth tiene soporte nativo de Gemma 4 (FastLanguageModel) y engancha la LoRA
de forma que los gradientes fluyen — al contrario que el hack de PEFT vanilla
sobre el .linear interno, que daba loss PLANO (no aprendía).

Notas Gemma 4:
  - Es multimodal: el "tokenizer" es un Gemma4Processor; el de texto está en
    processor.tokenizer (GemmaTokenizer). ORPOTrainer necesita el de TEXTO.

Uso:
  python finetune/train_orpo_unsloth.py --base-model unsloth/gemma-4-E2B-it \
    --dataset dataset/output/dpo_dataset_v2.jsonl \
    --output output/k8s-rca-orpo-gemma4-it --epochs 3 --lr 2e-5
"""

from __future__ import annotations

# ruff: noqa: I001  (orden de imports intencional: unsloth debe ir antes que transformers)

# Unsloth DEBE importarse antes que transformers/trl para aplicar sus parches.
from unsloth import FastLanguageModel  # noqa: E402  (import-order intencional)

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

from train_orpo import load_orpo_dataset  # noqa: E402  (reusa el loader del dataset)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Instruct (no el base pretrained): tiene chat_template nativa con rol system,
    # igual que el qwen ORPO usó Qwen2.5-1.5B-Instruct. El base pretrained aprendía
    # a repetir el prompt (echo) en vez de diagnosticar.
    p.add_argument("--base-model", default="unsloth/gemma-4-E2B-it")
    p.add_argument("--dataset", default="dataset/output/dpo_dataset_v2.jsonl")
    p.add_argument("--output", default="output/k8s-rca-orpo-gemma4-it")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lambda-orpo", type=float, default=0.1)
    # Prompts de eventos: mediana 933 tok, max 1457 con la plantilla nativa.
    # 1536/1792 evita truncar (antes 512/1024 truncaba el 97% de los prompts).
    p.add_argument("--max-seq-len", type=int, default=1792)
    p.add_argument("--max-prompt-len", type=int, default=1536)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from trl import ORPOConfig, ORPOTrainer

    print("═" * 60)
    print("  K8s-RCA-SLM — ORPO (Unsloth)")
    print(f"  Base   : {args.base_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  λ={args.lambda_orpo}  épocas={args.epochs}  LR={args.lr}")
    print(f"  Output : {args.output}")
    print("═" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Cargando modelo base (Unsloth, 4-bit)...")
    model, processor = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    # Gemma 4 multimodal → el tokenizer de TEXTO está en processor.tokenizer.
    text_tok = getattr(processor, "tokenizer", processor)
    if text_tok.pad_token is None:
        text_tok.pad_token = text_tok.eos_token

    print("[2/4] Adaptadores LoRA (Unsloth)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("[3/4] Dataset ORPO...")
    dataset = load_orpo_dataset(args.dataset, text_tok)

    orpo_config = ORPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        beta=args.lambda_orpo,
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_prompt_len,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=dataset,
        processing_class=text_tok,
    )

    print("[4/4] Entrenando ORPO (Unsloth)...\n")
    stats = trainer.train()
    runtime_min = stats.metrics["train_runtime"] / 60
    train_loss = stats.metrics.get("train_loss", -1)
    print(f"\n  Completado en {runtime_min:.1f} min — train_loss={train_loss:.4f}")

    model.save_pretrained(str(output_dir))
    text_tok.save_pretrained(str(output_dir))
    (output_dir / "training_summary.json").write_text(json.dumps({
        "method": "ORPO+Unsloth", "base_model": args.base_model,
        "dataset": args.dataset, "lambda_orpo": args.lambda_orpo,
        "lr": args.lr, "epochs": args.epochs,
        "train_loss": train_loss, "runtime_min": runtime_min,
    }, indent=2))
    print(f"\n  Adaptadores guardados en: {output_dir}")


if __name__ == "__main__":
    main()
