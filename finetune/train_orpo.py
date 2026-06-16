"""
ORPO — Odds Ratio Preference Optimization (Hong et al., 2024).

A diferencia de DPO, ORPO combina la pérdida SFT y la pérdida de preferencia en un
único paso de entrenamiento, sin necesitar un modelo de referencia separado:

    L_ORPO = L_SFT + λ · L_OR

El término L_SFT ancla el formato durante todo el entrenamiento, evitando el
mode collapse de formato que DPO produce cuando el modelo base es inestable.

Diferencias clave vs train_dpo.py:
  - Sin ref_model (el odds ratio no lo necesita)
  - Parte del modelo base Qwen2.5-1.5B directamente (no del checkpoint SFT)
  - ORPOConfig en lugar de DPOConfig (λ en vez de β)
  - L_SFT actúa como ancla de formato en cada step

Uso:
  python finetune/train_orpo.py
  python finetune/train_orpo.py --lambda-orpo 0.1 --epochs 3
  python finetune/train_orpo.py --no-gguf   # solo entrena, sin exportar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model",   default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Modelo base de HuggingFace (no checkpoint SFT — ORPO entrena desde base)")
    p.add_argument("--dataset",      default="dataset/output/dpo_dataset_v2.jsonl",
                   help="Dataset de pares chosen/rejected (mismo formato que DPO)")
    p.add_argument("--output",       default="finetune/output/k8s-rca-orpo")
    p.add_argument("--epochs",       type=int,   default=3,
                   help="Épocas de entrenamiento (ORPO es más estable que DPO, puede usar más)")
    p.add_argument("--batch-size",   type=int,   default=1)
    p.add_argument("--grad-accum",   type=int,   default=16,
                   help="Batch efectivo = batch_size × grad_accum = 16")
    p.add_argument("--lr",           type=float, default=8e-6,
                   help="LR más bajo que DPO (5e-5): ORPO combina dos losses, gradientes más fuertes")
    p.add_argument("--lambda-orpo",  type=float, default=0.1,
                   help="λ — peso de la loss de preferencia respecto a L_SFT. "
                        "0.1 = conservador (formato domina), 1.0 = agresivo (preferencia domina)")
    p.add_argument("--max-seq-len",  type=int,   default=1024)
    p.add_argument("--max-prompt-len", type=int, default=512)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="Dropout LoRA. Más bajo que SFT v2 (0.10): L_SFT ya regulariza el formato")
    p.add_argument("--no-gguf",      action="store_true",
                   help="Saltar cuantización GGUF (solo guardar adaptadores LoRA)")
    return p.parse_args()


def load_orpo_dataset(path: str, tokenizer):
    """
    Carga el JSONL de pares chosen/rejected para ORPOTrainer.

    TRL ORPOTrainer espera los mismos campos que DPOTrainer:
      prompt   = system + user en ChatML (con marcador de assistant al final)
      chosen   = contenido del assistant correcto (string)
      rejected = contenido del assistant incorrecto (string)
    """
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)

            prompt_msgs   = sample["prompt"]   # [system, user]
            chosen_msgs   = sample["chosen"]   # [assistant_correct]
            rejected_msgs = sample["rejected"] # [assistant_wrong]

            try:
                prompt_str = tokenizer.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, AttributeError):
                # Fallback ChatML manual
                result = ""
                for msg in prompt_msgs:
                    result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
                result += "<|im_start|>assistant\n"
                prompt_str = result

            records.append({
                "prompt":   prompt_str,
                "chosen":   chosen_msgs[0]["content"],
                "rejected": rejected_msgs[0]["content"],
            })

    dataset = Dataset.from_list(records)
    print(f"  Dataset ORPO cargado: {len(dataset)} pares chosen/rejected")
    return dataset


def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  K8s-RCA-SLM — ORPO Fine-tuning")
    print(f"  Base model : {args.base_model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  λ (lambda) : {args.lambda_orpo}  (balance L_SFT / L_OR)")
    print(f"  Épocas     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Output     : {args.output}")
    print("═" * 60 + "\n")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import ORPOConfig, ORPOTrainer

    # ── 1. Modelo base con QLoRA 4-bit ────────────────────────────────────────
    print("[1/4] Cargando modelo base con QLoRA 4-bit...")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config = bnb_cfg,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
        cache_dir           = str(output_dir.parent / ".cache"),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ── 2. Adaptadores LoRA ───────────────────────────────────────────────────
    print(f"[2/4] Configurando LoRA (r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout})...")

    lora_cfg = LoraConfig(
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        bias           = "none",
        task_type      = "CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("[3/4] Cargando dataset ORPO...")
    dataset = load_orpo_dataset(args.dataset, tokenizer)

    orpo_config = ORPOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 20,
        beta                        = args.lambda_orpo,   # en ORPOConfig, beta = λ
        fp16                        = False,
        bf16                        = True,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        max_length                  = args.max_seq_len,
        max_prompt_length           = args.max_prompt_len,
        logging_steps               = 10,
        save_strategy               = "epoch",
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = 42,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
    )

    # Compatibilidad PEFT 0.19.x + TRL 0.24.x: ORPOTrainer intenta acceder a
    # model.warnings_issued que no existe en el LoraModel wrapper.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    # ORPOTrainer: sin ref_model — el odds ratio lo calcula internamente
    # TRL >= 0.11 usa processing_class en lugar de tokenizer
    trainer = ORPOTrainer(
        model            = model,
        args             = orpo_config,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )

    # ── 4. Entrenar ───────────────────────────────────────────────────────────
    print("[4/4] Entrenando con ORPO (L_SFT + λ·L_OR)...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    train_loss  = trainer_stats.metrics.get("train_loss", -1)
    print(f"\n  Completado en {runtime_min:.1f} min")
    print(f"  Train loss final: {train_loss:.4f}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adaptadores LoRA ORPO guardados en: {output_dir}")

    summary = {
        "method":       "ORPO",
        "base_model":   args.base_model,
        "dataset":      args.dataset,
        "lambda_orpo":  args.lambda_orpo,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "train_loss":   train_loss,
        "runtime_min":  runtime_min,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))


def quantize_to_gguf(lora_path: str, output_dir: str) -> None:
    """Fusiona LoRA con modelo base y exporta GGUF Q8_0."""
    import shutil
    import subprocess

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(output_dir)
    tmp_dir = out_dir / "_merged_tmp_orpo"

    print("  Fusionando adaptadores LoRA con modelo base...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    base = AutoModelForCausalLM.from_pretrained(
        lora_path,
        torch_dtype = torch.float16,
        device_map  = "auto",
    )
    merged = PeftModel.from_pretrained(base, lora_path)
    merged = merged.merge_and_unload()
    if hasattr(merged, "_hf_peft_config_loaded"):
        merged._hf_peft_config_loaded = False
    merged.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    del merged, base
    print(f"  Modelo fusionado en: {tmp_dir}")

    # Intentar Q4_K_M con llama.cpp quantize; fallback a Q8_0 con convert script
    llama_quantize = Path("/home/sonar/llama.cpp/build/bin/quantize")
    llama_convert  = Path("/home/sonar/llama.cpp/tools/convert_hf_to_gguf.py")

    gguf_name = "k8s-rca-orpo"

    if llama_quantize.is_file():
        # Ruta óptima: convertir a f16 → quantize a Q4_K_M
        f16_path = out_dir / f"{gguf_name}-f16.gguf"
        q4_path  = out_dir / f"{gguf_name}-Q4_K_M.gguf"
        subprocess.run(
            ["python3", str(llama_convert), str(tmp_dir),
             "--outfile", str(f16_path), "--outtype", "f16"],
            check=True
        )
        subprocess.run(
            [str(llama_quantize), str(f16_path), str(q4_path), "Q4_K_M"],
            check=True
        )
        f16_path.unlink(missing_ok=True)
        gguf_final = q4_path
    else:
        # Fallback: Q8_0 directamente (calidad comparable, ~1.6 GB)
        q8_path = out_dir / f"{gguf_name}-Q8_0.gguf"
        print("  [!] llama.cpp quantize no disponible — usando Q8_0")
        subprocess.run(
            ["python3", str(llama_convert), str(tmp_dir),
             "--outfile", str(q8_path), "--outtype", "q8_0"],
            check=True
        )
        gguf_final = q8_path

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n  GGUF listo: {gguf_final}")
    print("\n  Registrar en Ollama:")
    print("    cd finetune && ollama create k8s-rca-orpo -f Modelfile_orpo")


def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {args.dataset}\n"
            f"Ejecuta primero:\n"
            f"  python3 finetune/generate_dpo_dataset_v2.py"
        )

    train(args)

    if not args.no_gguf:
        print("\n" + "─" * 60)
        print("  Cuantizando a GGUF...")
        print("─" * 60)
        quantize_to_gguf(args.output, str(Path(args.output).parent))

    print("\n" + "═" * 60)
    print("  ORPO completado.")
    print("  Siguientes pasos:")
    print("    1. ollama create k8s-rca-orpo -f finetune/Modelfile_orpo")
    print("    2. python3 eval/run_eval.py --models orpo,sft,dpo_v2,baseline")
    print("    3. Actualizar EXPERIMENTS.md con resultados")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
