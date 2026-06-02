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
    p.add_argument("--batch-size", type=int,   default=1,
                   help="DPO necesita chosen+rejected+ref simultáneo: batch=1 para 24GB")
    p.add_argument("--grad-accum", type=int,   default=16,
                   help="Batch efectivo = batch_size * grad_accum = 16 por defecto")
    p.add_argument("--lr",         type=float, default=5e-5,
                   help="LR más bajo que SFT (2e-4) para estabilidad DPO")
    p.add_argument("--max-seq-len",type=int,   default=1024)
    p.add_argument("--lora-r",     type=int,   default=16)
    p.add_argument("--lora-alpha", type=int,   default=32)
    p.add_argument("--no-gguf",    action="store_true")
    return p.parse_args()


# ── Carga y formateo del dataset DPO ─────────────────────────────────────────

def _apply_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    """Aplica el formato ChatML de Qwen2.5 manualmente (sin depender de tokenizer.chat_template)."""
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
    return result


def load_dpo_dataset(path: str, tokenizer):
    """
    Carga el JSONL de pares DPO y los formatea para TRL DPOTrainer.

    TRL espera tres campos string:
      prompt   = system + user en ChatML (con marcador de assistant al final)
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

            prompt_msgs   = sample["prompt"]    # [system, user]
            chosen_msgs   = sample["chosen"]    # [assistant_correct]
            rejected_msgs = sample["rejected"]  # [assistant_wrong]

            # Intentar apply_chat_template; fallback a ChatML manual
            try:
                prompt_str = tokenizer.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, AttributeError):
                prompt_str = _apply_chatml(prompt_msgs, add_generation_prompt=True)

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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    from trl import DPOTrainer, DPOConfig

    # ── 1. Cargar tokenizer y modelo SFT con QLoRA 4-bit ─────────────────────
    print("[1/4] Cargando modelo SFT con QLoRA 4-bit (PEFT + bitsandbytes)...")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_compute_dtype     = torch.bfloat16,
        bnb_4bit_use_double_quant  = True,
    )

    # El checkpoint SFT tiene campos propietarios de unsloth en adapter_config.json.
    # Cargamos el modelo base de HF directamente y aplicamos los pesos LoRA manualmente.
    import json, tempfile, shutil
    from peft import PeftModel

    # Leer el base_model del config SFT para saber qué modelo base usar
    sft_cfg_path = Path(args.sft_model) / "adapter_config.json"
    with open(sft_cfg_path) as f:
        sft_cfg = json.load(f)

    # unsloth usa su propio hub slug — mapeamos al original de HuggingFace
    base_model_id = sft_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    if "unsloth" in base_model_id:
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"  Base model: {base_model_id}")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config = bnb_cfg,
        device_map          = "auto",
        torch_dtype         = torch.bfloat16,
        cache_dir           = str(Path(args.sft_model).parent.parent / ".cache"),
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()

    # Cargar los pesos LoRA del SFT (los campos desconocidos se ignoran automáticamente en peft>=0.19)
    model = PeftModel.from_pretrained(base, args.sft_model, is_trainable=True)
    print("  Adaptadores SFT cargados correctamente.")

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
        max_prompt_length           = 512,
        logging_steps               = 10,
        save_strategy               = "epoch",
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = 42,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
        precompute_ref_log_probs    = True,  # Precomputa logprobs de referencia 1 vez → mitad de memoria en training
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
    """Fusiona LoRA + exporta GGUF usando unsloth (solo para export, no training)."""
    import shutil
    import subprocess

    out_dir = Path(output_dir)
    tmp_dir = out_dir / "_merged_tmp"

    print("  Fusionando adaptadores LoRA con modelo base (fp16)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(lora_path)
    base = AutoModelForCausalLM.from_pretrained(
        lora_path, torch_dtype=torch.float16, device_map="auto"
    )
    merged = PeftModel.from_pretrained(base, lora_path)
    merged = merged.merge_and_unload()
    merged.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    del merged, base
    print(f"  Modelo fusionado en: {tmp_dir}")

    # Convertir a GGUF con llama.cpp (debe estar instalado en el sistema)
    gguf_path = out_dir / "k8s-rca-dpo-Q4_K_M.gguf"
    print("  Convirtiendo a GGUF Q4_K_M con llama.cpp...")
    result = subprocess.run(
        ["python3", "convert_hf_to_gguf.py", str(tmp_dir),
         "--outfile", str(gguf_path), "--outtype", "q4_k_m"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [!] Error en conversión GGUF: {result.stderr[:500]}")
        print(f"  Modelo fusionado disponible en: {tmp_dir}")
        print(f"  Convierte manualmente con llama.cpp")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  GGUF listo: {gguf_path}")

    print(f"\n  Registrar en Ollama:")
    print(f"    cd finetune && ollama create k8s-rca-dpo -f Modelfile_dpo")


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
