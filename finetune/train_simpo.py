"""
Fase 2 SimPO: fine-tuning con Simple Preference Optimization sobre el SFT.

Diferencia clave respecto al DPO v1:
  - No usa ref_model (CPOTrainer con loss_type="simpo")
  - La señal viene de la probabilidad absoluta normalizada por longitud
  - Añade un término NLL pequeño (cpo_alpha) para anclar el formato SFT

Esto resuelve el fallo del DPO v1 donde π_θ ≈ π_ref → gradiente ≈ 0.

Uso:
  python finetune/train_simpo.py
  python finetune/train_simpo.py --beta 2.0 --gamma 1.0 --epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sft-model",   default="finetune/output/k8s-rca-slm",
                   help="Checkpoint SFT del que partir")
    p.add_argument("--dataset",     default="dataset/output/dpo_dataset.jsonl",
                   help="Dataset de pares chosen/rejected (mismo que DPO)")
    p.add_argument("--output",      default="finetune/output/k8s-rca-simpo")
    p.add_argument("--epochs",      type=int,   default=1,
                   help="1 época — más épocas acumulan ruido con señal pequeña")
    p.add_argument("--beta",        type=float, default=2.0,
                   help="β SimPO — escala la diferencia de log-probs normalizados. "
                        "Paper usa 2.0 para tareas de instrucción.")
    p.add_argument("--gamma",       type=float, default=1.0,
                   help="γ SimPO — margen objetivo entre chosen y rejected. "
                        "Default paper: 0.5. Usamos 1.0 para forzar separación real.")
    p.add_argument("--cpo-alpha",   type=float, default=0.1,
                   help="Peso del término NLL sobre chosen. 0.1 ancla el formato SFT "
                        "sin dominar la señal SimPO.")
    p.add_argument("--batch-size",  type=int,   default=1)
    p.add_argument("--grad-accum",  type=int,   default=16,
                   help="Batch efectivo = batch_size * grad_accum = 16")
    p.add_argument("--lr",          type=float, default=3e-5,
                   help="Más conservador que DPO v1 (5e-5)")
    p.add_argument("--max-seq-len", type=int,   default=1024)
    p.add_argument("--lora-r",      type=int,   default=16)
    p.add_argument("--lora-alpha",  type=int,   default=32)
    p.add_argument("--no-gguf",     action="store_true")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

def _apply_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    result = ""
    for msg in messages:
        result += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    if add_generation_prompt:
        result += "<|im_start|>assistant\n"
    return result


def load_dataset(path: str, tokenizer):
    """
    Carga el JSONL de pares DPO/SimPO.
    CPOTrainer espera los mismos campos que DPOTrainer: prompt, chosen, rejected.
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

            try:
                prompt_str = tokenizer.apply_chat_template(
                    prompt_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (ValueError, AttributeError):
                prompt_str = _apply_chatml(prompt_msgs, add_generation_prompt=True)

            records.append({
                "prompt":   prompt_str,
                "chosen":   chosen_msgs[0]["content"],
                "rejected": rejected_msgs[0]["content"],
            })

    dataset = Dataset.from_list(records)
    print(f"  Dataset SimPO cargado: {len(dataset)} pares")
    return dataset


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  K8s-RCA-SLM — SimPO Fine-tuning")
    print(f"  SFT base   : {args.sft_model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  β (beta)   : {args.beta}")
    print(f"  γ (gamma)  : {args.gamma}")
    print(f"  cpo_alpha  : {args.cpo_alpha}")
    print(f"  Épocas     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Output     : {args.output}")
    print("═" * 60 + "\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel
    from trl import CPOTrainer, CPOConfig

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("[1/4] Cargando tokenizer y modelo SFT con QLoRA 4-bit...")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Modelo base + adaptadores SFT ─────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = True,
    )

    sft_cfg_path = Path(args.sft_model) / "adapter_config.json"
    with open(sft_cfg_path) as f:
        sft_cfg = json.load(f)

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

    model = PeftModel.from_pretrained(base, args.sft_model, is_trainable=True)
    print("  Adaptadores SFT cargados.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Parámetros entrenables: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("\n[3/4] Preparando dataset SimPO...")
    dataset = load_dataset(args.dataset, tokenizer)

    # ── 4. Configuración SimPO ────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpo_config = CPOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 10,
        # SimPO-specific
        loss_type                   = "simpo",
        beta                        = args.beta,
        simpo_gamma                 = args.gamma,
        cpo_alpha                   = args.cpo_alpha,
        # Formato y memoria
        max_length                  = args.max_seq_len,
        max_prompt_length           = 512,
        fp16                        = False,
        bf16                        = True,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        # Logging
        logging_steps               = 10,
        save_strategy               = "epoch",
        save_total_limit            = 2,
        report_to                   = "none",
        seed                        = 42,
        remove_unused_columns       = False,
        gradient_checkpointing      = True,
    )

    trainer = CPOTrainer(
        model         = model,
        args          = cpo_config,
        train_dataset = dataset,
        tokenizer     = tokenizer,
    )

    # ── 5. Entrenar ───────────────────────────────────────────────────────────
    print("\n[4/4] Entrenando con SimPO...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    print(f"\n  Entrenamiento completado en {runtime_min:.1f} min")
    print(f"  Loss final: {trainer_stats.metrics.get('train_loss', 'N/A')}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adaptadores LoRA SimPO guardados en: {output_dir}")


# ── Cuantización GGUF ─────────────────────────────────────────────────────────

def quantize_to_gguf(lora_path: str, output_dir: str) -> None:
    """Fusiona LoRA SimPO + exporta GGUF Q8_0."""
    import shutil
    import subprocess
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    out_dir = Path(output_dir)
    tmp_dir = out_dir / "_simpo_merged_tmp"

    print("  Fusionando adaptadores LoRA SimPO con modelo base (fp16)...")

    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    # Cargar base en fp16 limpio (sin bitsandbytes) para fusión correcta
    import json
    sft_cfg_path = Path(lora_path) / "adapter_config.json"
    with open(sft_cfg_path) as f:
        sft_cfg = json.load(f)
    base_model_id = sft_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    if "unsloth" in base_model_id:
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype = torch.float16,
        device_map  = "auto",
        cache_dir   = str(Path(lora_path).parent.parent / ".cache"),
    )
    merged = PeftModel.from_pretrained(base, lora_path)
    merged = merged.merge_and_unload()

    if hasattr(merged, "_hf_peft_config_loaded"):
        merged._hf_peft_config_loaded = False

    # Romper weight tying antes de guardar
    import torch.nn as nn
    merged.model.embed_tokens.weight = nn.Parameter(
        merged.lm_head.weight.detach().clone()
    )
    merged.config.tie_word_embeddings = False

    merged.save_pretrained(str(tmp_dir))
    tokenizer.save_pretrained(str(tmp_dir))
    del merged, base
    print(f"  Modelo fusionado en: {tmp_dir}")

    gguf_path = out_dir / "k8s-rca-simpo-Q8_0.gguf"
    print("  Convirtiendo a GGUF Q8_0...")
    result = subprocess.run(
        ["python3", "convert_hf_to_gguf.py", str(tmp_dir),
         "--outfile", str(gguf_path), "--outtype", "q8_0"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [!] Error GGUF: {result.stderr[:500]}")
        print(f"  Modelo fusionado disponible en: {tmp_dir}")
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  GGUF listo: {gguf_path}")

    print(f"\n  Registrar en Ollama:")
    print(f"    cd finetune && ollama create k8s-rca-simpo -f Modelfile_simpo")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {args.dataset}\n"
            f"Ejecuta primero: python finetune/generate_dpo_dataset.py"
        )
    if not Path(args.sft_model).exists():
        raise FileNotFoundError(f"Modelo SFT no encontrado: {args.sft_model}")

    train(args)

    if not args.no_gguf:
        print("\n" + "─" * 60)
        print("  Cuantizando a GGUF Q8_0...")
        print("─" * 60)
        quantize_to_gguf(args.output, str(Path(args.output).parent))

    print("\n" + "═" * 60)
    print("  SimPO completado.")
    print("  Siguiente paso:")
    print("    1. ollama create k8s-rca-simpo -f finetune/Modelfile_simpo")
    print("    2. python eval/run_eval.py --models sft,simpo,baseline")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
