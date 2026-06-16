"""
KTO — Kahneman-Tversky Optimization (Ethayarajh et al., 2023).

A diferencia de DPO y ORPO, KTO no requiere pares chosen/rejected.
Cada muestra se etiqueta independientemente como deseable (label=True)
o indeseable (label=False), basándose en la prospect theory:

    L_KTO = λ_d · E[1 − σ( β·(log[Pθ/Pref] − z_ref) )]   ← deseables
           + λ_u · E[1 − σ( β·(z_ref − log[Pθ/Pref]) )]   ← indeseables

    z_ref = KL(Pθ || Pref)   ← divergencia KL como punto de referencia

Diferencias clave vs ORPO:
  - Necesita ref_model (como DPO), pero sin pares
  - Sin L_SFT explícita → posible format drift si base débil
  - Permite datos no pareados → más flexible para datasets reales
  - Fundamento teórico distinto: prospect theory vs Bradley-Terry

Uso:
  python finetune/train_kto.py
  python finetune/train_kto.py --beta 0.1 --epochs 3
  python finetune/train_kto.py --no-gguf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model",    default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Modelo base (mismo que ORPO para comparación limpia)")
    p.add_argument("--dataset",       default="dataset/output/kto_dataset.jsonl",
                   help="Dataset KTO con campos prompt, completion, label (bool)")
    p.add_argument("--output",        default="finetune/output/k8s-rca-kto")
    p.add_argument("--epochs",        type=int,   default=3)
    p.add_argument("--batch-size",    type=int,   default=2,
                   help="KTO requiere batch_size > 1 para calcular el término KL correctamente")
    p.add_argument("--grad-accum",    type=int,   default=8,
                   help="batch efectivo = batch_size × grad_accum = 16")
    p.add_argument("--lr",            type=float, default=8e-6,
                   help="Igual que ORPO para comparabilidad directa")
    p.add_argument("--beta",          type=float, default=0.1,
                   help="β — divergencia KL respecto al ref model (mismo que DPO/ORPO)")
    p.add_argument("--max-seq-len",   type=int,   default=1024)
    p.add_argument("--max-prompt-len",type=int,   default=512)
    p.add_argument("--lora-r",        type=int,   default=16)
    p.add_argument("--lora-alpha",    type=int,   default=32)
    p.add_argument("--lora-dropout",  type=float, default=0.05)
    p.add_argument("--no-gguf",       action="store_true")
    return p.parse_args()


def load_kto_dataset(path: str):
    """
    Carga el JSONL KTO. KTOTrainer espera:
      prompt     = string con el prompt completo (ChatML hasta <|im_start|>assistant\\n)
      completion = string con la respuesta del assistant
      label      = bool — True deseable, False indeseable
    """
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            records.append({
                "prompt":     s["prompt"],
                "completion": s["completion"],
                "label":      bool(s["label"]),
            })

    dataset = Dataset.from_list(records)
    n_pos = sum(1 for r in records if r["label"])
    n_neg = sum(1 for r in records if not r["label"])
    print(f"  Dataset KTO: {len(records)} muestras — {n_pos} deseables / {n_neg} indeseables")
    return dataset


def train(args: argparse.Namespace) -> None:
    print("\n" + "═" * 60)
    print("  K8s-RCA-SLM — KTO Fine-tuning")
    print(f"  Base model : {args.base_model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  β (beta)   : {args.beta}  (divergencia KL ref)")
    print(f"  Épocas     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Output     : {args.output}")
    print("═" * 60 + "\n")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import KTOConfig, KTOTrainer

    # ── 1. Modelo base con QLoRA 4-bit ─────────────────────────────────────────
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

    # ── 2. Adaptadores LoRA ────────────────────────────────────────────────────
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

    # ── 3. Dataset KTO ─────────────────────────────────────────────────────────
    print("[3/4] Cargando dataset KTO...")
    dataset = load_kto_dataset(args.dataset)

    kto_config = KTOConfig(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_steps                = 20,
        beta                        = args.beta,
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
        # KTO-specific: deseable vs indeseable deben estar balanceados
        # desirable_weight y undesirable_weight controlan λ_d y λ_u
        desirable_weight            = 1.0,
        undesirable_weight          = 1.0,
    )

    # Compatibilidad PEFT wrapper
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = KTOTrainer(
        model            = model,
        args             = kto_config,
        train_dataset    = dataset,
        processing_class = tokenizer,
    )

    # ── 4. Entrenar ────────────────────────────────────────────────────────────
    print("[4/4] Entrenando con KTO (prospect theory)...\n")
    trainer_stats = trainer.train()

    runtime_min = trainer_stats.metrics["train_runtime"] / 60
    train_loss  = trainer_stats.metrics.get("train_loss", -1)
    print(f"\n  Completado en {runtime_min:.1f} min")
    print(f"  Train loss final: {train_loss:.4f}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n  Adaptadores LoRA KTO guardados en: {output_dir}")

    summary = {
        "method":      "KTO",
        "base_model":  args.base_model,
        "dataset":     args.dataset,
        "beta":        args.beta,
        "lr":          args.lr,
        "epochs":      args.epochs,
        "train_loss":  train_loss,
        "runtime_min": runtime_min,
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
    tmp_dir = out_dir / "_merged_tmp_kto"

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

    # Strip PEFT prefix si existe (mismo fix que ORPO)
    _strip_peft_prefix(tmp_dir)

    llama_quantize = Path("/home/sonar/llama.cpp/build/bin/quantize")
    llama_convert  = Path("/home/sonar/llama.cpp/convert_hf_to_gguf.py")
    gguf_name      = "k8s-rca-kto"

    if llama_quantize.is_file():
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
    print("    cd finetune && ollama create k8s-rca-kto -f Modelfile_kto")


def _strip_peft_prefix(model_dir: Path) -> None:
    """Elimina el prefijo 'base_model.model.' de los tensores si existe."""
    from safetensors.torch import load_file, save_file

    sf_files = list(model_dir.glob("*.safetensors"))
    if not sf_files:
        return

    for sf_path in sf_files:
        tensors = load_file(str(sf_path))
        needs_fix = any(k.startswith("base_model.model.") for k in tensors)
        if not needs_fix:
            print("  Tensores OK — sin prefijo PEFT")
            return

        print(f"  Stripping PEFT prefix de {len(tensors)} tensores...")
        clean = {k.replace("base_model.model.", "", 1): v for k, v in tensors.items()}
        save_file(clean, str(sf_path))
        print(f"  Limpio: {sf_path.name}")


def main() -> None:
    args = parse_args()

    if not Path(args.dataset).exists():
        raise FileNotFoundError(
            f"Dataset no encontrado: {args.dataset}\n"
            f"Ejecuta primero:\n"
            f"  python3 finetune/generate_kto_dataset.py"
        )

    train(args)

    if not args.no_gguf:
        print("\n" + "─" * 60)
        print("  Cuantizando a GGUF...")
        print("─" * 60)
        quantize_to_gguf(args.output, str(Path(args.output).parent))

    print("\n" + "═" * 60)
    print("  KTO completado.")
    print("  Siguientes pasos:")
    print("    1. ollama create k8s-rca-kto -f finetune/Modelfile_kto")
    print("    2. python3 eval/run_eval.py --models kto,orpo,sft,baseline")
    print("    3. Actualizar EXPERIMENTS.md con resultados")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
