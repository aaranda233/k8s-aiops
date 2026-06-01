"""
Evaluacion de calidad del dataset antes del fine-tuning.

Tres métricas:

  1. DIVERSIDAD SEMANTICA
     Embeds todos los user_msg con un modelo ligero (TF-IDF o sentence-transformers)
     y calcula la distancia media entre pares. Un dataset bueno tiene samples
     bien distribuidos en el espacio semántico, no apiñados en clusters.

  2. PERPLEXIDAD DEL MODELO BASE
     Mide cuánto le cuesta al modelo base predecir las respuestas del dataset.
     Perplexidad alta  → el modelo NO sabe esto → hay señal nueva que aprender ✓
     Perplexidad baja  → el modelo ya lo sabe   → fine-tuning innecesario ✗
     (requiere transformers + torch, se puede saltar con --no-perplexity)

  3. CALIDAD DE OUTPUTS
     Análisis léxico de las respuestas del assistant:
     - Longitud media de ROOT CAUSE y KUBECTL
     - Vocabulario único (type-token ratio)
     - Comandos kubectl distintos
     - Namespaces y deployments reales vs sintéticos

Uso:
  python dataset/evaluate_dataset.py dataset/output/combined.jsonl
  python dataset/evaluate_dataset.py dataset/output/combined.jsonl --no-perplexity
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", nargs="?", default="dataset/output/combined.jsonl")
    p.add_argument("--no-perplexity", action="store_true",
                   help="Saltar cálculo de perplexidad (necesita torch+transformers)")
    p.add_argument("--perplexity-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="Modelo base para calcular perplexidad")
    p.add_argument("--sample-perplexity", type=int, default=50,
                   help="Cuántos samples usar para perplexidad (es lento en CPU)")
    return p.parse_args()


# ── Carga ─────────────────────────────────────────────────────────────────────

def load(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ── 1. Diversidad semántica ───────────────────────────────────────────────────

def evaluate_diversity(samples: list[dict]) -> dict:
    """
    TF-IDF + cosine similarity entre todos los user_msg.
    Distancia media alta = dataset diverso.
    Distancia media baja = samples muy similares entre sí.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import pairwise_distances
    import numpy as np

    user_msgs = [s["messages"][1]["content"] for s in samples]

    # TF-IDF sobre los user_msg (rápido, sin GPU)
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vec.fit_transform(user_msgs)

    # Distancia coseno entre muestra aleatoria de pares (O(n²) es caro para 1k)
    rng = np.random.default_rng(42)
    n = len(samples)
    idx = rng.choice(n, size=min(300, n), replace=False)
    X_sample = X[idx]

    dists = pairwise_distances(X_sample, metric="cosine")
    upper = dists[np.triu_indices_from(dists, k=1)]

    mean_dist  = float(np.mean(upper))
    median_dist = float(np.median(upper))
    p10 = float(np.percentile(upper, 10))

    # Porcentaje de pares con distancia < 0.1 (casi duplicados)
    near_dupes_pct = float(np.mean(upper < 0.1) * 100)

    return {
        "mean_cosine_dist":    mean_dist,
        "median_cosine_dist":  median_dist,
        "p10_cosine_dist":     p10,
        "near_duplicate_pct":  near_dupes_pct,
    }


# ── 2. Perplexidad del modelo base ────────────────────────────────────────────

def evaluate_perplexity(samples: list[dict], model_name: str, n_samples: int) -> dict:
    """
    Calcula la perplexidad del modelo base sobre las respuestas (assistant turn).
    Solo evaluamos el assistant turn porque es lo que aprende el fine-tuning.

    PPL > 50  → señal nueva, fine-tuning útil ✓
    PPL < 20  → el modelo ya sabe esto, revisar si el dataset aporta algo ✗
    """
    import random
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Cargando {model_name} en {device} para calcular PPL...")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )
    model.eval()

    random.seed(42)
    subset = random.sample(samples, min(n_samples, len(samples)))

    ppls = []
    with torch.no_grad():
        for s in subset:
            # Formatear como chat completo
            chat = tok.apply_chat_template(
                s["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            ids = tok(chat, return_tensors="pt", max_length=1024, truncation=True).input_ids.to(device)

            # Solo calcular loss sobre el turn del assistant
            # (los tokens del system+user se enmascaran con -100)
            assistant_text = s["messages"][2]["content"]
            assistant_ids = tok(assistant_text, return_tensors="pt").input_ids[0]
            n_assistant = len(assistant_ids)

            labels = ids.clone()
            labels[0, :-n_assistant] = -100  # ignorar system + user

            out = model(ids, labels=labels)
            if not torch.isnan(out.loss):
                ppls.append(math.exp(out.loss.item()))

    return {
        "mean_perplexity":   round(sum(ppls) / len(ppls), 2) if ppls else None,
        "median_perplexity": round(sorted(ppls)[len(ppls)//2], 2) if ppls else None,
        "n_evaluated":       len(ppls),
    }


# ── 3. Calidad de outputs ─────────────────────────────────────────────────────

def evaluate_outputs(samples: list[dict]) -> dict:
    root_causes = []
    kubectls    = []
    all_words   = []

    for s in samples:
        assistant = s["messages"][2]["content"]

        rc_match = re.search(r"ROOT CAUSE:\s*(.+?)(?:\nKUBECTL:|$)", assistant, re.DOTALL)
        kc_match = re.search(r"KUBECTL:\s*(.+)", assistant)

        if rc_match:
            rc = rc_match.group(1).strip()
            root_causes.append(rc)
            all_words.extend(rc.lower().split())
        if kc_match:
            kubectls.append(kc_match.group(1).strip())

    # Type-Token Ratio (vocabulario único / total palabras)
    ttr = len(set(all_words)) / len(all_words) if all_words else 0

    # Comandos kubectl únicos
    kubectl_base = [k.split()[0:3] for k in kubectls]  # primeras 3 palabras
    unique_kubectl = len(set(" ".join(p) for p in kubectl_base))

    # Longitudes
    rc_lengths  = [len(r.split()) for r in root_causes]
    kc_lengths  = [len(k.split()) for k in kubectls]

    # Comandos kubectl más frecuentes
    kubectl_verbs = Counter(k.split()[1] if len(k.split()) > 1 else k for k in kubectls)

    return {
        "samples_with_root_cause": len(root_causes),
        "samples_with_kubectl":    len(kubectls),
        "root_cause_avg_words":    round(sum(rc_lengths) / len(rc_lengths), 1) if rc_lengths else 0,
        "kubectl_avg_words":       round(sum(kc_lengths) / len(kc_lengths), 1) if kc_lengths else 0,
        "type_token_ratio":        round(ttr, 3),
        "unique_kubectl_patterns": unique_kubectl,
        "kubectl_verb_distribution": dict(kubectl_verbs.most_common(8)),
    }


# ── Interpretación automática ─────────────────────────────────────────────────

def interpret(diversity: dict, outputs: dict, perplexity: dict | None) -> list[str]:
    issues   = []
    warnings = []
    good     = []

    d = diversity["mean_cosine_dist"]
    if d > 0.5:
        good.append(f"Diversidad semántica alta ({d:.3f}) — samples bien distribuidos")
    elif d > 0.3:
        warnings.append(f"Diversidad semántica media ({d:.3f}) — considera añadir más variedad")
    else:
        issues.append(f"Diversidad semántica baja ({d:.3f}) — muchos samples muy similares")

    nd = diversity["near_duplicate_pct"]
    if nd > 15:
        issues.append(f"Alto % de near-duplicados ({nd:.1f}%) — deduplicar más agresivamente")
    elif nd > 5:
        warnings.append(f"Algunos near-duplicados ({nd:.1f}%)")
    else:
        good.append(f"Near-duplicados bajo control ({nd:.1f}%)")

    ttr = outputs["type_token_ratio"]
    if ttr > 0.15:
        good.append(f"Vocabulario rico (TTR={ttr:.3f})")
    elif ttr > 0.08:
        warnings.append(f"Vocabulario moderado (TTR={ttr:.3f})")
    else:
        issues.append(f"Vocabulario repetitivo (TTR={ttr:.3f}) — respuestas muy formulaicas")

    uk = outputs["unique_kubectl_patterns"]
    if uk > 20:
        good.append(f"{uk} patrones kubectl distintos — buena cobertura de comandos")
    elif uk > 10:
        warnings.append(f"Solo {uk} patrones kubectl únicos")
    else:
        issues.append(f"Muy pocos patrones kubectl únicos ({uk}) — poco aprendizaje de comandos")

    if perplexity:
        ppl = perplexity["mean_perplexity"]
        if ppl and ppl > 50:
            good.append(f"Perplexidad alta ({ppl:.1f}) — el modelo base no sabe esto, fine-tuning muy útil")
        elif ppl and ppl > 20:
            warnings.append(f"Perplexidad media ({ppl:.1f}) — el modelo ya conoce parte del patrón")
        elif ppl:
            issues.append(f"Perplexidad baja ({ppl:.1f}) — el modelo base ya sabe responder, revisar si el fine-tuning aporta algo")

    return good, warnings, issues


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(path: str, n: int, diversity: dict, outputs: dict, perplexity: dict | None) -> None:
    print(f"\n{'═'*60}")
    print(f"  Evaluación del dataset: {Path(path).name}")
    print(f"  Total samples: {n}")
    print(f"{'─'*60}")

    print(f"\n  [1] DIVERSIDAD SEMÁNTICA (TF-IDF cosine)")
    print(f"      Distancia media entre pares : {diversity['mean_cosine_dist']:.3f}  (1.0 = completamente distintos)")
    print(f"      Distancia mediana           : {diversity['median_cosine_dist']:.3f}")
    print(f"      Percentil 10                : {diversity['p10_cosine_dist']:.3f}  (los más similares)")
    print(f"      Near-duplicados (dist<0.1)  : {diversity['near_duplicate_pct']:.1f}%")

    print(f"\n  [2] CALIDAD DE OUTPUTS")
    print(f"      Samples con ROOT CAUSE      : {outputs['samples_with_root_cause']}")
    print(f"      Samples con KUBECTL         : {outputs['samples_with_kubectl']}")
    print(f"      Palabras media ROOT CAUSE   : {outputs['root_cause_avg_words']}")
    print(f"      Palabras media KUBECTL      : {outputs['kubectl_avg_words']}")
    print(f"      Type-Token Ratio            : {outputs['type_token_ratio']:.3f}  (0=repetitivo, 1=todo distinto)")
    print(f"      Patrones kubectl únicos     : {outputs['unique_kubectl_patterns']}")
    print(f"      Verbos kubectl más usados:")
    for verb, count in outputs["kubectl_verb_distribution"].items():
        bar = "█" * (count // 5)
        print(f"        kubectl {verb:<20} {bar} {count}")

    if perplexity:
        ppl = perplexity["mean_perplexity"]
        print(f"\n  [3] PERPLEXIDAD DEL MODELO BASE")
        print(f"      Samples evaluados           : {perplexity['n_evaluated']}")
        print(f"      PPL media                   : {ppl}")
        print(f"      PPL mediana                 : {perplexity['median_perplexity']}")
        ppl_label = "ALTA → fine-tuning muy útil ✓" if ppl and ppl > 50 else \
                    "MEDIA → fine-tuning moderadamente útil" if ppl and ppl > 20 else \
                    "BAJA → modelo ya conoce el patrón ✗"
        print(f"      Interpretación              : {ppl_label}")

    good, warnings, issues = interpret(diversity, outputs, perplexity)

    print(f"\n{'─'*60}")
    for g in good:
        print(f"  ✓  {g}")
    for w in warnings:
        print(f"  ⚠  {w}")
    for i in issues:
        print(f"  ✗  {i}")

    if not issues:
        verdict = "DATASET LISTO PARA FINE-TUNING" if not warnings else "DATASET ACEPTABLE"
    else:
        verdict = "REVISAR DATASET ANTES DE ENTRENAR"

    print(f"\n  → {verdict}")
    print(f"{'═'*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    samples = load(args.dataset)
    print(f"Evaluando {len(samples)} samples de {args.dataset}...\n")

    print("  [1/3] Calculando diversidad semántica...")
    diversity = evaluate_diversity(samples)

    print("  [2/3] Analizando calidad de outputs...")
    outputs = evaluate_outputs(samples)

    perplexity = None
    if not args.no_perplexity:
        print(f"  [3/3] Calculando perplexidad (modelo: {args.perplexity_model}, "
              f"n={args.sample_perplexity})...")
        try:
            perplexity = evaluate_perplexity(samples, args.perplexity_model, args.sample_perplexity)
        except ImportError:
            print("       torch/transformers no instalados — saltando perplexidad")
            print("       Instala con: pip install torch transformers")
    else:
        print("  [3/3] Perplexidad saltada (--no-perplexity)")

    print_report(args.dataset, len(samples), diversity, outputs, perplexity)


if __name__ == "__main__":
    main()
