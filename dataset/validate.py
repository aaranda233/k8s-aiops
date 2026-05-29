"""
Validador del dataset antes de fine-tuning.

Comprueba:
  - Formato correcto (messages: system/user/assistant)
  - Longitud de tokens aproximada (evitar muestras demasiado largas)
  - Diversidad de escenarios
  - Output siempre contiene ROOT CAUSE y KUBECTL
  - Sin duplicados exactos
"""

import json
import sys
from collections import Counter
from pathlib import Path


MAX_CHARS = 4000   # aprox 1000 tokens — límite seguro para modelos 1.5b


def validate(path: Path) -> bool:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        print(f"ERROR: {path} está vacío.")
        return False

    errors   = []
    warnings = []
    scenario_ids = []
    seen_hashes  = set()
    token_lengths = []

    for i, line in enumerate(lines, 1):
        try:
            sample = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Línea {i}: JSON inválido — {e}")
            continue

        msgs = sample.get("messages", [])

        # Estructura
        if len(msgs) != 3:
            errors.append(f"Línea {i}: se esperan 3 mensajes (system/user/assistant), hay {len(msgs)}")
            continue
        if msgs[0]["role"] != "system":
            errors.append(f"Línea {i}: primer mensaje debe ser 'system'")
        if msgs[1]["role"] != "user":
            errors.append(f"Línea {i}: segundo mensaje debe ser 'user'")
        if msgs[2]["role"] != "assistant":
            errors.append(f"Línea {i}: tercer mensaje debe ser 'assistant'")

        assistant = msgs[2]["content"]

        # Formato de salida
        if "ROOT CAUSE:" not in assistant:
            errors.append(f"Línea {i}: falta ROOT CAUSE en assistant")
        if "KUBECTL:" not in assistant:
            errors.append(f"Línea {i}: falta KUBECTL en assistant")

        # Longitud
        total_chars = sum(len(m["content"]) for m in msgs)
        token_lengths.append(total_chars)
        if total_chars > MAX_CHARS:
            warnings.append(f"Línea {i}: muestra muy larga ({total_chars} chars > {MAX_CHARS})")

        # Duplicados
        key = hash(msgs[1]["content"][:200])
        if key in seen_hashes:
            warnings.append(f"Línea {i}: posible duplicado")
        seen_hashes.add(key)

        # Metadata
        meta = sample.get("metadata", {})
        if "scenario_id" in meta:
            scenario_ids.append(meta["scenario_id"])

    # Resumen
    print(f"\n{'═'*55}")
    print(f"  Dataset: {path.name}")
    print(f"  Total samples: {len(lines)}")
    print(f"{'─'*55}")

    if token_lengths:
        avg = sum(token_lengths) / len(token_lengths)
        print(f"  Longitud media: {avg:.0f} chars (~{avg//4:.0f} tokens)")
        print(f"  Longitud máx:   {max(token_lengths)} chars")

    if scenario_ids:
        print(f"\n  Distribución de escenarios:")
        for sid, count in Counter(scenario_ids).most_common():
            bar = "█" * (count // 2)
            print(f"    {sid:<30} {bar} {count}")

    if warnings:
        print(f"\n  ⚠ Advertencias ({len(warnings)}):")
        for w in warnings[:5]:
            print(f"    {w}")
        if len(warnings) > 5:
            print(f"    ... y {len(warnings)-5} más")

    if errors:
        print(f"\n  ✗ Errores ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        print(f"\n  DATASET INVÁLIDO")
        print(f"{'═'*55}\n")
        return False

    print(f"\n  ✓ Dataset válido — listo para fine-tuning")
    print(f"{'═'*55}\n")
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        "dataset/output/synthetic.jsonl",
        "dataset/output/labeled.jsonl",
        "dataset/output/combined.jsonl",
    ]
    ok = True
    for t in targets:
        p = Path(t)
        if p.exists():
            ok = validate(p) and ok
        else:
            print(f"No existe: {p}")
    sys.exit(0 if ok else 1)
