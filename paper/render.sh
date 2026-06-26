#!/usr/bin/env bash
# Genera un .qmd desde el paper markdown (Graphviz → figura) y lo renderiza.
# Uso:  render.sh <SRC.md> <OUT.qmd> <LANG> <PDF_NAME> [pdf|html]
set -euo pipefail

SRC="${1:-../RESEARCH.md}"
OUT="${2:-_generated.qmd}"
LANG="${3:-en}"
PDF="${4:-paper.pdf}"
FORMAT="${5:-pdf}"

echo "▸ Preprocesando $SRC → $OUT (lang=$LANG)"
python3 build_qmd.py "$SRC" "$OUT" "$LANG"

echo "▸ Renderizando $OUT → $PDF ($FORMAT)"
quarto render "$OUT" --to "$FORMAT" --output "$PDF"

echo "▸ Listo."
ls -la "$PDF" 2>/dev/null || true
