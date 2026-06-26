#!/usr/bin/env bash
# Regenera _generated.qmd desde RESEARCH.md y lo renderiza con Quarto.
# Uso:  render.sh [pdf|html|all]   (por defecto: pdf)
set -euo pipefail

FORMAT="${1:-pdf}"

echo "▸ Preprocesando RESEARCH.md → _generated.qmd"
python3 build_qmd.py

case "$FORMAT" in
  pdf)  quarto render _generated.qmd --to pdf  --output paper.pdf ;;
  html) quarto render _generated.qmd --to html --output paper.html ;;
  all)  quarto render _generated.qmd --to pdf  --output paper.pdf
        quarto render _generated.qmd --to html --output paper.html ;;
  *)    echo "formato no soportado: $FORMAT (usa pdf|html|all)"; exit 1 ;;
esac

echo "▸ Listo. Salida en paper/"
ls -la paper.pdf paper.html 2>/dev/null || true
