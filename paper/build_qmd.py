#!/usr/bin/env python3
"""Genera `paper/_generated.qmd` desde `RESEARCH.md` — única fuente de verdad.

El paper NO se edita aparte: se escribe en RESEARCH.md y este script extrae el
título (primer H1) y el abstract (sección `## Abstract`) al front-matter YAML de
Quarto, elimina del cuerpo el título, las líneas de metadatos
(`**Status/Model/Hardware**`) y la sección Abstract, y sanea unos pocos glifos
unicode que el motor xelatex+DejaVu no cubre. Reproducible: sin fechas dinámicas.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
RESEARCH = HERE.parent / "RESEARCH.md"
OUT = HERE / "_generated.qmd"

# ── Configuración editable (autores, fecha, keywords) ───────────────────────
AUTHORS = [
    {"name": "Antonio Aranda Hernández", "email": "aaranda@hortichuelas.es"},
]
DATE = "2026-06-26"
KEYWORDS = [
    "AIOps", "Kubernetes", "Small Language Model", "Root Cause Analysis",
    "Anomaly Detection", "Automated Remediation", "ORPO", "ReAct Agent",
]

# Emoji/glifos que DejaVu no cubre → equivalentes seguros (las flechas, cajas,
# checks ✓/✗, ⚠, ▶ y matemáticos sí están en DejaVu y se conservan).
_GLYPHS = {
    "⭐": "★",          # estrella emoji → estrella negra (sí está en DejaVu)
    "✅": "✓",          # check emoji → check simple
    "❌": "✗",          # cruz emoji → cruz simple
    "⟳": "(retry)",     # flecha circular no cubierta
    "◻": "[ ]",         # cuadro medio → ascii
    "️": "",       # variation selector-16 (emoji), invisible y molesto
}

_META_PREFIXES = ("**Status:**", "**Model:**", "**Hardware:**")


def _sanitize(text: str) -> str:
    for bad, good in _GLYPHS.items():
        text = text.replace(bad, good)
    return text


def _yaml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def parse_research(md: str) -> tuple[str, list[str], list[str]]:
    """Devuelve (título, líneas_abstract, líneas_cuerpo)."""
    lines = md.splitlines()
    title = ""
    abstract: list[str] = []
    body: list[str] = []

    i = 0
    in_abstract = False
    while i < len(lines):
        line = lines[i]
        if not title and line.startswith("# "):
            title = line[2:].strip()
            i += 1
            continue
        if line.startswith(_META_PREFIXES):
            i += 1
            continue
        if line.strip() == "## Abstract":
            in_abstract = True
            i += 1
            continue
        if in_abstract:
            # el abstract termina en el siguiente encabezado o separador
            if line.startswith("## ") or line.startswith("# ") or line.strip() == "---":
                in_abstract = False
                # no consumir esta línea: cae al cuerpo en la próxima iteración
            else:
                if line.strip():
                    abstract.append(line.rstrip())
                i += 1
                continue
        body.append(line)
        i += 1

    # recortar separadores/blancos sobrantes al inicio del cuerpo
    while body and (not body[0].strip() or body[0].strip() == "---"):
        body.pop(0)
    return title, abstract, body


def build() -> None:
    md = RESEARCH.read_text(encoding="utf-8")
    title, abstract, body = parse_research(md)

    abstract_text = _sanitize(" ".join(abstract)).strip()
    body_text = _sanitize("\n".join(body)).strip()

    authors_yaml = "\n".join(
        f'  - name: "{_yaml_escape(a["name"])}"\n    email: "{a["email"]}"'
        for a in AUTHORS
    )
    keywords_yaml = "\n".join(f"  - {k}" for k in KEYWORDS)

    # El abstract va indentado bajo un bloque literal YAML.
    abstract_block = "\n".join("  " + ln for ln in abstract_text.split("\n"))

    front = f"""---
title: "{_yaml_escape(title)}"
author:
{authors_yaml}
date: "{DATE}"
abstract: |
{abstract_block}
keywords:
{keywords_yaml}
format:
  pdf:
    pdf-engine: xelatex
    papersize: a4
    geometry: [margin=2.3cm]
    mainfont: "DejaVu Serif"
    sansfont: "DejaVu Sans"
    monofont: "DejaVu Sans Mono"
    monofontoptions: [Scale=0.8]
    fontsize: 10pt
    linestretch: 1.05
    toc: true
    toc-depth: 3
    number-sections: true
    colorlinks: true
    code-block-bg: true
    highlight-style: github
  html:
    toc: true
    toc-depth: 3
    number-sections: true
    theme: cosmo
    embed-resources: true
---

"""
    OUT.write_text(front + body_text + "\n", encoding="utf-8")
    print(f"escrito {OUT.relative_to(HERE.parent)} "
          f"(título: {len(title)} chars · abstract: {len(abstract_text)} chars · "
          f"cuerpo: {len(body_text)} chars)")


if __name__ == "__main__":
    build()
