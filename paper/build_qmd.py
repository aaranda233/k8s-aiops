#!/usr/bin/env python3
"""Genera un `.qmd` de Quarto desde un paper en Markdown (fuente de verdad).

Uso:
    build_qmd.py [SRC.md] [OUT.qmd] [LANG]
        SRC.md   ruta al paper markdown   (default: ../RESEARCH.md)
        OUT.qmd  salida qmd               (default: _generated.qmd)
        LANG     en|es                    (default: en)

Extrae el título (primer `# `) y el abstract (`## Abstract` / `## Resumen`) al
front-matter YAML, elimina del cuerpo el título/metadatos/abstract, sanea glifos
emoji que xelatex+DejaVu no cubre, y **renderiza los bloques ```dot``` a PDF con
Graphviz** embebiéndolos como figura (evita la dependencia de Chrome de Quarto).
Reproducible: sin fechas dinámicas.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

# ── Configuración editable (autores, fecha, keywords) ───────────────────────
AUTHORS = [
    {"name": "Antonio Aranda Hernández", "email": "aaranda@hortichuelas.es"},
]
DATE = "2026-06-26"
KEYWORDS = [
    "AIOps", "Kubernetes", "Small Language Model", "Root Cause Analysis",
    "Anomaly Detection", "Automated Remediation", "ORPO", "ReAct Agent",
]

# Emoji/glifos que DejaVu no cubre → equivalentes seguros (flechas, cajas,
# checks ✓/✗, ⚠, ▶ y matemáticos sí están en DejaVu y se conservan).
_GLYPHS = {
    "⭐": "★", "✅": "✓", "❌": "✗", "⟳": "(retry)", "◻": "[ ]", "️": "",
}

_META_PREFIXES = ("**Status:**", "**Model:**", "**Hardware:**")
_ABSTRACT_HEADINGS = ("## Abstract", "## Resumen")


def _sanitize(text: str) -> str:
    for bad, good in _GLYPHS.items():
        text = text.replace(bad, good)
    return text


def _yaml_escape(s: str) -> str:
    return s.replace('"', '\\"')


def parse_research(md: str) -> tuple[str, list[str], list[str]]:
    """Devuelve (título, líneas_abstract, líneas_cuerpo)."""
    lines = md.splitlines()
    title, abstract, body = "", [], []
    in_abstract = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if not title and line.startswith("# "):
            title = line[2:].strip(); i += 1; continue
        if line.startswith(_META_PREFIXES):
            i += 1; continue
        if line.strip() in _ABSTRACT_HEADINGS:
            in_abstract = True; i += 1; continue
        if in_abstract:
            if line.startswith("## ") or line.startswith("# ") or line.strip() == "---":
                in_abstract = False  # no consumir: cae al cuerpo
            else:
                if line.strip():
                    abstract.append(line.rstrip())
                i += 1; continue
        body.append(line); i += 1

    while body and (not body[0].strip() or body[0].strip() == "---"):
        body.pop(0)
    return title, abstract, body


def render_dot_blocks(body: str) -> str:
    """Renderiza cada bloque ```dot``` a PDF con Graphviz y lo sustituye por una
    figura. Cada bloque puede llevar directivas `//| fig-id:` y `//| fig-cap:`.
    Si `dot` no está disponible, deja el bloque como código (no rompe el build).
    """
    if subprocess.run(["which", "dot"], capture_output=True).returncode != 0:
        return body

    def _replace(m: re.Match) -> str:
        inner = m.group(1)
        fig_id, fig_cap, dot_lines = "fig", "", []
        for ln in inner.splitlines():
            s = ln.strip()
            if s.startswith("//| fig-id:"):
                fig_id = s.split(":", 1)[1].strip()
            elif s.startswith("//| fig-cap:"):
                fig_cap = s.split(":", 1)[1].strip().strip('"')
            else:
                dot_lines.append(ln)
        dot_src = "\n".join(dot_lines).strip()
        out_pdf = HERE / f"{fig_id}.pdf"
        subprocess.run(["dot", "-Tpdf", "-o", str(out_pdf)],
                       input=dot_src, text=True, check=True)
        cap = _yaml_escape(fig_cap)
        return f'![{fig_cap}]({fig_id}.pdf){{#{fig_id} fig-align="center" width=85%}}'

    return re.sub(r"```dot\n(.*?)\n```", _replace, body, flags=re.DOTALL)


def build(src: Path, out: Path, lang: str) -> None:
    md = src.read_text(encoding="utf-8")
    title, abstract, body = parse_research(md)

    abstract_text = _sanitize(" ".join(abstract)).strip()
    body_text = render_dot_blocks(_sanitize("\n".join(body)).strip())

    authors_yaml = "\n".join(
        f'  - name: "{_yaml_escape(a["name"])}"\n    email: "{a["email"]}"'
        for a in AUTHORS
    )
    keywords_yaml = "\n".join(f"  - {k}" for k in KEYWORDS)
    abstract_block = "\n".join("  " + ln for ln in abstract_text.split("\n"))

    front = f"""---
title: "{_yaml_escape(title)}"
author:
{authors_yaml}
date: "{DATE}"
lang: {lang}
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
    out.write_text(front + body_text + "\n", encoding="utf-8")
    print(f"escrito {out.name} desde {src.name} (lang={lang} · título {len(title)}c · "
          f"abstract {len(abstract_text)}c · cuerpo {len(body_text)}c)")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "RESEARCH.md"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "_generated.qmd"
    lang = sys.argv[3] if len(sys.argv) > 3 else "en"
    build(src, out, lang)
