# Imagen autocontenida para generar el paper desde RESEARCH.md con Quarto.
#   - Quarto (markdown académico → PDF/HTML)
#   - TeX Live vía apt (xelatex + paquetes); nativo por arquitectura, sin depender
#     de la instalación de TinyTeX (que falla bajo emulación amd64↔arm64)
#   - Fuentes DejaVu (cubren flechas, checks, caracteres de caja de los diagramas)
#   - python3 (ejecuta el preprocesador build_qmd.py)
FROM ghcr.io/quarto-dev/quarto:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        fonts-dejavu \
        fonts-dejavu-extra \
        fontconfig \
        texlive-xetex \
        texlive-latex-recommended \
        texlive-latex-extra \
        texlive-fonts-recommended \
        texlive-plain-generic \
        lmodern \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paper
