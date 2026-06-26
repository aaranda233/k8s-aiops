# Paper — generación con Quarto (Docker)

Genera el paper en PDF/HTML con estilo académico **desde `RESEARCH.md`**, que es
la única fuente de verdad. No se edita ningún documento aparte: se escribe en
`RESEARCH.md` y aquí se renderiza.

## Cómo funciona

```
RESEARCH.md ──(build_qmd.py)──▶ _generated.qmd ──(Quarto + xelatex)──▶ paper.pdf
```

`build_qmd.py` extrae el título (primer `# `) y el abstract (`## Abstract`) al
front-matter YAML de Quarto, quita del cuerpo el título/metadatos/abstract y
genera `_generated.qmd` (autores, keywords, formato y plantilla). Quarto lo
compila a PDF con TinyTeX/xelatex y fuentes DejaVu (cubren flechas, checks y los
caracteres de caja de los diagramas).

## Uso

Requiere Docker (el daemon corriendo). Todo va dentro del contenedor.

```bash
cd paper
make pdf      # → paper/paper.pdf      (inglés, desde RESEARCH.md)
make pdf-es   # → paper/paper-es.pdf   (español, desde RESEARCH_es.md)
make html     # → paper/paper.html     (inglés)
make all      # EN + ES (PDF)
make shell    # shell para depurar dentro del contenedor
make clean    # borra artefactos generados
```

La primera `make` construye la imagen (Quarto + TeX Live + Graphviz + fuentes),
tarda unos minutos; las siguientes solo renderizan.

## Diagramas

Los bloques ` ```dot ` en el markdown se renderizan a PDF vectorial con Graphviz
(en `build_qmd.py`, vía `dot -Tpdf`) y se embeben como figura — sin depender del
navegador headless que Quarto pide por defecto. Cada bloque admite directivas
`//| fig-id:` y `//| fig-cap:`.

## Personalización

- **Autores / fecha / keywords:** editar las constantes al principio de
  `build_qmd.py`.
- **Estilo / formato:** el bloque `format:` que escribe `build_qmd.py`
  (márgenes, fuentes, tamaño, TOC). Para una plantilla de revista (IEEE, ACM,
  Elsevier, Springer LNCS) se añade su extensión de Quarto:
  `quarto add quarto-journals/ieee` y se cambia `format: ieee-pdf`.
- **Citas:** añadir un `references.bib` y `bibliography: references.bib` al
  front-matter; usar `[@clave]` en `RESEARCH.md`.

## Salida

`paper.pdf` y `paper.html` quedan en `paper/` (gitignored — se regeneran).
