# Generación del paper con Quarto en Docker.
#   make build    → construye la imagen (una vez)
#   make pdf      → paper/paper.pdf      (inglés, desde RESEARCH.md)
#   make pdf-es   → paper/paper-es.pdf   (español, desde RESEARCH_es.md)
#   make html     → paper/paper.html     (inglés)
#   make all      → EN + ES (PDF)
#   make shell    → shell dentro del contenedor (depurar)
#   make clean    → borra artefactos generados
#
# Los .md son la única fuente de verdad: el paper se edita ahí.

IMAGE := k8s-aiops-paper
REPO  := $(abspath ..)
RUN   := docker run --rm -v "$(REPO):/work" -w /work/paper $(IMAGE)

.PHONY: build pdf pdf-es html all shell clean

build:
	docker build -t $(IMAGE) .

pdf: build
	$(RUN) bash render.sh ../RESEARCH.md _generated.qmd en paper.pdf pdf

pdf-es: build
	$(RUN) bash render.sh ../RESEARCH_es.md _generated_es.qmd es paper-es.pdf pdf

html: build
	$(RUN) bash render.sh ../RESEARCH.md _generated.qmd en paper.html html

all: pdf pdf-es

shell: build
	docker run --rm -it -v "$(REPO):/work" -w /work/paper $(IMAGE) bash

clean:
	rm -f _generated*.qmd paper.pdf paper-es.pdf paper.html fig-*.pdf
	rm -rf .quarto _generated_files _generated_es_files
