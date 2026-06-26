# Generación del paper con Quarto en Docker.
#   make build   → construye la imagen (una vez)
#   make pdf     → genera paper/paper.pdf desde RESEARCH.md
#   make html    → genera paper/paper.html
#   make all     → ambos
#   make shell   → shell dentro del contenedor (depurar)
#
# RESEARCH.md es la única fuente de verdad: el paper se edita ahí.

IMAGE := k8s-aiops-paper
REPO  := $(abspath ..)
RUN   := docker run --rm -v "$(REPO):/work" -w /work/paper $(IMAGE)

.PHONY: build pdf html all shell clean

build:
	docker build -t $(IMAGE) .

pdf: build
	$(RUN) bash render.sh pdf

html: build
	$(RUN) bash render.sh html

all: build
	$(RUN) bash render.sh all

shell: build
	docker run --rm -it -v "$(REPO):/work" -w /work/paper $(IMAGE) bash

clean:
	rm -f _generated.qmd paper.pdf paper.html
	rm -rf .quarto _generated_files
