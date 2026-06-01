# Base: PyTorch oficial de Docker Hub — torch 2.6 + CUDA 12.4 preinstalados
# Sin autenticación, ~8GB (vs 20GB de NGC)
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/workspace/.cache/huggingface

# Instalar git y unsloth desde PyPI
RUN apt-get update -qq && apt-get install -y -qq git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir unsloth
# Nota: "from unsloth import FastLanguageModel" necesita GPU — solo funciona en runtime con --gpus all

# Copiar scripts y dataset
COPY train.py evaluate_dataset.py ./
COPY output/ ./output/

VOLUME ["/workspace/output", "/workspace/.cache"]

CMD ["python", "train.py", "--dataset", "output/combined.jsonl"]
