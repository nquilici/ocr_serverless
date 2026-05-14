# syntax=docker/dockerfile:1
# Dockerfile — vLLM + Qwen3-VL OCR (Runpod Serverless)
# 
# Runpod serverless detecta Dockerfile na raiz do repo GitHub.
# Build: docker build -t ocr-vllm .
# O modelo é embedado na imagem para cold-start rápido.

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
ENV HF_HOME=/models HF_HUB_ENABLE_HF_TRANSFER=1

# Sistema
RUN apt-get update -qq && apt-get install -y -qq \
    python3.12 python3.12-venv python3-pip curl && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# vLLM
RUN pip install --no-cache-dir \
    vllm==0.8.3 \
    torch==2.6.0 \
    transformers==4.51.0 \
    pillow requests

# ═══ Pré-download do modelo (embedado na imagem) ═══
# Troque MODEL_ID para alternar entre 4B/8B/DeepSeek
ARG MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
RUN python -c "\
from transformers import AutoProcessor; \
print(f'📦 Baixando {MODEL_ID}...'); \
AutoProcessor.from_pretrained('${MODEL_ID}', trust_remote_code=True); \
print('✅ Cache OK')"

# Script de entrada
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENV MODEL_ID=${MODEL_ID}
ENV GPU_MEMORY_UTILIZATION=0.95
ENV MAX_MODEL_LEN=4096

ENTRYPOINT ["/entrypoint.sh"]
