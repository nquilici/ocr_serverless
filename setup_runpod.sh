#!/bin/bash
# setup_runpod.sh — Executar DENTRO do pod Runpod
# Faz tudo: instalar vLLM, baixar modelo, iniciar servidor

set -e

MODEL_ID=${1:-Qwen/Qwen3-VL-8B-Instruct}
PORT=${2:-8000}

echo "╔══════════════════════════════════════════════╗"
echo "║   RUNPOD OCR SETUP                          ║"
echo "╠══════════════════════════════════════════════╣"
echo "║ Modelo: $MODEL_ID"
echo "║ GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "║ VRAM:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "╚══════════════════════════════════════════════╝"

echo ""
echo "📦 Instalando vLLM..."
pip install -q vllm==0.8.3

echo ""
echo "📦 Baixando modelo (cache)..."
python -c "
from transformers import AutoProcessor
AutoProcessor.from_pretrained('$MODEL_ID', trust_remote_code=True)
print('✅ Modelo em cache')
"

echo ""
echo "🚀 Iniciando vLLM..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-num-seqs 32 \
    --limit-mm-per-prompt image=1
