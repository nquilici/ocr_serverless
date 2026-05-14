#!/bin/bash
# entrypoint.sh — Inicia vLLM com Qwen3-VL para OCR
set -e

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-4B-Instruct}
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEMORY_UTILIZATION:-0.95}
MAX_LEN=${MAX_MODEL_LEN:-4096}

echo "╔══════════════════════════════════════════╗"
echo "║   OCR SERVER — vLLM + Qwen3-VL          ║"
echo "╠══════════════════════════════════════════╣"
echo "║ Modelo: $MODEL_ID"
echo "║ Porta:  $PORT"
echo "║ GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "║ VRAM:   $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'none')"
echo "╚══════════════════════════════════════════╝"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --trust-remote-code \
    --enable-prefix-caching \
    --max-num-seqs 32 \
    --limit-mm-per-prompt image=1
