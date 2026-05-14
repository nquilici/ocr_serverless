# OCR Serverless — vLLM + Qwen3-VL

[![Runpod Serverless](https://img.shields.io/badge/Runpod-Serverless-6C47FF?logo=runpod)](https://runpod.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://hub.docker.com/r/nquilici/ocr-serverless)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

High-throughput OCR for legal documents using vLLM + Qwen3-VL-8B.

## Quick Start

```bash
# Local test
docker run --gpus all -p 8000:8000 nquilici/ocr-serverless

# Use with client
python vllm_client.py document.pdf --server http://localhost:8000 --workers 16
```

## Runpod Serverless

1. Create template from this repo
2. Create endpoint (A40 GPU)
3. Connect: `python vllm_client.py --server https://api.runpod.ai/v2/<ID>`

## Performance (A40, 16 workers)

| Pages | Time | Cost |
|-------|------|------|
| 1,000 | ~20 min | ~$0.53 |
| 8,000 | ~2.7 h | ~$2.13 |
| 19,654 | ~6.6 h | ~$5.21 |
