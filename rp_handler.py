"""
rp_handler.py — Runpod Serverless Handler for vLLM OCR

Runpod invokes this handler for each request.
It initializes the vLLM model once and reuses it across requests.
"""

import os, base64, time
from io import BytesIO
from PIL import Image

# ═══════════════════════════════════════════
# INIT (runs once on cold start)
# ═══════════════════════════════════════════

print("📦 Loading vLLM model...", flush=True)

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

llm = LLM(
    model=MODEL_ID,
    trust_remote_code=True,
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    max_num_seqs=32,
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
)

print(f"✅ Model loaded: {MODEL_ID}", flush=True)


# ═══════════════════════════════════════════
# HANDLER
# ═══════════════════════════════════════════

def handler(event):
    """
    Event format (OpenAI-compatible):
    {
      "input": {
        "messages": [{
          "role": "user",
          "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "Transcribe this page..."}
          ]
        }]
      }
    }
    """
    try:
        messages = event.get("input", event).get("messages", [])
        
        # Extract prompt and images
        prompt = ""
        image_data = None
        
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    prompt += part.get("text", "") + "\n"
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image"):
                        image_data = url.split(",", 1)[1]
        
        if not image_data:
            return {"error": "No image provided"}
        
        prompt = prompt.strip() or "Transcribe exactly all visible text in this document page."
        
        # Build vLLM prompt with image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))
        
        messages_vllm = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]
        }]
        
        t0 = time.time()
        outputs = llm.chat(
            messages=messages_vllm,
            sampling_params=sampling_params,
        )
        elapsed = time.time() - t0
        
        text = outputs[0].outputs[0].text.strip()
        
        return {
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": len(text.split()),
                "total_tokens": len(text.split()),
            },
            "time": round(elapsed, 2),
        }
    
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════
# For local testing
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import json, sys
    event = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {
        "input": {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What do you see?"}
                ]
            }]
        }
    }
    result = handler(event)
    print(json.dumps(result, indent=2))
