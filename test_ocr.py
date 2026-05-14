#!/usr/bin/env python3
"""Quick test — sends a test image to the vLLM server."""

import base64, requests, sys, os
from PIL import Image
from io import BytesIO

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")

# Create test image with text
img = Image.new("RGB", (400, 100), (255, 255, 255))
buf = BytesIO()
img.save(buf, "PNG")
img_b64 = base64.b64encode(buf.getvalue()).decode()

headers = {"Content-Type": "application/json"}
if API_KEY:
    headers["Authorization"] = f"Bearer {API_KEY}"

resp = requests.post(f"{SERVER.rstrip('/')}/v1/chat/completions", json={
    "model": "default",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "What do you see in this image? Reply with one word."}
        ]
    }],
    "max_tokens": 10,
    "temperature": 0.0,
}, headers=headers, timeout=30)

print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Response: {data['choices'][0]['message']['content']}")
    print("✅ Test passed!")
else:
    print(f"Error: {resp.text[:200]}")
    sys.exit(1)
