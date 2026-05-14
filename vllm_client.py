#!/usr/bin/env python3
"""
vllm_client.py — Envia páginas do PDF para o servidor vLLM

Conecta no endpoint OpenAI-compatible do vLLM e processa em alta velocidade.
Suporta paralelismo: até 32 requisições simultâneas (continuous batching).

Uso:
  python vllm_client.py fisico.pdf --server http://GPU_IP:8000 --start 1500 --count 8000 --workers 16
"""

import os, sys, json, time, argparse, base64
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import fitz  # pymupdf
from PIL import Image
import requests

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════

PROMPT = (
    "Transcreva exatamente todo o texto visível nesta página de documento. "
    "Preserve parágrafos, pontuação, números e estrutura. Não resuma, não explique."
)

OVERLAY_FONTS = ("Helvetica-Bold",)
OVERLAY_SIZES = (8, 9)
DPI = 150

_log_lock = threading.Lock()

def log(msg: str):
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════
# EXTRAÇÃO DE IMAGENS (local, rápida)
# ═══════════════════════════════════════════

def extract_page_image(doc, page_num, temp_dir):
    page = doc[page_num]
    for img_ref in page.get_images():
        if img_ref[2] > 400 and img_ref[3] > 400:
            try:
                pix = fitz.Pixmap(doc, img_ref[0])
                if pix.n > 4: pix = fitz.Pixmap(fitz.csRGB, pix)
                path = f"{temp_dir}/pg_{page_num:04d}.png"
                pix.save(path)
                if os.path.getsize(path) > 100: return path
            except: pass
    safe = fitz.Rect(15, 70, page.rect.width - 30, page.rect.height - 120)
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(clip=safe, matrix=mat)
    if pix.n > 4: pix = fitz.Pixmap(fitz.csRGB, pix)
    path = f"{temp_dir}/pg_{page_num:04d}.png"
    pix.save(path)
    return path if os.path.getsize(path) > 100 else None


def classify_page(page):
    blocks = page.get_text("dict")["blocks"]
    tc, hi = 0, False
    for b in blocks:
        bb = b["bbox"]; sk = False
        for rx,ry,rx1,ry1 in [(0,0,250,60),(580,0,700,1000),(0,860,600,1100)]:
            if bb[0]>=rx and bb[1]>=ry and bb[2]<=rx1 and bb[3]<=ry1 and b["type"]==0:
                ao = True
                for l in b.get("lines",[]):
                    for s in l.get("spans",[]):
                        fnt,sz = s.get("font",""), s.get("size",0)
                        if not ((fnt in OVERLAY_FONTS and sz in OVERLAY_SIZES) or (fnt=="ArialNarrow" and abs(sz-11.04)<0.1)):
                            ao = False
                if ao: sk = True
        if sk: continue
        if b["type"] == 0:
            for l in b.get("lines",[]):
                for s in l.get("spans",[]):
                    fnt,sz = s.get("font",""), s.get("size",0)
                    if not ((fnt in OVERLAY_FONTS and sz in OVERLAY_SIZES) or (fnt=="ArialNarrow" and abs(sz-11.04)<0.1)):
                        tc += len(s.get("text","").strip())
        elif b["type"] == 1 and bb[2]-bb[0] > 200: hi = True
    return "mixed" if tc>80 and hi else ("text" if tc>80 else ("image" if hi else "empty"))


# ═══════════════════════════════════════════
# COMUNICAÇÃO COM VLLM
# ═══════════════════════════════════════════

class VLLMClient:
    def __init__(self, server_url: str, max_tokens: int = 512, timeout: int = 120):
        self.api_url = server_url.rstrip("/") + "/v1/chat/completions"
        self.max_tokens = max_tokens
        self.timeout = timeout

    def check_health(self) -> bool:
        try:
            r = requests.get(self.api_url.replace("/chat/completions", "/models"), timeout=10)
            return r.status_code == 200
        except:
            return False

    def ocr_page(self, image_path: str) -> dict:
        """Envia imagem ao vLLM e retorna texto."""
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        t0 = time.time()
        try:
            resp = requests.post(self.api_url, json={
                "model": "default",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": PROMPT},
                    ]
                }],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            }, timeout=self.timeout)

            elapsed = time.time() - t0

            if resp.status_code != 200:
                return {"text": "", "error": f"HTTP {resp.status_code}", "chars": 0, "time": elapsed}

            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()

            return {"text": text, "chars": len(text), "time": round(elapsed, 1)}

        except Exception as e:
            return {"text": "", "error": str(e)[:60], "chars": 0, "time": round(time.time() - t0, 1)}


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="vLLM OCR Client")
    parser.add_argument("pdf", help="PDF a processar")
    parser.add_argument("--server", required=True, help="URL do servidor vLLM (http://IP:8000)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", default="output_vllm")
    parser.add_argument("--workers", type=int, default=16, help="Requisições paralelas")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output)
    temp_dir = output_dir / "_temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(exist_ok=True)

    log(f"\n{'='*60}")
    log(f"VLLM OCR CLIENT — {args.workers} workers")
    log(f"{'='*60}")
    log(f"  PDF:     {args.pdf}")
    log(f"  Server:  {args.server}")
    log(f"  Páginas: {args.start}–{args.start+args.count-1}")
    log(f"  Workers: {args.workers}")
    log(f"{'='*60}\n")

    # Verificar servidor
    client = VLLMClient(args.server, args.max_tokens)
    if not client.check_health():
        log(f"❌ Servidor inacessível: {args.server}")
        sys.exit(1)
    log("✅ vLLM server online")

    # Classificar páginas (local)
    doc = fitz.open(args.pdf)
    log("📋 Classificando páginas...")
    pages_to_ocr = []
    stats = {"text": 0, "image": 0, "mixed": 0, "empty": 0}

    for i in range(args.count):
        page_num = args.start + i
        page = doc[page_num]
        ptype = classify_page(page)
        stats[ptype] = stats.get(ptype, 0) + 1

        if ptype in ("image", "mixed"):
            img_path = extract_page_image(doc, page_num, str(temp_dir))
            if img_path:
                pages_to_ocr.append((page_num, ptype, img_path))
            else:
                (output_dir / f"page_{page_num:04d}.txt").write_text("[PÁGINA EM BRANCO]", encoding="utf-8")
        elif ptype == "text":
            text = page.get_text("text")
            (output_dir / f"page_{page_num:04d}.txt").write_text(text or "", encoding="utf-8")
        else:
            (output_dir / f"page_{page_num:04d}.txt").write_text("", encoding="utf-8")

    doc.close()
    log(f"  OCR: {len(pages_to_ocr)} | Texto nativo: {stats['text']} | Vazias: {stats['empty']}")

    # OCR paralelo via vLLM
    log(f"\n🚀 Iniciando OCR paralelo ({len(pages_to_ocr)} páginas, {args.workers} workers)...")
    t_start = time.time()
    done = 0
    chars_total = 0

    def process_one(page_num, ptype, img_path):
        result = client.ocr_page(img_path)
        if result["text"]:
            (output_dir / f"page_{page_num:04d}.txt").write_text(result["text"], encoding="utf-8")
        else:
            err = result.get("error", "desconhecido")
            (output_dir / f"page_{page_num:04d}.txt").write_text(f"[ERRO: {err}]", encoding="utf-8")
        return page_num, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, pn, pt, ip): pn for pn, pt, ip in pages_to_ocr}
        
        for fut in as_completed(futures):
            page_num, result = fut.result()
            done += 1
            chars_total += result["chars"]
            elapsed = time.time() - t_start
            
            if done % 50 == 0 or done <= 5:
                ppm = done / (elapsed / 60)
                eta = (len(pages_to_ocr) - done) / ppm if ppm > 0 else 0
                log(f"  {done}/{len(pages_to_ocr)} | {ppm:.0f} pág/min | "
                    f"ETA: {eta:.0f} min | {result['time']:.0f}s/pág")

    elapsed = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"CONCLUÍDO em {elapsed/60:.1f} min")
    log(f"  Páginas: {done} | {chars_total:,} caracteres")
    log(f"  Ritmo:   {done/(elapsed/60):.0f} pág/min")
    log(f"  Output:  {output_dir}/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
