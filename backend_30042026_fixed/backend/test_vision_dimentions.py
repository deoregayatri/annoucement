import base64
import io
import time
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import requests

# ---------------- CONFIGURATION ----------------
OLLAMA_URL = "http://localhost:11434/api/chat"  # Set to your localhost Ollama endpoint
MODEL_NAME = "qwen2.5vl:7b"                    # Vision model
PDF_PATH = "C:/Users/Gayatri/Downloads/aesthetik_22032025.pdf"
PAGES_TO_TEST = [1, 2]
DIMENSIONS_TO_TEST = [1024, 1536, 2048]
JPEG_QUALITY = 85
OUTPUT_DIR = Path("./vision_dimension_benchmark_results")
# -----------------------------------------------

OUTPUT_DIR.mkdir(exist_ok=True)


def prepare_image(
    image: Image.Image, max_dim: int, use_jpeg: bool = True
) -> tuple[str, float]:
    """Resize image and return base64 string + size in KB."""
    img = image.copy()
    
    # 1. Handle transparency by pasting over white background
    if img.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img.convert("RGB"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # 2. Resize with aspect ratio preserved
    if max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    # 3. Compress to buffer
    buffer = io.BytesIO()
    if use_jpeg:
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    else:
        img.save(buffer, format="PNG")

    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    size_kb = len(buffer.getvalue()) / 1024
    return b64_str, size_kb


def query_vision_llm(image_b64: str) -> tuple[str, float, str]:
    """Send image to Ollama VLM and return (extracted_text, latency, error_message)."""
    prompt = """Extract ALL visible text from this document image line by line.
Rules:
1. Extract every single line, heading, table content, and notice.
2. Preserve original reading order and column layout as much as possible.
3. Output only the extracted text, nothing else."""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt, "images": [image_b64]}
        ],
        "stream": False,
    }

    start = time.time()
    try:
        resp = requests.post(
            OLLAMA_URL, json=payload, timeout=300, verify=False
        )
        latency = time.time() - start
        if resp.status_code == 200:
            content = resp.json().get("message", {}).get("content", "")
            return content, latency, ""
        else:
            return "", latency, f"HTTP {resp.status_code}: {resp.text[:100]}"
    except Exception as e:
        return "", time.time() - start, str(e)


import sys
# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def run_benchmark():
    global MODEL_NAME, OLLAMA_URL

    # Auto-detect which localhost port (11434, 11435) is active and check model name
    candidate_urls = [
        OLLAMA_URL,
        "http://localhost:11434/api/chat",
        "http://localhost:11435/api/chat",
        "https://localhost:11435/api/chat",
    ]
    for url in candidate_urls:
        try:
            tags_url = url.replace("/api/chat", "/api/tags")
            r = requests.get(tags_url, timeout=2, verify=False)
            if r.status_code == 200:
                OLLAMA_URL = url
                available = [m.get("name", "") for m in r.json().get("models", [])]
                for m in available:
                    if "qwen2.5-vl" in m.lower() or "qwen" in m.lower():
                        MODEL_NAME = m
                        break
                break
        except Exception:
            continue

    doc = fitz.open(PDF_PATH)
    print(f"\n{'='*80}")
    print(f"BENCHMARKING VISION EXTRACTION FOR: {Path(PDF_PATH).name}")
    print(f"Target URL   : {OLLAMA_URL}")
    print(f"Target Model : {MODEL_NAME}")
    print(f"{'='*80}\n")

    for page_num in PAGES_TO_TEST:
        page = doc[page_num - 1]
        pix = page.get_pixmap(dpi=300)
        pil_img = Image.open(io.BytesIO(pix.tobytes("png")))

        print(
            f"\n--- Testing Page {page_num} (Original Size: {pil_img.size[0]}x{pil_img.size[1]}) ---"
        )
        print(
            f"{'Dimension':<12} | {'Format':<6} | {'Payload':<10} | {'Status':<10} | {'Lines':<8} | {'Chars':<8} | {'Time (s)':<8}"
        )
        print("-" * 80)

        # 1. Test raw PNG
        png_b64, png_kb = prepare_image(pil_img, max_dim=None, use_jpeg=False)
        png_text, png_time, png_err = query_vision_llm(png_b64)
        png_status = "OK" if not png_err else "FAILED"
        print(
            f"{'Original':<12} | {'PNG':<6} | {png_kb:>7.1f} KB | {png_status:<10} | {len(png_text.splitlines()):<8} | {len(png_text):<8} | {png_time:>7.1f}s"
        )
        if png_err:
            print(f"   -> [Error Detail]: {png_err}")

        # 2. Test different JPEG dimensions
        for dim in DIMENSIONS_TO_TEST:
            jpg_b64, jpg_kb = prepare_image(
                pil_img, max_dim=dim, use_jpeg=True
            )
            text, elapsed, err = query_vision_llm(jpg_b64)

            status = "OK" if not err else "FAILED"
            lines_count = len(
                [line for line in text.splitlines() if line.strip()]
            )
            chars_count = len(text)

            print(
                f"{f'{dim}px':<12} | {'JPEG':<6} | {jpg_kb:>7.1f} KB | {status:<10} | {lines_count:<8} | {chars_count:<8} | {elapsed:>7.1f}s"
            )
            if err:
                print(f"   -> [Error Detail]: {err}")

            if text:
                out_file = OUTPUT_DIR / f"page_{page_num}_{dim}px_extracted.txt"
                out_file.write_text(text, encoding="utf-8")

    print(
        f"\n[DONE] All extracted texts saved to: {OUTPUT_DIR.resolve()} for comparison."
    )


if __name__ == "__main__":
    run_benchmark()

