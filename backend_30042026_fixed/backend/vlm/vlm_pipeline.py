"""
vlm/vlm_pipeline.py
===================
Orchestrates the full VLM image processing flow for a single PDF.

Flow:
  extract_images_from_pdf()
      ↓
  For each image:
      compute_image_hash()
      ├── check per-PDF seen_hashes → SKIP (intra-doc duplicate)
      ├── check global cache       → REUSE (cross-doc cache hit)
      └── call_vlm_sync()          → store in cache
      ↓
  Return list[ImageResult]

Metrics logged per PDF:
  total_images, unique_images, cache_hits, per_doc_dedup, total_vlm_ms

Fully self-contained: no FastAPI / event-loop dependency.
Can be called directly from ProcessPoolExecutor or ThreadPoolExecutor workers.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default VLM prompt for financial document images
_VLM_PROMPT = (
    "You are analyzing an image extracted from a financial market announcement PDF. "
    "Describe all visible content concisely: text, numbers, charts, tables, logos, "
    "or structural elements. Focus on information relevant to market surveillance, "
    "company disclosures, or financial data. If the image contains no useful "
    "financial information (e.g. a plain logo), say so briefly."
)


def _pil_to_base64(pil_image, max_dim: int = 1024, quality: int = 85) -> str:
    """
    Resize image to max_dim (preserving aspect ratio) and encode as base64 PNG.
    
    Args:
        pil_image: PIL.Image in any mode.
        max_dim: Maximum pixel dimension on the longer side.
        quality: JPEG quality for resize step (PNG is lossless regardless).

    Returns:
        Base64-encoded PNG string.
    """
    from PIL import Image

    img = pil_image.convert("RGB")
    w, h = img.size
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        logger.debug(f"[VLMPipeline] Resized {w}x{h} → {img.width}x{img.height}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_pdf_images(
    pdf_path: Path,
    ann_metadata: Optional[Dict[str, Any]] = None,
    cache=None,                        # ImageResultCache | None
    prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Run the complete VLM image pipeline for one PDF.

    Args:
        pdf_path:     Path to the PDF file.
        ann_metadata: Optional Oracle metadata dict (used for log context only).
        cache:        Shared ImageResultCache instance. If None, the global
                      singleton from image_deduplicator is used.
        prompt:       VLM prompt override; uses default financial-analysis prompt
                      if not supplied.

    Returns:
        List of result dicts, one per qualifying image:
          {
            "page_number":   int,
            "image_index":   int,
            "image_hash":    str,     (MD5 hex)
            "description":   str,     (VLM output or empty string on error)
            "vlm_latency_ms": float,  (0.0 if from cache)
            "from_cache":    bool,
            "skipped_intra": bool,    (True = duplicate within the same PDF)
          }
    """
    from vlm.image_extractor     import extract_images_from_pdf
    from vlm.image_deduplicator  import compute_image_hash, get_global_vlm_cache
    from agents.llm_client       import call_vlm_sync
    from config                  import settings

    if not getattr(settings, "vlm_enabled", True):
        logger.info(f"[VLMPipeline] VLM disabled in settings — skipping '{pdf_path.name}'")
        return []

    if cache is None:
        cache = get_global_vlm_cache()

    vlm_prompt  = prompt or _VLM_PROMPT
    min_size    = int(getattr(settings, "vlm_image_min_size",     64))
    max_dim     = int(getattr(settings, "vlm_image_max_dim",    1024))
    batch_size  = max(1, min(3, int(getattr(settings, "vlm_images_per_request", 1))))
    model       = getattr(settings, "ollama_model_vision", "minicpm-v:latest")

    ann_seq     = (ann_metadata or {}).get("ann_seq", "?")
    doc_label   = f"{pdf_path.name} (ann_seq={ann_seq})"

    # ── Step 1: Extract images ─────────────────────────────────────────────
    t_extract = time.perf_counter()
    images    = extract_images_from_pdf(pdf_path, min_size=min_size)
    extract_ms = (time.perf_counter() - t_extract) * 1000

    total_images = len(images)
    if total_images == 0:
        logger.info(f"[VLMPipeline] No qualifying images in '{doc_label}'")
        return []

    logger.info(
        f"[VLMPipeline] '{doc_label}': {total_images} images extracted "
        f"in {extract_ms:.0f}ms — starting dedup + VLM"
    )

    # ── Step 2: Dedup + VLM ────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    seen_hashes: set = set()           # per-PDF intra-document dedup
    total_vlm_ms = 0.0
    cache_hits   = 0
    intra_skips  = 0
    vlm_calls    = 0
    failures     = 0

    # Process images in mini-batches of batch_size
    for batch_start in range(0, total_images, batch_size):
        batch = images[batch_start : batch_start + batch_size]

        # Hash each image in the batch first
        batch_with_hash = []
        for img_info in batch:
            h = compute_image_hash(img_info["pil_image"])
            batch_with_hash.append({**img_info, "image_hash": h})

        # Build the list of images we actually need to call VLM for
        to_call: List[Dict] = []
        skip_results: List[Dict] = []

        for img_info in batch_with_hash:
            h = img_info["image_hash"]
            base_result = {
                "page_number":    img_info["page_number"],
                "image_index":    img_info["image_index"],
                "image_hash":     h,
                "description":    "",
                "vlm_latency_ms": 0.0,
                "from_cache":     False,
                "skipped_intra":  False,
            }

            # 2a. Per-PDF intra-document dedup
            if h in seen_hashes:
                intra_skips += 1
                base_result["skipped_intra"] = True
                skip_results.append(base_result)
                logger.debug(
                    f"[VLMPipeline] Intra-doc SKIP hash={h[:8]} "
                    f"page={img_info['page_number']} idx={img_info['image_index']}"
                )
                continue

            seen_hashes.add(h)

            # 2b. Global cache hit
            cached = cache.get(h)
            if cached is not None:
                cache_hits += 1
                base_result["description"] = cached
                base_result["from_cache"]  = True
                skip_results.append(base_result)
                logger.debug(
                    f"[VLMPipeline] Cache HIT hash={h[:8]} "
                    f"page={img_info['page_number']} idx={img_info['image_index']}"
                )
                continue

            # Need VLM call
            to_call.append({**img_info, "_base_result": base_result})

        results.extend(skip_results)

        if not to_call:
            continue

        # 2c. Call VLM (batched up to batch_size images per request)
        b64_images = []
        for item in to_call:
            try:
                b64_images.append(_pil_to_base64(item["pil_image"], max_dim=max_dim))
            except Exception as enc_exc:
                logger.warning(
                    f"[VLMPipeline] Base64 encode failed "
                    f"page={item['page_number']} idx={item['image_index']}: {enc_exc}"
                )
                b64_images.append(None)

        # Send only non-None images but track positions
        valid_b64   = [b for b in b64_images if b is not None]
        t_vlm_start = time.perf_counter()

        if valid_b64:
            try:
                description = call_vlm_sync(
                    images  = valid_b64,
                    prompt  = vlm_prompt,
                    model   = model,
                )
                vlm_calls  += 1
            except Exception as vlm_exc:
                description = ""
                failures   += 1
                logger.error(
                    f"[VLMPipeline] VLM call failed for batch "
                    f"(pages={[i['page_number'] for i in to_call]}): {vlm_exc}"
                )
        else:
            description = ""

        call_ms      = (time.perf_counter() - t_vlm_start) * 1000
        total_vlm_ms += call_ms

        # Distribute result back to each image in the batch
        # (all images in a batched call share the same description for now)
        for idx, item in enumerate(to_call):
            img_result = item["_base_result"]
            img_result["description"]    = description
            img_result["vlm_latency_ms"] = call_ms / max(len(to_call), 1)
            img_result["from_cache"]     = False

            # Store in global cache
            if description:
                cache.put(item["image_hash"], description)

            logger.info(
                f"[VLMPipeline] VLM OK | page={item['page_number']} "
                f"idx={item['image_index']} hash={item['image_hash'][:8]} "
                f"latency={img_result['vlm_latency_ms']:.0f}ms "
                f"chars={len(description)}"
            )
            results.append(img_result)

    # ── Step 3: Metrics summary ────────────────────────────────────────────
    unique_images = total_images - intra_skips
    dedup_ratio   = (intra_skips / total_images * 100) if total_images else 0.0
    cache_stats   = cache.stats()

    logger.info(
        f"[VLMPipeline] DONE '{doc_label}' | "
        f"total={total_images} unique={unique_images} "
        f"intra_skip={intra_skips} ({dedup_ratio:.0f}%) "
        f"cache_hits={cache_hits} vlm_calls={vlm_calls} failures={failures} "
        f"total_vlm_ms={total_vlm_ms:.0f} "
        f"cache={cache_stats}"
    )

    return results


def format_image_descriptions_as_text(
    image_results: List[Dict[str, Any]],
) -> str:
    """
    Convert a list of image result dicts into a text string suitable
    for appending to a chunk's text content.

    Only includes results with non-empty descriptions that were not
    per-PDF intra-doc duplicates.

    Returns:
        Text block like:
          [IMAGE page=2 idx=1] <description>
          [IMAGE page=3 idx=2] <description>
    """
    lines = []
    for r in image_results:
        if r.get("skipped_intra"):
            continue
        desc = r.get("description", "").strip()
        if not desc:
            continue
        lines.append(
            f"[IMAGE page={r['page_number']} idx={r['image_index']}] {desc}"
        )
    return "\n".join(lines)
