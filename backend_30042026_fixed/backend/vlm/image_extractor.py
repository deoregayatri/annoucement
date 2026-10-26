"""
vlm/image_extractor.py
======================
Extract embedded images from a PDF file using PyMuPDF (fitz).

Responsibilities:
  - Open the PDF
  - Iterate pages; collect embedded image xrefs
  - Filter by minimum dimension (skip tiny logos/icons)
  - Return PIL Images with full positional metadata

This module performs NO VLM calls — pure extraction only.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def extract_images_from_pdf(
    pdf_path: Path,
    min_size: int = 64,
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
        min_size: Minimum width AND height in pixels. Images smaller than this
                  on either axis are skipped (avoids logos, rule-lines, etc.).

    Returns:
        List of dicts, one per qualifying image:
          {
              "page_number":   int,    # 1-indexed
              "image_index":   int,    # 1-indexed within the page
              "pil_image":     Image,  # PIL.Image in RGB mode
              "width":         int,
              "height":        int,
          }

    Returns [] on any error (caller must not crash on image extraction failure).
    """
    results: List[Dict[str, Any]] = []

    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import io

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        logger.info(
            f"[ImageExtractor] Scanning {num_pages} pages in '{pdf_path.name}'"
        )

        for page_num in range(1, num_pages + 1):
            page = doc[page_num - 1]
            image_list = page.get_images(full=True)

            img_idx = 0
            for xref_info in image_list:
                xref = xref_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    w = base_image.get("width", 0)
                    h = base_image.get("height", 0)

                    # Filter tiny images
                    if w < min_size or h < min_size:
                        logger.debug(
                            f"[ImageExtractor] Skipping small image "
                            f"(page={page_num} idx={img_idx + 1} size={w}x{h})"
                        )
                        continue

                    # Convert to PIL RGB
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    img_idx += 1

                    results.append(
                        {
                            "page_number": page_num,
                            "image_index": img_idx,
                            "pil_image":   pil_img,
                            "width":       pil_img.width,
                            "height":      pil_img.height,
                        }
                    )
                except Exception as img_exc:
                    logger.debug(
                        f"[ImageExtractor] Could not extract xref={xref} "
                        f"page={page_num}: {img_exc}"
                    )

        doc.close()
        logger.info(
            f"[ImageExtractor] '{pdf_path.name}': "
            f"extracted {len(results)} qualifying images (min_size={min_size}px)"
        )

    except ImportError:
        logger.warning(
            "[ImageExtractor] PyMuPDF (fitz) not installed — skipping image extraction."
        )
    except Exception as exc:
        logger.error(
            f"[ImageExtractor] Failed to extract images from '{pdf_path}': {exc}",
            exc_info=True,
        )

    return results
