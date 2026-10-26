"""
Document processing utilities for extracting text from various file formats.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

from config import settings

logger = logging.getLogger(__name__)


def extract_pages_from_pdf(file_path: Path) -> list:
    """
    Extract text AND visual content from PDF file by page using PyMuPDF + Gemini Vision.

    Returns:
        List of dicts: {'page_number': int, 'text': str, 'source': str}
    """
    pages_data = []

    try:
        import fitz  # PyMuPDF
        from core.vision_llm import get_vision_llm

        logger.info(f" Starting multimodal extraction for: {file_path.name}")
        doc = fitz.open(str(file_path))
        num_pages = len(doc)

        # Initialize Vision LLM if enabled
        vision_service = None
        if settings.use_vision_llm:
            vision_service = get_vision_llm()
            if vision_service and vision_service.initialized:
                logger.info(f"[OK] Vision LLM ENABLED for image extraction")
            else:
                vision_service = None

        if vision_service:
            logger.info(f" Starting multimodal extraction for: {file_path.name}")
            logger.info("[OK] Vision LLM ENABLED for image extraction")

        # PHASE 1: Extract all text and collect all images from all pages
        page_texts = {}
        images_by_page = {}  # {page_num: [Image, ...]}

        for page_num in range(1, num_pages + 1):
            logger.info(f" Processing page {page_num}/{num_pages}...")
            page = doc[page_num - 1]

            # 1. Extract text using PyMuPDF Layout (if available) or fallback to basic text
            try:
                import pymupdf
                import pymupdf.layout

                # Use PyMuPDF layout analysis for better reading order and tabular data
                # Typically, this is done by analyzing the page. Let's try markdown output
                # or block merging depending on version. The user requested `import pymupdf.layout`.
                try:
                    text = page.get_text("markdown")
                except Exception:
                    # In case "markdown" is not directly supported in this PyMuPDF version,
                    # Fallback to block sorted text
                    text = page.get_text("text", sort=True)
            except ImportError:
                text = page.get_text("text", sort=True)

            # OCR FALLBACK: If text is empty or very sparse, use OCR
            # This handles scanned PDFs
            if len(text.strip()) < 50 and settings.use_ocr:
                try:
                    logger.info(
                        f"[WARNING] Page {page_num} has sparse text ({len(text.strip())} chars). Attempting OCR..."
                    )
                    from core.ocr_service import get_ocr_service

                    ocr = get_ocr_service()

                    # Render page as image for OCR
                    pix = page.get_pixmap(dpi=300)
                    img_data = pix.tobytes("png")

                    import io
                    from PIL import Image

                    image = Image.open(io.BytesIO(img_data))

                    ocr_text = ocr.extract_text_from_pil_image(
                        image, image_name=f"page_{page_num}"
                    )
                    if len(ocr_text) > len(text.strip()):
                        logger.info(
                            f"[OK] OCR extracted {len(ocr_text)} chars (vs {len(text.strip())} raw)"
                        )
                        text = ocr_text  # Replace sparse text with OCR text
                    else:
                        logger.info(f"OCR yielded less/same text. Keeping raw.")
                except Exception as e:
                    logger.warning(f"OCR fallback failed for page {page_num}: {e}")

            if text.strip():
                if page_num not in page_texts:
                    page_texts[page_num] = []
                page_texts[page_num].append(text)

            # 2. Extract embedded images for VLM (minicpm-v)
            # Strategy: Only use VLM for specifically embedded visuals (charts, graphs),
            # saving enormous compute rather than passing full scanned pages to LLM.

            if vision_service:
                images = vision_service.extract_images_from_pdf_page(
                    file_path,
                    page_num,
                    min_size=getattr(settings, "vision_min_image_size", 100),
                )

                if images:
                    num_images = min(len(images), settings.vision_max_images_per_page)
                    images_by_page[page_num] = images[:num_images]
                    logger.info(
                        f"  Found {len(images)} images (passing top {num_images} to VLM)"
                    )

        doc.close()

        # PHASE 2: Smart batch vision analysis across ALL pages
        if vision_service and images_by_page:
            from utils.smart_vision_batcher import SmartVisionBatcher

            total_images = sum(len(imgs) for imgs in images_by_page.values())
            logger.info(
                f" Smart batching: {total_images} images from {len(images_by_page)} pages"
            )

            batcher = SmartVisionBatcher(vision_service)
            descriptions_by_page = batcher.process_all_images(
                images_by_page, context=file_path.stem
            )

            # Add vision descriptions to page texts
            for page_num, descriptions in descriptions_by_page.items():
                for idx, description in enumerate(descriptions, 1):
                    if description and description.strip():
                        if page_num not in page_texts:
                            page_texts[page_num] = []
                        page_texts[page_num].append(
                            f"\n\n[IMAGE {idx} ON PAGE {page_num} - VISUAL CONTENT EXTRACTED BY GEMINI VISION]\n{description}\n"
                        )

        # Assemble results
        for page_num in range(1, num_pages + 1):
            if page_num in page_texts:
                combined_page_text = "\n\n".join(page_texts[page_num])
                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": combined_page_text,
                        "source": "multimodal" if vision_service else "text_only",
                    }
                )

        logger.info(
            f"[OK] Extracted {len(pages_data)} pages with {'multimodal' if vision_service else 'text-only'} content from {file_path.name}"
        )
        return pages_data

    except Exception as e:
        logger.error(f"Failed to extract pages from {file_path.name}: {e}")
        # Return empty list or re-raise?
        # For backward compatibility, maybe return empty and let caller handle.
        return []


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF file with smart routing.
    Backward compatible wrapper around extract_pages_from_pdf.
    """
    try:
        pages = extract_pages_from_pdf(file_path)

        if not pages:
            # Fallback to original holistic OCR logic if pages extraction found nothing useful?
            # Or just rely on the fact that extract_pages currently covers PyPDF + VLM.
            # If standard extraction failed, we might want to try the OCR route which expects file path.

            # Re-implementing specific "is_scanned" check here to trigger OCR if page extraction yielded nothing significant.
            if settings.use_ocr:
                from core.ocr_service import get_ocr_service

                ocr = get_ocr_service()
                # Check if file seems scanned (size vs text content?)
                # or just try OCR if text is empty.
                # Assuming OCR processor handles it.
                logger.info("Page extraction yielded no text. Attempting full OCR...")
                return ocr.extract_text_from_pdf(file_path)

            return ""

        # Check total text length to decide on OCR fallback (Scanning detection)
        full_text = "\n\n".join([p["text"] for p in pages])

        if len(full_text) < 100 and settings.use_ocr:
            from core.ocr_service import get_ocr_service

            ocr = get_ocr_service()

            # Instead of is_scanned_pdf logic, we can just OCR the document if it lacks text
            logger.info("PDF appears scanned. Using OCR for full text.")
            return ocr.extract_text_from_pdf_images(file_path, max_images=50)

        return full_text

    except Exception as e:
        logger.error(f"Failed to extract text from PDF {file_path.name}: {e}")
        raise ValueError(f"Could not process PDF file: {str(e)}")


def extract_text_from_docx(file_path: Path) -> str:
    """
    Extract text from DOCX file.

    Args:
        file_path: Path to DOCX file

    Returns:
        Extracted text content
    """
    try:
        from docx import Document

        doc = Document(str(file_path))
        text_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        full_text = "\n\n".join(text_parts)
        logger.info(
            f"Extracted {len(full_text)} characters from DOCX ({len(doc.paragraphs)} paragraphs)"
        )

        return full_text

    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise ValueError(f"Could not process DOCX file: {str(e)}")


def extract_text_from_txt(file_path: Path) -> str:
    """
    Extract text from TXT file.

    Args:
        file_path: Path to TXT file

    Returns:
        File content
    """
    try:
        # Try UTF-8 first, fallback to other encodings
        encodings = ["utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                logger.info(
                    f"Extracted {len(text)} characters from TXT (encoding: {encoding})"
                )
                return text
            except UnicodeDecodeError:
                continue

        raise ValueError("Could not decode text file with any supported encoding")

    except Exception as e:
        logger.error(f"Failed to extract text from TXT: {e}")
        raise ValueError(f"Could not process TXT file: {str(e)}")


def extract_text_from_image(file_path: Path) -> str:
    """
    Extract text from image file using VLM and/or OCR.
    Tries VLM first for better results, falls back to OCR.

    Args:
        file_path: Path to image file

    Returns:
        Extracted text content
    """
    try:
        text = ""

        # Try Vision LLM first if enabled
        if settings.use_vision_llm:
            try:
                from core.vision_llm import get_vision_llm

                vlm = get_vision_llm()

                if vlm.initialized:
                    text = vlm.extract_text_from_image(file_path)
                    logger.info(f"VLM extracted {len(text)} characters from image")

                    # If VLM got good text, return it
                    if len(text) > 50:
                        return text

            except Exception as e:
                logger.warning(f"VLM extraction failed: {e}")

        # Try OCR if VLM didn't get much text or failed
        if settings.use_ocr and len(text) < 50:
            try:
                from core.ocr_service import get_ocr_service

                ocr = get_ocr_service()
                ocr_text = ocr.extract_text_from_image(file_path)
                logger.info(f"OCR extracted {len(ocr_text)} characters from image")

                # Use whichever got more text
                if len(ocr_text) > len(text):
                    text = ocr_text

            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")

        if not text.strip():
            raise ValueError("No text could be extracted from image")

        return text

    except Exception as e:
        logger.error(f"Failed to extract text from image: {e}")
        raise ValueError(f"Could not process image file: {str(e)}")


def extract_pages_from_docx(file_path: Path) -> list:
    try:
        text = extract_text_from_docx(file_path)
        return [{"page_number": 1, "text": text, "source": "docx"}]
    except Exception:
        return []


def extract_pages_from_txt(file_path: Path) -> list:
    try:
        text = extract_text_from_txt(file_path)
        return [{"page_number": 1, "text": text, "source": "txt"}]
    except Exception:
        return []


def extract_pages_from_file(file_path: Path) -> list:
    """
    Extract content from file structured by pages.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pages_from_pdf(file_path)
    elif suffix == ".docx":
        return extract_pages_from_docx(file_path)
    elif suffix == ".txt":
        return extract_pages_from_txt(file_path)
    else:
        # Fallback to general text extraction
        try:
            text, _ = extract_text_from_file(file_path)
            return [{"page_number": 1, "text": text, "source": "generic"}]
        except Exception:
            return []


def extract_text_from_file(file_path: Path) -> Tuple[str, str]:
    """
    Extract text from file based on extension.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (extracted_text, file_type)

    Raises:
        ValueError: If file type is not supported
    """
    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    extractors = {
        ".pdf": ("pdf", extract_text_from_pdf),
        ".docx": ("docx", extract_text_from_docx),
        ".txt": ("txt", extract_text_from_txt),
        ".png": ("image", extract_text_from_image),
        ".jpg": ("image", extract_text_from_image),
        ".jpeg": ("image", extract_text_from_image),
    }

    if suffix not in extractors:
        raise ValueError(
            f"Unsupported file type: {suffix}. Supported: {', '.join(extractors.keys())}"
        )

    file_type, extractor = extractors[suffix]

    try:
        text = extractor(file_path)

        if not text.strip():
            raise ValueError("No text content found in file")

        return text, file_type

    except Exception as e:
        logger.error(f"Failed to extract text from {file_path}: {e}")
        raise
