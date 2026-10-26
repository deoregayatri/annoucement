"""
Vision Language Model processor for extracting text from images.
Uses Ollama API with minicpm-v model.
"""

import logging
import base64
import json
from pathlib import Path
from typing import List, Optional
from PIL import Image
import io
import requests

from config import settings

logger = logging.getLogger(__name__)


class VisionLLM:
    """
    Vision Language Model processor using Ollama.
    Handles text extraction from images using multimodal LLM.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize Vision LLM processor.
        
        Args:
            model: Model name (default: minicpm-v:latest)
            base_url: Ollama API base URL
        """
        # Model is now configurable via .env - OLLAMA_MODEL_VISION (default: minicpm-v:latest)
        self.model    = settings.ollama_model_vision
        self.base_url = settings.ollama_base_url
        self.provider = "ollama"
        self.initialized = False
        
        logger.info(f"Initializing Local Vision LLM (minicpm-v) with base_url={self.base_url}")
        
        try:
            self._test_connection()
            self.initialized = True
            logger.info(f"Local Vision LLM (minicpm-v) initialized successfully")
        except Exception as e:
            logger.error(f"Vision LLM initialization failed: {e}")
            self.initialized = False
    
    def _test_connection(self):
        """Test Ollama connection and model availability."""
        try:
            # Check if Ollama is running
            url = f"{self.base_url}/api/tags"
            logger.info(f"Testing Ollama connection: {url}")
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            logger.info(f"Available Ollama models: {model_names}")
            
            # Simple check for minicpm-v presence
            found = False
            for m in model_names:
                if "minicpm-v" in m:
                    self.model = m # Use exact name found
                    found = True
                    break
            
            if not found:
                 logger.warning(f"Model minicpm-v not found in Ollama list. Please run: ollama pull minicpm-v")
                 # We still set initialized=True to allow attempts, or maybe fail?
                 # Let's verify if we should fail. User said "Using minicpm-v local vision model".
                 # If not found, it won't work.
            
            logger.info(f"Will use VLM model: {self.model}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
        except Exception as e:
            # raise Exception(f"Ollama connection test failed: {e}")
            logger.error(f"Ollama connection test failed: {e}")
            # Don't crash app, just disable VLM
            self.initialized = False

    
    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 string."""
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise
    
    def _encode_pil_image(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        try:
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode PIL image: {e}")
            raise
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """Extract text from image using Vision LLM."""
        if not self.initialized:
            logger.warning("Vision LLM not initialized, cannot extract text")
            return ""
        
        try:
            image_b64 = self._encode_image(image_path)
            return self._generate_text(image_b64)
        except Exception as e:
            logger.error(f"Vision LLM extraction failed: {e}")
            return ""

    def extract_text_from_pil_image(self, image: Image.Image, image_name: str = "image") -> str:
        """Extract text from PIL Image using Vision LLM."""
        if not self.initialized:
            logger.warning("Vision LLM not initialized, cannot extract text")
            return ""
        
        try:
            image_b64 = self._encode_pil_image(image)
            return self._generate_text(image_b64)
        except Exception as e:
            logger.error(f"Vision LLM extraction failed: {e}")
            return ""

    def analyze_images_batch(self, images: List[Image.Image], context: str = "") -> List[str]:
        """
        Analyze a batch of images.
        For minicpm-v on Ollama, simply processing sequentially is often more reliable 
        than sending 10 images at once and asking for structured output.
        
        Args:
            images: List of PIL images
            context: Context string (e.g. "page 1")
            
        Returns:
            List of text descriptions/extractions
        """
        results = []
        for idx, img in enumerate(images):
            try:
                # We reuse extract_text_from_pil_image but maybe with a slightly different prompt if needed?
                # The generic extract method uses a "Extract all visible text" prompt.
                # If context is provided, we might want to log it but the extraction is structural.
                text = self.extract_text_from_pil_image(img, image_name=f"batch_img_{idx}")
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to analyze image {idx} in batch: {e}")
                results.append("")
        return results

    def _generate_text(self, image_b64: str) -> str:
        """Call Ollama /api/chat with retry on 404 (model-swap collision)."""
        import time

        prompt = """Extract all visible text from this image.
Rules:
1. Include ALL text you can see, maintaining the original order
2. Preserve formatting (headings, lists, tables) as much as possible
3. Only output the extracted text
Text:"""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64]
                }
            ],
            "stream": False
        }

        # Retry up to 4 times on 404 - happens when Ollama temporarily unloads
        # minicpm-v to serve a concurrent mistral/text request. Wait and retry
        # as Ollama will reload the model automatically.
        max_retries = 4
        retry_wait = 10  # seconds between retries

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"[VLM] Sending image to Ollama (attempt {attempt}/{max_retries})...")
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=3600
                )
                if response.status_code == 404 and attempt < max_retries:
                    logger.warning(
                        f"VLM got 404 (model likely unloaded), retrying in {retry_wait}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(retry_wait)
                    continue
                response.raise_for_status()
                result = response.json()
                text = result.get("message", {}).get("content", "").strip()
                logger.info(f"Vision LLM extracted {len(text)} characters")
                return text
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[VLM] Transient error (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {retry_wait}s..."
                    )
                    time.sleep(retry_wait)
                    continue
                logger.error(f"[VLM] Failed after {max_retries} retries: {e}")
                return ""

        logger.error("VLM failed after all retries")
        return ""



    # ... Keep the PDF extraction helpers as they were useful ...
    def has_embedded_images(self, pdf_path: Path) -> bool:
        """Check if PDF contains embedded images."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            for page in doc:
                if page.get_images():
                    doc.close()
                    return True
            doc.close()
            return False
        except:
            return False

    def render_page_as_image(self, pdf_path: Path, page_num: int) -> Optional[Image.Image]:
        """Render a specific PDF page as a PIL Image."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc): return None
            page = doc[page_num - 1]
            # 1.5x gives good quality for text extraction while being much faster on CPU
            # (vs 2x = ~44% fewer pixels to process)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            logger.error(f"Failed to render page {page_num}: {e}")
            return None

    def extract_images_from_pdf_page(self, pdf_path: Path, page_num: int, min_size: int = 100) -> List[Image.Image]:
        """Extract images from a specific PDF page."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if page_num < 1 or page_num > len(doc): return []
            page = doc[page_num - 1]
            images = []
            for img in page.get_images():
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    pil_image = Image.open(io.BytesIO(base_image["image"]))
                    # Normalize color mode so downstream PNG encoding never fails.
                    # PNG cannot encode CMYK (or other non-RGB modes like P/LA),
                    # which was causing "cannot write mode CMYK as PNG" and
                    # aborting extraction for the whole PDF.
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    if pil_image.width >= min_size and pil_image.height >= min_size:
                        images.append(pil_image)
                except:
                    continue
            return images
        except Exception as e:
            logger.error(f"Failed to extract images from page {page_num}: {e}")
            return []


# Global Vision LLM instance
_vision_llm: Optional[VisionLLM] = None

def get_vision_llm() -> VisionLLM:
    """Get or create global Vision LLM instance."""
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = VisionLLM()
        logger.info("Created new Vision LLM instance")
    return _vision_llm

def invalidate_vision_llm():
    """Invalidate cached Vision LLM to force reload."""
    global _vision_llm
    _vision_llm = None
    logger.info("Vision LLM cache invalidated")
