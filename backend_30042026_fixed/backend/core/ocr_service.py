"""
OCR Service using PaddleOCR for text extraction from images and scanned documents.
Replaces the Ollama-based vision_llm for OCR functionality.
"""

import logging
import io
from pathlib import Path
from typing import List, Optional, Dict
from PIL import Image

logger = logging.getLogger(__name__)


class OCRService:
    """
    PaddleOCR-based text extraction service.
    Handles text extraction from images and PDFs without requiring Ollama.
    """
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Initialize PaddleOCR service.
        
        Args:
            lang: Language code (default: 'en' for English)
            use_gpu: Whether to use GPU acceleration (default: False)
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.initialized = False
        self.ocr = None
        
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang=lang,
                use_gpu=use_gpu,
                show_log=False  # Reduce console output
            )
            self.initialized = True
            logger.info(f" PaddleOCR initialized successfully (lang={lang}, gpu={use_gpu})")
            
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
            self.initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            self.initialized = False
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """
        Extract text from image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text content
        """
        if not self.initialized:
            logger.warning("PaddleOCR not initialized, cannot extract text")
            return ""
        
        try:
            # Run OCR
            result = self.ocr.ocr(str(image_path), cls=True)
            
            # Extract text from results
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # text is in format: [bbox, (text, confidence)]
                        text_lines.append(text)
            
            extracted_text = '\n'.join(text_lines)
            logger.info(f"PaddleOCR extracted {len(extracted_text)} characters from {image_path.name}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed for {image_path}: {e}")
            return ""
    
    def extract_text_from_pil_image(self, image: Image.Image, image_name: str = "image") -> str:
        """
        Extract text from PIL Image.
        
        Args:
            image: PIL Image object
            image_name: Name for logging
            
        Returns:
            Extracted text content
        """
        if not self.initialized:
            logger.warning("PaddleOCR not initialized, cannot extract text")
            return ""
        
        try:
            # Convert PIL Image to numpy array for PaddleOCR
            import numpy as np
            image_array = np.array(image)
            
            # Run OCR
            result = self.ocr.ocr(image_array, cls=True)
            
            # Extract text from results
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        text_lines.append(text)
            
            extracted_text = '\n'.join(text_lines)
            logger.info(f"PaddleOCR extracted {len(extracted_text)} characters from {image_name}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed for {image_name}: {e}")
            return ""
    
    def has_embedded_images(self, pdf_path: Path) -> bool:
        """
        Check if PDF contains embedded images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if PDF contains images, False otherwise
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                if image_list:
                    doc.close()
                    logger.info(f"PDF contains {len(image_list)} images on page {page_num + 1}")
                    return True
            
            doc.close()
            return False
            
        except ImportError:
            logger.warning("PyMuPDF not installed, cannot detect images in PDF")
            return False
        except Exception as e:
            logger.error(f"Failed to check for images in PDF: {e}")
            return False
    
    def extract_images_from_pdf(self, pdf_path: Path, max_images: int = 10) -> List[Image.Image]:
        """
        Extract embedded images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            max_images: Maximum number of images to extract
            
        Returns:
            List of PIL Image objects
        """
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(str(pdf_path))
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    if len(images) >= max_images:
                        logger.info(f"Reached max_images limit ({max_images})")
                        break
                    
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Convert to PIL Image
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        
                        # Fix Bug 8: Normalize color mode so PaddleOCR/PIL doesn't fail on CMYK
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")
                            
                        images.append(pil_image)
                        
                        logger.debug(f"Extracted image {len(images)} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
                
                if len(images) >= max_images:
                    break
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from PDF")
            
            return images
            
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return []
        except Exception as e:
            logger.error(f"Failed to extract images from PDF: {e}")
            return []
    
    def extract_text_from_pdf_images(self, pdf_path: Path, max_images: int = 10) -> str:
        """
        Extract text from all images in a PDF.
        
        Args:
            pdf_path: Path to PDF file
            max_images: Maximum number of images to process
            
        Returns:
            Combined extracted text from all images
        """
        if not self.initialized:
            logger.warning("PaddleOCR not initialized, cannot extract text from PDF images")
            return ""
        
        try:
            images = self.extract_images_from_pdf(pdf_path, max_images)
            
            if not images:
                logger.info(f"No images found in PDF: {pdf_path.name}")
                return ""
            
            all_text = []
            for i, image in enumerate(images, 1):
                logger.debug(f"Processing image {i}/{len(images)} from PDF...")
                text = self.extract_text_from_pil_image(image, f"{pdf_path.name}_image_{i}")
                if text:
                    all_text.append(f"--- Image {i} ---\n{text}")
            
            combined_text = '\n\n'.join(all_text)
            logger.info(f"Extracted {len(combined_text)} characters from {len(images)} images in PDF")
            return combined_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from PDF images: {e}")
            return ""


# Global OCR service instance (lazy loading)
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get or create global OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
        logger.info("Created new OCR service instance")
    return _ocr_service


def invalidate_ocr_service():
    """Invalidate cached OCR service to force reload."""
    global _ocr_service
    _ocr_service = None
    logger.info("OCR service cache invalidated")
