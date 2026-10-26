"""
Smart multi-page vision batching for efficient context window usage.
Optimized for local Ollama (CPU-only) - no rate limiting, reduced image sizes,
per-image progress logging, and timeout protection.
"""

import base64
import io
import logging
import time
from typing import List, Dict, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# Max image dimension for CPU processing - smaller = much faster inference
CPU_MAX_IMAGE_DIM = 1024


def resize_for_cpu(image: Image.Image, max_dim: int = CPU_MAX_IMAGE_DIM) -> Image.Image:
    """
    Resize image to fit within max_dim while preserving aspect ratio.
    This dramatically speeds up CPU inference (e.g. 2048x2048  1024x1024 = ~4x faster).
    """
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image

    if w > h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)

    resized = image.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h} for CPU processing")
    return resized


class SmartVisionBatcher:
    """
    Intelligently batch images from multiple pages to maximize context window usage.
    Optimized for local Ollama CPU inference - no rate limiting.
    """

    def __init__(self, vision_service, max_tokens_per_batch=14000):
        self.vision_service = vision_service
        self.max_tokens_per_batch = max_tokens_per_batch
        self.tokens_per_image_avg = 1000

    def estimate_image_tokens(self, image: Image.Image) -> int:
        """Estimate tokens for an image based on size."""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_size = len(base64.b64encode(buffered.getvalue()))
        tokens = int((base64_size / 4) * 1.3)
        return min(tokens, 2000)

    def create_batches(
        self,
        images_with_metadata: List[Dict]
    ) -> List[List[Dict]]:
        """
        Group images into optimal batches based on token limits.
        """
        batches = []
        current_batch = []
        current_tokens = 500

        for img_meta in images_with_metadata:
            img_tokens = self.estimate_image_tokens(img_meta['image'])

            if current_tokens + img_tokens > self.max_tokens_per_batch and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 500

            current_batch.append(img_meta)
            current_tokens += img_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def process_batch(
        self,
        batch: List[Dict],
        context: str,
        batch_idx: int = 0,
        total_batches: int = 0
    ) -> List[Dict]:
        """
        Process a batch of images and return descriptions.
        Each image is processed individually with progress logging and timeout.
        No rate limiting for local Ollama.
        """
        if not batch:
            return []

        results = []
        for img_idx, item in enumerate(batch):
            page_num = item['page_num']
            image_idx = item['image_idx']

            try:
                # Resize for CPU efficiency
                resized_img = resize_for_cpu(item['image'])

                logger.info(
                    f"  Processing image {img_idx + 1}/{len(batch)} "
                    f"(page {page_num}) [batch {batch_idx}/{total_batches}]"
                )

                start_time = time.time()

                # Call VLM for single image
                text = self.vision_service.extract_text_from_pil_image(
                    resized_img,
                    image_name=f"page_{page_num}_img_{image_idx}"
                )

                elapsed = time.time() - start_time
                chars = len(text) if text else 0
                logger.info(
                    f"[OK] Page {page_num} image {image_idx}: "
                    f"extracted {chars} chars in {elapsed:.1f}s"
                )

                results.append({
                    'page_num': page_num,
                    'image_idx': image_idx,
                    'description': text
                })

            except Exception as e:
                logger.error(
                    f"[FAILED] Failed page {page_num} image {image_idx}: {e}"
                )
                results.append({
                    'page_num': page_num,
                    'image_idx': image_idx,
                    'description': ''
                })

        return results

    def process_all_images(
        self,
        images_by_page: Dict[int, List[Image.Image]],
        context: str
    ) -> Dict[int, List[str]]:
        """
        Process all images from multiple pages efficiently.
        No rate limiting - designed for local Ollama on CPU.
        """
        # Flatten all images with metadata
        all_images = []
        for page_num, images in images_by_page.items():
            for idx, image in enumerate(images, 1):
                all_images.append({
                    'image': image,
                    'page_num': page_num,
                    'image_idx': idx
                })

        if not all_images:
            return {}

        total_images = len(all_images)
        logger.info(f" Processing {total_images} images from {len(images_by_page)} pages (CPU mode, no rate limiting)")

        # Create optimal batches
        batches = self.create_batches(all_images)
        logger.info(f"[REPORT] Created {len(batches)} batches (avg {total_images/len(batches):.1f} images/batch)")

        # Process all batches sequentially (Ollama handles one request at a time)
        all_results = []
        overall_start = time.time()

        for batch_idx, batch in enumerate(batches, 1):
            logger.info(f" Starting batch {batch_idx}/{len(batches)} ({len(batch)} images)")
            results = self.process_batch(
                batch, context,
                batch_idx=batch_idx,
                total_batches=len(batches)
            )
            all_results.extend(results)

            # Progress summary
            done = len(all_results)
            elapsed = time.time() - overall_start
            rate = elapsed / done if done else 0
            remaining = (total_images - done) * rate
            logger.info(
                f" Progress: {done}/{total_images} images done "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

        # Reorganize by page
        descriptions_by_page = {}
        for result in all_results:
            page_num = result['page_num']
            if page_num not in descriptions_by_page:
                descriptions_by_page[page_num] = {}
            descriptions_by_page[page_num][result['image_idx']] = result['description']

        # Convert to lists
        output = {}
        for page_num, desc_dict in descriptions_by_page.items():
            output[page_num] = [desc_dict[i] for i in sorted(desc_dict.keys())]

        total_time = time.time() - overall_start
        logger.info(
            f"[OK] Completed: {len(all_results)} images in {total_time:.0f}s "
            f"({total_time/len(all_results):.1f}s/image avg)"
        )
        return output
