"""
vlm/image_deduplicator.py
=========================
Hash-based image deduplication and result caching for VLM calls.

Two layers of deduplication:
  1. Per-PDF  set   (seen_hashes)   — skip repeated images within 1 document.
  2. Global cache   (ImageResultCache) — reuse VLM results across all documents
                                         in the same process run (e.g. logos shared
                                         across many company PDFs).

Hash algorithm:
  normalize → RGB → resize(128×128) → MD5 of raw pixel bytes

Thread safety:
  ImageResultCache uses a threading.Lock for all cache mutations.
"""

import hashlib
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Normalisation constants (keep small → fast hashing, insensitive to JPEG artifacts)
_HASH_DIM = 128


def compute_image_hash(pil_image) -> str:
    """
    Compute a deterministic content hash for a PIL image.

    Steps:
      1. Convert to RGB (drops alpha channel, normalises colour space)
      2. Resize to 128×128 using LANCZOS (resolution-invariant)
      3. MD5 of raw pixel bytes (fast, sufficient for dedup — not cryptographic)

    Returns:
        32-character hex string (MD5).
    """
    from PIL import Image

    try:
        img = pil_image.convert("RGB").resize((_HASH_DIM, _HASH_DIM), Image.LANCZOS)
        pixel_bytes = img.tobytes()
        return hashlib.md5(pixel_bytes).hexdigest()
    except Exception as exc:
        logger.warning(f"[ImageDedup] Hash computation failed: {exc}")
        # Fallback: use id() so the image is treated as unique
        return f"fallback-{id(pil_image)}"


class ImageResultCache:
    """
    Thread-safe, in-memory cache mapping image_hash → VLM description string.

    Designed to be shared as a module-level singleton across all concurrent
    ingestion workers in the same process.  Workers that miss the cache call
    VLM and then store their result; subsequent workers with the same hash
    get an instant cache hit.

    Metrics tracked:
      hits   — number of times a cached result was returned
      misses — number of times the hash was absent (VLM called)
      size   — current number of entries
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._lock  = threading.Lock()
        self._hits   = 0
        self._misses = 0

    def get(self, image_hash: str) -> Optional[str]:
        """Return cached VLM description or None."""
        with self._lock:
            result = self._store.get(image_hash)
            if result is not None:
                self._hits += 1
                logger.debug(f"[ImageCache] HIT  hash={image_hash[:8]}... → {len(result)} chars")
            else:
                self._misses += 1
                logger.debug(f"[ImageCache] MISS hash={image_hash[:8]}...")
            return result

    def put(self, image_hash: str, description: str) -> None:
        """Store a VLM result keyed by image hash."""
        with self._lock:
            self._store[image_hash] = description

    def stats(self) -> dict:
        """Return a snapshot of cache performance metrics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total else 0.0
            return {
                "hits":      self._hits,
                "misses":    self._misses,
                "size":      len(self._store),
                "hit_rate":  f"{hit_rate:.1f}%",
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton shared across all workers in the same process
# ---------------------------------------------------------------------------
_global_vlm_cache: Optional[ImageResultCache] = None
_cache_create_lock = threading.Lock()


def get_global_vlm_cache() -> ImageResultCache:
    """Return (or lazily create) the shared VLM result cache singleton."""
    global _global_vlm_cache
    if _global_vlm_cache is None:
        with _cache_create_lock:
            if _global_vlm_cache is None:
                _global_vlm_cache = ImageResultCache()
                logger.info("[ImageCache] Global VLM result cache created.")
    return _global_vlm_cache
