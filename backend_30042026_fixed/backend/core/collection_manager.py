"""
Collection Manager - dual-collection singleton registry.

Manages two Qdrant collections:
  - announcements_json  : chunks from Excel/URL/JSON ingestion pipeline
  - announcements_docs  : chunks from PDF/DOCX/TXT document uploads

Usage:
    from core.collection_manager import get_json_store, get_doc_store
    json_store = get_json_store()
    doc_store  = get_doc_store()
"""

import logging
from typing import Optional, Dict, Any

from config import settings
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection name constants - driven by .env via settings
# ---------------------------------------------------------------------------
JSON_COLLECTION_NAME = settings.qdrant_json_collection  # default: announcements_json


class CollectionManager:
    """
    Singleton manager for both Qdrant collections.

    Instantiated once at application startup via get_collection_manager().
    Provides unified access to json_store and doc_store.
    """

    def __init__(self) -> None:
        logger.info("[CollectionManager] Initializing dual-collection stores ...")

        self.json_store = VectorStore(
            collection_name=JSON_COLLECTION_NAME, enable_quantization=True
        )

        logger.info("[CollectionManager] Required collections ready.")
        self._print_summary()

    def _print_summary(self) -> None:
        """Print collection configuration summary to logs at startup."""
        json_info = self.json_store.get_collection_info()

        logger.info(
            "\n"
            "\n"
            "          QDRANT DUAL-COLLECTION CONFIGURATION SUMMARY        \n"
            "\n"
            f"  Collection                {JSON_COLLECTION_NAME:<33}\n"
            f"  Total vectors             {json_info['total_vectors']:<33}\n"
            f"  Active vectors            {json_info['active_vectors']:<33}\n"
            f"  Dimension                 {json_info['dimension']:<33}\n"
            f"  Quantization              {json_info['quantization']:<33}\n"
            f"  HNSW m / ef_construct     {json_info['hnsw_m']} / {json_info['hnsw_ef_construct']:<28}\n"
            ""
        )

    def get_summary(self) -> Dict[str, Any]:
        """Return combined status dict (used by /health endpoint)."""
        return {"json_collection": self.json_store.get_collection_info()}


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_collection_manager: Optional[CollectionManager] = None
_json_store: Optional[VectorStore] = None
_doc_store: Optional[VectorStore] = None


def get_collection_manager() -> CollectionManager:
    """Get or create the shared CollectionManager instance."""
    global _collection_manager
    if _collection_manager is None:
        _collection_manager = CollectionManager()
    return _collection_manager


def get_json_store() -> VectorStore:
    """
    Get the announcements_json VectorStore singleton.
    For Excel/URL/JSON ingestion pipeline.
    """
    global _json_store
    if _json_store is None:
        _json_store = get_collection_manager().json_store
    return _json_store


def get_doc_store() -> VectorStore:
    """
    Get the announcements_docs VectorStore singleton.
    For PDF/DOCX/TXT document uploads.
    """
    global _doc_store
    if _doc_store is None:
        _doc_store = VectorStore(
            collection_name="announcements_docs", enable_quantization=True
        )
        logger.info("[CollectionManager] Doc store initialized.")
    return _doc_store
