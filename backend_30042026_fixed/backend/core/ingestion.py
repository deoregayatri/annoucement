"""
Document Ingestion Pipeline - Stage 3 upgrade.

Changes from Stage 3:
  - Routes chunks to `announcements_docs` collection via CollectionManager
  - Uses token-based chunking (chunk_text_tokens, 500t / 50t overlap)
  - Adds 'active': True and 'document_id' to all chunk metadata
  - Extracts and structures text via Marker
  - Chunks via SemanticChunker
  - Embeds and indexes to Qdrant

Backward compat:
  - process_document() return signature unchanged: {"chunks", "metadata", "doc_metadata"}
  - The caller in api/routes.py is NOT broken - it still calls
    vector_store.add_documents(); we now call doc_store internally
"""

import hashlib
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from config import settings
from core.metadata import ChunkMetadata, ExtractionResult
from utils.document_processor import extract_pages_from_file, extract_text_from_file
from utils.text_chunker import chunk_text_tokens, create_chunks_from_pages_continuous, create_chunks_with_metadata

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Unified pipeline for processing document uploads (PDF / DOCX / TXT).

    Stage 3 additions:
      - Token-based chunking (500t / 50t)
      - Routes to announcements_docs via CollectionManager
      - active=True flag on all chunks
      - SHA-256 dedup: skip if already indexed
      - Qdrant upload
    """

    def __init__(self) -> None:
        pass

    async def process_document(
        self,
        file_path: Union[str, Path],
        file_content: Optional[bytes] = None,
        original_filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single document end-to-end.

        Args:
            file_path: Path to the file on disk.
            file_content: Raw bytes (optional).
            original_filename: Original filename if file_path is a temp path.

        Returns:
            Dict with keys:
              'chunks'       - list of chunk texts
              'metadata'     - list of metadata dicts
              'doc_metadata' - top-level extraction metadata
              'chunk_ids'    - list of Qdrant point UUIDs (new in Stage 3)
              'skipped'      - True if dedup skipped re-ingestion
        """
        file_path = Path(file_path)
        filename  = original_filename or file_path.name
        file_ext  = file_path.suffix.lower()

        logger.info(f"[IngestionPipeline] Starting ingestion: '{filename}'")

        #  1. Compute SHA-256 for dedup 
        file_hash = self._hash_file(file_path)

        #  2. Dedup check against doc_store 
        doc_store = self._get_doc_store()
        if doc_store.document_hash_exists(file_hash):
            logger.info(f"[IngestionPipeline] SKIP '{filename}' - already indexed (hash match).")
            return {
                "chunks": [], "metadata": [], "doc_metadata": {},
                "chunk_ids": [], "skipped": True,
            }

        try:
            #  3. Extract text 
            if file_ext == ".pdf":
                pages     = extract_pages_from_file(file_path)
                full_text = "\n\n".join(p["text"] for p in pages)
            else:
                full_text, _ = extract_text_from_file(file_path)
                pages = []

            if not full_text.strip():
                logger.warning(f"[IngestionPipeline] No text extracted from '{filename}'.")
                return {"chunks": [], "metadata": [], "doc_metadata": {}, "chunk_ids": [], "skipped": False}

            #  4. Metadata extraction 
            meta_result = metadata or {}

            # If standard ones are missing, try to extract minimal info from filename as fallback
            if "company_name" not in meta_result:
                meta_result["company_name"] = "Unknown"
            if "company_symbol" not in meta_result:
                meta_result["company_symbol"] = "Unknown"
            if "announcement_date" not in meta_result:
                meta_result["announcement_date"] = datetime.now().isoformat()

            logger.info(f"[IngestionPipeline] Using provided metadata: {meta_result}")

            #  5. Token-based chunking 
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            document_id  = file_hash  # Use hash as stable document_id

            if file_ext == ".pdf" and pages:
                # Token-based chunking across the FULL document (continuous text)
                raw_chunks = create_chunks_from_pages_continuous(
                    pages=pages,
                    filename=filename,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    file_type=file_ext.replace(".", ""),
                    file_size_mb=file_size_mb,
                )
            else:
                raw_chunks = create_chunks_with_metadata(
                    text=full_text,
                    filename=filename,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    file_type=file_ext.replace(".", ""),
                    file_size_mb=file_size_mb,
                )

            #  6. Validate metadata + build final lists 
            final_texts: List[str] = []
            final_metas: List[Dict[str, Any]] = []

            c_name = meta_result.get("company_name", "Unknown")
            c_sym  = meta_result.get("company_symbol", "Unknown")
            c_date = meta_result.get("announcement_date")

            if c_name in ("Unknown", None) or c_sym in ("Unknown", None):
                raise ValueError(
                    f"[IngestionPipeline] Critical metadata missing: company='{c_name}', symbol='{c_sym}'"
                )
            if not c_date:
                raise ValueError("[IngestionPipeline] Critical metadata missing: announcement_date")

            for chunk in raw_chunks:
                chunk_meta = {**chunk["metadata"]}
                chunk_meta.update({
                    "company_name"     : c_name,
                    "company_symbol"   : c_sym,
                    "company_id"       : c_sym,
                    "subject"          : meta_result.get("subject", "General Announcement"),
                    "announcement_date": self._date_to_ts(c_date),
                    "document_id"      : document_id,
                    "source_type"      : "document",
                    "active"           : True,
                    "file_hash"        : file_hash,
                })
                chunk_meta.setdefault("page_number",  1)
                chunk_meta.setdefault("total_pages",  len(pages) if pages else 1)

                final_texts.append(chunk["text"])
                final_metas.append(chunk_meta)

            #  7. Embed 
            from core.embeddings import get_embedding_model
            embedder   = get_embedding_model()
            embeddings = embedder.embed(final_texts)

            #  8. Store in announcements_docs 
            chunk_ids = doc_store.add_documents(
                texts=final_texts,
                embeddings=embeddings,
                metadata=final_metas,
                file_hash=file_hash,
                source_type="document",
            )

            # Skip sparse invalidation since local BM25 is removedss  # Non-critical

            logger.info(
                f"[IngestionPipeline] Indexed {len(chunk_ids)} chunks for '{filename}' "
                f" announcements_docs"
            )



            return {
                "chunks"      : final_texts,
                "metadata"    : final_metas,
                "doc_metadata": meta_result,
                "chunk_ids"   : chunk_ids,
                "skipped"     : False,
            }

        except Exception as exc:
            logger.error(f"[IngestionPipeline] Ingestion failed for '{filename}': {exc}")
            logger.debug(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _get_doc_store():
        """Lazy import to avoid circular imports."""
        from core.collection_manager import get_doc_store
        return get_doc_store()

    @staticmethod
    def _date_to_ts(date_val: Any) -> Optional[int]:
        """Convert various date formats to Unix int timestamp."""
        if date_val is None:
            return None
        if isinstance(date_val, (int, float)):
            return int(date_val)
        if isinstance(date_val, str):
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y"):
                try:
                    return int(datetime.strptime(date_val.strip(), fmt).timestamp())
                except ValueError:
                    continue
            try:
                return int(datetime.fromisoformat(date_val.replace("Z", "+00:00")).timestamp())
            except (ValueError, AttributeError):
                pass
        return None


 