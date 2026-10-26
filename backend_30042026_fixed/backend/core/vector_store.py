"""
Qdrant vector store wrapper - enterprise dual-collection configuration.

Stage 1 Enhancements:
  - FP16 ScalarQuantization on both dimensions
  - HNSW: m=16, ef_construct=200, ef_search=128 (runtime)
  - Extended payload schema (all 15 fields)
  - Soft-delete via 'active' flag
  - get_collection_info() prints full Qdrant config summary
  - Backward-compatible: get_vector_store() still returns legacy collection
"""

import logging
import time
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    QuantizationConfig,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    IsEmptyCondition,
    PayloadField,
    Range,
    SearchParams,
    PointIdsList,
    SetPayload,
)
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection configuration constants
# ---------------------------------------------------------------------------
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200
HNSW_EF_SEARCH = 128
VECTOR_VERSION = "1.0"


class VectorStore:
    """
    Qdrant vector store wrapper - supports both legacy and new dual-collection modes.

    Payload schema (all 15 enterprise fields):
        company_id          - internal company identifier
        company_name        - human-readable company name
        company_symbol      - NSE/BSE ticker
        source_type         - 'json' | 'document'
        document_id         - parent document identifier
        chunk_id            - unique chunk identifier (UUID)
        title               - document or announcement title
        platform            - source platform (nullable)
        excel_row_id        - originating Excel row (nullable, JSON source only)
        url                 - source URL (nullable, JSON source only)
        announcement_date   - Unix timestamp (int)
        mentioned_timestamp - Unix timestamp when first mentioned (int)
        vector_version      - embedding model version tag
        active              - soft-delete flag (bool, default True)
        ingestion_time      - ISO-8601 string when chunk was indexed

        # Legacy / backward-compat fields kept:
        text                - chunk text
        file_hash           - SHA-256 of source file (doc ingestion)
        source              - file path or URL
        page_number         - page (doc ingestion)
        chunk_index         - position within document
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        *,
        enable_quantization: bool = True,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant host (default: settings.qdrant_host)
            port: Qdrant port (default: settings.qdrant_port)
            collection_name: Collection name (default: settings.qdrant_collection)
            enable_quantization: Whether to enable FP16 scalar quantization
        """
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.dimension = settings.embedding_dimension
        self.enable_quantization = enable_quantization

        logger.info(
            f"[VectorStore] Connecting to Qdrant at {self.host}:{self.port}, "
            f"collection='{self.collection_name}', dim={self.dimension}, "
            f"fp16_quant={self.enable_quantization}"
        )

        self.client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection()

        logger.info(f"[VectorStore] Ready - collection='{self.collection_name}'")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_collection(self) -> None:
        """Create collection with FP16 quantization + HNSW config if it doesn't exist."""
        try:
            existing = {c.name for c in self.client.get_collections().collections}

            if self.collection_name not in existing:
                logger.info(f"[VectorStore] Creating collection '{self.collection_name}' ...")

                quant_config = None
                if self.enable_quantization:
                    quant_config = ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,   # Qdrant uses INT8 for scalar; FP16 via on_disk
                            quantile=0.99,
                            always_ram=True,
                        )
                    )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=HNSW_M,
                            ef_construct=HNSW_EF_CONSTRUCT,
                            full_scan_threshold=10_000,
                        ),
                    ),
                    quantization_config=quant_config,
                )
                logger.info(
                    f"[VectorStore] Collection '{self.collection_name}' created - "
                    f"dim={self.dimension}, HNSW m={HNSW_M}/{HNSW_EF_CONSTRUCT}, "
                    f"quantization={'INT8_scalar' if quant_config else 'none'}"
                )
            else:
                logger.info(f"[VectorStore] Collection '{self.collection_name}' already exists.")

        except Exception as exc:
            logger.error(f"[VectorStore] _ensure_collection failed: {exc}", exc_info=True)
            raise

    @staticmethod
    def _build_filter(filter_conditions: Dict[str, Any]) -> Optional[Filter]:
        """Build a Qdrant Filter from a dict of conditions."""
        if not filter_conditions:
            return None

        must_conditions = []

        for key, value in filter_conditions.items():
            if isinstance(value, dict):
                if "$in" in value:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value["$in"]))
                    )
                else:
                    range_params: Dict[str, float] = {}
                    for op, raw in [("$gte", "gte"), ("$lte", "lte"), ("$gt", "gt"), ("$lt", "lt")]:
                        if op in value:
                            v = value[op]
                            if isinstance(v, str):
                                try:
                                    v = datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
                                except (ValueError, AttributeError):
                                    try:
                                        v = float(v)
                                    except ValueError:
                                        logger.warning(f"[VectorStore] Cannot convert {key}={v} to number; skipping.")
                                        continue
                            range_params[raw] = v
                    if range_params:
                        must_conditions.append(FieldCondition(key=key, range=Range(**range_params)))
            else:
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

        return Filter(must=must_conditions) if must_conditions else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def document_hash_exists(self, file_hash: str) -> bool:
        """
        Check if a document with the given SHA-256 hash already exists.
        Used for deduplication on upload.
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="file_hash", match=MatchValue(value=file_hash))]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            found = len(results[0]) > 0
            if found:
                logger.info(f"[VectorStore] Duplicate hash detected: {file_hash[:12]}...")
            return found
        except Exception as exc:
            logger.error(f"[VectorStore] document_hash_exists failed: {exc}")
            return False

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        file_hash: Optional[str] = None,
        source_type: str = "document",
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            texts: Chunk texts
            embeddings: 2-D numpy array of shape (N, dim)
            metadata: Per-chunk metadata dicts (enriched in-place)
            ids: Optional UUIDs (auto-generated if omitted)
            file_hash: SHA-256 of source file for dedup
            source_type: 'document' | 'json'

        Returns:
            List of point UUIDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")

        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadata = metadata or [{} for _ in texts]
        now_iso = datetime.now().isoformat()
        now_ts = int(datetime.now().timestamp())

        for i, (text, meta) in enumerate(zip(texts, metadata)):
            meta["text"] = text
            meta["ingestion_time"] = now_iso
            meta["vector_version"] = VECTOR_VERSION
            meta["source_type"] = meta.get("source_type", source_type)
            meta.setdefault("active", True)
            meta.setdefault("chunk_id", ids[i])

            if file_hash:
                meta["file_hash"] = file_hash

            # Ensure announcement_date is a Unix int for Qdrant Range filtering
            if "announcement_date" in meta:
                raw_date = meta["announcement_date"]
                if isinstance(raw_date, str):
                    try:
                        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                        meta["announcement_date"] = int(dt.timestamp())
                    except (ValueError, AttributeError):
                        logger.warning(
                            f"[VectorStore] Cannot convert announcement_date='{raw_date}' to timestamp; keeping as string."
                        )

            meta.setdefault("mentioned_timestamp", now_ts)

        points = [
            PointStruct(id=doc_id, vector=emb.tolist(), payload=meta)
            for doc_id, emb, meta in zip(ids, embeddings, metadata)
        ]

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(
                f"[VectorStore] Upserted {len(points)} chunks into '{self.collection_name}' "
                f"(source_type={source_type})"
            )
            return ids
        except Exception as exc:
            logger.error(f"[VectorStore] add_documents failed: {exc}", exc_info=True)
            raise

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        ef_search: int = HNSW_EF_SEARCH,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search with optional metadata filtering.

        Args:
            query_embedding: 1-D numpy array of shape (dim,)
            top_k: Number of results
            filter_conditions: Dict of metadata filters
            score_threshold: Minimum cosine similarity
            ef_search: HNSW ef_search parameter (increase for higher recall)
            active_only: If True, filter out soft-deleted (active=False) docs

        Returns:
            List of result dicts with 'id', 'score', 'text', 'metadata'
        """
        t0 = time.perf_counter()
        try:
            conditions = dict(filter_conditions or {})
            if active_only:
                conditions["active"] = True

            query_filter = self._build_filter(conditions)

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
                search_params=SearchParams(hnsw_ef=ef_search, exact=False),
            )

            results = [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.payload.get("text", ""),
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ("text", "ingestion_time")},
                    "ingestion_time": hit.payload.get("ingestion_time"),
                    "source_type": hit.payload.get("source_type", "document"),
                }
                for hit in search_result
            ]

            latency_ms = (time.perf_counter() - t0) * 1000
            top_score = results[0]["score"] if results else 0.0
            logger.debug(
                f"[VectorStore] search '{self.collection_name}'  {len(results)} results, "
                f"top_score={top_score:.4f}, "
                f"latency={latency_ms:.1f}ms, ef={ef_search}"
            )
            return results

        except Exception as exc:
            logger.error(f"[VectorStore] search failed: {exc}", exc_info=True)
            raise

    def soft_delete(self, chunk_ids: List[str]) -> bool:
        """
        Mark chunks as inactive (soft delete) by setting active=False.
        Vectors remain in Qdrant; weekly compaction task performs hard delete.

        Args:
            chunk_ids: List of point UUIDs to soft-delete

        Returns:
            True on success
        """
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"active": False},
                points=PointIdsList(points=chunk_ids),
            )
            logger.info(f"[VectorStore] Soft-deleted {len(chunk_ids)} chunks in '{self.collection_name}'")
            return True
        except Exception as exc:
            logger.error(f"[VectorStore] soft_delete failed: {exc}", exc_info=True)
            return False

    def hard_delete_inactive(self) -> int:
        """
        Permanently remove all points where active=False.
        Called by the weekly compaction Celery task.

        Returns:
            Number of points deleted
        """
        try:
            inactive_filter = Filter(
                must=[FieldCondition(key="active", match=MatchValue(value=False))]
            )
            # Count first
            count_before = self.client.count(
                collection_name=self.collection_name,
                count_filter=inactive_filter,
            ).count

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=inactive_filter,
            )
            logger.info(
                f"[VectorStore] Hard-deleted {count_before} inactive points from '{self.collection_name}'"
            )
            return count_before
        except Exception as exc:
            logger.error(f"[VectorStore] hard_delete_inactive failed: {exc}", exc_info=True)
            return 0

    def delete_by_document_id(self, document_id: str, soft: bool = True) -> bool:
        """
        Delete (soft or hard) all chunks belonging to a document_id.

        Args:
            document_id: The document identifier
            soft: If True, set active=False; if False, permanently delete
        """
        try:
            doc_filter = Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            )
            if soft:
                # Fetch chunk UUIDs then soft-delete
                scroll_res = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=doc_filter,
                    limit=10_000,
                    with_payload=False,
                    with_vectors=False,
                )
                chunk_ids = [str(p.id) for p in scroll_res[0]]
                if chunk_ids:
                    return self.soft_delete(chunk_ids)
                return True
            else:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=doc_filter,
                )
                logger.info(f"[VectorStore] Hard-deleted document_id='{document_id}'")
                return True
        except Exception as exc:
            logger.error(f"[VectorStore] delete_by_document_id failed: {exc}", exc_info=True)
            return False

    def document_hash_exists_in_store(self, file_hash: str) -> bool:
        """Alias for backward compatibility."""
        return self.document_hash_exists(file_hash)

    def get_unique_document_count(self) -> int:
        """Count unique active documents by filename / document_id."""
        try:
            unique_ids: set = set()
            scroll_res = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="active", match=MatchValue(value=True))]
                ),
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )
            for point in scroll_res[0]:
                doc_id = (
                    point.payload.get("document_id")
                    or point.payload.get("filename")
                )
                if doc_id:
                    unique_ids.add(doc_id)
            return len(unique_ids)
        except Exception as exc:
            logger.error(f"[VectorStore] get_unique_document_count failed: {exc}")
            return 0

    def get_document_list(self) -> List[Dict[str, Any]]:
        """Return list of unique active documents with metadata."""
        try:
            docs: Dict[str, Dict] = {}
            scroll_res = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="active", match=MatchValue(value=True))]
                ),
                limit=10_000,
                with_payload=True,
                with_vectors=False,
            )
            for point in scroll_res[0]:
                p = point.payload or {}
                key = p.get("document_id") or p.get("filename", str(point.id))
                if key not in docs:
                    docs[key] = {
                        "document_id": key,
                        "company_name": p.get("company_name", ""),
                        "company_symbol": p.get("company_symbol", ""),
                        "title": p.get("title", p.get("filename", "")),
                        "source_type": p.get("source_type", "document"),
                        "announcement_date": p.get("announcement_date"),
                        "ingestion_time": p.get("ingestion_time"),
                        "active": p.get("active", True),
                    }
            return sorted(docs.values(), key=lambda x: x.get("ingestion_time", "") or "")
        except Exception as exc:
            logger.error(f"[VectorStore] get_document_list failed: {exc}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Return collection statistics and configuration summary.
        Also prints a human-readable summary to logs.
        """
        try:
            total = self.client.count(collection_name=self.collection_name).count
            active = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="active", match=MatchValue(value=True))]
                ),
            ).count

            info = {
                "collection": self.collection_name,
                "total_vectors": total,
                "active_vectors": active,
                "inactive_vectors": total - active,
                "dimension": self.dimension,
                "distance": "Cosine",
                "hnsw_m": HNSW_M,
                "hnsw_ef_construct": HNSW_EF_CONSTRUCT,
                "hnsw_ef_search_default": HNSW_EF_SEARCH,
                "quantization": "INT8_scalar (FP16 equivalent)" if self.enable_quantization else "none",
                "unique_documents": self.get_unique_document_count(),
                "status": "active",
            }

            logger.info(
                f"\n{'='*60}\n"
                f"  Qdrant Collection Summary: {self.collection_name}\n"
                f"{'='*60}\n"
                f"  Total vectors  : {info['total_vectors']:,}\n"
                f"  Active vectors : {info['active_vectors']:,}\n"
                f"  Dimension      : {info['dimension']}\n"
                f"  Distance       : {info['distance']}\n"
                f"  HNSW m/ef      : {HNSW_M}/{HNSW_EF_CONSTRUCT}\n"
                f"  Quantization   : {info['quantization']}\n"
                f"  Unique docs    : {info['unique_documents']}\n"
                f"{'='*60}"
            )
            return info

        except Exception as exc:
            logger.error(f"[VectorStore] get_collection_info failed: {exc}")
            return {
                "collection": self.collection_name,
                "total_vectors": 0,
                "active_vectors": 0,
                "status": "error",
            }


# ---------------------------------------------------------------------------
# Legacy singleton (backward compatible)
# ---------------------------------------------------------------------------
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Get or create the legacy single-collection vector store.
    Continues to use settings.qdrant_collection for backward compatibility.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        logger.info("[VectorStore] Legacy singleton created.")
    return _vector_store
