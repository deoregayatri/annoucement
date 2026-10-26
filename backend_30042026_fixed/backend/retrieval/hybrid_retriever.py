"""
Hybrid Retriever - Stage 4: 4-Signal Parallel Search Engine.

Architecture:
  Per query, fires 4 parallel retrieval signals:
    1. JSON dense   - announcements_json vector search
    2. JSON sparse  - BM25 over JSON corpus
    3. DOC dense    - announcements_docs vector search
    4. DOC sparse   - BM25 over DOC corpus

  Per-collection fusion  top 12 per collection  reranker  top 8 each
  Combined: max 16 chunks to LLM (8 JSON + 8 DOC)

  Retry escalation (max 2 cycles):
    Level 1: similarity < threshold  ef_search  256, retry
    Level 2: still weak  top_k  30, retry
    Level 3: still weak  trigger dictionary expansion

  Logs per-stage latency using time.perf_counter() and structured logger.

LangChain integration:
  - LangChain Document for unified chunk representation
  - Uses LangGraph-compatible return shape for Stage 6 graph nodes
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

# NOTE: _rrf_fuse is defined locally below (text-based dedup for this retriever).
# The canonical rrf_fuse from score_fusion.py uses ID-based dedup and is not used here.

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 12  # per signal before fusion
FINAL_TOP_K = 8  # after reranking, per collection
# RRF smoothing constant
RRF_K = 60
# Dense / sparse contribution weights
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
# Score threshold below which retry/expansion is triggered.
# RRF fused_score for rank-1 is max ~1/(60+1)=0.0161, so threshold must be in that range.
SIMILARITY_THRESHOLD = 0.008  # triggers only when truly no good results found
RETRY_EF_SEARCH = 256
RETRY_TOP_K = 30
MAX_RETRY_CYCLES = 2


# ---------------------------------------------------------------------------
# Score fusion helpers
# ---------------------------------------------------------------------------


def _rrf_fuse(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) for combining dense and sparse results.
    Chunks identified by 'text' content (simple dedup key).
    """
    scores: Dict[str, float] = {}
    meta_map: Dict[str, Dict] = {}

    for rank, hit in enumerate(dense_results):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + dense_weight * (1.0 / (k + rank + 1))
        meta_map[key] = hit

    for rank, hit in enumerate(sparse_results):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
        if key not in meta_map:
            meta_map[key] = hit

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in fused:
        item = dict(meta_map[key])
        item["fused_score"] = score
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    4-Signal parallel hybrid retriever.

    Stage 4 enhancements:
      - Dual-collection dense + sparse search (4 signals)
      - RRF score fusion per collection
      - Cross-encoder reranking (top 20  top 8 per collection)
      - 3-level retry escalation
      - Per-stage latency logging with structured logger
      - Returns LangGraph-compatible result dict
    """

    def __init__(self) -> None:
        # Lazy singletons - avoid heavy init at import time
        self._embedder = None
        self._json_store = None
        self._doc_store = None
        self._sparse = None
        self._reranker = None
        self._dict_store = None
        self._query_expander = None
        logger.info("[HybridRetriever] Initialised (lazy singletons).")

    # ------------------------------------------------------------------
    # Singleton accessors
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from core.embeddings import get_embedding_model

            self._embedder = get_embedding_model()
        return self._embedder

    def _get_json_store(self):
        if self._json_store is None:
            from core.collection_manager import get_json_store

            self._json_store = get_json_store()
        return self._json_store

    def _get_neaps_search(self):
        from retrieval.neaps_hybrid_search import search as neaps_search

        return neaps_search

    def _get_reranker(self):
        if self._reranker is None:
            from core.reranker import get_reranker

            self._reranker = get_reranker()
        return self._reranker

    def _get_dict_store(self):
        if self._dict_store is None:
            try:
                from vocabulary.dictionary_store import get_dictionary_store

                self._dict_store = get_dictionary_store()
            except ImportError:
                logger.debug(
                    "[HybridRetriever] vocabulary module not available, dictionary expansion disabled."
                )
                self._dict_store = False
        return self._dict_store if self._dict_store is not False else None

    def _get_query_expander(self):
        if self._query_expander is None:
            try:
                from vocabulary.query_expander import get_query_expander

                self._query_expander = get_query_expander()
            except ImportError:
                logger.debug(
                    "[HybridRetriever] vocabulary module not available, query expansion disabled."
                )
                self._query_expander = False
        return self._query_expander if self._query_expander is not False else None

    # ------------------------------------------------------------------
    # Main retrieval API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K,
        query_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        4-signal hybrid retrieval with retry escalation.

        Args:
            query: User query string
            filters: Optional metadata filters (company_symbol, date_range, etc.)
            top_k: Per-signal top-k (before fusion)
            query_id: Optional trace ID (auto-generated if omitted)

        Returns:
            Dict with:
              'json_chunks'  - top 8 chunks from JSON corpus (LangGraph-ready)
              'doc_chunks'   - top 8 chunks from DOC corpus
              'all_chunks'   - combined up to 16 chunks
              'query_id'     - trace UUID
              'latencies_ms' - per-stage timing dict
              'dict_used'    - whether dictionary expansion was triggered
              'retry_count'  - number of retry cycles used
        """
        query_id = query_id or str(uuid.uuid4())
        t_total = time.perf_counter()
        latencies: Dict[str, float] = {}

        try:
            from utils.structured_logger import get_structured_logger

            slog = get_structured_logger()
        except Exception:
            slog = None

        #  Stage 1: Embed query
        t0 = time.perf_counter()
        embedder = self._get_embedder()
        query_emb = embedder.embed_query(query)
        latencies["embed_ms"] = (time.perf_counter() - t0) * 1000

        #  Stage 2: 4-signal parallel search (sync via asyncio)
        t0 = time.perf_counter()
        company_filter = filters.get("company_symbol") if filters else None
        ef_search = (
            settings.qdrant_ef_search if hasattr(settings, "qdrant_ef_search") else 128
        )

        json_dense, json_sparse, doc_dense, doc_sparse = self._four_signal_search(
            query=query,
            query_emb=query_emb,
            top_k=top_k,
            filters=filters,
            ef_search=ef_search,
            company_filter=company_filter,
        )
        latencies["search_ms"] = (time.perf_counter() - t0) * 1000

        #  Stage 3: Per-collection RRF fusion  top 12
        t0 = time.perf_counter()
        json_fused = _rrf_fuse(json_dense, json_sparse)[:DEFAULT_TOP_K]
        doc_fused = _rrf_fuse(doc_dense, doc_sparse)[:DEFAULT_TOP_K]
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000

        #  Stage 4: Retry escalation
        retry_count = 0
        dict_used = False
        t0_retry = time.perf_counter()

        top_score_json = json_fused[0]["fused_score"] if json_fused else 0.0
        top_score_doc = doc_fused[0]["fused_score"] if doc_fused else 0.0
        max_score = max(top_score_json, top_score_doc)

        # L1 retry: widen ef_search
        if max_score < SIMILARITY_THRESHOLD:
            logger.info(
                f"[HybridRetriever] Low confidence ({max_score:.3f} < {SIMILARITY_THRESHOLD}), "
                f"retry L1: ef_search={RETRY_EF_SEARCH}"
            )
            retry_count += 1
            json_dense2, json_sparse2, doc_dense2, doc_sparse2 = (
                self._four_signal_search(
                    query=query,
                    query_emb=query_emb,
                    top_k=top_k,
                    filters=clean_filters,
                    ef_search=RETRY_EF_SEARCH,
                    company_filter=company_filter,
                )
            )
            json_fused = _rrf_fuse(json_dense2, json_sparse2)[:DEFAULT_TOP_K]
            doc_fused = _rrf_fuse(doc_dense2, doc_sparse2)[:DEFAULT_TOP_K]

            # Update max_score after L1 retry
            max_score = max(
                json_fused[0]["fused_score"] if json_fused else 0.0,
                doc_fused[0]["fused_score"] if doc_fused else 0.0,
            )

            # L2 retry: widen top_k (only if L1 is STILL insufficient)
            if max_score < SIMILARITY_THRESHOLD:
                logger.info(
                    f"[HybridRetriever] Still low confidence ({max_score:.3f}), "
                    f"retry L2: top_k={RETRY_TOP_K}"
                )
                retry_count += 1
                json_dense3, json_sparse3, doc_dense3, doc_sparse3 = (
                    self._four_signal_search(
                        query=query,
                        query_emb=query_emb,
                        top_k=RETRY_TOP_K,
                        filters=filters,
                        ef_search=RETRY_EF_SEARCH,
                        company_filter=company_filter,
                    )
                )
                json_fused = _rrf_fuse(json_dense3, json_sparse3)[:DEFAULT_TOP_K]
                doc_fused = _rrf_fuse(doc_dense3, doc_sparse3)[:DEFAULT_TOP_K]
                max_score = max(
                    json_fused[0]["fused_score"] if json_fused else 0.0,
                    doc_fused[0]["fused_score"] if doc_fused else 0.0,
                )

        latencies["retry_ms"] = (time.perf_counter() - t0_retry) * 1000

        #  Stage 5: Reranking  top 8 per collection
        t0 = time.perf_counter()
        json_chunks = self._rerank(query, json_fused, top_k=FINAL_TOP_K, source="json")
        doc_chunks = self._rerank(query, doc_fused, top_k=FINAL_TOP_K, source="doc")
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000

        latencies["total_ms"] = (time.perf_counter() - t_total) * 1000

        #  Log structured event
        if slog:
            try:
                slog.log_query_event(
                    query_id=query_id,
                    stage="retrieval_complete",
                    latency_ms=latencies["total_ms"],
                    json_chunks=len(json_chunks),
                    doc_chunks=len(doc_chunks),
                    max_score=max_score,
                    dict_used=dict_used,
                    retry_count=retry_count,
                )
            except Exception:
                pass

        logger.info(
            f"[HybridRetriever] query_id={query_id[:8]}  "
            f"json={len(json_chunks)}, doc={len(doc_chunks)}, "
            f"dict={dict_used}, retries={retry_count}, "
            f"total={latencies['total_ms']:.0f}ms"
        )

        all_chunks = json_chunks + doc_chunks
        all_chunks.sort(key=lambda c: c.get("rerank_score", c.get("fused_score", 0.0)), reverse=True)
        all_chunks = all_chunks[:top_k]
        return {
            "json_chunks": json_chunks,
            "doc_chunks": doc_chunks,
            "all_chunks": all_chunks,
            "query_id": query_id,
            "latencies_ms": latencies,
            "dict_used": dict_used,
            "retry_count": retry_count,
            "max_score": max_score,
        }

    # Legacy single-list API for backward compat with routes.py
    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible search() - returns flat list of top chunks.
        Calls retrieve() internally and merges both collections.
        """
        result = self.retrieve(query=query, filters=filters, top_k=top_k)
        return result["all_chunks"]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a summary of component states (used at startup logging)."""
        stats: dict = {}

        # Vector store counts
        try:
            json_store = self._get_json_store()
            stats["vector_store"] = {
                "points_count": getattr(json_store, "count", lambda: 0)()
                if callable(getattr(json_store, "count", None))
                else json_store.get_point_count()
                if callable(getattr(json_store, "get_point_count", None))
                else 0
            }
        except Exception as exc:
            logger.debug(f"[get_stats] vector_store error: {exc}")
            stats["vector_store"] = {"points_count": 0}

        # BM25 / sparse retrieval stats (native Qdrant sparse vectors)
        try:
            bm25_index_size = 0
            from config import settings as _settings

            bm25_path = getattr(_settings, "bm25_index_path", None)
            if bm25_path:
                import os

                bm25_index_size = (
                    os.path.getsize(bm25_path) if os.path.exists(bm25_path) else 0
                )
            stats["bm25"] = {
                "index_size_bytes": bm25_index_size,
                "backend": "qdrant_sparse_vectors",
            }
        except Exception as exc:
            logger.debug(f"[get_stats] bm25 error: {exc}")
            stats["bm25"] = {"index_size_bytes": 0, "backend": "qdrant_sparse_vectors"}

        # Embedding model cache stats
        try:
            from core.embeddings import get_embedding_model

            embedder = get_embedding_model()
            cache_info = getattr(embedder, "cache_info", {}) or {}
            stats["embedding_cache"] = (
                cache_info if isinstance(cache_info, dict) else {"entries": 0}
            )
        except Exception as exc:
            logger.debug(f"[get_stats] embedding_cache error: {exc}")
            stats["embedding_cache"] = {"entries": 0}

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _four_signal_search(
        self,
        query: str,
        query_emb,
        top_k: int,
        filters: Optional[Dict],
        ef_search: int,
        company_filter: Optional[str],
    ) -> Tuple[List, List, List, List]:
        """Run retrieval signals correctly for JSON and NEAPS."""
        json_store = self._get_json_store()
        neaps_search = self._get_neaps_search()

        def _normalize_company_symbols(raw: Any) -> List[str]:
            symbols: List[str] = []
            if raw is None:
                return symbols
            if isinstance(raw, dict) and "$in" in raw:
                symbols.extend([str(s).upper() for s in raw.get("$in", []) if s])
            elif isinstance(raw, (list, tuple, set)):
                symbols.extend([str(s).upper() for s in raw if s])
            else:
                symbols.append(str(raw).upper())
            # Deduplicate while preserving order
            seen = set()
            ordered: List[str] = []
            for s in symbols:
                if s not in seen:
                    seen.add(s)
                    ordered.append(s)
            return ordered

        # Determine company symbols from explicit argument or incoming filters
        incoming_company = (filters or {}).get("company_symbol")
        symbols = _normalize_company_symbols(company_filter or incoming_company)

        dense_filter = dict(filters) if filters else {}
        if symbols:
            dense_filter["company_symbol"] = {"$in": symbols}
            company_filter = symbols if len(symbols) > 1 else symbols[0]
        else:
            company_filter = None

        # JSON Dense search is SKIPPED   json_answer_node and doc_answer_node are disabled
        # in rag_graph.py (they return hardcoded strings), so json_chunks never reach the LLM.
        json_dense = []  # intentionally empty
        json_sparse = []  # deprecated

        # DOC retrieval   use neaps_search for dense+sparse hybrid fusion only.
        # NOTE: neaps_search() performs its own cross-encoder rerank internally.
        # HybridRetriever.retrieve() will also rerank the final merged set.
        # To prevent double-reranking: pass skip_rerank=True if supported,
        # otherwise set doc_dense to raw fused results (scored by RRF only).
        # The final authoritative rerank happens in reranker_node in rag_graph.py.
        try:
            doc_dense = neaps_search(
                query=query,
                company_symbol=company_filter,
                top_k=top_k,
                final_top_k=top_k,
                skip_rerank=True,  # skip internal cross-encoder   avoids double rerank
            )
        except TypeError:
            # Older version of neaps_search without skip_rerank   fall back gracefully
            doc_dense = neaps_search(
                query=query,
                company_symbol=company_filter,
                top_k=top_k,
                final_top_k=top_k,
            )
            logger.debug(
                "[HybridRetriever] neaps_search called without skip_rerank (legacy)"
            )
        except Exception as exc:
            logger.error(f"[HybridRetriever] DOC neaps search failed: {exc}")
            doc_dense = []

        doc_sparse = []  # sparse handled internally by neaps_search via BM25

        return json_dense, json_sparse, doc_dense, doc_sparse

    def _rerank(
        self,
        query: str,
        fused: List[Dict[str, Any]],
        top_k: int,
        source: str,
    ) -> List[Dict[str, Any]]:
        """Cross-encoder rerank fused results  top_k."""
        if not fused:
            return []
        try:
            reranker = self._get_reranker()
            if (
                reranker is None
                or not getattr(reranker, "is_available", lambda: True)()
            ):
                return fused[:top_k]

            # We duplicate the dicts to prevent side effects, though reranker also modifies them
            docs_copy = [dict(hit) for hit in fused]
            ranked = reranker.rerank_with_documents(
                query=query, documents=docs_copy, text_key="text", top_k=top_k
            )

            results = []
            for h in ranked:
                h["source_collection"] = source
                # Do NOT insert a fake score   leave as None so callers can distinguish
                # "no score available" from a real cross-encoder score.
                if "rerank_score" not in h:
                    h["rerank_score"] = None
                results.append(h)
            return results

        except Exception as exc:
            logger.warning(f"[HybridRetriever] Reranker failed ({source}): {exc}")
            return fused[:top_k]

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever() -> HybridRetriever:
    """Get or create the HybridRetriever singleton."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever


def invalidate_hybrid_retriever() -> None:
    """Discard the cached HybridRetriever singleton so it is re-created on the next call."""
    global _hybrid_retriever
    _hybrid_retriever = None
    logger.info(
        "[HybridRetriever] Singleton invalidated (will be re-created on next use)."
    )
 