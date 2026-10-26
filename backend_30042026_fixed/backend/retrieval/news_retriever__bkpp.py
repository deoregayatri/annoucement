"""
NewsRetriever — dedicated retriever for the company_descriptions Qdrant collection.
Completely independent from HybridRetriever and the announcements pipeline.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

from rank_bm25 import BM25Okapi
from config import settings

logger = logging.getLogger(__name__)

NEWS_COLLECTION = "company_descriptions"

_instance = None


def get_news_retriever() -> "NewsRetriever":
    global _instance
    if _instance is None:
        _instance = NewsRetriever()
    return _instance


# ---------------------------------------------------------------------------
# RRF fusion helper
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense: List[Dict],
    sparse: List[Dict],
    k: int = 60,
    dense_w: float = 0.6,
    sparse_w: float = 0.4,
) -> List[Dict]:
    scores:   Dict[str, float] = {}
    meta_map: Dict[str, Dict]  = {}
    dense_keys:  set = set()
    sparse_keys: set = set()

    for rank, hit in enumerate(dense):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + dense_w / (k + rank + 1)
        meta_map[key] = hit
        dense_keys.add(key)

    for rank, hit in enumerate(sparse):
        key = hit.get("text", "")[:200]
        scores[key] = scores.get(key, 0.0) + sparse_w / (k + rank + 1)
        if key not in meta_map:
            meta_map[key] = hit
        sparse_keys.add(key)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in fused:
        item = dict(meta_map[key])
        item["fused_score"] = score
        in_dense  = key in dense_keys
        in_sparse = key in sparse_keys
        if in_dense and in_sparse:
            item["_provenance"] = "overlapped"
        elif in_dense:
            item["_provenance"] = "dense_only"
        else:
            item["_provenance"] = "sparse_only"
        results.append(item)
    return results


# ---------------------------------------------------------------------------
# NewsRetriever
# ---------------------------------------------------------------------------

class NewsRetriever:
    """
    Self-contained retriever for news articles in company_descriptions.
    Provides dense search, BM25, RRF fusion, and optional reranking.
    """

    def __init__(self):
        self.client     = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.collection = NEWS_COLLECTION
        self._bm25:       Optional[BM25Okapi] = None
        self._bm25_texts: List[str]           = []
        self._bm25_meta:  List[Dict]          = []
        self._reranker    = None
        logger.info(f"[NewsRetriever] Initialised — collection='{self.collection}'")

    # ------------------------------------------------------------------
    # BM25 — built lazily from Qdrant scroll on first search
    # ------------------------------------------------------------------

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        logger.info("[NewsRetriever] Building in-memory BM25 from Qdrant scroll...")
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection,
                limit=500,
                offset=offset,
                with_payload=True,
            )
            for p in points:
                payload = p.payload or {}
                text = payload.get("text") or payload.get("chunk_text", "")
                if not text:
                    continue
                self._bm25_texts.append(text)
                self._bm25_meta.append({
                    "url":                  payload.get("url"),
                    "company_symbol":       (
                        payload.get("company_symbol") or payload.get("symbol")
                    ),
                    "matched_company_name": payload.get("matched_company_name"),
                    "source_type":          "news",
                })
            offset = next_offset
            if not points or offset is None:
                break
        if self._bm25_texts:
            self._bm25 = BM25Okapi([t.lower().split() for t in self._bm25_texts])
        logger.info(f"[NewsRetriever] BM25 ready — {len(self._bm25_texts)} docs")

    def invalidate_bm25(self):
        self._bm25       = None
        self._bm25_texts = []
        self._bm25_meta  = []
        logger.info("[NewsRetriever] BM25 index invalidated — will rebuild on next search")

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query_vector: List[float],
        top_k: int,
        company_symbols: Optional[List[str]] = None,
    ) -> List[Dict]:
        qdrant_filter = None
        if company_symbols:
            qdrant_filter = Filter(must=[FieldCondition(
                key="company_symbol",
                match=MatchAny(any=[s.upper() for s in company_symbols]),
            )])
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=top_k,
                score_threshold=0.45, # Added relevance threshold fix
                with_payload=True,
            )
        except Exception as exc:
            logger.warning(f"[NewsRetriever] Dense search failed: {exc}")
            return []
        return [
            {
                "id":           str(h.id),
                "text":         (
                    (h.payload or {}).get("text")
                    or (h.payload or {}).get("chunk_text", "")
                ),
                "score":        h.score,
                "vector_score": h.score,
                "bm25_score":   0.0,
                "metadata": {
                    "url":                  (h.payload or {}).get("url"),
                    "company_symbol":       (
                        (h.payload or {}).get("company_symbol")
                        or (h.payload or {}).get("symbol")
                    ),
                    "matched_company_name": (h.payload or {}).get("matched_company_name"),
                    "csv_company_name":     (h.payload or {}).get("csv_company_name"),
                    "chunk_index":          (h.payload or {}).get("chunk_index"),
                    "total_chunks":         (h.payload or {}).get("total_chunks"),
                    "source_type":          "news",
                },
            }
            for h in hits
        ]

    # ------------------------------------------------------------------
    # BM25 sparse search
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        company_symbols: Optional[List[str]] = None,
    ) -> List[Dict]:
        self._ensure_bm25()
        if not self._bm25:
            logger.info("[NewsRetriever] BM25 index is empty — skipping sparse search")
            return []

        symbols_upper = (
            [s.upper() for s in company_symbols] if company_symbols else None
        )
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if len(results) >= top_k:
                break
            if score <= 0:
                continue
            meta = self._bm25_meta[idx]
            if symbols_upper:
                stored_sym = (meta.get("company_symbol") or "").upper()
                if stored_sym not in symbols_upper:
                    continue
            results.append({
                "id":           f"bm25_{idx}",
                "text":         self._bm25_texts[idx],
                "score":        float(score),
                "vector_score": 0.0,
                "bm25_score":   float(score),
                "metadata":     {**meta, "source_type": "news"},
            })
        if not results:
            logger.info(
                f"[NewsRetriever] BM25 returned 0 results for query='{query[:60]}' "
                f"(low IDF or terms absent). Dense-only fusion will apply."
            )
        return results

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def _rerank_results(
        self,
        query: str,
        fused: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if self._reranker is None:
            try:
                from core.reranker import get_reranker
                self._reranker = get_reranker()
            except Exception:
                self._reranker = False
        if not self._reranker:
            logger.info("[NewsRetriever] Reranker disabled — returning fused top-k")
            return fused[:top_k]
        try:
            logger.info("[NewsRetriever] Reranking with MS-Marco cross-encoder")
            ranked = self._reranker.rerank_with_documents(
                query=query,
                documents=[dict(h) for h in fused],
                text_key="text",
                top_k=top_k,
            )
            for h in ranked:
                h["source_collection"] = "news"
            return ranked
        except Exception as exc:
            logger.warning(f"[NewsRetriever] Reranker failed (returning fused): {exc}")
            return fused[:top_k]

    # ------------------------------------------------------------------
    # Fetch and stitch all chunks for a list of URLs
    # ------------------------------------------------------------------

    def fetch_full_descriptions(self, urls: List[str]) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for url in urls:
            if not url:
                continue
            try:
                all_chunks = []
                offset = None
                while True:
                    points, next_offset = self.client.scroll(
                        collection_name=self.collection,
                        scroll_filter=Filter(must=[
                            FieldCondition(
                                key="url",
                                match=MatchValue(value=url),
                            )
                        ]),
                        limit=100,
                        offset=offset,
                        with_payload=True,
                    )
                    all_chunks.extend(points)
                    offset = next_offset
                    if not points or offset is None:
                        break

                if not all_chunks:
                    logger.warning(f"[NewsRetriever] No chunks found in Qdrant for url={url}")
                    result[url] = ""
                    continue

                # Sort by chunk_index so text reads in correct order
                all_chunks.sort(
                    key=lambda p: (p.payload or {}).get("chunk_index", 0)
                )

                full_text = " ".join(
                    (p.payload or {}).get("text") or (p.payload or {}).get("chunk_text", "")
                    for p in all_chunks
                )
                result[url] = full_text
                logger.info(
                    f"[NewsRetriever] Stitched {len(all_chunks)} chunks for url={url} "
                    f"({len(full_text)} chars)"
                )
            except Exception as exc:
                logger.warning(f"[NewsRetriever] fetch_full_descriptions failed for url={url}: {exc}")
                result[url] = ""
        return result

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def search_with_stats(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        company_symbols: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        t_total   = time.perf_counter()
        latencies: Dict[str, float] = {}

        t0    = time.perf_counter()
        dense = self._dense_search(query_vector, top_k=top_k * 2, company_symbols=company_symbols)
        latencies["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0     = time.perf_counter()
        sparse = self._bm25_search(query, top_k=top_k * 2, company_symbols=company_symbols)
        latencies["sparse_ms"] = (time.perf_counter() - t0) * 1000

        n_dense  = len(dense)
        n_sparse = len(sparse)

        if not dense and not sparse:
            logger.info(
                f"[NewsRetriever] Both dense and sparse returned 0 results "
                f"for query='{query[:60]}' filter={company_symbols}"
            )

        t0    = time.perf_counter()
        fused = _rrf_fuse(dense, sparse)
        latencies["fusion_ms"] = (time.perf_counter() - t0) * 1000

        n_unique      = len(fused)
        n_dense_only  = sum(1 for h in fused if h.get("_provenance") == "dense_only")
        n_sparse_only = sum(1 for h in fused if h.get("_provenance") == "sparse_only")
        n_overlapped  = sum(1 for h in fused if h.get("_provenance") == "overlapped")

        fused = fused[: top_k * 2]

        t0 = time.perf_counter()
        if use_reranker:
            hits = self._rerank_results(query, fused, top_k=top_k)
        else:
            hits = fused[:top_k]
        latencies["rerank_ms"] = (time.perf_counter() - t0) * 1000
        latencies["total_ms"]  = (time.perf_counter() - t_total) * 1000

        logger.info(
            f"[NewsRetriever] query dense={n_dense} sparse={n_sparse} "
            f"unique={n_unique} final={len(hits)} "
            f"total={latencies['total_ms']:.0f}ms"
        )

        return {
            "hits":          hits,
            "n_dense":       n_dense,
            "n_sparse":      n_sparse,
            "n_unique":      n_unique,
            "n_final":       len(hits),
            "n_dense_only":  n_dense_only,
            "n_sparse_only": n_sparse_only,
            "n_overlapped":  n_overlapped,
            "latencies_ms":  latencies,
        }

    # ------------------------------------------------------------------
    # Legacy API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        company_symbols: Optional[List[str]] = None,
        use_reranker: bool = True,
    ) -> List[Dict]:
        result = self.search_with_stats(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            company_symbols=company_symbols,
            use_reranker=use_reranker,
        )
        return result["hits"]