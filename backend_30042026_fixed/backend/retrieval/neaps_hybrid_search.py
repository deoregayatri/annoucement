"""
neaps_hybrid_search.py
======================
Hybrid search over the neaps_docs Qdrant collection.

Combines dense vector similarity and sparse BM25 keyword search,
then fuses scores using Reciprocal Rank Fusion (RRF).

Weights are defined as constants (can be moved to .env if needed).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# RRF contribution weights.
# A 0.7/0.3 split was too dense-biased, leading to 0 sparse-only chunks in provenance.
# 0.55/0.45 gives BM25 more competitive weight while dense still leads.
# Tune via .env if needed: NEAPS_DENSE_WEIGHT / NEAPS_SPARSE_WEIGHT
import os as _os

DENSE_WEIGHT = float(_os.environ.get("NEAPS_DENSE_WEIGHT", "0.55"))
SPARSE_WEIGHT = float(_os.environ.get("NEAPS_SPARSE_WEIGHT", "0.45"))


def _safe_normalize_query(query: str):
    """
    Safely apply domain query expansion.
    Wrapped in try/except so that if query_normalizer.py is missing or
    broken, search() always falls back to the original query and never crashes.

    Returns: (search_query, was_expanded)
    """
    try:
        from retrieval.query_normalizer import normalize_query
        return normalize_query(query)
    except Exception as e:
        logger.warning(f"[NeapsHybrid] query_normalizer unavailable ({e}), using raw query")
        return query, False


def search(
    query: str,
    company_symbol: Optional[str] = None,
    top_k: int = 5,
    final_top_k: int = 5,
    skip_rerank: bool = False,
) -> List[Dict[str, Any]]:
    """
    Run hybrid retrieval: Dense from Qdrant, Sparse from local BM25 .pkl index.
    Merges results using Reciprocal Rank Fusion (RRF).
    Reranking is done HERE so callers receive fully ranked results.
    """
    from embeddings.neaps_hybrid_embedder import embed_query_hybrid
    from vector_store.neaps_qdrant_store import get_neaps_qdrant_store
    from embeddings.neaps_sparse import get_bm25_index
    from retrieval.neaps_reranker import rerank

    # Step 0: Domain query expansion
    # Expand abbreviations like "cv"/"used cv" to "used CVs commercial vehicle loan"
    # BEFORE embed_query_hybrid so BOTH dense (Qdrant) and sparse (BM25) benefit.
    # Safe: falls back to raw query if query_normalizer.py is missing.
    search_query, was_expanded = _safe_normalize_query(query)
    if was_expanded:
        logger.info(f"[NeapsHybrid] Query expanded: '{query}' -> '{search_query}'")

    store = get_neaps_qdrant_store()
    company_label = company_symbol if company_symbol else "ALL"

    # Widen top_k for expanded/short queries so more candidates are available
    effective_top_k = max(top_k, 20) if was_expanded else top_k

    # 1. Dense Search (Qdrant) - embed the EXPANDED query
    emb = embed_query_hybrid(search_query)
    dense_results = store.search_dense(
        query_dense=emb["dense"],
        company_symbol=company_symbol,
        top_k=effective_top_k,
    )

    # 2. Sparse Search (Local BM25 .pkl index) - use EXPANDED query string
    sparse_results = get_bm25_index().search(
        query=search_query,
        company_symbol=company_symbol,
        top_k=effective_top_k,
    )

    # 3. Reciprocal Rank Fusion (RRF) + Provenance Tracking
    scores: Dict[str, Dict] = {}
    k_rrf = 60

    for rank, r in enumerate(dense_results):
        cid = r["id"]
        r_copy = dict(r)
        r_copy["dense_score"] = r["score"]
        r_copy["sparse_score"] = 0.0
        r_copy["fused_score"] = DENSE_WEIGHT / (k_rrf + rank + 1)
        r_copy["score"] = r_copy["fused_score"]
        r_copy["retrieval_method"] = "dense_only"
        scores[cid] = r_copy

    for rank, r in enumerate(sparse_results):
        cid = r["id"]
        rrf_val = SPARSE_WEIGHT / (k_rrf + rank + 1)
        if cid in scores:
            scores[cid]["sparse_score"] = r["score"]
            scores[cid]["fused_score"] += rrf_val
            scores[cid]["score"] = scores[cid]["fused_score"]
            scores[cid]["retrieval_method"] = "hybrid_overlap"
        else:
            r_copy = dict(r)
            r_copy["dense_score"] = 0.0
            r_copy["sparse_score"] = r["score"]
            r_copy["fused_score"] = rrf_val
            r_copy["score"] = r_copy["fused_score"]
            r_copy["retrieval_method"] = "sparse_only"
            scores[cid] = r_copy

    fused = list(scores.values())

    # 4. Cross-Encoder Reranking (skip when caller handles reranking to avoid double-rerank)
    # NOTE: pass original query to reranker (not expanded) for accurate relevance scoring
    if skip_rerank:
        fused.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)
        reranked = fused[:final_top_k]
    else:
        reranked = rerank(query=query, results=fused, top_k=final_top_k)

    # 5. Logging
    total_unique = len(fused)
    dense_only = sum(1 for c in reranked if c["retrieval_method"] == "dense_only")
    sparse_only = sum(1 for c in reranked if c["retrieval_method"] == "sparse_only")
    overlap = sum(1 for c in reranked if c["retrieval_method"] == "hybrid_overlap")

    logger.info("================ HYBRID SEARCH PIPELINE ================")
    logger.info(f"Query: '{query[:60]}' | Expanded: '{search_query[:60]}' | Filter: {company_label} | skip_rerank={skip_rerank}")
    logger.info(
        f"Layer 1 (Retrieval): Fetched {len(dense_results)} Dense + {len(sparse_results)} Sparse = {total_unique} Unique Chunks"
    )
    logger.info(
        f"Layer 2 (RRF Fusion & Rerank): Trimmed down to Final Top {len(reranked)}"
    )
    logger.info(
        f"LLM Context Provenance: {dense_only} Dense-only | {sparse_only} Sparse-only | {overlap} Overlapped"
    )
    logger.info("=========================================================")

    # Promote fused_score -> score for downstream consumers
    # Keep fused_score present too so API can fall back to it if rerank_score is missing.
    for item in reranked:
        fused = item.get("fused_score", item.get("score", 0.0))
        item["score"] = fused
        item.setdefault("fused_score", fused)  # ensure fused_score is always present

    return reranked