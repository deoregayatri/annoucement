"""
neaps_reranker.py
=================
Reranking wrapper for NEAPS retrieval results.

Delegates to the existing core/reranker.py cross-encoder.
Falls back gracefully if the reranker model is unavailable.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank a list of retrieval results using the cross-encoder reranker.

    Args:
        query:   Original user query.
        results: List of retrieval result dicts (must have 'text' key).
        top_k:   Number of results to return after reranking.

    Returns:
        Reranked and trimmed list of result dicts (up to top_k).
    """
    if not results:
        return []

    try:
        from core.reranker import get_reranker
        reranker = get_reranker()

        # Use rerank_with_documents which accepts dicts directly
        # and adds 'rerank_score' to each document
        reranked = reranker.rerank_with_documents(
            query=query,
            documents=results,
            text_key="text",
            top_k=top_k,
        )

        logger.info(
            f"[NeapsReranker] Reranked {len(results)}  top {len(reranked)} results."
        )
        return reranked

    except Exception as exc:
        logger.warning(
            f"[NeapsReranker] Reranker unavailable ({exc}), returning original order."
        )
        # Do NOT insert a fake rerank_score — preserve existing fused_score/score ranking.
        # core/reranker.py already populates rerank_score from the document's own score
        # when the model is disabled or unavailable.
        return results[:top_k]
