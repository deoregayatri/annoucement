"""
Score fusion for combining multiple retrieval signals.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class ScoreFusion:
    """
    Combine scores from multiple retrieval methods.
    """
    
    def __init__(
        self,
        weight_vector: Optional[float] = None,
        weight_bm25: Optional[float] = None,
        weight_dictionary: Optional[float] = None
    ):
        """
        Initialize score fusion.
        
        Args:
            weight_vector: Weight for vector similarity
            weight_bm25: Weight for BM25 score
            weight_dictionary: Weight for dictionary boost
        """
        self.weight_vector = weight_vector or settings.weight_vector
        self.weight_bm25 = weight_bm25 or settings.weight_bm25
        self.weight_dictionary = weight_dictionary or settings.weight_dictionary
        
        # Validate weights sum to 1.0
        total = self.weight_vector + self.weight_bm25 + self.weight_dictionary
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Weights sum to {total}, not 1.0. Normalizing..."
            )
            self.weight_vector /= total
            self.weight_bm25 /= total
            self.weight_dictionary /= total
        
        logger.info(
            f"Initialized score fusion (vector={self.weight_vector:.2f}, "
            f"bm25={self.weight_bm25:.2f}, dict={self.weight_dictionary:.2f})"
        )
    
    def fuse_scores(
        self,
        vector_score: float,
        bm25_score: float,
        dictionary_boost: float = 0.0
    ) -> float:
        """
        Combine scores with weighted average.
        
        Args:
            vector_score: Cosine similarity (0-1)
            bm25_score: Normalized BM25 score (0-1)
            dictionary_boost: Dictionary match boost (0-1)
        
        Returns:
            Fused score (0-1)
        """
        return (
            self.weight_vector * vector_score +
            self.weight_bm25 * bm25_score +
            self.weight_dictionary * dictionary_boost
        )
    
    @staticmethod
    def normalize_bm25_scores(scores: List[float]) -> List[float]:
        """
        Normalize BM25 scores to 0-1 range using min-max normalization.
        
        Args:
            scores: Raw BM25 scores
        
        Returns:
            Normalized scores (0-1)
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
    
    def merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        dictionary_boosts: Optional[Dict[str, float]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Merge and rerank results from multiple retrievers.
        
        Args:
            vector_results: Results from vector search (with 'id' and 'score')
            bm25_results: Results from BM25 search (with 'index' and 'score')
            dictionary_boosts: Optional dict mapping doc_id to boost score
            top_k: Number of results to return
        
        Returns:
            Merged and reranked results
        """
        # Normalize BM25 scores
        bm25_scores = [r["score"] for r in bm25_results]
        normalized_bm25 = self.normalize_bm25_scores(bm25_scores)
        
        # Create lookup for normalized BM25 scores
        bm25_lookup = {
            r["index"]: norm_score
            for r, norm_score in zip(bm25_results, normalized_bm25)
        }
        
        # Get dictionary boosts
        dict_boosts = dictionary_boosts or {}
        
        # Merge results
        merged = {}
        
        # Add vector results
        for result in vector_results:
            doc_id = result["id"]
            vector_score = result["score"]
            
            merged[doc_id] = {
                "id": doc_id,
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "vector_score": vector_score,
                "bm25_score": 0.0,
                "dictionary_boost": 0.0,
                "fused_score": 0.0
            }
        
        # Add/update with BM25 results
        for result in bm25_results:
            idx = result["index"]
            # Get document ID - try from result first, then metadata, finally fallback
            doc_id = result.get("id") or result.get("metadata", {}).get("id") or f"doc_{idx}"
            bm25_score = bm25_lookup.get(idx, 0.0)
            
            if doc_id in merged:
                # Update existing entry with BM25 score
                merged[doc_id]["bm25_score"] = bm25_score
            else:
                # Create new entry for BM25-only result
                merged[doc_id] = {
                    "id": doc_id,
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "vector_score": 0.0,
                    "bm25_score": bm25_score,
                    "dictionary_boost": 0.0,
                    "fused_score": 0.0
                }
        
        # Add dictionary boosts and calculate fused scores
        for doc_id in merged:
            merged[doc_id]["dictionary_boost"] = dict_boosts.get(doc_id, 0.0)
            
            merged[doc_id]["fused_score"] = self.fuse_scores(
                vector_score=merged[doc_id]["vector_score"],
                bm25_score=merged[doc_id]["bm25_score"],
                dictionary_boost=merged[doc_id]["dictionary_boost"]
            )
        
        # Convert to list and sort by fused score
        results = list(merged.values())
        results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        # Take top-k
        results = results[:top_k]
        
        logger.debug(
            f"Merged {len(vector_results)} vector + {len(bm25_results)} BM25 results "
            f"into {len(results)} final results"
        )
        
        return results


# Global score fusion instance
_score_fusion: Optional['ScoreFusion'] = None


def get_score_fusion() -> 'ScoreFusion':
    """Get or create global score fusion instance."""
    global _score_fusion
    if _score_fusion is None:
        _score_fusion = ScoreFusion()
    return _score_fusion


def merge_scores(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    dictionary_boosts: Optional[Dict[str, float]] = None,
    top_k: int = 10,
    weight_vector: float = 0.5,
    weight_bm25: float = 0.3,
    weight_dictionary: float = 0.2
) -> List[Dict[str, Any]]:
    """
    Convenience function to merge scores using custom weights.
    
    Args:
        vector_results: Results from vector search
        bm25_results: Results from BM25 search
        dictionary_boosts: Optional dictionary boosts
        top_k: Number of results to return
        weight_vector: Weight for vector scores
        weight_bm25: Weight for BM25 scores
        weight_dictionary: Weight for dictionary boost
    
    Returns:
        Merged and reranked results
    """
    fusion = ScoreFusion(
        weight_vector=weight_vector,
        weight_bm25=weight_bm25,
        weight_dictionary=weight_dictionary
    )
    return fusion.merge_results(
        vector_results=vector_results,
        bm25_results=bm25_results,
        dictionary_boosts=dictionary_boosts,
        top_k=top_k
    )


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def rrf_fuse(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion — the single canonical RRF implementation.

    Fuses dense (vector) and sparse (BM25) result lists by rank position.
    Each document's score: sum(weight / (k + rank + 1)).

    Args:
        dense_results:  List of dicts with 'id' key (ranked best-first).
        sparse_results: List of dicts with 'id' key (ranked best-first).
        k:              RRF smoothing constant (default=60, empirically validated).
        dense_weight:   Contribution weight of dense rankings.
        sparse_weight:  Contribution weight of sparse rankings.

    Returns:
        Merged list sorted by descending fused RRF score,
        with 'score' key replaced by the fused value.
    """
    scores: Dict[str, float] = {}
    docs:   Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results):
        doc_id = str(doc.get("id", "") or doc.get("_id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight * (1.0 / (k + rank + 1))
        docs[doc_id]   = doc

    for rank, doc in enumerate(sparse_results):
        doc_id = str(doc.get("id", "") or doc.get("_id", ""))
        if not doc_id:
            continue
        scores[doc_id] = scores.get(doc_id, 0.0) + sparse_weight * (1.0 / (k + rank + 1))
        docs.setdefault(doc_id, doc)

    fused = []
    for doc_id, fused_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = dict(docs[doc_id])
        entry["score"] = fused_score
        fused.append(entry)

    return fused
