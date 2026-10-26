"""
neaps_hybrid_embedder.py
========================
Combines dense and sparse embeddings for a batch of chunks
into the format expected by NeapsQdrantStore.

Output per chunk:
  {
    "dense":  np.ndarray of shape (dim,),       # float32 dense vector
    "sparse": {"indices": [...], "values": [...]}  # BM25 sparse vector
  }
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def embed_chunks_hybrid(
    texts: List[str],
) -> List[Dict[str, Any]]:
    """
    Generate dense embeddings only. Sparse search is now handled offline.
    Returns empty sparse structure to maintain compatibility with Qdrant schema
    without rewriting the collection.
    """
    from embeddings.neaps_dense import embed_chunks

    if not texts:
        return []

    # Dense embeddings (reuses existing model)
    dense_embs: np.ndarray = embed_chunks(texts)

    # Return empty sparse structures (sparse search now handled by LocalIncrementalBM25)
    combined = [
        {"dense": dense_embs[i], "sparse": {"indices": [], "values": []}}
        for i in range(len(texts))
    ]

    logger.info(
        f"[HybridEmbedder] Produced {len(combined)} dense + empty sparse embeddings "
        f"(dense dim={dense_embs.shape[1]})"
    )
    return combined


def embed_query_hybrid(query: str) -> Dict[str, Any]:
    """
    Generate dense embedding for query. Sparse is empty placeholder.
    """
    from embeddings.neaps_dense import embed_query as dense_q

    return {
        "dense" : dense_q(query),
        "sparse": {"indices": [], "values": []},
    }
