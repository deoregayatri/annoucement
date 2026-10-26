"""
neaps_dense.py
==============
Dense embedding wrapper for NEAPS chunks.

Reuses the existing core/embeddings.py singleton (BAAI/bge-large-en-v1.5).
No new model is loaded - avoids wasting memory.
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def embed_chunks(texts: List[str]) -> np.ndarray:
    """
    Generate dense embeddings for a list of chunk texts.

    Args:
        texts: List of chunk text strings.

    Returns:
        2-D numpy array of shape (N, embedding_dim).
    """
    from core.embeddings import get_embedding_model

    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model = get_embedding_model()
    embeddings = model.embed_documents(texts, show_progress=len(texts) > 10)
    logger.info(f"[NeapsDense] Embedded {len(texts)} chunks  shape={embeddings.shape}")
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Generate dense embedding for a single query string.

    Returns:
        1-D numpy array of shape (embedding_dim,).
    """
    from core.embeddings import get_embedding_model

    model = get_embedding_model()
    return model.embed_query(query)


def get_dense_embedder():
    """
    Return the raw SentenceTransformer model for subprocess workers.

    Used by neaps_ingestion_service._subprocess_init() to pre-load the
    embedding model once per subprocess, avoiding repeated model loads.

    Returns:
        sentence_transformers.SentenceTransformer instance
    """
    from core.embeddings import get_embedding_model

    wrapper = get_embedding_model()
    return wrapper.model
