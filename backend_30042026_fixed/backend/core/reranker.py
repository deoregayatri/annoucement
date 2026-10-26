"""
MS-Marco reranker using cross-encoder/ms-marco-MiniLM-L-6-v2.
Reranks retrieved chunks for improved relevance.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    MS-Marco reranker using cross-encoder for passage reranking.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_path: Optional[Path] = None,
        batch_size: int = 32,
    ):
        """
        Initialize MS-Marco reranker.

        Args:
            model_name: Model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
            model_path: Local model path (optional)
            batch_size: Batch size for reranking
        """
        self.model_name = (
            model_name
            or settings.reranker_model_name
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        
        # Only load model if reranking is enabled
        if settings.use_reranker:
            try:
                self.model_name = settings.reranker_model_name
                self.model_path = settings.reranker_model_path

                self._load_model()
                logger.info(f"Initialized MS-Marco reranker: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                logger.warning("Reranker will return neutral scores")
        else:
            logger.info("Reranker disabled in settings")
    
    def _load_model(self):
        """Load cross-encoder model."""
        from sentence_transformers import CrossEncoder
        
        try:
            logger.info(f'self.model_path {self.model_path}')
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading reranker from local path: {self.model_path}")
                self.model = CrossEncoder(str(self.model_path))
            else:
                logger.info(f"Loading reranker from HuggingFace: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
                
                # Save locally if path is configured
                if self.model_path:
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.model_path))
                    logger.info(f"Saved reranker model to: {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self, query: str, passages: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages using cross-encoder.

        Args:
            query: Query text
            passages: List of passage texts
            top_k: Number of top results to return

        Returns:
            List of dicts with 'index', 'score', 'text'
        """
        if not settings.use_reranker or self.model is None:
            logger.debug("Reranker disabled or not loaded, returning neutral scores")
            # Use index position as a proxy score � rank-0 gets highest score
            return [
                {"index": i, "score": 1.0 / (i + 1), "text": text}
                for i, text in enumerate(passages[:top_k] if top_k else passages)
            ]

        try:
            # Create query-passage pairs
            pairs = [[query, passage] for passage in passages]

            # Score with cross-encoder
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Create results with scores
            results = [
                {"index": i, "score": float(score), "text": text}
                for i, (score, text) in enumerate(zip(scores, passages))
            ]

            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)

            # Return top-k
            if top_k:
                results = results[:top_k]

            logger.debug(
                f"Reranked {len(passages)} passages, returning top {len(results)}"
            )

            return results

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original order with neutral scores
            return [
                {"index": i, "score": 0.5, "text": text}
                for i, text in enumerate(passages[:top_k] if top_k else passages)
            ]

    def rerank_with_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank document dicts using cross-encoder.

        Args:
            query: Query text
            documents: List of document dicts
            text_key: Key in dict containing text
            top_k: Number of top results to return

        Returns:
            List of documents sorted by rerank score
        """
        if not documents:
            return []

        if not settings.use_reranker or self.model is None:
            logger.debug(
                "Reranker disabled using existing fused/dense score as confidence proxy"
            )
            for doc in documents:
                # Prefer the actual vector/fused score over a hardcoded constant.
                # fused_score from RRF is ~[0, 0.016]; score from Qdrant dense is ~[0, 1].
                # We take whichever is available and keep it as-is so callers can
                # interpret it correctly (sigmoid in rag_graph.py handles the mapping).
                existing = doc.get("fused_score") or doc.get("score")
                doc["rerank_score"] = float(existing) if existing is not None else None
            return documents[:top_k] if top_k else documents

        try:
            # Extract texts
            texts = [doc.get(text_key, "") for doc in documents]

            # Create query-passage pairs
            pairs = [[query, text] for text in texts]

            # Score with cross-encoder
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Add rerank scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            # Sort by rerank score (descending)
            documents.sort(key=lambda x: x["rerank_score"], reverse=True)

            logger.debug(
                f"Reranked {len(documents)} documents, "
                f"top score: {documents[0]['rerank_score']:.3f}"
            )

            # Return top-k
            if top_k:
                return documents[:top_k]

            return documents

        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            # Fallback: return original order, preserving existing scores
            for doc in documents:
                existing = doc.get("fused_score") or doc.get("score")
                doc["rerank_score"] = float(existing) if existing is not None else None
            return documents[:top_k] if top_k else documents

    def get_info(self) -> Dict[str, Any]:
        """Get reranker information."""
        return {
            "enabled": settings.use_reranker,
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "batch_size": self.batch_size,
        }


# Global reranker instance (lazy loading)
_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    """Get or create global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
        logger.info("Created new MS-Marco reranker instance")
    return _reranker


def invalidate_reranker():
    """Invalidate cached reranker to force reload."""
    global _reranker
    _reranker = None
    logger.info("Reranker cache invalidated")
