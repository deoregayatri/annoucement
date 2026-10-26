"""
Embedding model wrapper supporting local models only.
"""

import logging
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
from cachetools import LRUCache
import hashlib

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for embeddings supporting:
    - Local: BAAI/bge-large-en-v1.5
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        model_name: Optional[str] = None,
        device: str = "cpu",
        enable_cache: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model_path: Local path to model (if using local)
            model_name: Model name
            device: Device to run model on ('cpu' or 'cuda')
            enable_cache: Enable LRU cache for embeddings
        """
        self.dimension = settings.embedding_dimension
        self.batch_size = settings.embedding_batch_size
        
        # Initialize cache
        self.enable_cache = enable_cache and settings.enable_memory_cache
        self.cache: Optional[LRUCache] = None
        if self.enable_cache:
            self.cache = LRUCache(maxsize=settings.cache_max_size)
        
        # Load local model
        self.model = self._load_local_model(model_path, model_name, device)
        logger.info(f"Initialized local embeddings: {model_name or settings.embedding_model_name}")
    
    def _load_local_model(self, model_path, model_name, device):
        """Load local sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        
        model_path = model_path or settings.embedding_model_path
        model_name = model_name or settings.embedding_model_name
        
        try:
            if model_path.exists():
                logger.info(f"Loading model from local path: {model_path}")
                return SentenceTransformer(str(model_path), device=device)
            else:
                logger.warning(f"Local model not found, downloading: {model_name}")
                model = SentenceTransformer(model_name, device=device)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))
                return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _embed_local(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Generate embeddings using local model."""
        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise
    
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length
            show_progress: Show progress bar for batch encoding
        
        Returns:
            Numpy array of embeddings (shape: [n_texts, dimension])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.enable_cache and self.cache is not None:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self._embed_local(uncached_texts, normalize, show_progress)
            
            # Cache new embeddings
            if self.enable_cache and self.cache is not None:
                for text, emb in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.cache[cache_key] = emb
            
            # Merge cached and new embeddings
            if embeddings:
                result = np.empty((len(texts), self.dimension), dtype=np.float32)
                cached_idx = 0
                uncached_idx = 0
                for i in range(len(texts)):
                    if i in uncached_indices:
                        result[i] = new_embeddings[uncached_idx]
                        uncached_idx += 1
                    else:
                        result[i] = embeddings[cached_idx]
                        cached_idx += 1
                return result
            else:
                return new_embeddings
        else:
            # All embeddings were cached
            return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for search query.
        
        Args:
            query: Search query text
        
        Returns:
            Query embedding (1D array)
        """
        return self.embed(query, normalize=True)[0]
    
    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of document texts
            show_progress: Show progress bar
        
        Returns:
            Document embeddings (2D array)
        """
        return self.embed(documents, normalize=True, show_progress=show_progress)
    
    def clear_cache(self):
        """Clear embedding cache."""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Embedding cache cleared")
    
    def get_cache_info(self) -> dict:
        """Get cache statistics."""
        if self.cache is None:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "mode": "Local"
        }


# Global embedding model instance (lazy loading)
_embedding_model: Optional[EmbeddingModel] = None


def invalidate_embedding_model():
    """Invalidate cached embedding model instance to force reload with new settings."""
    global _embedding_model
    _embedding_model = None
    logger.info("Embedding model cache invalidated")


def get_embedding_model(force_reload: bool = False) -> EmbeddingModel:
    """Get or create global embedding model instance."""
    global _embedding_model
    if _embedding_model is None or force_reload:
        _embedding_model = EmbeddingModel()
        logger.info(f"Created new embedding model instance")
    return _embedding_model
