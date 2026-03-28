"""
Embedding service for generating vector embeddings.
Uses sentence-transformers for semantic embeddings.
"""

import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating and managing text embeddings.
    Uses sentence-transformers for semantic similarity.
    """
    
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        """
        Initialize embedding service with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the embedding model.
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a vector embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            numpy array of embeddings
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If model is not loaded
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts into vector embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
            
        Raises:
            ValueError: If texts list is empty
            RuntimeError: If model is not loaded
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Encoded {len(texts)} texts to embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            raise RuntimeError("Embedding model not loaded")
        
        return self.model.get_sentence_embedding_dimension()


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """
    Get or create the global embedding service instance.
    
    Returns:
        Embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
