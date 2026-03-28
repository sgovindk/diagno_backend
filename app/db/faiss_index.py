"""
FAISS index management for vector storage and retrieval.
Handles creating, saving, loading, and searching FAISS indexes.
"""

import os
import pickle
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import faiss
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FAISSIndex:
    """
    Manages FAISS vector index with metadata persistence.
    Stores embeddings and retrieves similar vectors.
    """
    
    def __init__(self):
        """Initialize FAISS index manager."""
        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_dim: Optional[int] = None
        self._ensure_data_directory()
    
    def _ensure_data_directory(self) -> None:
        """
        Ensure data directory exists for saving indexes.
        """
        data_dir = Path(settings.FAISS_INDEX_PATH).parent
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Data directory ensured at: {data_dir}")
    
    def create_index(self, embedding_dim: int) -> None:
        """
        Create a new FAISS index with specified embedding dimension.
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        try:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.embedding_dim = embedding_dim
            self.metadata = []
            logger.info(f"Created new FAISS index with dimension: {embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ) -> None:
        """
        Add embeddings to the index with associated metadata.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            metadata_list: List of metadata dicts for each embedding
            
        Raises:
            ValueError: If dimensions don't match
            RuntimeError: If index is not initialized
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call create_index first.")
        
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings and metadata must match")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"index dimension {self.embedding_dim}"
            )
        
        try:
            # Ensure embeddings are float32 (required by FAISS)
            embeddings = embeddings.astype(np.float32)
            self.index.add(embeddings)
            self.metadata.extend(metadata_list)
            logger.info(f"Added {len(embeddings)} embeddings to index")
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> Tuple[List[Dict], List[float]]:
        """
        Search for top-k similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            Tuple of (metadata_list, distances)
            
        Raises:
            RuntimeError: If index is empty or not initialized
            ValueError: If query embedding dimension doesn't match
        """
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        
        if self.index.ntotal == 0:
            raise RuntimeError("FAISS index is empty. No documents added yet.")
        
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} does not match "
                f"index dimension {self.embedding_dim}"
            )
        
        try:
            # Ensure query embedding is float32
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            # Limit k to available items
            k = min(k, self.index.ntotal)
            
            distances, indices = self.index.search(query_embedding, k)
            
            # Extract metadata for retrieved indices
            results = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    results.append(self.metadata[idx])
            
            distances_list = distances[0].tolist()
            
            logger.debug(f"Search returned {len(results)} results for query")
            return results, distances_list
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def save_index(
        self,
        index_path: str = settings.FAISS_INDEX_PATH,
        metadata_path: str = settings.FAISS_METADATA_PATH
    ) -> None:
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            raise RuntimeError("No index to save")
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            logger.info(f"Saved FAISS index to: {index_path}")
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'embedding_dim': self.embedding_dim
                }, f)
            logger.info(f"Saved metadata to: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
            raise
    
    def load_index(
        self,
        index_path: str = settings.FAISS_INDEX_PATH,
        metadata_path: str = settings.FAISS_METADATA_PATH
    ) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata file
            
        Returns:
            True if loaded successfully, False if files don't exist
        """
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Index files not found at {index_path}")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from: {index_path}")
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', [])
                self.embedding_dim = data.get('embedding_dim')
            
            logger.info(f"Loaded metadata from: {metadata_path}")
            logger.info(f"Index contains {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            raise
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "num_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "num_metadata": len(self.metadata)
        }
    
    def is_empty(self) -> bool:
        """
        Check if index is empty.
        
        Returns:
            True if index is empty or not initialized
        """
        return self.index is None or self.index.ntotal == 0


# Global FAISS index instance
_faiss_index: Optional[FAISSIndex] = None


def get_faiss_index() -> FAISSIndex:
    """
    Get or create the global FAISS index instance.
    
    Returns:
        FAISS index instance
    """
    global _faiss_index
    if _faiss_index is None:
        _faiss_index = FAISSIndex()
    return _faiss_index
