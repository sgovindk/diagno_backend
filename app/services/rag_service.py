"""
RAG service - orchestrates RAG pipeline.
Handles query embedding, retrieval, and LLM generation.
"""

from typing import List, Tuple
from app.services.embedding_service import get_embedding_service
from app.services.groq_service import get_groq_service
from app.db.faiss_index import get_faiss_index
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.rag import Source

logger = get_logger(__name__)


class RAGService:
    """
    Orchestrates the RAG (Retrieval Augmented Generation) pipeline.
    """
    
    def __init__(self):
        """Initialize RAG service with dependencies."""
        self.embedding_service = get_embedding_service()
        self.groq_service = get_groq_service()
        self.faiss_index = get_faiss_index()
    
    async def query(self, question: str, top_k: int = None) -> Tuple[str, List[Source]]:
        """
        Execute RAG pipeline: embed query -> retrieve -> generate answer.
        
        Args:
            question: User question
            top_k: Number of top results to retrieve
            
        Returns:
            Tuple of (answer, sources)
            
        Raises:
            ValueError: If question is empty or index is empty
            RuntimeError: If any service fails
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if top_k is None:
            top_k = settings.TOP_K
        
        # Check if FAISS index is initialized
        if self.faiss_index.is_empty():
            raise RuntimeError(
                "FAISS index is empty. Please upload documents first using /v1/upload/manual"
            )
        
        try:
            logger.info(f"Processing RAG query: {question[:50]}...")
            
            # Step 1: Embed the query
            logger.debug("Step 1: Embedding query...")
            query_embedding = self.embedding_service.encode_text(question)
            
            # Step 2: Search FAISS for top-k similar chunks
            logger.debug(f"Step 2: Searching for top-{top_k} similar chunks...")
            metadata_list, distances = self.faiss_index.search(query_embedding, top_k)
            
            if not metadata_list:
                raise RuntimeError("No relevant documents found in index")
            
            # Step 3: Construct context from retrieved chunks
            logger.debug("Step 3: Constructing context...")
            context_parts = []
            sources = []
            
            for i, metadata in enumerate(metadata_list):
                chunk_text = metadata.get('text', '')
                source_name = metadata.get('source', 'Unknown')
                chunk_index = metadata.get('chunk_index', i)
                
                context_parts.append(f"[Document {i+1}] {chunk_text}")
                sources.append(Source(
                    text=chunk_text,
                    source=source_name,
                    chunk_index=chunk_index
                ))
            
            context = "\n\n".join(context_parts)
            
            # Step 4: Generate answer using Groq LLM
            logger.debug("Step 4: Generating answer with Groq LLM...")
            answer = await self.groq_service.generate_answer_async(
                query=question,
                context=context,
                temperature=0.7,
                max_tokens=500
            )
            
            logger.info(f"RAG query completed successfully")
            return answer, sources
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            raise


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance.
    
    Returns:
        RAG service instance
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
