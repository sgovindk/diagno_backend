"""
RAG query endpoint - Retrieval Augmented Generation.
Retrieves relevant documents and generates LLM responses.
"""

from fastapi import APIRouter, HTTPException, Query
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse
from app.services.rag_service import get_rag_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    RAG (Retrieval Augmented Generation) query endpoint.
    
    Accepts a user question, retrieves relevant documents from the FAISS index,
    and generates a contextual answer using Groq LLM.
    
    Args:
        request: RAG query request with user question
        
    Returns:
        RAGQueryResponse: Generated answer with source citations
        
    Raises:
        HTTPException: If query fails or index is not initialized
    """
    try:
        question = request.query.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"RAG query received: {question[:50]}...")
        
        # Get RAG service and execute query
        rag_service = get_rag_service()
        answer, sources = await rag_service.query(question)
        
        response = RAGQueryResponse(
            answer=answer,
            sources=sources,
            query=question
        )
        
        logger.debug(f"RAG query response prepared with {len(sources)} sources")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in RAG query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing RAG query"
        )
