"""
RAG (Retrieval Augmented Generation) schemas.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RAGQueryRequest(BaseModel):
    """
    Request schema for RAG query endpoint.
    
    Attributes:
        query: User question to answer using RAG
    """
    query: str = Field(..., min_length=1, max_length=500, description="User question")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Optional retrieval depth override")
    rules: List[str] = Field(default_factory=list, description="Optional app-provided response rules")
    definitions: List[str] = Field(default_factory=list, description="Optional app-provided domain definitions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the procedure for circuit breaker troubleshooting?",
                "top_k": 3,
                "rules": [
                    "Return steps as numbered list",
                    "Do not mention information outside context"
                ],
                "definitions": [
                    "MCB means miniature circuit breaker",
                    "Trip means automatic open due to fault"
                ]
            }
        }


class Source(BaseModel):
    """
    Source metadata for retrieved chunk.
    
    Attributes:
        text: The retrieved chunk text
        source: Source file or document name
        chunk_index: Index of the chunk
    """
    text: str
    source: Optional[str] = None
    chunk_index: Optional[int] = None


class RAGQueryResponse(BaseModel):
    """
    Response schema for RAG query endpoint.
    
    Attributes:
        answer: Generated answer from Groq LLM
        sources: List of source chunks used for generating the answer
        query: Original user query
    """
    answer: str = Field(..., description="Generated answer from LLM")
    sources: List[Source] = Field(default_factory=list, description="Source chunks used")
    query: str = Field(..., description="Original query")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Circuit breaker troubleshooting involves checking for tripped switches, verifying proper voltage...",
                "sources": [
                    {
                        "text": "Check if the breaker switch is in the OFF position...",
                        "source": "manual.pdf",
                        "chunk_index": 0
                    }
                ],
                "query": "What is the procedure for circuit breaker troubleshooting?"
            }
        }
