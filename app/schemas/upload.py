"""
Schemas for manual upload endpoint.
"""

from pydantic import BaseModel, Field


class ManualUploadResponse(BaseModel):
    """
    Response schema for manual upload ingestion.

    Attributes:
        message: Result message.
        source: Original file name.
        chunks_added: Number of chunks ingested.
        index_size: Total vectors in FAISS index after ingestion.
    """

    message: str = Field(..., description="Upload status message")
    source: str = Field(..., description="Uploaded file name")
    chunks_added: int = Field(..., ge=0, description="Number of chunks added")
    index_size: int = Field(..., ge=0, description="Current FAISS index size")
