"""
API v1 router - aggregates all v1 endpoints.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import ping, rag, upload

# Create router for v1 endpoints
router = APIRouter(prefix="/v1")

# Include ping endpoint
router.include_router(ping.router)

# Include RAG endpoint
router.include_router(rag.router, prefix="/rag")

# Include upload endpoint
router.include_router(upload.router, prefix="/upload")
