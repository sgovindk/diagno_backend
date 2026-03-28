"""
Ping endpoint - Ultra-lightweight online status check.
Used by Flutter app to determine online/offline mode.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from app.schemas.ping import PingRequest, PingResponse
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/ping", response_model=PingResponse, tags=["Health"])
async def ping() -> PingResponse:
    """
    Health check endpoint - returns server status and timestamp.
    
    Ultra-lightweight endpoint used by clients to check if the server is online.
    Responds immediately without any heavy processing.
    
    Returns:
        PingResponse: Server status and current UTC timestamp
        
    Raises:
        HTTPException: If there's an internal error
    """
    try:
        # Get current UTC timestamp
        current_time = datetime.now(timezone.utc)
        
        logger.debug("Ping request received")
        
        return PingResponse(
            status="online",
            timestamp=current_time
        )
    except Exception as e:
        logger.error(f"Ping endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
