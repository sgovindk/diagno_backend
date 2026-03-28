"""
Ping schema for status checking endpoint.
"""

from pydantic import BaseModel
from datetime import datetime


class PingRequest(BaseModel):
    """
    Request schema for ping endpoint.
    """
    pass


class PingResponse(BaseModel):
    """
    Response schema for ping endpoint.
    
    Attributes:
        status: Server status ("online" or "offline")
        timestamp: UTC timestamp of the response
    """
    status: str
    timestamp: datetime
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "status": "online",
                "timestamp": "2026-03-28T10:30:00Z"
            }
        }
