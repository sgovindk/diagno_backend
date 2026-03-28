"""
Security utilities for DiagnosticPilot Hybrid Copilot.
Includes CORS, rate limiting, and other security middleware.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address


def add_cors_middleware(app: FastAPI, origins: list[str]) -> None:
    """
    Add CORS middleware to the application.
    
    Args:
        app: FastAPI application instance
        origins: List of allowed origins
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def get_rate_limiter() -> Limiter:
    """
    Create and return a rate limiter instance.
    
    Returns:
        Configured Limiter instance
    """
    return Limiter(key_func=get_remote_address)
