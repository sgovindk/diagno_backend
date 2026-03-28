"""
Dependency injection module for DiagnosticPilot Hybrid Copilot.
Provides shared dependencies for endpoints.
"""

from typing import Generator
from app.core.logging import get_logger


async def get_logger_dependency() -> Generator:
    """
    Dependency to provide logger to endpoints.
    
    Yields:
        Logger instance
    """
    yield get_logger(__name__)
