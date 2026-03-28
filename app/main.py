"""
Main FastAPI application entry point.
DiagnosticPilot Hybrid Copilot Backend.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.security import add_cors_middleware
from app.api.v1 import router as v1_router
from app.db.faiss_index import get_faiss_index

# Setup logging
logger = setup_logging(
    level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE,
    app_name=settings.APP_NAME
)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="A hybrid AI system for electrical diagnostic assistance",
    version=settings.API_VERSION,
    debug=settings.DEBUG
)

# Add CORS middleware
add_cors_middleware(app, settings.CORS_ORIGINS)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions.
    
    Args:
        request: Request object
        exc: Exception instance
        
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# Include routers
app.include_router(v1_router.router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - returns API information.
    
    Returns:
        Dictionary with API metadata
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.API_VERSION,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint - comprehensive status check.
    
    Returns:
        Dictionary with health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.API_VERSION
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    Called when the application starts.
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.API_VERSION}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Server will run on {settings.HOST}:{settings.PORT}")

    faiss_index = get_faiss_index()
    loaded = faiss_index.load_index()
    if loaded:
        info = faiss_index.get_index_info()
        logger.info(
            "Loaded persisted FAISS index with %s vectors",
            info.get("num_vectors", 0),
        )
    else:
        logger.info("No persisted FAISS index found. Waiting for manual uploads.")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    Called when the application shuts down.
    """
    logger.info(f"Shutting down {settings.APP_NAME}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
