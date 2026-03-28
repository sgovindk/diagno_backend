"""
Configuration module for DiagnosticPilot Hybrid Copilot.
Loads settings from environment variables using Pydantic.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    
    # API Configuration
    API_VERSION: str = "v1"
    APP_NAME: str = "DiagnosticPilot Hybrid Copilot"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Groq API Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TIMEOUT: int = 30
    GROQ_SYSTEM_PROMPT: str = (
        "You are an expert electrical diagnostic assistant. "
        "Answer only using the provided context. "
        "If unsure, say you don't know. "
        "Keep responses concise, accurate, and safe."
    )
    GROQ_UNKNOWN_ANSWER: str = "I don't know based on the provided documents."
    
    # RAG Configuration
    TOP_K: int = 3
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    MAX_CONTEXT_CHARS: int = 6000
    
    # FAISS Configuration
    FAISS_INDEX_PATH: str = "data/faiss_index.pkl"
    FAISS_METADATA_PATH: str = "data/metadata.pkl"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
