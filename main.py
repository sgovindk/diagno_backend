"""
Railway ASGI entrypoint compatibility module.
Allows platforms expecting `main:app` to start the FastAPI app.
"""

from app.main import app
