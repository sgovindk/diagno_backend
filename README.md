# DiagnosticPilot Hybrid Copilot Backend

Scalable FastAPI backend for hybrid diagnostics (offline TinyLlama app + online Groq RAG).

## Status

🚧 Work in progress — architecture is being expanded for production scale.

## Current Capabilities

- Health and connectivity check (`POST /v1/ping`)
- User manual ingestion (`POST /v1/upload/manual`)
- Groq + FAISS RAG querying (`POST /v1/rag/query`)

## Run

1. Install dependencies: `pip install -r requirements.txt`
2. Start server: `uvicorn app.main:app --reload`
