"""
Upload endpoint for user manuals.
Supports PDF and TXT ingestion for RAG retrieval.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.logging import get_logger
from app.schemas.upload import ManualUploadResponse
from app.services.file_service import get_file_service

router = APIRouter()
logger = get_logger(__name__)


@router.post("/manual", response_model=ManualUploadResponse, status_code=status.HTTP_201_CREATED, tags=["Upload"])
async def upload_manual(file: UploadFile = File(...)) -> ManualUploadResponse:
    """
    Upload and index a user manual for RAG retrieval.

    Args:
        file: Uploaded PDF or TXT file.

    Returns:
        ManualUploadResponse with ingestion summary.

    Raises:
        HTTPException: If file is invalid or indexing fails.
    """
    try:
        if file.filename is None or not file.filename.strip():
            raise HTTPException(status_code=400, detail="File name is required")

        logger.info("Manual upload received: %s", file.filename)

        file_service = get_file_service()
        result = await file_service.ingest_manual(file)

        return ManualUploadResponse(**result)
    except ValueError as exc:
        logger.warning("Upload validation error: %s", str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Upload processing error: %s", str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected upload error: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Unexpected error while uploading manual") from exc
