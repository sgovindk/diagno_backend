"""
File ingestion service for user-uploaded manuals.
Extracts text, chunks content, embeds chunks, and stores vectors in FAISS.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional

from fastapi import UploadFile
from pypdf import PdfReader

from app.core.config import settings
from app.core.logging import get_logger
from app.db.faiss_index import get_faiss_index
from app.services.embedding_service import get_embedding_service
from app.utils.text_splitter import RecursiveTextSplitter

logger = get_logger(__name__)


class FileService:
    """
    Service that handles user manual ingestion into the RAG knowledge base.
    """

    def __init__(self) -> None:
        """Initialize file ingestion dependencies."""
        self.embedding_service = get_embedding_service()
        self.faiss_index = get_faiss_index()
        self.text_splitter = RecursiveTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    async def ingest_manual(self, upload_file: UploadFile) -> Dict[str, Any]:
        """
        Ingest a user-uploaded manual into FAISS.

        Args:
            upload_file: Uploaded PDF or text file.

        Returns:
            Ingestion summary.

        Raises:
            ValueError: If the file type/content is invalid.
            RuntimeError: If ingestion fails.
        """
        source_name = upload_file.filename or "unknown"
        extension = self._get_extension(source_name)

        if extension not in {".pdf", ".txt"}:
            raise ValueError("Only PDF and TXT files are supported")

        try:
            text_content = await self._extract_text(upload_file, extension)
            if not text_content.strip():
                raise ValueError("Uploaded file does not contain extractable text")

            chunks = self.text_splitter.split_text(text_content)
            if not chunks:
                raise ValueError("No valid chunks produced from uploaded file")

            embeddings = self.embedding_service.encode_texts(chunks)
            embedding_dim = int(embeddings.shape[1])

            if self.faiss_index.index is None:
                self.faiss_index.create_index(embedding_dim)
            elif self.faiss_index.embedding_dim != embedding_dim:
                raise RuntimeError("Embedding dimension mismatch with existing FAISS index")

            metadata = [
                {
                    "text": chunk,
                    "source": source_name,
                    "chunk_index": idx,
                }
                for idx, chunk in enumerate(chunks)
            ]

            self.faiss_index.add_embeddings(embeddings, metadata)
            self.faiss_index.save_index()

            index_size = int(self.faiss_index.index.ntotal) if self.faiss_index.index else 0
            logger.info(
                "Ingested file '%s' into FAISS with %s chunks. Index size: %s",
                source_name,
                len(chunks),
                index_size,
            )

            return {
                "message": "Manual uploaded and indexed successfully",
                "source": source_name,
                "chunks_added": len(chunks),
                "index_size": index_size,
            }
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Manual ingestion failed for '%s': %s", source_name, str(exc), exc_info=True)
            raise RuntimeError("Failed to process uploaded manual") from exc

    async def _extract_text(self, upload_file: UploadFile, extension: str) -> str:
        """
        Extract text from supported upload file types.

        Args:
            upload_file: Uploaded file.
            extension: File extension.

        Returns:
            Extracted text content.
        """
        content = await upload_file.read()

        if extension == ".txt":
            return self._decode_text(content)

        if extension == ".pdf":
            return self._extract_pdf_text(content)

        raise ValueError("Unsupported file type")

    def _extract_pdf_text(self, content: bytes) -> str:
        """
        Extract text from PDF bytes.

        Args:
            content: PDF raw bytes.

        Returns:
            Combined PDF text.
        """
        reader = PdfReader(BytesIO(content))
        page_texts = []
        for page in reader.pages:
            page_texts.append(page.extract_text() or "")
        return "\n".join(page_texts)

    def _decode_text(self, content: bytes) -> str:
        """
        Decode plain text file content.

        Args:
            content: Raw file bytes.

        Returns:
            Decoded text string.
        """
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore")

    def _get_extension(self, filename: str) -> str:
        """
        Extract lowercase file extension.

        Args:
            filename: Original filename.

        Returns:
            File extension including dot.
        """
        dot_index = filename.rfind(".")
        if dot_index == -1:
            return ""
        return filename[dot_index:].lower()


_file_service: Optional[FileService] = None


def get_file_service() -> FileService:
    """
    Get or create the global file ingestion service.

    Returns:
        FileService instance.
    """
    global _file_service
    if _file_service is None:
        _file_service = FileService()
    return _file_service
