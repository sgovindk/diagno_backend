"""
Text splitting utilities for document chunking.
Provides a recursive splitter with configurable chunk size and overlap.
"""

from __future__ import annotations

from typing import List


class RecursiveTextSplitter:
    """
    Recursively split text using a separator hierarchy.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] | None = None,
    ) -> None:
        """
        Initialize the recursive text splitter.

        Args:
            chunk_size: Maximum chunk length in characters.
            chunk_overlap: Overlap length between adjacent chunks.
            separators: Ordered separator priority for recursive splitting.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive separator fallback.

        Args:
            text: Full text content.

        Returns:
            List of cleaned chunks.
        """
        if not text or not text.strip():
            return []

        raw_chunks = self._split_recursive(text.strip(), self.separators)
        cleaned = [chunk.strip() for chunk in raw_chunks if chunk and chunk.strip()]
        return self._apply_overlap(cleaned)

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text with fallback separators.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of chunks not exceeding chunk size.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split_recursive(text, separators[1:])

        chunks: List[str] = []
        current = ""

        for part in parts:
            candidate = part if not current else f"{current}{separator}{part}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(self._split_recursive(current, separators[1:]))

            if len(part) <= self.chunk_size:
                current = part
            else:
                chunks.extend(self._split_recursive(part, separators[1:]))
                current = ""

        if current:
            chunks.extend(self._split_recursive(current, separators[1:]))

        return chunks

    def _hard_split(self, text: str) -> List[str]:
        """
        Fallback splitter using fixed-size windows.

        Args:
            text: Text to split.

        Returns:
            Fixed-size chunks.
        """
        if len(text) <= self.chunk_size:
            return [text]

        step = max(1, self.chunk_size - self.chunk_overlap)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap to recursively split chunks.

        Args:
            chunks: Chunks without overlap.

        Returns:
            Chunks with overlap context prepended.
        """
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped: List[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk[: self.chunk_size])
                continue

            prev_tail = chunks[idx - 1][-self.chunk_overlap :]
            merged = f"{prev_tail} {chunk}".strip()
            overlapped.append(merged[: self.chunk_size])

        return overlapped
