"""
Document chunking engine with character-offset and provenance tracking.
"""

import hashlib
import re
from typing import List
from trustworthy_rag.schemas import DocumentChunk


class TextChunker:
    """Sliding-window text chunker preserving character offsets and provenance."""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, doc_id: str, text: str) -> List[DocumentChunk]:
        """Split text into overlapping chunks, tracking start/end character offsets."""
        clean_text = text.strip()
        if not clean_text:
            return []

        # Find word boundaries with character offsets using regex
        word_matches = list(re.finditer(r"\S+", text))
        if not word_matches:
            return []

        chunks: List[DocumentChunk] = []
        total_words = len(word_matches)
        step = self.chunk_size - self.chunk_overlap
        idx = 0

        for start_idx in range(0, total_words, step):
            end_idx = min(start_idx + self.chunk_size, total_words)
            if start_idx >= end_idx:
                break

            first_match = word_matches[start_idx]
            last_match = word_matches[end_idx - 1]

            start_char = first_match.start()
            end_char = last_match.end()
            chunk_slice = text[start_char:end_char]
            token_count = end_idx - start_idx

            chunk_id = f"{doc_id}_c{idx:04d}"
            sha256_hash = hashlib.sha256(chunk_slice.encode("utf-8")).hexdigest()

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_slice,
                start_char_offset=start_char,
                end_char_offset=end_char,
                token_count=token_count,
                sha256_hash=sha256_hash,
            )
            chunks.append(chunk)
            idx += 1

            if end_idx >= total_words:
                break

        return chunks
