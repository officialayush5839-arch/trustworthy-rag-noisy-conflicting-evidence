"""
Unit tests for the sliding-window TextChunker.
"""

import pytest
from trustworthy_rag.data.chunker import TextChunker


def test_chunker_offset_reconstruction():
    text = (
        "Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of "
        "LLM responses by grounding the model on external sources of knowledge."
    )
    chunker = TextChunker(chunk_size=10, chunk_overlap=3)
    chunks = chunker.chunk_text(doc_id="test_doc", text=text)

    assert len(chunks) > 0

    for chunk in chunks:
        # Verify slice matches text exactly using offsets
        extracted = text[chunk.start_char_offset:chunk.end_char_offset]
        assert extracted == chunk.text
        assert chunk.token_count <= 10
        assert chunk.chunk_id.startswith("test_doc_c")
        assert len(chunk.sha256_hash) == 64


def test_chunker_empty_text():
    chunker = TextChunker()
    assert chunker.chunk_text("d1", "") == []
    assert chunker.chunk_text("d1", "   \n\t  ") == []


def test_chunker_invalid_parameters():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, chunk_overlap=15)
