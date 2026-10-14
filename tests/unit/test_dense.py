"""
Unit tests for DenseRetriever.
"""

from trustworthy_rag.retrieval.dense import DenseRetriever
from trustworthy_rag.schemas import DocumentChunk


def test_dense_retrieval_ranking():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Artificial intelligence and machine learning models process massive data.",
            start_char_offset=0,
            end_char_offset=74,
            token_count=9,
            sha256_hash="h1",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            text="Photosynthesis is the biological process used by plants to convert light energy.",
            start_char_offset=0,
            end_char_offset=80,
            token_count=11,
            sha256_hash="h2",
        ),
    ]

    retriever = DenseRetriever(use_mock_fallback=True)
    retriever.index(chunks)

    results = retriever.retrieve("Tell me about machine learning intelligence", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].dense_score is not None
    assert 0.0 <= results[0].dense_score <= 1.0


def test_dense_empty_retrieval():
    retriever = DenseRetriever(use_mock_fallback=True)
    assert retriever.retrieve("") == []
