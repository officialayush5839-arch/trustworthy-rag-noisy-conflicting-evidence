"""
Unit tests for BM25Retriever.
"""

from trustworthy_rag.retrieval.bm25 import BM25Retriever
from trustworthy_rag.schemas import DocumentChunk


def test_bm25_retrieval_ranking():
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            text="The capital of France is Paris. It is known for the Eiffel Tower.",
            start_char_offset=0,
            end_char_offset=65,
            token_count=13,
            sha256_hash="h1",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            text="Tokyo is the capital of Japan. It is known for its bustling districts.",
            start_char_offset=0,
            end_char_offset=70,
            token_count=13,
            sha256_hash="h2",
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            text="Quantum computing utilizes qubits and superposition to process calculations.",
            start_char_offset=0,
            end_char_offset=76,
            token_count=9,
            sha256_hash="h3",
        ),
    ]

    retriever = BM25Retriever()
    retriever.index(chunks)

    # Query matching Paris
    results = retriever.retrieve("What is the capital of France?", top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].sparse_score is not None
    assert results[0].sparse_score > 0
    assert results[0].rank == 1


def test_bm25_empty_query():
    retriever = BM25Retriever()
    retriever.index([])
    assert retriever.retrieve("") == []
