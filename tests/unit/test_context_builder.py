"""
Unit tests for ContextBuilder and citation extraction.
"""

from trustworthy_rag.generation.context_builder import ContextBuilder
from trustworthy_rag.generation.generator import AnswerGenerator
from trustworthy_rag.schemas import RetrievedPassage


def test_context_builder_chunk_headers():
    passages = [
        RetrievedPassage(chunk_id="doc_01_c0001", doc_id="doc_01", text="Mercury is the smallest planet.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="doc_02_c0001", doc_id="doc_02", text="Venus has a dense atmosphere.", fused_score=0.8, rank=2, retrieval_method="hybrid_rrf"),
    ]

    builder = ContextBuilder()
    assembled = builder.assemble(passages)

    assert "[Chunk ID: doc_01_c0001]" in assembled.formatted_text
    assert "[Chunk ID: doc_02_c0001]" in assembled.formatted_text
    assert "doc_01_c0001" in assembled.chunk_offsets
    assert "doc_02_c0001" in assembled.chunk_offsets


def test_generator_extract_citations():
    generator = AnswerGenerator(backend="mock")
    text = "Mercury is smallest [doc_01_c0001], but Pluto was reclassified [doc_99_c0001]."
    citations = generator.extract_citations(text, valid_chunk_ids=["doc_01_c0001"])

    assert len(citations) == 2
    assert citations[0].chunk_id == "doc_01_c0001"
    assert citations[0].is_verified is True
    assert citations[1].chunk_id == "doc_99_c0001"
    assert citations[1].is_verified is False
