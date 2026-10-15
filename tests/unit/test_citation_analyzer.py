"""
Unit tests for CitationGroundingAnalyzer.
"""

from trustworthy_rag.evidence.citation import CitationGroundingAnalyzer
from trustworthy_rag.schemas import RetrievedPassage


def test_citation_grounding_valid():
    passages = [
        RetrievedPassage(chunk_id="doc_tech_c0001", doc_id="d1", text="Gemini announced December 2023.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="doc_tech_c0002", doc_id="d1", text="Transformer proposed in 2017.", fused_score=0.9, rank=2, retrieval_method="hybrid_rrf"),
    ]
    text = "Gemini launched in late 2023 [doc_tech_c0001], following transformer foundations [doc_tech_c0002]."

    analyzer = CitationGroundingAnalyzer()
    score, citations = analyzer.audit_citations(text, passages)

    assert score == 1.0
    assert len(citations) == 2
    assert all(c.is_verified for c in citations)


def test_citation_grounding_unverified_citation():
    passages = [
        RetrievedPassage(chunk_id="doc_tech_c0001", doc_id="d1", text="Gemini announced December 2023.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
    ]
    # Includes hallucinated chunk citation doc_fake_c9999
    text = "Gemini was announced [doc_tech_c0001] but ungrounded claim here [doc_fake_c9999]."

    analyzer = CitationGroundingAnalyzer()
    score, citations = analyzer.audit_citations(text, passages)

    assert score == 0.5
    assert citations[0].is_verified is True
    assert citations[1].is_verified is False
