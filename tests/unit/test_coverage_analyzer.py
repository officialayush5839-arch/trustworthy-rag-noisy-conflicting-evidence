"""
Unit tests for EvidenceCoverageAnalyzer and RelevanceAnalyzer.
"""

from trustworthy_rag.evidence.coverage import EvidenceCoverageAnalyzer
from trustworthy_rag.evidence.relevance import RelevanceAnalyzer
from trustworthy_rag.schemas import RetrievedPassage


def test_coverage_analyzer():
    analyzer = EvidenceCoverageAnalyzer()
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="Gemini was announced on December 6, 2023 by Google DeepMind.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf")
    ]

    query = "When was Gemini announced?"
    ratio, covered, missing = analyzer.compute_coverage(query, passages)
    assert ratio >= 0.8
    assert "gemini" in covered
    assert "announced" in covered


def test_coverage_analyzer_disjoint():
    analyzer = EvidenceCoverageAnalyzer()
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="Baking sourdough bread requires precise temperature.", fused_score=0.2, rank=1, retrieval_method="hybrid_rrf")
    ]

    query = "Who won the Nobel prize in Chemistry for CRISPR-Cas9?"
    ratio, covered, missing = analyzer.compute_coverage(query, passages)
    assert ratio == 0.0
    assert len(missing) > 0


def test_relevance_analyzer_decay():
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="text 1", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="c2", doc_id="d2", text="text 2", fused_score=0.2, rank=2, retrieval_method="hybrid_rrf"),
    ]
    score = RelevanceAnalyzer.compute_relevance(passages)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # Rank 1 carries higher log2 weight
