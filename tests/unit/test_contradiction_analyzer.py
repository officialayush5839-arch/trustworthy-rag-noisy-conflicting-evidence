"""
Unit tests for ContradictionAnalyzer.
"""

from trustworthy_rag.evidence.contradiction import ContradictionAnalyzer
from trustworthy_rag.schemas import RetrievedPassage


def test_contradiction_detected_on_conflict():
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="Apollo 11 landed humans on the Moon on July 20, 1969.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="conflict_c1", doc_id="d1", text="Apollo 11 failed to land on the Moon due to an aborted descent on July 20, 1984.", fused_score=0.98, rank=2, retrieval_method="contradiction_injection"),
    ]

    analyzer = ContradictionAnalyzer(use_mock_fallback=True)
    con_score, findings = analyzer.analyze(passages)

    assert con_score >= 0.85
    assert len(findings) > 0
    assert findings[0].contradiction_probability >= 0.85


def test_consistent_passages_low_contradiction():
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="The James Webb Telescope was launched in 2021.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="c2", doc_id="d1", text="JWST observes the universe in infrared wavelengths.", fused_score=0.9, rank=2, retrieval_method="hybrid_rrf"),
    ]

    analyzer = ContradictionAnalyzer(use_mock_fallback=True)
    con_score, findings = analyzer.analyze(passages)

    assert con_score < 0.20
    assert len(findings) == 0
