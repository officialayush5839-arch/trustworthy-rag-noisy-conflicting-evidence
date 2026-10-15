"""
Unit tests for ClaimFaithfulnessEvaluator and atomic claim decomposition.
"""

from trustworthy_rag.evaluation.claims import ClaimDecomposer, ClaimFaithfulnessEvaluator
from trustworthy_rag.schemas import ClaimStatus, RetrievedPassage


def test_claim_decomposer():
    text = "Apollo 11 landed on the Moon in 1969 [doc_01_c0001]. Neil Armstrong was commander [doc_01_c0002]."
    decomposer = ClaimDecomposer()
    claims = decomposer.extract_atomic_claims(text)

    assert len(claims) == 2
    assert "Apollo 11 landed" in claims[0][0]
    assert "doc_01_c0001" in claims[0][1]
    assert "Neil Armstrong was commander" in claims[1][0]
    assert "doc_01_c0002" in claims[1][1]


def test_claim_faithfulness_supported():
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="Apollo 11 landed on the Moon in 1969.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf")
    ]
    text = "Apollo 11 landed on the Moon in 1969 [c1]."
    evaluator = ClaimFaithfulnessEvaluator()
    faithfulness, atomic_claims = evaluator.evaluate_claims(text, passages)

    assert faithfulness == 1.0
    assert len(atomic_claims) == 1
    assert atomic_claims[0].status == ClaimStatus.SUPPORTED


def test_claim_faithfulness_contradicted():
    passages = [
        RetrievedPassage(chunk_id="c1", doc_id="d1", text="Apollo 11 landed on the Moon in 1969.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf")
    ]
    text = "Apollo 11 failed to land on the Moon [c1]."
    evaluator = ClaimFaithfulnessEvaluator()
    faithfulness, atomic_claims = evaluator.evaluate_claims(text, passages)

    assert faithfulness == 0.0
    assert len(atomic_claims) == 1
    assert atomic_claims[0].status == ClaimStatus.CONTRADICTED
