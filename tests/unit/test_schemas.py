"""
Unit tests for Pydantic data schemas and contract enforcement.
"""

import pytest
from pydantic import ValidationError

from trustworthy_rag.schemas import (
    AtomicClaim,
    Citation,
    ClaimStatus,
    ContradictionFinding,
    DecisionAction,
    DecisionResult,
    Document,
    DocumentChunk,
    EvaluationResult,
    EvidenceQualityScore,
    GenerationResult,
    PerturbationMetadata,
    PerturbationType,
    Question,
    RetrievedPassage,
)


def test_document_chunk_valid():
    chunk = DocumentChunk(
        chunk_id="doc_001_c0001",
        doc_id="doc_001",
        text="The quick brown fox jumps over the lazy dog.",
        start_char_offset=0,
        end_char_offset=44,
        token_count=9,
        sha256_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    assert chunk.chunk_id == "doc_001_c0001"
    assert chunk.token_count == 9

    # Check immutability (frozen config)
    with pytest.raises(ValidationError):
        chunk.token_count = 10  # type: ignore


def test_document_chunk_negative_offset_fails():
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="doc_001_c0001",
            doc_id="doc_001",
            text="invalid offsets",
            start_char_offset=-1,
            end_char_offset=10,
            token_count=2,
            sha256_hash="abc",
        )


def test_retrieved_passage_rank_bounds():
    with pytest.raises(ValidationError):
        RetrievedPassage(
            chunk_id="c1",
            doc_id="d1",
            text="sample text",
            fused_score=0.9,
            rank=0,  # Rank must be >= 1
            retrieval_method="bm25",
        )


def test_evidence_quality_score_ranges():
    valid_score = EvidenceQualityScore(
        relevance_score=0.85,
        coverage_score=0.90,
        citation_support_score=0.95,
        contradiction_score=0.05,
        composite_quality_score=0.88,
        is_contradictory=False,
    )
    assert valid_score.is_contradictory is False

    with pytest.raises(ValidationError):
        EvidenceQualityScore(
            relevance_score=1.5,  # Must be <= 1.0
            coverage_score=0.9,
            contradiction_score=0.1,
            composite_quality_score=0.8,
            is_contradictory=False,
        )


def test_decision_result_tri_state():
    score = EvidenceQualityScore(
        relevance_score=0.9,
        coverage_score=0.85,
        contradiction_score=0.02,
        composite_quality_score=0.88,
        is_contradictory=False,
    )
    decision = DecisionResult(
        action=DecisionAction.ANSWER,
        decision_reason="Strong relevant evidence, no contradictions",
        evidence_score=score,
        applied_thresholds={"tau_ans_rel": 0.65},
    )
    assert decision.action == DecisionAction.ANSWER
    assert decision.action.value in ["answer", "clarify", "abstain"]


def test_atomic_claim_and_evaluation_result():
    claim = AtomicClaim(
        claim_id="clm_001",
        claim_text="The company acquired the startup in 2021.",
        associated_citation_ids=["cit_001"],
        status=ClaimStatus.SUPPORTED,
        entailment_score=0.96,
        evaluator_method="nli_model",
    )
    eval_res = EvaluationResult(
        run_id="run_001",
        question_id="q_001",
        perturbation_type=PerturbationType.CLEAN,
        method_name="proposed_evidence_aware",
        is_answered=True,
        is_correct=True,
        claim_faithfulness=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        retrieval_recall=1.0,
        atomic_claims=[claim],
    )
    assert eval_res.claim_faithfulness == 1.0
    assert len(eval_res.atomic_claims) == 1
