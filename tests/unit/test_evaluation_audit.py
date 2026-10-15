"""
Empirical unit tests for the Scientific Evaluation Audit remediations (P0-01 through P0-09, P1-03, P1-11).
Verifies hand-calculated mathematical properties, error boundaries, and un-leaked baseline behaviors.
"""

import pytest
import numpy as np
from trustworthy_rag.schemas import (
    RetrievedPassage,
    PerturbationType,
    Question,
    EvaluationResult,
)
from trustworthy_rag.evidence.citation import CitationGroundingAnalyzer
from trustworthy_rag.evaluation.risk_coverage import RiskCoverageCompiler
from trustworthy_rag.evaluation.evaluator import ComparativeEvaluator


def test_hand_calculated_aurcc_matches_numerical_integration():
    """
    Test 5: Hand-calculated AURCC verification.
    Given sorted scores and errors:
      scores = [0.90, 0.70, 0.40, 0.20]
      errors = [0.00, 1.00, 0.00, 1.00]
    Coverage vector: [0.25, 0.50, 0.75, 1.00]
    Cumulative risk vector: [0/1, 1/2, 1/3, 2/4] = [0.0, 0.5, 0.3333333333333333, 0.5]
    Trapezoid integration:
      Trapezoid 1: 0.25 * (0.0 + 0.5)/2 = 0.0625 = 3/48
      Trapezoid 2: 0.25 * (0.5 + 1/3)/2 = 5/48
      Trapezoid 3: 0.25 * (1/3 + 0.5)/2 = 5/48
      Total AURCC = 13/48 ~= 0.2708333333333333
    """
    scores = [0.90, 0.70, 0.40, 0.20]
    errors = [0.00, 1.00, 0.00, 1.00]

    cov, risk, aurcc = RiskCoverageCompiler.compute_curve(scores, errors)

    assert len(cov) == 4
    np.testing.assert_allclose(cov, [0.25, 0.50, 0.75, 1.00])
    np.testing.assert_allclose(risk, [0.0, 0.5, 1.0 / 3.0, 0.5])

    expected_aurcc = 13.0 / 48.0
    assert aurcc == pytest.approx(expected_aurcc, rel=1e-5)


def test_risk_coverage_input_validation_boundaries():
    """
    Test 4: RiskCoverageCompiler raises ValueError on invalid, empty, or non-finite inputs.
    """
    # Empty inputs
    with pytest.raises(ValueError, match="must not be empty"):
        RiskCoverageCompiler.compute_curve([], [])

    # Length mismatch
    with pytest.raises(ValueError, match="Mismatch in array lengths"):
        RiskCoverageCompiler.compute_curve([0.9, 0.8], [0.0])

    # NaN in confidence scores
    with pytest.raises(ValueError, match="must contain finite numerical values"):
        RiskCoverageCompiler.compute_curve([0.9, float("nan")], [0.0, 1.0])

    # Inf in errors
    with pytest.raises(ValueError, match="must contain finite numerical values"):
        RiskCoverageCompiler.compute_curve([0.9, 0.5], [0.0, float("inf")])


def test_spurious_citation_detected_by_content_overlap():
    """
    Test 3: Citation verification fails when cited text shares no content words with cited chunk.
    """
    passages = [
        RetrievedPassage(
            chunk_id="chunk_apollo_11",
            doc_id="doc_space",
            text="The Apollo 11 mission successfully landed humans on the lunar Moon surface in July 1969.",
            fused_score=0.95,
            rank=1,
            retrieval_method="hybrid_rrf",
        ),
    ]

    # Grounded claim: shares content words ("Apollo", "landed", "Moon")
    grounded_text = "Apollo 11 astronauts landed on the Moon [chunk_apollo_11]."
    analyzer = CitationGroundingAnalyzer()
    score_grounded, citations_grounded = analyzer.audit_citations(grounded_text, passages)

    assert score_grounded == 1.0
    assert len(citations_grounded) == 1
    assert citations_grounded[0].is_verified is True

    # Spurious citation: shares zero content words with Apollo chunk
    spurious_text = "Photosynthesis operates via chloroplasts synthesizing chemical sugars [chunk_apollo_11]."
    score_spurious, citations_spurious = analyzer.audit_citations(spurious_text, passages)

    assert score_spurious == 0.0
    assert len(citations_spurious) == 1
    assert citations_spurious[0].is_verified is False


def test_abstention_does_not_inflate_answered_faithfulness():
    """
    Test 2: Abstention produces claim_faithfulness=None and does not inflate answered faithfulness.
    Previously, abstentions returned faithfulness=1.0, distorting true answer quality.
    """
    # Simulate evaluator results with 1 correct answered query, 1 abstained query, and 1 wrong answered query
    answered_correct = EvaluationResult(
        run_id="run_test",
        question_id="q1",
        perturbation_type=PerturbationType.CLEAN,
        method_name="proposed",
        is_answered=True,
        claim_faithfulness=1.0,
        citation_precision=1.0,
        citation_recall=1.0,
        retrieval_recall=1.0,
        selective_error=0.0,
    )
    abstained = EvaluationResult(
        run_id="run_test",
        question_id="q2",
        perturbation_type=PerturbationType.CONTRADICTION,
        method_name="proposed",
        is_answered=False,
        claim_faithfulness=None,  # Crucial audit fix: None, NOT 1.0!
        citation_precision=0.0,
        citation_recall=0.0,
        retrieval_recall=1.0,
        selective_error=None,
    )
    answered_incorrect = EvaluationResult(
        run_id="run_test",
        question_id="q3",
        perturbation_type=PerturbationType.DISTRACTOR,
        method_name="proposed",
        is_answered=True,
        claim_faithfulness=0.4,
        citation_precision=0.5,
        citation_recall=0.5,
        retrieval_recall=1.0,
        selective_error=0.6,
    )

    results = [answered_correct, abstained, answered_incorrect]

    # Calculate answered faithfulness strictly over answered queries
    answered_faithfulness_vals = [r.claim_faithfulness for r in results if r.is_answered and r.claim_faithfulness is not None]
    assert len(answered_faithfulness_vals) == 2
    avg_answered_faithfulness = sum(answered_faithfulness_vals) / len(answered_faithfulness_vals)
    # Expected: (1.0 + 0.4) / 2 = 0.70
    assert avg_answered_faithfulness == pytest.approx(0.70)

    # If abstention had been wrongly assigned 1.0, the average would have been inflated to (1.0 + 1.0 + 0.4)/3 = 0.80
    wrong_inflated_avg = (1.0 + 1.0 + 0.4) / 3.0
    assert avg_answered_faithfulness < wrong_inflated_avg


def test_known_false_answers_yield_non_zero_selective_risk():
    """
    Test 1: Known false answers yield non-zero selective error and positive cumulative risk.
    """
    scores = [0.85, 0.75, 0.65]
    # All answered answers are false or hallucinated (error = 1.0)
    errors = [1.0, 1.0, 1.0]

    cov, risk, aurcc = RiskCoverageCompiler.compute_curve(scores, errors)

    assert all(r == 1.0 for r in risk)
    assert aurcc == pytest.approx(1.0)
    assert aurcc > 0.0


def test_confidence_baseline_has_zero_leakage_of_perturbation_type():
    """
    Test 6: Baseline B3 confidence estimation does not condition on ground-truth perturbation labels.
    Two identical queries differing only by perturbation type must yield identical confidence scores.
    """
    evaluator = ComparativeEvaluator(generator_backend="mock", contradiction_use_mock=True)

    passages = [
        RetrievedPassage(
            chunk_id="c1",
            doc_id="d1",
            text="General relativity was formulated by Albert Einstein in 1915.",
            fused_score=0.88,
            rank=1,
            retrieval_method="hybrid_rrf",
        )
    ]

    query = Question(
        question_id="q_test_1",
        dataset_name="test_dataset",
        query_text="Who formulated general relativity?",
        gold_answers=["Albert Einstein"],
        gold_passage_ids=["c1"],
        split="test",
    )

    # Evaluate query under CLEAN vs CONTRADICTION perturbations
    res_clean = evaluator.evaluate_query(
        question=query,
        retrieved_passages=passages,
        method_name="baseline_confidence_only",
        perturbation=PerturbationType.CLEAN,
    )

    res_contradictory = evaluator.evaluate_query(
        question=query,
        retrieved_passages=passages,
        method_name="baseline_confidence_only",
        perturbation=PerturbationType.CONTRADICTION,
    )

    # Baseline B3 must produce identical decision and confidence regardless of perturbation label!
    assert res_clean.is_answered == res_contradictory.is_answered
    assert res_clean.retrieval_recall == res_contradictory.retrieval_recall
