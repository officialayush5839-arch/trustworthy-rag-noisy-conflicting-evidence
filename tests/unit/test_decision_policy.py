"""
Unit tests for DecisionPolicyEngine (Answer, Clarify, Abstain).
"""

from trustworthy_rag.config import PolicyThresholds
from trustworthy_rag.policy.decision import DecisionPolicyEngine
from trustworthy_rag.schemas import DecisionAction


def test_decision_policy_answer_branch():
    """All signals meet or exceed thresholds -> ANSWER."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.85,
        s_cov=0.90,
        s_con=0.05,
        s_cit=0.95,
    )
    assert result.action == DecisionAction.ANSWER
    assert result.evidence_score.composite_quality_score >= 0.70
    assert not result.evidence_score.is_contradictory


def test_decision_policy_contradiction_abstain():
    """High contradiction signal -> ABSTAIN with conflict reason."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.85,
        s_cov=0.90,
        s_con=0.88,  # > tau_con_max (0.70)
        s_cit=0.95,
    )
    assert result.action == DecisionAction.ABSTAIN
    assert "contradiction" in result.decision_reason.lower()
    assert result.evidence_score.is_contradictory


def test_decision_policy_irrelevant_abstain():
    """Low relevance signal -> ABSTAIN with relevance reason."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.40,  # < tau_ans_rel (0.65)
        s_cov=0.90,
        s_con=0.05,
        s_cit=0.95,
    )
    assert result.action == DecisionAction.ABSTAIN
    assert "relevance" in result.decision_reason.lower()


def test_decision_policy_missing_coverage_abstain():
    """Very low coverage -> ABSTAIN with coverage reason."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.80,
        s_cov=0.20,  # < tau_clar_min (0.40)
        s_con=0.05,
        s_cit=0.95,
        missing_entities=["CRISPR", "Nobel"],
    )
    assert result.action == DecisionAction.ABSTAIN
    assert "coverage" in result.decision_reason.lower()
    assert "CRISPR" in result.decision_reason


def test_decision_policy_partial_coverage_clarify():
    """Partial coverage between tau_clar_min and tau_cov_min -> CLARIFY."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.80,
        s_cov=0.50,  # in [0.40, 0.60)
        s_con=0.05,
        s_cit=0.95,
        missing_entities=["second Lagrange point"],
    )
    assert result.action == DecisionAction.CLARIFY
    assert "clarification" in result.decision_reason.lower()


def test_decision_policy_ungrounded_citations_abstain():
    """Draft answer fails citation entailment audit -> ABSTAIN."""
    engine = DecisionPolicyEngine()
    result = engine.evaluate(
        s_rel=0.85,
        s_cov=0.90,
        s_con=0.05,
        s_cit=0.50,  # < tau_cit_min (0.70)
    )
    assert result.action == DecisionAction.ABSTAIN
    assert "citation" in result.decision_reason.lower()
