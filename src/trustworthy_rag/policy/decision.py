"""
Evidence-aware tri-state decision policy engine (Answer, Clarify, Abstain).
"""

from typing import Dict, List, Optional
from trustworthy_rag.config import PolicyThresholds
from trustworthy_rag.schemas import (
    DecisionAction,
    DecisionResult,
    EvidenceQualityScore,
)


class DecisionPolicyEngine:
    """Evaluates multi-signal evidence vectors against calibrated thresholds to emit policy actions."""

    def __init__(self, thresholds: Optional[PolicyThresholds] = None):
        self.thresholds = thresholds or PolicyThresholds()

    def evaluate(
        self,
        s_rel: float,
        s_cov: float,
        s_con: float,
        s_cit: float = 1.0,
        missing_entities: Optional[List[str]] = None,
    ) -> DecisionResult:
        """Evaluate evidence signals and return deterministic DecisionResult."""
        t = self.thresholds

        # Calculate composite quality score
        composite = (0.35 * s_rel) + (0.35 * s_cov) + (0.30 * s_cit) - (0.50 * s_con)
        composite_score = float(max(0.0, min(1.0, composite)))
        is_contradictory = s_con >= t.tau_con_max

        weights = {"rel": 0.35, "cov": 0.35, "cit": 0.30, "con": -0.50}
        evidence_score = EvidenceQualityScore(
            relevance_score=float(s_rel),
            coverage_score=float(s_cov),
            citation_support_score=float(s_cit),
            contradiction_score=float(s_con),
            composite_quality_score=composite_score,
            is_contradictory=is_contradictory,
            feature_weights=weights,
        )

        applied_thresholds = {
            "tau_ans_rel": t.tau_ans_rel,
            "tau_cov_min": t.tau_cov_min,
            "tau_clar_min": t.tau_clar_min,
            "tau_con_max": t.tau_con_max,
            "tau_cit_min": t.tau_cit_min,
        }

        # Policy Branch 1: Factual contradiction in evidence -> ABSTAIN
        if is_contradictory:
            return DecisionResult(
                action=DecisionAction.ABSTAIN,
                decision_reason=(
                    f"Irreconcilable factual contradiction detected among retrieved evidence passages "
                    f"(contradiction_score={s_con:.3f} >= threshold={t.tau_con_max:.2f})"
                ),
                evidence_score=evidence_score,
                applied_thresholds=applied_thresholds,
            )

        # Policy Branch 2: Irrelevant evidence -> ABSTAIN
        if s_rel < t.tau_ans_rel:
            return DecisionResult(
                action=DecisionAction.ABSTAIN,
                decision_reason=(
                    f"Insufficient evidence relevance to support a factual answer "
                    f"(relevance_score={s_rel:.3f} < threshold={t.tau_ans_rel:.2f})"
                ),
                evidence_score=evidence_score,
                applied_thresholds=applied_thresholds,
            )

        # Policy Branch 3: Insufficient coverage -> ABSTAIN
        if s_cov < t.tau_clar_min:
            missing_str = f" Missing: {', '.join(missing_entities)}" if missing_entities else ""
            return DecisionResult(
                action=DecisionAction.ABSTAIN,
                decision_reason=(
                    f"Essential query entities and factual predicates absent from evidence "
                    f"(coverage_score={s_cov:.3f} < threshold={t.tau_clar_min:.2f}).{missing_str}"
                ),
                evidence_score=evidence_score,
                applied_thresholds=applied_thresholds,
            )

        # Policy Branch 4: Partial coverage -> CLARIFY
        if s_cov < t.tau_cov_min:
            missing_str = f" Query regarding {', '.join(missing_entities)} is ambiguous." if missing_entities else ""
            return DecisionResult(
                action=DecisionAction.CLARIFY,
                decision_reason=(
                    f"Partial evidence retrieved; clarification required to resolve ambiguity "
                    f"(coverage_score={s_cov:.3f} in [{t.tau_clar_min:.2f}, {t.tau_cov_min:.2f})).{missing_str}"
                ),
                evidence_score=evidence_score,
                applied_thresholds=applied_thresholds,
            )

        # Policy Branch 5: Citation ungrounded draft answer -> ABSTAIN
        if s_cit < t.tau_cit_min:
            return DecisionResult(
                action=DecisionAction.ABSTAIN,
                decision_reason=(
                    f"Draft answer failed post-generation citation entailment audit "
                    f"(citation_support={s_cit:.3f} < threshold={t.tau_cit_min:.2f})"
                ),
                evidence_score=evidence_score,
                applied_thresholds=applied_thresholds,
            )

        # Policy Branch 6: Fully grounded, relevant, and consistent evidence -> ANSWER
        return DecisionResult(
            action=DecisionAction.ANSWER,
            decision_reason="Evidence is consistent, query fully covered, and citations verified.",
            evidence_score=evidence_score,
            applied_thresholds=applied_thresholds,
        )
