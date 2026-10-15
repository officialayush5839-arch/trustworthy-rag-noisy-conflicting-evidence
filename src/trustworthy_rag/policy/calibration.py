"""
Adaptive Dynamic Evidence Router and Continuous Policy Calibrator.
Dynamically scales decision thresholds based on query entropy and retriever score dispersion,
preventing false abstentions on complex paraphrased queries (mitigating ERR-FA-01).
"""

from __future__ import annotations

import math
from typing import List, Tuple
from pydantic import BaseModel

from trustworthy_rag.config import PolicyThresholds
from trustworthy_rag.schemas import RetrievedPassage


class AdaptiveThresholds(BaseModel):
    tau_ans_rel: float
    tau_cov_min: float
    tau_con_max: float
    query_entropy: float
    score_dispersion: float


class DynamicEvidenceRouter:
    """Computes query-specific entropy and score dispersion to dynamically calibrate thresholds."""

    def __init__(self, base_thresholds: PolicyThresholds):
        self.base = base_thresholds

    @staticmethod
    def compute_query_entropy(query: str) -> float:
        """Compute Shannon entropy of the query token distribution."""
        tokens = [t.lower() for t in query.split() if t.isalnum()]
        if not tokens:
            return 0.0

        freqs: dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1

        entropy = 0.0
        n = len(tokens)
        for count in freqs.values():
            p = count / n
            entropy -= p * math.log2(p)

        return float(entropy)

    @staticmethod
    def compute_score_dispersion(passages: List[RetrievedPassage]) -> float:
        """Compute score gap between top-1 candidate and trailing candidate distribution."""
        if not passages:
            return 0.0
        if len(passages) == 1:
            return float(passages[0].fused_score)

        top_score = passages[0].fused_score
        trailing_mean = sum(p.fused_score for p in passages[1:]) / len(passages[1:])
        dispersion = max(0.0, top_score - trailing_mean)
        return float(dispersion)

    def calibrate(
        self,
        query: str,
        passages: List[RetrievedPassage],
    ) -> AdaptiveThresholds:
        """Dynamically adapt decision thresholds based on query and retrieval dynamics."""
        entropy = self.compute_query_entropy(query)
        dispersion = self.compute_score_dispersion(passages)

        # High dispersion (strong top-1 hit) allows a minor reduction in tau_ans_rel
        # to prevent unnecessary abstention on valid paraphrases
        dispersion_factor = math.tanh(dispersion * 2.0) * 0.15
        adapted_tau_ans = max(0.20, self.base.tau_ans_rel - dispersion_factor)

        # High query entropy (complex multi-token question) scales coverage requirement
        entropy_factor = min(0.10, entropy * 0.02)
        adapted_tau_cov = max(0.30, self.base.tau_cov_min - entropy_factor)

        return AdaptiveThresholds(
            tau_ans_rel=round(adapted_tau_ans, 4),
            tau_cov_min=round(adapted_tau_cov, 4),
            tau_con_max=round(self.base.tau_con_max, 4),
            query_entropy=round(entropy, 4),
            score_dispersion=round(dispersion, 4),
        )
