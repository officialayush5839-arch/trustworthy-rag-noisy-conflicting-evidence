"""
Retrieval relevance scoring and rank decay aggregator.
"""

import math
from typing import List
from trustworthy_rag.schemas import RetrievedPassage


class RelevanceAnalyzer:
    """Computes aggregate relevance score s_rel from top-K retrieval candidates."""

    @staticmethod
    def compute_relevance(passages: List[RetrievedPassage], top_k: int = 5) -> float:
        """Compute rank-decayed aggregate retrieval relevance score in [0.0, 1.0]."""
        if not passages:
            return 0.0

        candidates = passages[:top_k]
        weighted_sum = 0.0
        weight_normalizer = 0.0

        for i, p in enumerate(candidates, start=1):
            weight = 1.0 / math.log2(i + 1)
            score = max(0.0, min(1.0, p.fused_score))
            weighted_sum += weight * score
            weight_normalizer += weight

        if weight_normalizer == 0.0:
            return 0.0

        return float(min(1.0, max(0.0, weighted_sum / weight_normalizer)))
