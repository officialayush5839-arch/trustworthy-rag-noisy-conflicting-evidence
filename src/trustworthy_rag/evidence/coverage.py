"""
Evidence entity and predicate coverage analyzer.
"""

import re
from typing import List, Set, Tuple
from trustworthy_rag.schemas import RetrievedPassage

STOP_WORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "and", "or", "not", "can", "could", "should",
    "detail", "factual", "specifics", "according", "evidence", "confirm",
}


class EvidenceCoverageAnalyzer:
    """Analyzes whether essential query entities and predicate tokens are represented in retrieved context."""

    @staticmethod
    def extract_salient_tokens(query: str) -> Set[str]:
        """Extract content tokens (excluding stopwords and punctuation) from query."""
        tokens = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())
        salient = {t for t in tokens if t not in STOP_WORDS}
        return salient

    def compute_coverage(self, query: str, passages: List[RetrievedPassage]) -> Tuple[float, Set[str], Set[str]]:
        """Compute coverage score = |Query Entities in Context| / |Total Query Entities|."""
        salient_tokens = self.extract_salient_tokens(query)
        if not salient_tokens:
            return 1.0, set(), set()

        if not passages:
            return 0.0, set(), salient_tokens

        combined_context = " ".join(p.text.lower() for p in passages)
        covered_tokens = {t for t in salient_tokens if t in combined_context}
        missing_tokens = salient_tokens - covered_tokens

        coverage_ratio = len(covered_tokens) / len(salient_tokens)
        return float(min(1.0, max(0.0, coverage_ratio))), covered_tokens, missing_tokens
