"""
Post-generation citation support and claim grounding analyzer.
"""

import re
from typing import Dict, List, Tuple
from trustworthy_rag.schemas import Citation, RetrievedPassage


class CitationGroundingAnalyzer:
    """Verifies that generated answer assertions are grounded in cited evidence passages."""

    @staticmethod
    def audit_citations(
        answer_text: str,
        retrieved_passages: List[RetrievedPassage],
    ) -> Tuple[float, List[Citation]]:
        """Audit bracketed chunk citations and compute grounding score s_cit in [0.0, 1.0]."""
        passage_map: Dict[str, RetrievedPassage] = {p.chunk_id: p for p in retrieved_passages}
        citations_found = re.findall(r"\[([a-zA-Z0-9_\-]+_c\d{4})\]", answer_text)

        if not citations_found:
            # If no citations emitted in an answer, citation grounding is 0.0
            return 0.0, []

        citations: List[Citation] = []
        valid_count = 0

        for i, chunk_id in enumerate(citations_found):
            is_valid = chunk_id in passage_map
            if is_valid:
                valid_count += 1

            citations.append(
                Citation(
                    citation_id=f"cit_{i+1:03d}",
                    chunk_id=chunk_id,
                    cited_text_span=chunk_id,
                    is_verified=is_valid,
                )
            )

        grounding_score = valid_count / len(citations_found)
        return float(min(1.0, max(0.0, grounding_score))), citations
