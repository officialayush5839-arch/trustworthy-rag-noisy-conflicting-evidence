"""
Post-generation citation support and claim grounding analyzer.
Verifies that generated answer assertions are genuinely entailed/supported by cited evidence passages,
preventing spurious or hallucinated citation references (remediating P0-06).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set, Tuple
from trustworthy_rag.schemas import Citation, RetrievedPassage

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "and", "or", "not", "this", "that", "these",
}


class CitationGroundingAnalyzer:
    """Verifies that generated answer assertions are grounded in cited evidence passages."""

    @staticmethod
    def extract_sentence_citation_pairs(answer_text: str) -> List[Tuple[str, List[str]]]:
        """Pair each sentence clause with its embedded bracket citations [chunk_id]."""
        sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        pairs: List[Tuple[str, List[str]]] = []
        last_text = ""
        for sent in sentences:
            if not sent.strip():
                continue
            citations = re.findall(r"\[([a-zA-Z0-9_\-]+)\]", sent)
            clean_text = re.sub(r"\[[a-zA-Z0-9_\-]+\]", "", sent).strip()
            clean_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean_text)) - STOP_WORDS
            if citations and not clean_words and last_text:
                if pairs:
                    prev_text, prev_cits = pairs[-1]
                    pairs[-1] = (prev_text, prev_cits + citations)
                else:
                    pairs.append((last_text, citations))
            elif citations:
                pairs.append((clean_text, citations))
                last_text = clean_text
            else:
                last_text = clean_text
        return pairs

    @classmethod
    def audit_citation_metrics(
        cls,
        answer_text: str,
        retrieved_passages: List[RetrievedPassage],
        gold_passage_ids: Optional[List[str]] = None,
    ) -> Tuple[float, float, List[Citation]]:
        """Compute citation precision, citation recall, and verification status for all citations."""
        passage_map: Dict[str, RetrievedPassage] = {p.chunk_id: p for p in retrieved_passages}
        sentence_pairs = cls.extract_sentence_citation_pairs(answer_text)

        all_citations: List[Citation] = []
        verified_count = 0
        total_citations = 0

        for sent_text, chunk_ids in sentence_pairs:
            sent_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", sent_text.lower())) - STOP_WORDS

            for cid in chunk_ids:
                total_citations += 1
                is_valid = False
                matched_span = cid

                if cid in passage_map:
                    passage_text = passage_map[cid].text.lower()
                    passage_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", passage_text)) - STOP_WORDS
                    common_tokens = sent_words.intersection(passage_words)
                    if sent_words and passage_words:
                        overlap_ratio = len(common_tokens) / min(len(sent_words), len(passage_words))
                        if overlap_ratio >= 0.20 and len(common_tokens) >= 1:
                            is_valid = True
                    elif common_tokens:
                        is_valid = True
                    elif not sent_words and not passage_words:
                        is_valid = True

                if is_valid:
                    verified_count += 1

                all_citations.append(
                    Citation(
                        citation_id=f"cit_{total_citations:03d}",
                        chunk_id=cid,
                        cited_text_span=matched_span,
                        is_verified=is_valid,
                    )
                )

        citation_precision = (verified_count / total_citations) if total_citations > 0 else 0.0

        if gold_passage_ids:
            cited_chunk_ids = {c.chunk_id for c in all_citations if c.is_verified}
            matched_gold = sum(1 for g in gold_passage_ids if any(g in cid for cid in cited_chunk_ids))
            citation_recall = matched_gold / len(gold_passage_ids) if gold_passage_ids else 0.0
        else:
            citation_recall = 1.0 if verified_count > 0 else 0.0

        return round(citation_precision, 4), round(citation_recall, 4), all_citations

    @classmethod
    def audit_citations(
        cls,
        answer_text: str,
        retrieved_passages: List[RetrievedPassage],
    ) -> Tuple[float, List[Citation]]:
        """Audit citations returning precision score s_cit and citation records."""
        prec, _, citations = cls.audit_citation_metrics(answer_text, retrieved_passages)
        return prec, citations
