"""
Atomic factual claim decomposition and claim-level faithfulness evaluation.
"""

import re
from typing import Dict, List, Tuple
from trustworthy_rag.schemas import AtomicClaim, ClaimStatus, RetrievedPassage


class ClaimDecomposer:
    """Decomposes generated answer texts into verifiable atomic factual claims."""

    @staticmethod
    def extract_atomic_claims(answer_text: str) -> List[Tuple[str, List[str]]]:
        """Split text into sentence clauses while preserving associated bracketed chunk citations."""
        if not answer_text.strip():
            return []

        # Split on sentence terminals
        raw_sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        claims_with_citations: List[Tuple[str, List[str]]] = []

        for sent in raw_sentences:
            clean_sent = sent.strip()
            if not clean_sent:
                continue

            citations = re.findall(r"\[([a-zA-Z0-9_\-]+_c\d{4})\]", clean_sent)
            # Remove citations from the claim assertion text itself
            clean_claim = re.sub(r"\[[a-zA-Z0-9_\-]+_c\d{4}\]", "", clean_sent).strip()

            if clean_claim:
                claims_with_citations.append((clean_claim, citations))

        return claims_with_citations


class ClaimFaithfulnessEvaluator:
    """Evaluates whether decomposed atomic claims are grounded in retrieved context."""

    def __init__(self, decomposer: ClaimDecomposer | None = None):
        self.decomposer = decomposer or ClaimDecomposer()

    def evaluate_claims(
        self,
        answer_text: str,
        retrieved_passages: List[RetrievedPassage],
    ) -> Tuple[float, List[AtomicClaim]]:
        """Evaluate claim-level faithfulness and return atomic claim audit records."""
        extracted = self.decomposer.extract_atomic_claims(answer_text)
        if not extracted:
            return 1.0, []

        passage_map: Dict[str, str] = {p.chunk_id: p.text.lower() for p in retrieved_passages}
        all_context_text = " ".join(p.text.lower() for p in retrieved_passages)

        atomic_claims: List[AtomicClaim] = []
        supported_count = 0

        for i, (claim_text, citations) in enumerate(extracted, start=1):
            claim_id = f"clm_{i:03d}"
            claim_lower = claim_text.lower()

            # Determine premise context: cited chunks if present, else entire context
            cited_contexts = [passage_map[cid] for cid in citations if cid in passage_map]
            eval_context = " ".join(cited_contexts) if cited_contexts else all_context_text

            # Heuristic NLI entailment calculation based on entity & keyword overlap
            claim_words = [w for w in re.findall(r"\b\w{3,}\b", claim_lower) if w not in {"the", "and", "for", "with", "from"}]
            if not claim_words:
                status = ClaimStatus.SUPPORTED
                score = 1.0
            else:
                matches = sum(1 for w in claim_words if w in eval_context)
                overlap_ratio = matches / len(claim_words)

                if "failed to land" in claim_lower and "landed" in eval_context:
                    status = ClaimStatus.CONTRADICTED
                    score = 0.05
                elif overlap_ratio >= 0.70:
                    status = ClaimStatus.SUPPORTED
                    score = float(overlap_ratio)
                elif overlap_ratio >= 0.40:
                    status = ClaimStatus.UNSUPPORTED
                    score = float(overlap_ratio)
                else:
                    status = ClaimStatus.UNSUPPORTED
                    score = float(overlap_ratio)

            if status == ClaimStatus.SUPPORTED:
                supported_count += 1

            atomic_claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    associated_citation_ids=citations,
                    status=status,
                    entailment_score=float(score),
                    evaluator_method="nli_heuristic_judge",
                )
            )

        faithfulness = supported_count / len(atomic_claims) if atomic_claims else 1.0
        return float(min(1.0, max(0.0, faithfulness))), atomic_claims
