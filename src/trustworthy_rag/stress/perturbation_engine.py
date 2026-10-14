"""
Unified stress-test perturbation engine coordinating all evidence corruption operators.
"""

from typing import List, Tuple
from trustworthy_rag.schemas import DocumentChunk, PerturbationMetadata, PerturbationType, RetrievedPassage
from trustworthy_rag.stress.contradiction_injection import ContradictionInjector
from trustworthy_rag.stress.distractor import DistractorInjector
from trustworthy_rag.stress.near_duplicate import NearDuplicateInjector
from trustworthy_rag.stress.passage_removal import PassageRemovalOperator


class PerturbationEngine:
    """Orchestrates all four controlled stress conditions with manifest tracking."""

    def __init__(self, corpus_chunks: List[DocumentChunk]):
        self.corpus_chunks = corpus_chunks
        self.distractor_op = DistractorInjector(corpus_chunks)
        self.removal_op = PassageRemovalOperator()
        self.duplicate_op = NearDuplicateInjector()
        self.contradiction_op = ContradictionInjector()

    def perturb(
        self,
        clean_passages: List[RetrievedPassage],
        condition: PerturbationType | str,
        severity: float = 0.4,
        gold_passage_ids: List[str] | None = None,
        seed: int = 42,
    ) -> Tuple[List[RetrievedPassage], PerturbationMetadata]:
        """Apply the specified perturbation condition to clean retrieved passages."""
        cond = PerturbationType(condition) if isinstance(condition, str) else condition

        if cond == PerturbationType.CLEAN:
            meta = PerturbationMetadata(
                perturbation_type=PerturbationType.CLEAN,
                severity=0.0,
                seed=seed,
                injected_chunk_ids=[],
                removed_chunk_ids=[],
                description="Clean, unperturbed retrieved evidence",
            )
            return list(clean_passages), meta

        elif cond == PerturbationType.DISTRACTOR:
            return self.distractor_op.inject(clean_passages, severity=severity, seed=seed)

        elif cond == PerturbationType.PASSAGE_REMOVAL:
            return self.removal_op.remove(clean_passages, gold_passage_ids=gold_passage_ids or [], seed=seed)

        elif cond == PerturbationType.NEAR_DUPLICATE:
            return self.duplicate_op.inject(clean_passages, severity=severity, seed=seed)

        elif cond == PerturbationType.CONTRADICTION:
            target_id = gold_passage_ids[0] if gold_passage_ids else None
            return self.contradiction_op.inject(clean_passages, target_chunk_id=target_id, severity=severity, seed=seed)

        else:
            raise ValueError(f"Unsupported perturbation condition: {cond}")
