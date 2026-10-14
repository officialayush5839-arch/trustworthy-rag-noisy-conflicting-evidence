"""
Key supporting passage deletion operator.
"""

from typing import List, Tuple
from trustworthy_rag.schemas import PerturbationMetadata, PerturbationType, RetrievedPassage


class PassageRemovalOperator:
    """Removes essential gold supporting passages to test model abstention when evidence is missing."""

    def remove(
        self,
        clean_passages: List[RetrievedPassage],
        gold_passage_ids: List[str],
        seed: int = 42,
    ) -> Tuple[List[RetrievedPassage], PerturbationMetadata]:
        """Excise gold passage IDs from candidate set and re-rank."""
        gold_set = set(gold_passage_ids)
        removed_ids = []
        surviving: List[RetrievedPassage] = []

        for p in clean_passages:
            if p.chunk_id in gold_set or any(g in p.chunk_id for g in gold_set):
                removed_ids.append(p.chunk_id)
            else:
                surviving.append(p)

        # If no explicit gold chunk was found in the retrieved candidates, remove the top-1 chunk as surrogate
        if not removed_ids and clean_passages:
            target = clean_passages[0]
            removed_ids.append(target.chunk_id)
            surviving = clean_passages[1:]

        re_ranked = [
            RetrievedPassage(
                chunk_id=p.chunk_id,
                doc_id=p.doc_id,
                text=p.text,
                sparse_score=p.sparse_score,
                dense_score=p.dense_score,
                fused_score=p.fused_score,
                rank=new_rank,
                retrieval_method=p.retrieval_method,
            )
            for new_rank, p in enumerate(surviving, start=1)
        ]

        metadata = PerturbationMetadata(
            perturbation_type=PerturbationType.PASSAGE_REMOVAL,
            severity=1.0,
            seed=seed,
            injected_chunk_ids=[],
            removed_chunk_ids=removed_ids,
            description=f"Excised {len(removed_ids)} supporting passage(s): {', '.join(removed_ids)}",
        )
        return re_ranked, metadata
