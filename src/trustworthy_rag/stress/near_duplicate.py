"""
Near-duplicate passage injection operator.
"""

import random
from typing import List, Tuple
from trustworthy_rag.schemas import PerturbationMetadata, PerturbationType, RetrievedPassage


class NearDuplicateInjector:
    """Injects paraphrased, semantically redundant duplicate chunks into context."""

    @staticmethod
    def paraphrase_text(text: str) -> str:
        """Create a realistic semantic paraphrase of an existing chunk."""
        prefixes = [
            "In other words, ",
            "To reiterate, ",
            "According to parallel documentation, ",
            "Stated differently, ",
        ]
        chosen_prefix = random.choice(prefixes)
        return f"{chosen_prefix}{text.strip()}"

    def inject(
        self,
        clean_passages: List[RetrievedPassage],
        severity: float = 0.3,
        seed: int = 42,
    ) -> Tuple[List[RetrievedPassage], PerturbationMetadata]:
        """Inject duplicate passages for a fraction of top passages."""
        if not clean_passages:
            meta = PerturbationMetadata(
                perturbation_type=PerturbationType.NEAR_DUPLICATE,
                severity=severity,
                seed=seed,
                injected_chunk_ids=[],
                removed_chunk_ids=[],
                description="Empty passage list",
            )
            return [], meta

        rng = random.Random(seed)
        num_dups = max(1, int(len(clean_passages) * severity))
        targets = clean_passages[:num_dups]

        injected_ids = []
        perturbed_list = list(clean_passages)

        for target in targets:
            dup_id = f"dup_{target.chunk_id}"
            injected_ids.append(dup_id)
            paraphrased = self.paraphrase_text(target.text)

            dup_passage = RetrievedPassage(
                chunk_id=dup_id,
                doc_id=target.doc_id,
                text=paraphrased,
                sparse_score=target.sparse_score,
                dense_score=target.dense_score,
                fused_score=target.fused_score * 0.95,
                rank=len(perturbed_list) + 1,
                retrieval_method="near_duplicate_injection",
            )
            # Insert adjacent to or shortly after original
            insert_pos = min(target.rank + 1, len(perturbed_list))
            perturbed_list.insert(insert_pos, dup_passage)

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
            for new_rank, p in enumerate(perturbed_list, start=1)
        ]

        metadata = PerturbationMetadata(
            perturbation_type=PerturbationType.NEAR_DUPLICATE,
            severity=severity,
            seed=seed,
            injected_chunk_ids=injected_ids,
            removed_chunk_ids=[],
            description=f"Injected {len(injected_ids)} near-duplicate passages",
        )
        return re_ranked, metadata
