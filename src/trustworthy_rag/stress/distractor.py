"""
Distractor passage injection operator.
"""

import random
from typing import List, Tuple
from trustworthy_rag.schemas import DocumentChunk, PerturbationMetadata, PerturbationType, RetrievedPassage


class DistractorInjector:
    """Injects irrelevant distractor passages into retrieved candidate pools."""

    def __init__(self, all_corpus_chunks: List[DocumentChunk]):
        self.corpus_chunks = all_corpus_chunks

    def inject(
        self,
        clean_passages: List[RetrievedPassage],
        severity: float = 0.4,
        seed: int = 42,
    ) -> Tuple[List[RetrievedPassage], PerturbationMetadata]:
        """Inject distractors at ratio alpha = severity."""
        rng = random.Random(seed)
        existing_chunk_ids = {p.chunk_id for p in clean_passages}
        candidate_distractors = [c for c in self.corpus_chunks if c.chunk_id not in existing_chunk_ids]

        if not candidate_distractors or not clean_passages:
            meta = PerturbationMetadata(
                perturbation_type=PerturbationType.DISTRACTOR,
                severity=severity,
                seed=seed,
                injected_chunk_ids=[],
                removed_chunk_ids=[],
                description="No distractors available or clean passages empty",
            )
            return list(clean_passages), meta

        num_to_inject = max(1, int(len(clean_passages) * severity))
        sampled = rng.sample(candidate_distractors, min(num_to_inject, len(candidate_distractors)))

        injected_chunk_ids = []
        perturbed_list = list(clean_passages)

        for chunk in sampled:
            distractor_passage = RetrievedPassage(
                chunk_id=f"distractor_{chunk.chunk_id}",
                doc_id=chunk.doc_id,
                text=chunk.text,
                sparse_score=0.1,
                dense_score=0.1,
                fused_score=0.1,
                rank=len(perturbed_list) + 1,
                retrieval_method="distractor_injection",
            )
            injected_chunk_ids.append(distractor_passage.chunk_id)
            # Insert at randomized rank position
            insert_pos = rng.randint(0, len(perturbed_list))
            perturbed_list.insert(insert_pos, distractor_passage)

        # Re-index ranks
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
            perturbation_type=PerturbationType.DISTRACTOR,
            severity=severity,
            seed=seed,
            injected_chunk_ids=injected_chunk_ids,
            removed_chunk_ids=[],
            description=f"Injected {len(injected_chunk_ids)} distractor passages at severity {severity}",
        )
        return re_ranked, metadata
