"""
Explicit factual contradiction injection operator.
"""

import re
from typing import List, Tuple
from trustworthy_rag.schemas import PerturbationMetadata, PerturbationType, RetrievedPassage


class ContradictionInjector:
    """Injects mutually conflicting passages into retrieved context to test contradiction detection."""

    @staticmethod
    def mutate_facts(text: str) -> str:
        """Deterministically mutate numerical, temporal, or entity facts to create direct conflict."""
        mutated = text

        # Replace 4-digit years: e.g. 2023 -> 1998, 2017 -> 2005, 1969 -> 1984
        def year_repl(m):
            yr = int(m.group(0))
            return str(yr - 15 if yr > 2000 else yr + 20)

        mutated = re.sub(r"\b(19\d{2}|20\d{2})\b", year_repl, mutated)

        # Replace percentages: e.g. 5.25% -> 0.15%
        mutated = re.sub(r"\b\d+\.\d+%\b", "0.05%", mutated)

        # Mutate affirmative statements to negative or vice versa
        if "landed" in mutated:
            mutated = mutated.replace("landed humans on the Moon", "failed to land on the Moon due to an aborted descent")
        elif "succeeded" in mutated:
            mutated = mutated.replace("succeeded PaLM 2", "was canceled prior to release and never succeeded PaLM 2")
        elif "awarded the" in mutated:
            mutated = mutated.replace("awarded the", "denied the")
        else:
            mutated = f"Contradictory investigation asserts: In contrast to previous claims, {mutated}"

        return mutated

    def inject(
        self,
        clean_passages: List[RetrievedPassage],
        target_chunk_id: str | None = None,
        severity: float = 1.0,
        seed: int = 42,
    ) -> Tuple[List[RetrievedPassage], PerturbationMetadata]:
        """Inject an explicitly contradicting passage alongside original evidence."""
        if not clean_passages:
            meta = PerturbationMetadata(
                perturbation_type=PerturbationType.CONTRADICTION,
                severity=severity,
                seed=seed,
                injected_chunk_ids=[],
                removed_chunk_ids=[],
                description="Empty passage list",
            )
            return [], meta

        # Find target chunk or default to top-1
        target_passage = clean_passages[0]
        if target_chunk_id:
            for p in clean_passages:
                if p.chunk_id == target_chunk_id:
                    target_passage = p
                    break

        contradictory_text = self.mutate_facts(target_passage.text)
        conflict_chunk_id = f"conflict_{target_passage.chunk_id}"

        conflicting_passage = RetrievedPassage(
            chunk_id=conflict_chunk_id,
            doc_id=target_passage.doc_id,
            text=contradictory_text,
            sparse_score=target_passage.sparse_score,
            dense_score=target_passage.dense_score,
            fused_score=target_passage.fused_score * 0.98,
            rank=target_passage.rank + 1,
            retrieval_method="contradiction_injection",
        )

        perturbed_list = list(clean_passages)
        perturbed_list.insert(target_passage.rank, conflicting_passage)

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
            perturbation_type=PerturbationType.CONTRADICTION,
            severity=severity,
            seed=seed,
            injected_chunk_ids=[conflict_chunk_id],
            removed_chunk_ids=[],
            description=f"Injected conflicting passage {conflict_chunk_id} targeting {target_passage.chunk_id}",
        )
        return re_ranked, metadata
