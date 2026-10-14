"""
Reciprocal Rank Fusion (RRF) and hybrid retrieval combiner.
"""

from typing import Dict, List, Optional
from trustworthy_rag.schemas import RetrievedPassage


class RankFusionEngine:
    """Combines disparate ranked candidate lists using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        self.rrf_k = rrf_k

    def fuse(
        self,
        sparse_results: List[RetrievedPassage],
        dense_results: List[RetrievedPassage],
        top_k: int = 10,
    ) -> List[RetrievedPassage]:
        """Merge sparse and dense candidate lists using Reciprocal Rank Fusion."""
        # Maps chunk_id -> dict of passage details and accumulators
        fused_candidates: Dict[str, Dict] = {}

        for passage in sparse_results:
            cid = passage.chunk_id
            if cid not in fused_candidates:
                fused_candidates[cid] = {
                    "doc_id": passage.doc_id,
                    "text": passage.text,
                    "sparse_score": passage.sparse_score,
                    "dense_score": None,
                    "rrf_score": 0.0,
                }
            else:
                fused_candidates[cid]["sparse_score"] = passage.sparse_score

            fused_candidates[cid]["rrf_score"] += 1.0 / (self.rrf_k + passage.rank)

        for passage in dense_results:
            cid = passage.chunk_id
            if cid not in fused_candidates:
                fused_candidates[cid] = {
                    "doc_id": passage.doc_id,
                    "text": passage.text,
                    "sparse_score": None,
                    "dense_score": passage.dense_score,
                    "rrf_score": 0.0,
                }
            else:
                fused_candidates[cid]["dense_score"] = passage.dense_score

            fused_candidates[cid]["rrf_score"] += 1.0 / (self.rrf_k + passage.rank)

        # Sort candidates descending by RRF score
        sorted_cands = sorted(
            fused_candidates.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        selected = sorted_cands[:top_k]
        if not selected:
            return []

        # Min-max normalize fused score to [0.0, 1.0]
        max_rrf = max(c["rrf_score"] for _, c in selected)
        min_rrf = min(c["rrf_score"] for _, c in selected)
        rrf_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

        fused_results: List[RetrievedPassage] = []
        for rank, (chunk_id, data) in enumerate(selected, start=1):
            norm_score = (data["rrf_score"] - min_rrf) / rrf_range if rrf_range > 0 else 1.0
            fused_results.append(
                RetrievedPassage(
                    chunk_id=chunk_id,
                    doc_id=data["doc_id"],
                    text=data["text"],
                    sparse_score=data["sparse_score"],
                    dense_score=data["dense_score"],
                    fused_score=float(norm_score),
                    rank=rank,
                    retrieval_method="hybrid_rrf",
                )
            )

        return fused_results
