"""
Unit tests for Reciprocal Rank Fusion (RRF).
"""

from trustworthy_rag.retrieval.fusion import RankFusionEngine
from trustworthy_rag.schemas import RetrievedPassage


def test_rrf_combines_ranks():
    p1 = RetrievedPassage(chunk_id="c1", doc_id="d1", text="text 1", sparse_score=10.0, fused_score=1.0, rank=1, retrieval_method="bm25")
    p2 = RetrievedPassage(chunk_id="c2", doc_id="d2", text="text 2", sparse_score=5.0, fused_score=0.5, rank=2, retrieval_method="bm25")

    p_d1 = RetrievedPassage(chunk_id="c2", doc_id="d2", text="text 2", dense_score=0.9, fused_score=0.9, rank=1, retrieval_method="dense")
    p_d2 = RetrievedPassage(chunk_id="c3", doc_id="d3", text="text 3", dense_score=0.7, fused_score=0.7, rank=2, retrieval_method="dense")

    engine = RankFusionEngine(rrf_k=60)
    fused = engine.fuse([p1, p2], [p_d1, p_d2], top_k=3)

    assert len(fused) == 3
    # c2 appeared in both (rank 2 in sparse, rank 1 in dense) -> should have highest combined RRF score
    assert fused[0].chunk_id == "c2"
    assert fused[0].rank == 1
    assert fused[0].retrieval_method == "hybrid_rrf"
    assert 0.0 <= fused[0].fused_score <= 1.0
