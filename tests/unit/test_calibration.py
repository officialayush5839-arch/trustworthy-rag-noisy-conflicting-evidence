"""
Unit tests for adaptive dynamic evidence router and continuous policy calibration.
"""

from trustworthy_rag.config import PolicyThresholds
from trustworthy_rag.policy.calibration import DynamicEvidenceRouter
from trustworthy_rag.schemas import RetrievedPassage


def test_dynamic_evidence_router():
    base = PolicyThresholds(tau_ans_rel=0.65, tau_cov_min=0.60, tau_con_max=0.70)
    router = DynamicEvidenceRouter(base)

    passages = [
        RetrievedPassage(
            chunk_id="chunk_top",
            doc_id="doc_1",
            text="High confidence match.",
            fused_score=0.95,
            rank=1,
            retrieval_method="hybrid_rrf",
        ),
        RetrievedPassage(
            chunk_id="chunk_tail",
            doc_id="doc_2",
            text="Weak trailing candidate.",
            fused_score=0.20,
            rank=2,
            retrieval_method="hybrid_rrf",
        ),
    ]

    calibrated = router.calibrate(
        query="What is the exact aerodynamic lift mechanism for avian gliding?",
        passages=passages,
    )

    assert calibrated.score_dispersion > 0.5
    assert calibrated.query_entropy > 0.0
    # Threshold should dynamically adapt slightly lower to allow strong top-1 hit
    assert calibrated.tau_ans_rel < base.tau_ans_rel
    assert calibrated.tau_cov_min <= base.tau_cov_min
