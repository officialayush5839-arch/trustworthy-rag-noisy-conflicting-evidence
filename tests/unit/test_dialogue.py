"""
Unit tests for the multi-turn clarification dialogue manager.
"""

from trustworthy_rag.policy.dialogue import ClarificationDialogueManager
from trustworthy_rag.schemas import RetrievedPassage


def test_dialogue_initiate():
    mgr = ClarificationDialogueManager()
    passages = [
        RetrievedPassage(
            chunk_id="chunk_01",
            doc_id="doc_acme",
            text="ACME Global was founded in 1994 in Seattle.",
            fused_score=0.9,
            rank=1,
            retrieval_method="hybrid_rrf",
        ),
        RetrievedPassage(
            chunk_id="chunk_02",
            doc_id="doc_acme_corp",
            text="ACME Corporation was founded in 1998 in Chicago.",
            fused_score=0.85,
            rank=2,
            retrieval_method="hybrid_rrf",
        ),
    ]

    state = mgr.initiate_clarification(
        session_id="session_test_01",
        query="When was ACME founded?",
        missing_entities=["location", "incorporation_type"],
        candidate_passages=passages,
    )

    assert state.session_id == "session_test_01"
    assert not state.is_resolved
    assert state.turn_count == 1
    assert len(state.disambiguation_options) > 0


def test_dialogue_resolve():
    mgr = ClarificationDialogueManager()
    passages = [
        RetrievedPassage(
            chunk_id="c1",
            doc_id="d1",
            text="General organization facts.",
            fused_score=0.7,
            rank=1,
            retrieval_method="hybrid_rrf",
        )
    ]
    mgr.initiate_clarification(
        session_id="sess_02",
        query="Tell me about the contract",
        missing_entities=["signatory"],
        candidate_passages=passages,
    )

    resolved = mgr.resolve_clarification("sess_02", "I am asking about the 2021 European contract")
    assert resolved.is_resolved
    assert resolved.turn_count == 2
    assert "2021 European contract" in resolved.clarified_query

    synthetic_passage = mgr.synthesize_disambiguated_passage(resolved)
    assert synthetic_passage.chunk_id.startswith("clarify_")
    assert synthetic_passage.fused_score == 1.0
