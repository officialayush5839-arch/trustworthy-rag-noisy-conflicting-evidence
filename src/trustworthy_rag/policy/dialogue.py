"""
Multi-turn conversational clarification dialogue manager.
Resolves ambiguous or partially covered evidence by formulating targeted disambiguation prompts,
incorporating user responses, and transitioning from CLARIFY to grounded ANSWER.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from trustworthy_rag.schemas import DecisionAction, DecisionResult, RetrievedPassage


class ClarificationState(BaseModel):
    session_id: str
    original_query: str
    missing_entities: List[str] = Field(default_factory=list)
    disambiguation_options: List[str] = Field(default_factory=list)
    user_clarification: Optional[str] = None
    clarified_query: Optional[str] = None
    turn_count: int = 1
    is_resolved: bool = False


class ClarificationDialogueManager:
    """Orchestrates multi-turn dialogue when evidence triggers the CLARIFY action."""

    def __init__(self):
        self.active_sessions: Dict[str, ClarificationState] = {}

    def initiate_clarification(
        self,
        session_id: str,
        query: str,
        missing_entities: List[str],
        candidate_passages: List[RetrievedPassage],
    ) -> ClarificationState:
        """Formulate targeted clarification prompts based on missing query entities and candidate snippets."""
        options = []
        for p in candidate_passages[:3]:
            # Extract distinctive entity references from candidates
            words = [w for w in p.text.split() if len(w) > 4 and w[0].isupper()]
            if words:
                options.append(f"Regarding {' '.join(words[:2])} in chunk [{p.chunk_id}]")

        if not options and missing_entities:
            options = [f"Specify details regarding '{ent}'" for ent in missing_entities]

        state = ClarificationState(
            session_id=session_id,
            original_query=query,
            missing_entities=missing_entities,
            disambiguation_options=options,
            turn_count=1,
            is_resolved=False,
        )
        self.active_sessions[session_id] = state
        return state

    def resolve_clarification(
        self,
        session_id: str,
        user_response: str,
    ) -> ClarificationState:
        """Incorporate user response into clarified query and mark dialogue as resolved."""
        state = self.active_sessions.get(session_id)
        if not state:
            raise KeyError(f"No active clarification session for id: {session_id}")

        cleaned_response = user_response.strip()
        state.user_clarification = cleaned_response
        state.clarified_query = f"{state.original_query} (Specified: {cleaned_response})"
        state.turn_count += 1
        state.is_resolved = True
        return state

    def synthesize_disambiguated_passage(
        self,
        state: ClarificationState,
    ) -> RetrievedPassage:
        """Generate a synthetic grounded clarification constraint passage to augment retrieval context."""
        return RetrievedPassage(
            chunk_id=f"clarify_{state.session_id[:8]}",
            doc_id="user_clarification_constraint",
            text=f"User clarified context constraint: {state.user_clarification}. Original query scope: {state.original_query}.",
            sparse_score=1.0,
            dense_score=1.0,
            fused_score=1.0,
            rank=1,
            retrieval_method="dialogue_clarification",
        )
