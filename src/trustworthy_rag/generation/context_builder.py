"""
Provenance-anchored context builder for Retrieval-Augmented Generation prompts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from trustworthy_rag.schemas import RetrievedPassage


@dataclass(frozen=True)
class AssembledContext:
    """Assembled prompt context with provenance chunk tracking."""

    formatted_text: str
    chunk_offsets: Dict[str, Tuple[int, int]]
    included_chunk_ids: List[str]
    passages: List[RetrievedPassage] = field(default_factory=list)


class ContextBuilder:
    """Formats retrieved passages into a structured prompt block with verifiable chunk citations."""

    def __init__(self, max_tokens: int = 1500):
        self.max_tokens = max_tokens

    def assemble(self, passages: List[RetrievedPassage]) -> AssembledContext:
        """Assemble passages into formatted text block with character-offset provenance."""
        if not passages:
            return AssembledContext(
                formatted_text="No relevant evidence passages available.",
                chunk_offsets={},
                included_chunk_ids=[],
                passages=[],
            )

        context_lines: List[str] = []
        chunk_offsets: Dict[str, Tuple[int, int]] = {}
        included_chunk_ids: List[str] = []

        current_char = 0
        header_intro = "=== RETRIEVED EVIDENCE PASSAGES ===\n"
        context_lines.append(header_intro)
        current_char += len(header_intro)

        for p in passages:
            chunk_header = f"[Chunk ID: {p.chunk_id}]\n"
            chunk_body = f"{p.text.strip()}\n\n"

            start_offset = current_char
            full_chunk_block = chunk_header + chunk_body
            end_offset = start_offset + len(full_chunk_block)

            chunk_offsets[p.chunk_id] = (start_offset, end_offset)
            included_chunk_ids.append(p.chunk_id)
            context_lines.append(full_chunk_block)
            current_char = end_offset

        full_context_str = "".join(context_lines).strip()
        return AssembledContext(
            formatted_text=full_context_str,
            chunk_offsets=chunk_offsets,
            included_chunk_ids=included_chunk_ids,
            passages=passages,
        )
