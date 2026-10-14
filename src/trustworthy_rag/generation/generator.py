"""
Generative answer formulation engine and baseline conventional RAG runner.
"""

import logging
import os
import re
import time
from typing import List, Optional

from trustworthy_rag.generation.context_builder import AssembledContext
from trustworthy_rag.schemas import (
    Citation,
    DecisionAction,
    DecisionResult,
    EvidenceQualityScore,
    GenerationResult,
)

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = """You are a trustworthy research assistant. Answer the user's question using ONLY the provided evidence passages.
Every factual statement must cite the specific chunk ID in square brackets, e.g., [doc_001_c0001].
If the evidence is insufficient to answer the question, state clearly that the evidence does not provide enough information."""


class AnswerGenerator:
    """Generates answers from assembled evidence passages using pinned models or APIs."""

    def __init__(
        self,
        backend: str = "mock",
        model_identifier: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.backend = backend.lower()
        self.model_identifier = model_identifier
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_prompt(self, query: str, context: AssembledContext) -> str:
        """Construct full prompt combining system directive, evidence passages, and query."""
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"{context.formatted_text}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

    def extract_citations(self, text: str, valid_chunk_ids: List[str]) -> List[Citation]:
        """Extract bracketed chunk citations [doc_xxx_c0000] from text."""
        raw_matches = re.findall(r"\[([a-zA-Z0-9_\-]+_c\d{4})\]", text)
        citations: List[Citation] = []
        valid_set = set(valid_chunk_ids)

        for i, chunk_id in enumerate(raw_matches):
            citations.append(
                Citation(
                    citation_id=f"cit_{i+1:03d}",
                    chunk_id=chunk_id,
                    cited_text_span=chunk_id,
                    is_verified=(chunk_id in valid_set),
                )
            )

        return citations

    def generate(self, query_id: str, query: str, context: AssembledContext) -> GenerationResult:
        """Execute generation and return structured GenerationResult."""
        prompt = self.build_prompt(query, context)
        start_time = time.perf_counter()

        response_text = ""
        prompt_tokens = len(prompt.split())
        completion_tokens = 0

        if self.backend == "mock" or os.environ.get("USE_MOCK_GENERATOR", "0") == "1":
            # Deterministic, grounded mock generation for testing and offline environments
            if context.included_chunk_ids:
                first_chunk = context.included_chunk_ids[0]
                response_text = f"Based on the provided evidence, the documented answer is substantiated [{first_chunk}]."
            else:
                response_text = "The provided evidence does not contain sufficient information to answer the question."
            completion_tokens = len(response_text.split())
        elif self.backend == "gemini":
            try:
                from google import genai
                client = genai.Client()
                resp = client.models.generate_content(
                    model=self.model_identifier,
                    contents=prompt,
                )
                response_text = resp.text or ""
                completion_tokens = len(response_text.split())
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                response_text = f"[ERROR: Generation failed: {e}]"
        elif self.backend == "openai":
            try:
                from openai import OpenAI
                client = OpenAI()
                resp = client.chat.completions.create(
                    model=self.model_identifier,
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": f"{context.formatted_text}\n\nQuestion: {query}"},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = resp.choices[0].message.content or ""
                completion_tokens = resp.usage.completion_tokens if resp.usage else len(response_text.split())
                prompt_tokens = resp.usage.prompt_tokens if resp.usage else prompt_tokens
            except Exception as e:
                logger.error(f"OpenAI generation error: {e}")
                response_text = f"[ERROR: Generation failed: {e}]"
        else:
            # Fallback mock
            response_text = f"Mock response for query: {query}"
            completion_tokens = len(response_text.split())

        elapsed_sec = time.perf_counter() - start_time
        citations = self.extract_citations(response_text, context.included_chunk_ids)

        # Baseline always emits ANSWER action (unconditional generation)
        dummy_score = EvidenceQualityScore(
            relevance_score=1.0,
            coverage_score=1.0,
            contradiction_score=0.0,
            composite_quality_score=1.0,
            is_contradictory=False,
        )
        baseline_decision = DecisionResult(
            action=DecisionAction.ANSWER,
            decision_reason="Baseline conventional RAG unconditional generation",
            evidence_score=dummy_score,
            applied_thresholds={},
        )

        return GenerationResult(
            query_id=query_id,
            raw_response=response_text,
            cleaned_answer=response_text.strip(),
            decision=baseline_decision,
            citations=citations,
            model_identifier=self.model_identifier,
            latency_seconds=float(elapsed_sec),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
