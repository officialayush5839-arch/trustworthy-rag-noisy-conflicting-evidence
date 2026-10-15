"""
Core Pydantic v2 schemas and data contracts for the Trustworthy RAG platform.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class PerturbationType(str, Enum):
    CLEAN = "clean"
    DISTRACTOR = "distractor"
    PASSAGE_REMOVAL = "passage_removal"
    NEAR_DUPLICATE = "near_duplicate"
    CONTRADICTION = "contradiction"


class DecisionAction(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


# --- Source Data Schemas ---


class Document(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = None
    source_uri: Optional[str] = None
    corpus_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sha256_hash: str


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., description="Unique chunk identifier formatted as {doc_id}_c{idx:04d}")
    doc_id: str
    text: str
    start_char_offset: int = Field(..., ge=0)
    end_char_offset: int = Field(..., ge=0)
    token_count: int = Field(..., ge=0)
    sha256_hash: str


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    question_id: str
    dataset_name: str
    query_text: str
    gold_answers: List[str] = Field(default_factory=list)
    gold_passage_ids: List[str] = Field(default_factory=list)
    split: str = Field(..., description="dev | validation | test")


# --- Retrieval & Stress Schemas ---


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    text: str
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None
    fused_score: float
    rank: int = Field(..., ge=1)
    retrieval_method: str = Field(..., description="bm25 | dense | hybrid_rrf")


class PerturbationMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    perturbation_type: PerturbationType
    severity: float = Field(0.0, ge=0.0, le=1.0)
    seed: int
    injected_chunk_ids: List[str] = Field(default_factory=list)
    removed_chunk_ids: List[str] = Field(default_factory=list)
    description: str


# --- Evidence Reasoning Schemas ---


class ContradictionFinding(BaseModel):
    model_config = ConfigDict(frozen=True)

    premise_chunk_id: str
    hypothesis_chunk_id: str
    contradiction_probability: float = Field(..., ge=0.0, le=1.0)
    entailment_probability: float = Field(..., ge=0.0, le=1.0)
    neutral_probability: float = Field(..., ge=0.0, le=1.0)


class EvidenceQualityScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    relevance_score: float = Field(..., ge=0.0, le=1.0)
    coverage_score: float = Field(..., ge=0.0, le=1.0)
    citation_support_score: float = Field(0.0, ge=0.0, le=1.0)
    contradiction_score: float = Field(..., ge=0.0, le=1.0)
    composite_quality_score: float = Field(..., ge=0.0, le=1.0)
    is_contradictory: bool
    feature_weights: Dict[str, float] = Field(default_factory=dict)


class DecisionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: DecisionAction
    decision_reason: str
    evidence_score: EvidenceQualityScore
    applied_thresholds: Dict[str, float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Generation & Citation Schemas ---


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    chunk_id: str
    cited_text_span: str
    claim_id: Optional[str] = None
    is_verified: bool = False


class GenerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    raw_response: str
    cleaned_answer: Optional[str] = None
    decision: DecisionResult
    citations: List[Citation] = Field(default_factory=list)
    model_identifier: str
    latency_seconds: float = Field(..., ge=0.0)
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)


# --- Evaluation Schemas ---


class AtomicClaim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    claim_text: str
    associated_citation_ids: List[str] = Field(default_factory=list)
    status: ClaimStatus
    entailment_score: float = Field(..., ge=0.0, le=1.0)
    evaluator_method: str = Field(..., description="nli_model | llm_judge | human")


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    question_id: str
    perturbation_type: PerturbationType
    method_name: str
    is_answered: bool
    is_correct: Optional[bool] = None
    claim_faithfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    selective_error: Optional[float] = Field(None, ge=0.0, le=1.0)
    citation_precision: float = Field(0.0, ge=0.0, le=1.0)
    citation_recall: float = Field(0.0, ge=0.0, le=1.0)
    retrieval_recall: float = Field(..., ge=0.0, le=1.0)
    atomic_claims: List[AtomicClaim] = Field(default_factory=list)
    backend_identifier: str = Field("default", description="Model backend and revision")

