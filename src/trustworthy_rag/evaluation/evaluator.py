"""
Comparative evaluation and ablation matrix orchestrator.
Implements leak-free baselines (P0-08), separate abstention scoring (P0-01),
genuine citation verification (P0-06), and exact retrieval recall matching (P0-07).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from trustworthy_rag.config import GeneratorConfig, PolicyThresholds
from trustworthy_rag.evaluation.claims import ClaimFaithfulnessEvaluator
from trustworthy_rag.evaluation.risk_coverage import RiskCoverageCompiler
from trustworthy_rag.evidence import (
    CitationGroundingAnalyzer,
    ContradictionAnalyzer,
    EvidenceCoverageAnalyzer,
    RelevanceAnalyzer,
)
from trustworthy_rag.generation.context_builder import ContextBuilder
from trustworthy_rag.generation.generator import AnswerGenerator
from trustworthy_rag.policy.decision import DecisionPolicyEngine
from trustworthy_rag.schemas import (
    DecisionAction,
    EvaluationResult,
    PerturbationType,
    Question,
    RetrievedPassage,
)

logger = logging.getLogger(__name__)


class ComparativeEvaluator:
    """Executes comparative evaluation across baselines, proposed policy, and ablation variants."""

    def __init__(
        self,
        thresholds: Optional[PolicyThresholds] = None,
        generator_backend: str = "mock",
        contradiction_use_mock: bool = True,
        generator_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    ):
        self.thresholds = thresholds or PolicyThresholds()
        self.generator_backend = generator_backend
        self.generator_model_id = generator_model_id
        self.policy_engine = DecisionPolicyEngine(self.thresholds)
        self.relevance_analyzer = RelevanceAnalyzer()
        self.coverage_analyzer = EvidenceCoverageAnalyzer()
        self.contradiction_analyzer = ContradictionAnalyzer(use_mock_fallback=contradiction_use_mock)
        self.citation_analyzer = CitationGroundingAnalyzer()
        self.claim_evaluator = ClaimFaithfulnessEvaluator()
        self.context_builder = ContextBuilder()
        self.generator = AnswerGenerator(
            backend=self.generator_backend,
            model_identifier=self.generator_model_id,
        )

    def evaluate_query(
        self,
        question: Question,
        retrieved_passages: List[RetrievedPassage],
        method_name: str = "proposed_full",
        perturbation: PerturbationType = PerturbationType.CLEAN,
    ) -> EvaluationResult:
        """Evaluate a single query under the specified method variant without test condition leakage."""
        # Exact retrieval recall matching against gold passages (P0-07)
        gold_ids = set(question.gold_passage_ids)
        retrieved_ids = {p.chunk_id for p in retrieved_passages} | {p.doc_id for p in retrieved_passages}
        retrieval_recall = 1.0 if (gold_ids and bool(gold_ids.intersection(retrieved_ids))) else (0.0 if gold_ids else 1.0)

        # 1. Compute multi-signal evidence vector
        s_rel = self.relevance_analyzer.compute_relevance(retrieved_passages)
        s_cov, covered, missing = self.coverage_analyzer.compute_coverage(question.query_text, retrieved_passages)
        s_con, con_findings = self.contradiction_analyzer.analyze(retrieved_passages)

        # 2. Apply method-specific gating or ablation
        is_answered = True
        decision = None

        if method_name == "baseline_unconditional":
            # Baseline B1: Unconditional RAG (always attempts answer)
            is_answered = True

        elif method_name == "baseline_retrieval_only":
            # Baseline B2: Abstain if max retrieval relevance < threshold
            is_answered = s_rel >= self.thresholds.tau_ans_rel

        elif method_name == "baseline_confidence_only":
            # Baseline B3: Model observable retrieval/verbalized confidence gating (NO PERTURBATION LEAKAGE) (P0-08)
            # Purely based on top-1 candidate fused score and coverage
            top_score = retrieved_passages[0].fused_score if retrieved_passages else 0.0
            observable_confidence = (0.70 * top_score) + (0.30 * s_cov)
            is_answered = observable_confidence >= 0.55

        elif method_name == "ablation_no_contradiction":
            decision = self.policy_engine.evaluate(s_rel=s_rel, s_cov=s_cov, s_con=0.0, missing_entities=list(missing))
            is_answered = decision.action == DecisionAction.ANSWER

        elif method_name == "ablation_no_coverage":
            decision = self.policy_engine.evaluate(s_rel=s_rel, s_cov=1.0, s_con=s_con)
            is_answered = decision.action == DecisionAction.ANSWER

        elif method_name == "ablation_no_relevance":
            decision = self.policy_engine.evaluate(s_rel=1.0, s_cov=s_cov, s_con=s_con, missing_entities=list(missing))
            is_answered = decision.action == DecisionAction.ANSWER

        else:
            # Proposed full method: Multi-signal policy gating
            decision = self.policy_engine.evaluate(s_rel=s_rel, s_cov=s_cov, s_con=s_con, missing_entities=list(missing))
            is_answered = decision.action == DecisionAction.ANSWER

        # 3. Handle Abstention (P0-01)
        # An abstained query generates NO claims. Its answered faithfulness is None, NOT 1.0!
        if not is_answered:
            return EvaluationResult(
                run_id=f"run_{method_name}",
                question_id=question.question_id,
                perturbation_type=perturbation,
                method_name=method_name,
                is_answered=False,
                is_correct=None,
                claim_faithfulness=None,  # Separated outcome: None on abstention (remediates P0-01)
                selective_error=None,
                citation_precision=0.0,
                citation_recall=0.0,
                retrieval_recall=retrieval_recall,
                atomic_claims=[],
                backend_identifier=f"{self.generator.backend}:{self.generator.model_identifier}",
            )

        # 4. Generate answer on answered cases
        context = self.context_builder.assemble(retrieved_passages)
        gen = self.generator.generate(question.question_id, question.query_text, context)

        # Claim-level faithfulness evaluation
        faithfulness, atomic_claims = self.claim_evaluator.evaluate_claims(gen.raw_response, retrieved_passages)

        # Genuine citation verification (P0-06)
        cit_prec, cit_rec, citations = self.citation_analyzer.audit_citation_metrics(
            gen.raw_response,
            retrieved_passages,
            gold_passage_ids=question.gold_passage_ids,
        )

        # Post-generation citation grounding audit for proposed method (P0-09)
        if method_name == "proposed_full" and cit_prec < self.thresholds.tau_cit_min:
            # Fallback abstention if generated citations are ungrounded
            return EvaluationResult(
                run_id=f"run_{method_name}",
                question_id=question.question_id,
                perturbation_type=perturbation,
                method_name=method_name,
                is_answered=False,
                is_correct=None,
                claim_faithfulness=None,
                selective_error=None,
                citation_precision=cit_prec,
                citation_recall=cit_rec,
                retrieval_recall=retrieval_recall,
                atomic_claims=[],
                backend_identifier=f"{self.generator.backend}:{self.generator.model_identifier}",
            )

        selective_error = round(1.0 - faithfulness, 4)
        is_correct = faithfulness >= 0.70

        return EvaluationResult(
            run_id=f"run_{method_name}",
            question_id=question.question_id,
            perturbation_type=perturbation,
            method_name=method_name,
            is_answered=True,
            is_correct=is_correct,
            claim_faithfulness=round(faithfulness, 4),
            selective_error=selective_error,
            citation_precision=cit_prec,
            citation_recall=cit_rec,
            retrieval_recall=retrieval_recall,
            atomic_claims=atomic_claims,
            backend_identifier=f"{self.generator.backend}:{self.generator.model_identifier}",
        )
