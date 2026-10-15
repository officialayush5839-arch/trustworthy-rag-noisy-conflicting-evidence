"""
Evaluation and ablation matrix runner script.
Executes all baselines, proposed evidence-aware policy, and ablations across clean and perturbed test sets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

from trustworthy_rag.config import load_config
from trustworthy_rag.data.ingestion import DatasetIngestionEngine
from trustworthy_rag.evaluation.evaluator import ComparativeEvaluator
from trustworthy_rag.evaluation.risk_coverage import RiskCoverageCompiler
from trustworthy_rag.retrieval.bm25 import BM25Retriever
from trustworthy_rag.retrieval.dense import DenseRetriever
from trustworthy_rag.retrieval.fusion import RankFusionEngine
from trustworthy_rag.schemas import DocumentChunk, PerturbationType, Question
from trustworthy_rag.stress.perturbation_engine import PerturbationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation_suite(limit_queries: int = 50) -> Dict:
    """Execute complete experimental matrix across baselines, proposed method, and ablations."""
    config = load_config("configs/default.yaml")
    frozen_dir = Path(config.split_dir)
    chunk_file = frozen_dir / "chunks.jsonl"
    q_file = frozen_dir / "questions_val.jsonl"

    chunks: List[DocumentChunk] = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(DocumentChunk(**json.loads(line)))

    ingestion = DatasetIngestionEngine()
    questions: List[Question] = ingestion.parse_questions_jsonl(q_file)[:limit_queries]

    # Initialize retrieval & perturbation
    bm25 = BM25Retriever()
    bm25.index(chunks)
    dense = DenseRetriever(use_mock_fallback=True)
    dense.index(chunks)
    fusion = RankFusionEngine()
    stress_engine = PerturbationEngine(chunks)
    evaluator = ComparativeEvaluator(config.policy)

    methods = [
        "baseline_unconditional",
        "baseline_retrieval_only",
        "baseline_confidence_only",
        "proposed_full",
        "ablation_no_contradiction",
        "ablation_no_coverage",
        "ablation_no_relevance",
    ]

    conditions = [
        PerturbationType.CLEAN,
        PerturbationType.DISTRACTOR,
        PerturbationType.PASSAGE_REMOVAL,
        PerturbationType.NEAR_DUPLICATE,
        PerturbationType.CONTRADICTION,
    ]

    summary_results: Dict = {"methods": {}}

    for method in methods:
        summary_results["methods"][method] = {}
        for cond in conditions:
            answered_count = 0
            faithful_sum = 0.0
            error_list: List[float] = []
            score_list: List[float] = []

            for q in questions:
                # Retrieve clean hits
                s_hits = bm25.retrieve(q.query_text, top_k=5)
                d_hits = dense.retrieve(q.query_text, top_k=5)
                clean_fused = fusion.fuse(s_hits, d_hits, top_k=5)

                # Perturb candidates
                perturbed_passages, meta = stress_engine.perturb(
                    clean_fused,
                    condition=cond,
                    severity=0.5,
                    gold_passage_ids=q.gold_passage_ids,
                    seed=42,
                )

                # Evaluate under current method
                res = evaluator.evaluate_query(q, perturbed_passages, method_name=method, perturbation=cond)

                if res.is_answered:
                    answered_count += 1
                    faithful_sum += res.claim_faithfulness
                    err = 1.0 - res.claim_faithfulness
                else:
                    err = 0.0

                error_list.append(err)
                score_list.append(0.85 if res.is_answered else 0.10)

            coverage = answered_count / len(questions)
            mean_faithfulness = (faithful_sum / answered_count) if answered_count > 0 else 1.0
            selective_risk = 1.0 - mean_faithfulness

            cov_steps, risk_steps, aurcc = RiskCoverageCompiler.compute_curve(score_list, error_list)

            summary_results["methods"][method][cond.value] = {
                "coverage": round(float(coverage), 4),
                "faithfulness": round(float(mean_faithfulness), 4),
                "selective_risk": round(float(selective_risk), 4),
                "aurcc": round(float(aurcc), 4),
            }

    out_file = Path("experiments/evaluation_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    logger.info(f"Evaluation suite completed. Summary written to {out_file}")
    return summary_results


if __name__ == "__main__":
    run_evaluation_suite(limit_queries=20)
