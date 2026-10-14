"""
Baseline conventional RAG execution script.
Runs sparse BM25 + dense retrieval + hybrid RRF + unconditional answer generation.
"""

import json
import logging
from pathlib import Path
from typing import List

from trustworthy_rag.config import load_config
from trustworthy_rag.data.ingestion import DatasetIngestionEngine
from trustworthy_rag.generation.context_builder import ContextBuilder
from trustworthy_rag.generation.generator import AnswerGenerator
from trustworthy_rag.retrieval.bm25 import BM25Retriever
from trustworthy_rag.retrieval.dense import DenseRetriever
from trustworthy_rag.retrieval.fusion import RankFusionEngine
from trustworthy_rag.schemas import DocumentChunk, Question

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baseline_pipeline(
    config_path: str = "configs/default.yaml",
    split: str = "validation",
    limit: int = 50,
) -> Path:
    """Execute conventional RAG baseline on the chosen frozen dataset split."""
    config = load_config(config_path)
    logger.info(f"Loaded config: {config.experiment_name}")

    # Load frozen chunks and questions
    frozen_dir = Path(config.split_dir)
    chunk_file = frozen_dir / "chunks.jsonl"
    q_file = frozen_dir / f"questions_{'val' if split == 'validation' else split}.jsonl"

    if not chunk_file.is_file() or not q_file.is_file():
        raise FileNotFoundError(f"Missing frozen data at {frozen_dir}. Run scripts/freeze_datasets.py first.")

    ingestion = DatasetIngestionEngine()
    chunks: List[DocumentChunk] = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(DocumentChunk(**json.loads(line)))

    questions: List[Question] = ingestion.parse_questions_jsonl(q_file)[:limit]
    logger.info(f"Loaded {len(chunks)} chunks and {len(questions)} evaluation questions from {split} split.")

    # Initialize retrieval layer
    logger.info("Initializing BM25 and Dense retrievers...")
    bm25 = BM25Retriever(k1=config.retrieval.bm25_k1, b=config.retrieval.bm25_b)
    bm25.index(chunks)

    dense = DenseRetriever(model_name=config.retrieval.dense_model_name, use_mock_fallback=True)
    dense.index(chunks)

    fusion = RankFusionEngine(rrf_k=config.retrieval.rrf_k)
    context_builder = ContextBuilder(max_tokens=config.generator.max_tokens)
    generator = AnswerGenerator(backend="mock", model_identifier=config.generator.model_identifier)

    # Output directory
    out_dir = Path(config.output_dir) / "EXP_BASELINE"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = out_dir / "predictions.jsonl"

    logger.info(f"Executing baseline RAG queries. Writing outputs to {predictions_file}...")
    with open(predictions_file, "w", encoding="utf-8") as out_f:
        for i, q in enumerate(questions, start=1):
            sparse_hits = bm25.retrieve(q.query_text, top_k=config.retrieval.top_k)
            dense_hits = dense.retrieve(q.query_text, top_k=config.retrieval.top_k)
            fused_hits = fusion.fuse(sparse_hits, dense_hits, top_k=config.retrieval.top_k)

            context = context_builder.assemble(fused_hits)
            gen_result = generator.generate(query_id=q.question_id, query=q.query_text, context=context)

            out_f.write(gen_result.model_dump_json() + "\n")

    logger.info(f"Baseline pipeline completed successfully for {len(questions)} queries.")
    return predictions_file


if __name__ == "__main__":
    run_baseline_pipeline()
