"""
BM25 sparse keyword retriever implementation.
"""

import re
from typing import Dict, List
from rank_bm25 import BM25Okapi

from trustworthy_rag.retrieval.base import BaseRetriever
from trustworthy_rag.schemas import DocumentChunk, RetrievedPassage


class BM25Retriever(BaseRetriever):
    """Okapi BM25 sparse retriever operating over tokenized document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.chunk_map: Dict[str, DocumentChunk] = {}
        self.bm25: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Normalize and tokenize text into lowercase word tokens."""
        return re.findall(r"\w+", text.lower())

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index over provided document chunks."""
        self.chunks = list(chunks)
        self.chunk_map = {c.chunk_id: c for c in chunks}
        corpus = [self.tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if corpus else None

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedPassage]:
        """Retrieve top_k chunks matching query in descending BM25 score order."""
        if not self.bm25 or not self.chunks:
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        scored_pairs = [(self.chunks[i], float(score)) for i, score in enumerate(scores)]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        selected = scored_pairs[:top_k]
        if not selected:
            return []

        max_score = max(s for _, s in selected) if selected else 1.0
        min_score = min(s for _, s in selected) if selected else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0

        results: List[RetrievedPassage] = []
        for rank, (chunk, raw_score) in enumerate(selected, start=1):
            normalized = (raw_score - min_score) / score_range if score_range > 0 else 1.0
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    sparse_score=float(raw_score),
                    dense_score=None,
                    fused_score=float(normalized),
                    rank=rank,
                    retrieval_method="bm25",
                )
            )

        return results
