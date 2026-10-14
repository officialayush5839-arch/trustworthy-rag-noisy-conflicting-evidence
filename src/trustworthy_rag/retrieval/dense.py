"""
Dense embedding vector retriever implementation.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from trustworthy_rag.retrieval.base import BaseRetriever
from trustworthy_rag.schemas import DocumentChunk, RetrievedPassage

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense vector retriever utilizing neural sentence embeddings and cosine similarity."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", use_mock_fallback: bool = False):
        self.model_name = model_name
        self.use_mock_fallback = use_mock_fallback
        self.chunks: List[DocumentChunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        self._model = None

    def _get_model(self):
        """Lazy load sentence transformer model with fallback."""
        if self._model is None and not self.use_mock_fallback:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading dense embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer '{self.model_name}': {e}. Using deterministic embedding fallback.")
                self.use_mock_fallback = True
        return self._model

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode list of strings to normalized unit vectors."""
        model = self._get_model()
        if model is not None and not self.use_mock_fallback:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)

        # Deterministic lightweight n-gram/hash embedding fallback for offline/isolated tests
        dim = 384
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = text.lower().split()
            for w in words:
                idx = hash(w) % dim
                vectors[i, idx] += 1.0
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
            else:
                vectors[i, 0] = 1.0
        return vectors

    def index(self, chunks: List[DocumentChunk]) -> None:
        """Encode and index all document chunks into memory."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.chunk_embeddings = None
            return

        texts = [c.text for c in self.chunks]
        self.chunk_embeddings = self._encode_text(texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedPassage]:
        """Retrieve top_k most semantically similar chunks for the query."""
        if not self.chunks or self.chunk_embeddings is None or not query.strip():
            return []

        q_vec = self._encode_text([query])[0]
        # Inner product with unit vectors equals cosine similarity
        similarities = np.dot(self.chunk_embeddings, q_vec)

        # Clip similarities to [0.0, 1.0] for safety
        normalized_sims = (np.clip(similarities, -1.0, 1.0) + 1.0) / 2.0

        top_indices = np.argsort(-normalized_sims)[:top_k]

        results: List[RetrievedPassage] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            sim = float(normalized_sims[idx])
            results.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    sparse_score=None,
                    dense_score=sim,
                    fused_score=sim,
                    rank=rank,
                    retrieval_method="dense",
                )
            )

        return results
