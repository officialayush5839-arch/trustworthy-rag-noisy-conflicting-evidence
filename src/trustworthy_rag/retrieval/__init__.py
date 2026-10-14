"""
Retrieval engine modules: Sparse BM25, Dense Embeddings, and Hybrid Rank Fusion.
"""

from trustworthy_rag.retrieval.base import BaseRetriever
from trustworthy_rag.retrieval.bm25 import BM25Retriever
from trustworthy_rag.retrieval.dense import DenseRetriever
from trustworthy_rag.retrieval.fusion import RankFusionEngine

__all__ = [
    "BaseRetriever",
    "BM25Retriever",
    "DenseRetriever",
    "RankFusionEngine",
]
