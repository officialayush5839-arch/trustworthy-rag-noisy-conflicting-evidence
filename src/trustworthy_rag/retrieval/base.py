"""
Abstract base retriever contract for the Trustworthy RAG platform.
"""

from abc import ABC, abstractmethod
from typing import List
from trustworthy_rag.schemas import DocumentChunk, RetrievedPassage


class BaseRetriever(ABC):
    """Abstract interface for sparse and dense retrieval engines."""

    @abstractmethod
    def index(self, chunks: List[DocumentChunk]) -> None:
        """Index a list of document chunks into memory or vector store."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedPassage]:
        """Retrieve top_k passages for a given query string in descending score order."""
        pass
