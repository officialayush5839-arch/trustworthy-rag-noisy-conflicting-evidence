"""
Controlled perturbation stress suite for RAG evidence corruption.
"""

from trustworthy_rag.stress.contradiction_injection import ContradictionInjector
from trustworthy_rag.stress.distractor import DistractorInjector
from trustworthy_rag.stress.near_duplicate import NearDuplicateInjector
from trustworthy_rag.stress.passage_removal import PassageRemovalOperator
from trustworthy_rag.stress.perturbation_engine import PerturbationEngine

__all__ = [
    "ContradictionInjector",
    "DistractorInjector",
    "NearDuplicateInjector",
    "PassageRemovalOperator",
    "PerturbationEngine",
]
