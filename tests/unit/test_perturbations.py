"""
Unit tests for the four controlled perturbation stress operators.
"""

from trustworthy_rag.schemas import DocumentChunk, PerturbationType, RetrievedPassage
from trustworthy_rag.stress import PerturbationEngine


def create_sample_passages():
    clean_passages = [
        RetrievedPassage(chunk_id="chunk_01", doc_id="doc_01", text="The merger concluded on March 14, 2023 for $4.2 billion.", fused_score=1.0, rank=1, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="chunk_02", doc_id="doc_02", text="Company headquarters were established in Zurich in 1995.", fused_score=0.8, rank=2, retrieval_method="hybrid_rrf"),
        RetrievedPassage(chunk_id="chunk_03", doc_id="doc_03", text="Annual revenue surpassed expectations across all sectors.", fused_score=0.6, rank=3, retrieval_method="hybrid_rrf"),
    ]
    corpus_chunks = [
        DocumentChunk(chunk_id="distract_99", doc_id="doc_distract", text="Baking sourdough bread requires precise fermentation temperatures.", start_char_offset=0, end_char_offset=67, token_count=8, sha256_hash="h99"),
        DocumentChunk(chunk_id="distract_98", doc_id="doc_distract", text="Deep sea trenches exhibit extreme hydrostatic pressure.", start_char_offset=0, end_char_offset=56, token_count=7, sha256_hash="h98"),
    ]
    return clean_passages, corpus_chunks


def test_clean_perturbation():
    clean, corpus = create_sample_passages()
    engine = PerturbationEngine(corpus)
    perturbed, meta = engine.perturb(clean, condition=PerturbationType.CLEAN)

    assert len(perturbed) == len(clean)
    assert meta.perturbation_type == PerturbationType.CLEAN
    assert meta.injected_chunk_ids == []
    assert meta.removed_chunk_ids == []


def test_distractor_injection():
    clean, corpus = create_sample_passages()
    engine = PerturbationEngine(corpus)
    perturbed, meta = engine.perturb(clean, condition=PerturbationType.DISTRACTOR, severity=0.5, seed=42)

    assert len(perturbed) > len(clean)
    assert meta.perturbation_type == PerturbationType.DISTRACTOR
    assert len(meta.injected_chunk_ids) > 0
    # Clean list remained unmodified
    assert len(clean) == 3


def test_passage_removal():
    clean, corpus = create_sample_passages()
    engine = PerturbationEngine(corpus)
    perturbed, meta = engine.perturb(clean, condition=PerturbationType.PASSAGE_REMOVAL, gold_passage_ids=["chunk_01"])

    assert len(perturbed) == len(clean) - 1
    assert "chunk_01" in meta.removed_chunk_ids
    assert all(p.chunk_id != "chunk_01" for p in perturbed)
    # Check consecutive ranks
    assert [p.rank for p in perturbed] == [1, 2]


def test_near_duplicate_injection():
    clean, corpus = create_sample_passages()
    engine = PerturbationEngine(corpus)
    perturbed, meta = engine.perturb(clean, condition=PerturbationType.NEAR_DUPLICATE, severity=0.5)

    assert len(perturbed) > len(clean)
    assert meta.perturbation_type == PerturbationType.NEAR_DUPLICATE
    assert len(meta.injected_chunk_ids) > 0
    assert any("dup_" in p.chunk_id for p in perturbed)


def test_contradiction_injection():
    clean, corpus = create_sample_passages()
    engine = PerturbationEngine(corpus)
    perturbed, meta = engine.perturb(clean, condition=PerturbationType.CONTRADICTION, gold_passage_ids=["chunk_01"])

    assert len(perturbed) == len(clean) + 1
    assert meta.perturbation_type == PerturbationType.CONTRADICTION
    assert len(meta.injected_chunk_ids) == 1
    conflict_chunk = next(p for p in perturbed if p.chunk_id == meta.injected_chunk_ids[0])
    assert "conflict_chunk_01" == conflict_chunk.chunk_id
    # Assert year was mutated from 2023
    assert "2023" not in conflict_chunk.text
