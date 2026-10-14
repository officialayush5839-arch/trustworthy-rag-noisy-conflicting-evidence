# Master Prompt: Phase 0 — Specification, Environment Audit & Core Foundation

**Phase:** Phase 0  
**Target:** Python Package Foundation, Directory Scaffolding, Pydantic v2 Schemas, and Test Harness  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Scaffold Python package structure:
   - `src/trustworthy_rag/`
   - `src/trustworthy_rag/schemas/`
   - `src/trustworthy_rag/retrieval/`
   - `src/trustworthy_rag/stress/`
   - `src/trustworthy_rag/evidence/`
   - `src/trustworthy_rag/policy/`
   - `src/trustworthy_rag/generation/`
   - `src/trustworthy_rag/evaluation/`
   - `tests/unit/`, `tests/integration/`
   - `configs/`, `data/frozen/`, `experiments/runs/`
2. Define `pyproject.toml` and locked dependencies:
   - Python `>=3.11`
   - `pydantic>=2.7.0`
   - `rank-bm25>=0.2.2`
   - `sentence-transformers>=3.0.0`
   - `transformers>=4.42.0`
   - `faiss-cpu>=1.8.0`
   - `pytest>=8.2.0`, `pytest-cov>=5.0.0`
   - `pyyaml>=6.0.1`
   - `loguru>=0.7.2`
3. Implement core immutable data models matching `architecture.md`:
   - `Document`, `DocumentChunk`, `Question`
   - `RetrievedPassage`, `PerturbationMetadata`
   - `ContradictionFinding`, `EvidenceQualityScore`, `DecisionResult`
   - `Citation`, `GenerationResult`, `AtomicClaim`, `EvaluationResult`
4. Implement configuration loader (`src/trustworthy_rag/config.py`) parsing YAML into validated `ExperimentConfig`.
5. Unit tests validating schema serialization, immutability, and boundary constraints (`tests/unit/test_schemas.py`, `tests/unit/test_config.py`).

---

## 2. Verification Gates
- `pip install -e .` succeeds in the local environment.
- `pytest tests/unit/` runs with 100% pass rate.
- Zero schema validation regressions.
