# Master Prompt: Phase 1 — Data & Benchmark Preparation

**Phase:** Phase 1  
**Target:** Data Ingestion Engine, Sliding-Window Text Chunker with Offset Tracking, and Frozen Auditable Benchmark Splits  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Implement `src/trustworthy_rag/data/chunker.py`:
   - Text segmentation with configurable `chunk_size` and `chunk_overlap`.
   - Exact `start_char_offset` and `end_char_offset` tracking against parent text.
   - Deterministic chunk ID generator: `{doc_id}_c{idx:04d}`.
   - Cryptographic SHA-256 content hashing for every chunk.
2. Implement `src/trustworthy_rag/data/ingestion.py`:
   - Ingestion parsers for benchmark formats (RAGTruth, Natural Questions / HotpotQA).
   - Validates incoming raw records into typed Pydantic `Document` and `Question` schemas.
   - Curates an auditable benchmark corpus with realistic question-context-answer triples.
3. Implement `scripts/freeze_datasets.py`:
   - Partitions data into frozen, disjoint subsets: `dev` ($N=50$), `validation` ($N=100$), `test` ($N=250$).
   - Generates immutable cryptographic manifest `data/splits/manifest_sha256.json`.
   - Verifies split disjointness (zero data leakage between validation and test).
4. Unit tests:
   - `tests/unit/test_chunker.py`: verifies character offset reconstruction, overlap math, and SHA-256 hashing.
   - `tests/unit/test_dataset_validation.py`: validates ingestion parsing, bad record rejection, and schema enforcement.
   - `tests/unit/test_splits.py`: tests split disjointness, manifest verification, and tamper detection.

---

## 2. Verification Gates
- `pytest tests/unit/` passes 100% with chunker, ingestion, and split validation tests.
- `manifest_sha256.json` created and verified.
