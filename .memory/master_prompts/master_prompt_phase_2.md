# Master Prompt: Phase 2 — Conventional RAG Baseline

**Phase:** Phase 2  
**Target:** Sparse BM25 Retriever, Dense Embedding Retriever, Hybrid RRF Fusion, Context Builder, and Baseline Generation Harness  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Implement `src/trustworthy_rag/retrieval/base.py`:
   - Abstract `BaseRetriever` interface declaring `index(chunks)` and `retrieve(query, top_k) -> List[RetrievedPassage]`.
2. Implement `src/trustworthy_rag/retrieval/bm25.py`:
   - Standard Okapi BM25 ranking over chunk tokens (`rank_bm25`).
   - Normalization of BM25 scores and rank assignments.
3. Implement `src/trustworthy_rag/retrieval/dense.py`:
   - Vector embeddings using `sentence-transformers` (`BAAI/bge-small-en-v1.5` or `all-MiniLM-L6-v2`) with cosine similarity matrix/FAISS fallback.
   - Deterministic vector cache and index lookup.
4. Implement `src/trustworthy_rag/retrieval/fusion.py`:
   - Reciprocal Rank Fusion ($k_{rrf}=60$) combining BM25 and Dense result sets.
   - Unified rank sorting and score normalization.
5. Implement `src/trustworthy_rag/generation/context_builder.py`:
   - Provenance-anchored context assembler with chunk headers `[Chunk ID: {chunk_id}]`.
   - Character offset tracking within assembled prompt.
6. Implement `src/trustworthy_rag/generation/generator.py`:
   - Baseline generator executing prompt context against pinned generative models (API or local open model).
   - Generates responses with latency and token telemetry.
7. Implement `scripts/run_baseline.py`:
   - Executes baseline RAG over frozen validation split ($N=100$) and records outputs to `experiments/runs/EXP_BASELINE/predictions.jsonl`.
8. Testing:
   - Unit tests: `test_bm25.py`, `test_dense.py`, `test_fusion.py`, `test_context_builder.py`.
   - Integration test: `test_baseline_pipeline.py`.

---

## 2. Verification Gates
- All retrieval, fusion, context, and baseline unit/integration tests pass with 100% success rate.
- `scripts/run_baseline.py` executes end-to-end and outputs structured JSONL logs.
