# Product Requirements Document (PRD)

**Project Title:** Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence  
**Document ID:** PRD-SPEC-001  
**Status:** Approved Requirements  
**Version:** 1.0.0  
**Related Documents:** [architecture.md](./architecture.md), [goal.md](./goal.md), [phases.md](./phases.md), [rules.md](./rules.md), [memory.md](./memory.md), [task.md](./task.md)

---

## 1. Product Overview

The Trustworthy RAG System is an experimental research and software platform designed to evaluate, diagnose, and enhance the faithfulness of Retrieval-Augmented Generation (RAG) models under severe, controlled evidence corruption. 

Unlike conventional RAG setups that unconditionally generate text from whatever context is retrieved, this system implements an **evidence-aware decision policy engine** that audits retrieval relevance, evidence coverage, cross-passage factual contradiction, and citation entailment. Based on this multi-signal audit, the platform makes calibrated decisions to **Answer** (with verifiable chunk-level citations), **Request Clarification**, or **Abstain** (refuse to answer ungrounded queries).

The software is purpose-built to execute controlled empirical studies, compute empirical Risk–Coverage Pareto frontiers, conduct systematic ablations, and produce auditable scientific evidence suitable for publication in premier peer-reviewed AI venues.

---

## 2. Target Stakeholders and Core User Workflows

### 2.1 Target Stakeholders
1. **AI Researchers & RAG Specialists**: Design experiments, configure perturbation severities, execute ablation studies, and analyze risk–coverage curves.
2. **Machine Learning Engineers**: Maintain retrieval backends, optimize pairwise NLI inference, integrate local and API LLM generators, and maintain CI/CD testing pipelines.
3. **Research Evaluators & Peer Reviewers**: Audit raw generation logs, verify cryptographic dataset hashes, run independent reproduction scripts, and inspect error taxonomies.

### 2.2 End-to-End Operational Workflow
```
[Select Dataset & Frozen Split]
              |
              v
[Configure Retrieval & Model Parameters]
              |
              v
[Select Condition: Clean vs. Perturbed (Distractor, Removal, Duplicate, Contradiction)]
              |
              v
[Execute Pipeline: Baseline Control vs. Proposed Evidence-Aware Policy]
              |
              v
[Log Structured Generations, Chunk Provenance, & Intermediate Scores]
              |
              v
[Evaluate Atomic Claims, NLI Entailment, & Citation Precision]
              |
              v
[Compile Risk–Coverage Curves & Statistical Confidence Intervals]
              |
              v
[Export Reproducible Result Tables, LaTeX Figures, & Audit Trails]
```

---

## 3. Functional Requirements

### 3.1 Ingestion and Preprocessing
- **FR-001: Benchmark Dataset Ingestion**
  - *Description*: Ingest RAGTruth and Natural Questions / HotpotQA benchmark datasets from JSONL files into validated Pydantic objects.
  - *Rationale*: Clean data contracts prevent schema drift across downstream pipeline modules.
  - *Priority*: P0 (Mandatory) | *Dependencies*: None
  - *Acceptance Criteria*: Correctly parses questions, gold answers, and source passage metadata; rejects malformed records with descriptive validation errors.
- **FR-002: Auditable Dataset Split Freezing**
  - *Description*: Partition datasets into frozen `dev`, `validation`, and `test` splits and generate an immutable SHA-256 checksum manifest.
  - *Rationale*: Guarantees zero data leakage and ensures benchmark reproducibility across machines.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-001
  - *Acceptance Criteria*: Generates `manifest_sha256.json`; verification script asserts 100% hash consistency.

### 3.2 Storage and Retrieval
- **FR-003: Document Chunking and Provenance Anchoring**
  - *Description*: Segment corpus documents into chunks with standardized ID formatting (`{doc_id}_c{idx:04d}`), tracking character start/end offsets and token lengths.
  - *Rationale*: Precise chunk identifiers are required for fine-grained citation tracking and attribution.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-001
  - *Acceptance Criteria*: Every chunk references its parent document ID and exact character boundaries.
- **FR-004: Sparse Retrieval Engine (BM25)**
  - *Description*: Construct and query an in-memory or disk-backed BM25 index over all corpus chunks.
  - *Rationale*: Mandatory baseline retriever; provides transparent keyword matching.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-003
  - *Acceptance Criteria*: Returns top-$K$ chunks with rank positions and raw BM25 scores.
- **FR-005: Dense Retrieval Engine (Pinned Embeddings)**
  - *Description*: Encode chunks and queries using a pinned open embedding model (`BAAI/bge-small-en-v1.5`) and index via FAISS.
  - *Rationale*: Mandatory baseline retriever; provides semantic similarity matching.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-003
  - *Acceptance Criteria*: Computes cosine similarities; returns top-$K$ chunks in descending similarity order.
- **FR-006: Reciprocal Rank Fusion (RRF)**
  - *Description*: Combine sparse and dense ranked candidate lists using parameter $k_{rrf}=60$.
  - *Rationale*: Produces robust hybrid retrieval candidates combining lexical and semantic strengths.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-004, FR-005
  - *Acceptance Criteria*: Merges disjoint lists; outputs unified ranks and fused scores.

### 3.3 Controlled Stress Perturbation Suite
- **FR-007: Distractor Passage Injection**
  - *Description*: Inject topically similar but non-answering distractor passages into retrieved context at controlled ratios ($\alpha \in [0.2, 0.8]$).
  - *Rationale*: Tests model resilience against irrelevant context flooding.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Modifies context; logs injected chunk IDs in perturbation manifest.
- **FR-008: Key Supporting Passage Deletion**
  - *Description*: Identify and remove the essential gold supporting chunk from retrieved context.
  - *Rationale*: Tests whether the system abstains or hallucinates when evidence is absent.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Excises designated gold passage; verifies passage absence in context prompt.
- **FR-009: Near-Duplicate Passage Injection**
  - *Description*: Inject semantically redundant paraphrases of existing context chunks.
  - *Rationale*: Tests sensitivity to redundant distractors and prompt capacity dilution.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Injected duplicates reflect $\ge 0.85$ semantic similarity to target chunks.
- **FR-010: Explicit Contradiction Injection**
  - *Description*: Mutate key factual entities or numerical values in supporting chunks to create direct factual conflict.
  - *Rationale*: Tests cross-passage contradiction detection and refusal behavior.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Injected passages achieve pairwise NLI contradiction probability $> 0.75$ against original chunks.

### 3.4 Evidence Reasoning and Policy Engine
- **FR-011: Cross-Passage Contradiction Analyzer**
  - *Description*: Perform pairwise NLI classification across retrieved passages using `cross-encoder/nli-deberta-v3-large`.
  - *Rationale*: Prevents generating synthetic answers when evidence passages mutually disagree.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Outputs $K \times K$ contradiction probability matrix and aggregate score $s_{con}$.
- **FR-012: Relevance and Evidence Coverage Analyzers**
  - *Description*: Quantify query-passage relevance ($s_{rel}$) and query entity/predicate coverage ($s_{cov}$).
  - *Rationale*: Detects missing information before invoking generative models.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006
  - *Acceptance Criteria*: Outputs normalized scores in $[0.0, 1.0]$ with component diagnostic breakdown.
- **FR-013: Tri-State Decision Policy Engine**
  - *Description*: Evaluate score vector $\mathbf{s} = \langle s_{rel}, s_{cov}, s_{cit}, s_{con} \rangle$ against validation-calibrated thresholds to choose `ANSWER`, `CLARIFY`, or `ABSTAIN`.
  - *Rationale*: Core scientific contribution of the project.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-011, FR-012
  - *Acceptance Criteria*: Deterministically outputs decision enum with full textual explanation.
- **FR-014: Citation Support Grounding Audit**
  - *Description*: Parse generated answers for inline citations (`[doc_id_c001]`), decompose answers into atomic claims, and verify claim entailment against cited chunks.
  - *Rationale*: Ensures that claims emitted under the `ANSWER` decision are actually grounded.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-013
  - *Acceptance Criteria*: Computes citation grounding score $s_{cit}$; triggers abstention if $s_{cit} < \tau_{cit}$.

### 3.5 Baselines, Evaluation, and Reporting
- **FR-015: Mandatory Baseline Execution Harness**
  - *Description*: Execute and record results for Conventional RAG (unconditional), Retrieval-Only Gating, and Model Verbalized Confidence Gating.
  - *Rationale*: Mandatory controls required to scientifically evaluate the proposed policy.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-006, FR-013
  - *Acceptance Criteria*: Executes all baselines across identical evaluation queries and seeds.
- **FR-016: Signal Ablation Matrix**
  - *Description*: Execute individual signal ablation runs removing $s_{con}$, $s_{cov}$, $s_{cit}$, and $s_{rel}$ one at a time.
  - *Rationale*: Isolates the specific empirical contribution of each evidence signal.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-013, FR-015
  - *Acceptance Criteria*: Evaluates all ablation permutations on validation and test splits.
- **FR-017: Risk–Coverage Curve Compilation**
  - *Description*: Compute selective risk $\mathcal{R}$ across varying answer coverage levels $\mathcal{C} \in [0.1, 1.0]$ and calculate Area Under Risk–Coverage Curve (AURCC).
  - *Rationale*: Standard scientific metric for selective classification and calibrated abstention.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-015
  - *Acceptance Criteria*: Outputs numerical arrays and Pareto curves with bootstrap confidence intervals.
- **FR-018: End-to-End Reproduction CLI**
  - *Description*: Provide single CLI command `python run_reproduction.py` reproducing frozen test results.
  - *Rationale*: Guarantees peer-review audibility and public reproducibility.
  - *Priority*: P0 (Mandatory) | *Dependencies*: FR-017
  - *Acceptance Criteria*: Clean execution in fresh environment produces matching results within Monte Carlo variance.

---

## 4. Non-Functional Requirements (NFRs)

- **NFR-001: Strict Reproducibility**: Given an identical configuration file, dataset split, and random seed, the pipeline must produce bit-for-bit identical intermediate scoring vectors and deterministic decisions.
- **NFR-002: Modular Architecture**: All retrieval, scoring, decision, and evaluation modules must be decoupled via clean Python interfaces (`BaseRetriever`, `BaseScorer`, `BaseEvaluator`).
- **NFR-003: Test Coverage & Verification**: The codebase must maintain $>85\%$ automated unit and integration test coverage (`pytest --cov`).
- **NFR-004: Observability and Structured Logging**: Every query execution must log input IDs, candidate chunk ranks, scoring vectors, decision reasons, latency, and token consumption to structured JSONL logs.
- **NFR-005: Configuration Management**: All experimental parameters (model IDs, temperatures, $k$ values, thresholds, seeds) must be specified in declarative YAML files; no hard-coded magic numbers.
- **NFR-006: Graceful Failure Recovery**: System must handle API rate limits, model timeouts, and malformed inputs with explicit exponential backoff retries and descriptive error fallbacks.
- **NFR-007: Credential Security**: API keys (Gemini, OpenAI, Hugging Face) must be loaded strictly from local environment variables (`.env`); zero credentials committed to version control.
- **NFR-008: Resource Awareness & Latency Budget**: Mean end-to-end evidence scoring latency must not exceed 2.5 seconds per query on GPU (or 8.0 seconds on standard CPU).
- **NFR-009: Cross-Platform Portability**: Core codebase must execute natively on Windows 11 and Linux without requiring proprietary operating system drivers or external enterprise infrastructure.
- **NFR-010: Data Immutability**: Raw benchmark datasets and frozen splits must be mounted read-only during experiment runs; writing to raw data directories raises immediate permission exceptions.

---

## 5. Machine-Readable Interfaces and Output Formats

All primary pipeline outputs are exported as validated JSON/JSONL records matching the schemas in `architecture.md`.

### 5.1 Decision Result Record Example (`decision_record.json`)
```json
{
  "query_id": "ragtruth_qa_00142",
  "decision": "abstain",
  "reason": "Direct factual contradiction detected between retrieved evidence chunks",
  "evidence_score": {
    "relevance_score": 0.842,
    "coverage_score": 0.750,
    "citation_support_score": 0.0,
    "contradiction_score": 0.914,
    "composite_quality_score": 0.312,
    "is_contradictory": true,
    "feature_weights": {"rel": 0.25, "cov": 0.25, "cit": 0.25, "con": 0.25}
  },
  "applied_thresholds": {
    "tau_ans_rel": 0.65,
    "tau_cov_min": 0.60,
    "tau_con_max": 0.70,
    "tau_cit_min": 0.70
  },
  "contradiction_findings": [
    {
      "premise_chunk_id": "doc_102_c0001",
      "hypothesis_chunk_id": "doc_102_c0004",
      "contradiction_probability": 0.914,
      "entailment_probability": 0.021,
      "neutral_probability": 0.065
    }
  ],
  "timestamp": "2026-10-15T10:14:22Z"
}
```

### 5.2 Citation-Aware Answer Record Example (`answer_record.json`)
```json
{
  "query_id": "ragtruth_qa_00088",
  "decision": "answer",
  "reason": "Evidence consistent, query fully covered, draft claims verified",
  "generated_answer": "The merger between Company A and Company B was completed on March 14, 2023 [doc_045_c0002], for a total valuation of $4.2 billion [doc_045_c0003].",
  "citations": [
    {
      "citation_id": "cit_001",
      "chunk_id": "doc_045_c0002",
      "cited_text_span": "merger completed on March 14, 2023",
      "is_verified": true
    },
    {
      "citation_id": "cit_002",
      "chunk_id": "doc_045_c0003",
      "cited_text_span": "transaction valued at $4.2 billion",
      "is_verified": true
    }
  ],
  "latency_seconds": 1.42,
  "prompt_tokens": 612,
  "completion_tokens": 48
}
```

---

## 6. Comprehensive Acceptance Criteria Checklist

- [ ] **Data Pipeline**: RAGTruth and general QA benchmark subsets are ingested, chunked with offset tracking, and frozen with SHA-256 manifests.
- [ ] **Dual Retrieval**: BM25 and dense retrieval execute independently and fuse via RRF with verified ranking tests.
- [ ] **Stress Suite**: All four perturbations (distractor, removal, duplicate, contradiction) generate auditable perturbed sets without altering clean data.
- [ ] **Evidence Policy**: Multi-signal scoring engine computes $s_{rel}, s_{cov}, s_{con}, s_{cit}$ and emits deterministic `ANSWER`, `CLARIFY`, and `ABSTAIN` decisions.
- [ ] **Baselines**: Conventional RAG, retrieval-threshold gating, and self-reported confidence gating are executed across identical splits and seeds.
- [ ] **Ablation Matrix**: All four individual signal ablations are evaluated to prove signal utility.
- [ ] **Risk–Coverage Curves**: Empirical Pareto curves are compiled and plotted with 95% bootstrap confidence intervals.
- [ ] **Human Validation**: Automated claim-level judge aligns with human labels on $N \ge 100$ cases with Cohen's $\kappa \ge 0.70$.
- [ ] **Reproduction CLI**: Standalone reproduction script executes from terminal and replicates primary findings.

---

## 7. Minimal Operator Interface (CLI Specification)

A graphical user interface is explicitly out of scope. The platform is operated via a robust Command Line Interface (`rag-trust`):
```bash
# Ingest and freeze benchmark data
python -m trustworthy_rag.cli ingest --dataset ragtruth --split-dev 50 --split-val 100 --split-test 250

# Run baseline RAG on clean test set
python -m trustworthy_rag.cli run --config configs/baseline_rag.yaml --condition clean

# Run perturbation stress tests
python -m trustworthy_rag.cli stress --condition contradiction --severity 0.6 --seed 42

# Run full evidence-aware evaluation and ablation sweep
python -m trustworthy_rag.cli evaluate --policy configs/policy_calibrated.yaml --ablations all

# Replicate paper figures and tables
python -m trustworthy_rag.cli report --out reports/
```

---

## 8. Out-of-Scope Features

To protect research velocity and scientific focus, the following features are strictly deferred:
1. Multi-tenant web user interfaces, authentication, and role-based access control.
2. Production microservice orchestration (Kubernetes clusters, distributed message queues).
3. Unconstrained, multi-turn conversational memory systems.
4. Model pre-training or fine-tuning from scratch.
5. Real-time dynamic web scraping pipelines.
