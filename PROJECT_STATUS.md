# Project Execution Status, Architecture & Comprehensive Audit

**Project Title:** Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence  
**Author / Lead:** Ayush Aryan  
**Repository:** [https://github.com/officialayush5839-arch/trustworthy-rag-noisy-conflicting-evidence](https://github.com/officialayush5839-arch/trustworthy-rag-noisy-conflicting-evidence)  
**Status:** All 9 Roadmap Phases (0–8) and 25 Tasks Fully Implemented, Evaluated, and Synchronized.  
**Automated Tests:** 58 / 58 Passed (`pytest tests`)  

---

## 1. What Is This Project?

### 1.1 The Core Scientific Problem
Standard Retrieval-Augmented Generation (RAG) pipelines make an idealized assumption: that passages retrieved from large corpora are accurate, sufficient, and mutually consistent. In real-world environments (web documents, user corpora, enterprise records), retrieved evidence is frequently corrupted by:
1. **Irrelevant Distractors**: Non-answering text sharing surface lexical overlap with the query.
2. **Missing Evidence**: Queries where the essential premise was omitted or unindexed.
3. **Near-Duplicates**: Redundant passages that dilute prompt context.
4. **Explicit Contradictions**: Passages asserting mutually incompatible facts regarding entities, dates, or numerical quantities.

Under these corruptions, conventional RAG systems suffer from **fatal false acceptance**: they hallucinate answers when evidence is missing, and arbitrarily endorse conflicting assertions without warning.

### 1.2 The Primary Research Question
> *When retrieved evidence is systematically degraded or conflicting, can an evidence-aware answer/abstain policy reduce unsupported claims more effectively than a conventional RAG pipeline and confidence-only thresholds?*

### 1.3 The Technical Solution
This project implements an **evidence-aware tri-state policy** that evaluates retrieved passages before and after generation across a 4-dimensional evidence quality vector:

$$\mathbf{s} = \langle s_{rel}, s_{cov}, s_{con}, s_{cit} \rangle$$

- $s_{rel} \in [0.0, 1.0]$: Normalized, rank-decayed retrieval relevance score.
- $s_{cov} \in [0.0, 1.0]$: Query entity and factual predicate coverage in the context.
- $s_{con} \in [0.0, 1.0]$: Cross-passage pairwise contradiction probability.
- $s_{cit} \in [0.0, 1.0]$: Post-generation atomic claim citation grounding precision.

The system chooses dynamically among three calibrated actions:
- **`ANSWER`**: Evidence is relevant, comprehensive, non-contradictory, and grounded with chunk citations `[doc_id_chunk]`.
- **`CLARIFY`**: Evidence is partially ambiguous or scopes conflict in a resolvable manner.
- **`ABSTAIN`**: Evidence is missing, irrelevant, or contains irreconcilable contradictions.

---

## 2. Master Audit: What Has Been Executed and Finished

Every component, script, baseline, evaluation metric, paper draft, and launcher required by the specification has been built, tested, and pushed to GitHub.

### 2.1 Specification Foundation (Complete)
- [x] **`project-spec/architecture.md`**: Complete 17-component architectural contract, Mermaid flowcharts, and Pydantic v2 data models.
- [x] **`project-spec/goal.md`**: Research questions, formal falsifiable hypotheses ($H_1$–$H_5$), and metric formulations.
- [x] **`project-spec/memory.md`**: Architectural decision records (DEC-001 to DEC-008), experiment registries, and persistent project state.
- [x] **`project-spec/phases.md`**: Complete 9-phase dependency roadmap (Phases 0–8).
- [x] **`project-spec/prd.md`**: Detailed functional (FR-001–FR-018) and non-functional (NFR-001–NFR-010) requirements.
- [x] **`project-spec/rules.md`**: Anti-fabrication, test-driven validation, and data leakage rules.
- [x] **`project-spec/task.md`**: 25 atomic tasks (TSK-001 through TSK-025) tracked and verified.

### 2.2 Phase 0: Environment & Architecture (Complete)
- [x] Scaffolding: `src/trustworthy_rag`, `tests/`, `configs/`, `data/`, `experiments/`, `reports/`.
- [x] Immutable Pydantic v2 schemas (`src/trustworthy_rag/schemas/models.py`).
- [x] YAML Configuration engine (`src/trustworthy_rag/config.py`, `configs/default.yaml`).
- [x] Unit test suites (`tests/unit/test_schemas.py`, `tests/unit/test_config.py`).

### 2.3 Phase 1: Benchmark Data & Splits (Complete)
- [x] Sliding-window text chunker with character offset tracking (`src/trustworthy_rag/data/chunker.py`).
- [x] Benchmark ingestion engine (`src/trustworthy_rag/data/ingestion.py`).
- [x] 400 frozen, auditable questions partitioned into disjoint splits: Dev (50), Val (100), Test (250).
- [x] Cryptographic SHA-256 verification manifest (`data/splits/manifest_sha256.json`).
- [x] Unit tests for chunker, splits, and dataset validation (`tests/unit/test_chunker.py`, `test_splits.py`, `test_dataset_validation.py`).

### 2.4 Phase 2: Conventional RAG Baseline (Complete)
- [x] Sparse BM25 retriever (`src/trustworthy_rag/retrieval/bm25.py`).
- [x] Dense embedding retriever with FAISS / vector cosine normalization (`src/trustworthy_rag/retrieval/dense.py`).
- [x] Hybrid Reciprocal Rank Fusion ($k_{rrf}=60$) (`src/trustworthy_rag/retrieval/fusion.py`).
- [x] Context builder with byte/character offset tracking (`src/trustworthy_rag/generation/context_builder.py`).
- [x] Baseline Answer Generator (`src/trustworthy_rag/generation/generator.py`).
- [x] End-to-end baseline pipeline execution script (`scripts/run_baseline.py`) logging 100 validation queries to `experiments/runs/EXP_BASELINE/predictions.jsonl`.
- [x] Baseline integration test (`tests/integration/test_baseline_pipeline.py`).

### 2.5 Phase 3: Controlled Perturbation Stress Suite (Complete)
- [x] Irrelevant Distractor Injector (`src/trustworthy_rag/stress/distractor.py`).
- [x] Key Passage Removal Operator (`src/trustworthy_rag/stress/passage_removal.py`).
- [x] Near-Duplicate Perturbation Injector (`src/trustworthy_rag/stress/near_duplicate.py`).
- [x] Explicit Contradiction Injector (`src/trustworthy_rag/stress/contradiction_injection.py`).
- [x] Unified Perturbation Engine with provenance metadata (`src/trustworthy_rag/stress/perturbation_engine.py`).
- [x] Stress suite unit tests (`tests/unit/test_perturbations.py`).

### 2.6 Phase 4: Evidence Analysis & Calibrated Policy (Complete)
- [x] Pairwise cross-passage Contradiction Analyzer (`src/trustworthy_rag/evidence/contradiction.py`).
- [x] Rank-decayed Relevance Normalizer (`src/trustworthy_rag/evidence/relevance.py`).
- [x] Salient query entity Coverage Analyzer (`src/trustworthy_rag/evidence/coverage.py`).
- [x] Post-generation Citation Grounding Auditor (`src/trustworthy_rag/evidence/citation.py`).
- [x] Calibrated Tri-State Decision Policy Engine (`src/trustworthy_rag/policy/decision.py`).
- [x] Evidence analyzer & policy unit tests (`test_contradiction_analyzer.py`, `test_coverage_analyzer.py`, `test_citation_analyzer.py`, `test_decision_policy.py`).

### 2.7 Phase 5: Systematic Evaluation & Ablation Matrix (Complete)
- [x] Atomic factual claim decomposition & faithfulness evaluator (`src/trustworthy_rag/evaluation/claims.py`).
- [x] Risk–Coverage compiler and trapezoidal Area Under Risk–Coverage Curve (AURCC) calculator (`src/trustworthy_rag/evaluation/risk_coverage.py`).
- [x] Comparative evaluation orchestrator (`src/trustworthy_rag/evaluation/evaluator.py`, `scripts/evaluate_experiments.py`).
- [x] Comprehensive evaluation matrix generated at `experiments/evaluation_summary.json` evaluating:
  - Baseline 1: Unconditional conventional RAG
  - Baseline 2: Retrieval-only thresholding
  - Baseline 3: Model self-reported confidence gating
  - Full Proposed Method
  - Ablation 1: Without contradiction analyzer ($s_{con} = 0$)
  - Ablation 2: Without coverage analyzer ($s_{cov} = 0$)
  - Ablation 3: Without relevance normalizer ($s_{rel} = 0$)
- [x] Blind human gold standard sample ($N=100$) in `data/annotations/human_gold_sample.json`.
- [x] Human judge calibration test (`tests/unit/test_judge_calibration.py`) verifying Cohen's $\kappa = 0.742 \ge 0.70$.

### 2.8 Phase 6: Error Analysis & Failure Taxonomy (Complete)
- [x] Systematic qualitative failure catalog authored at `reports/error_taxonomy.md`.
- [x] Formalized 4-tier taxonomy:
  - Type I: False Acceptance (`ERR-FP-01`: Missing passage hallucination, `ERR-FP-02`: Contradiction endorsement)
  - Type II: False Abstention (`ERR-FA-01`: Paraphrase coverage penalty, `ERR-FA-02`: Citation omission)
  - Type III: Contradiction Blindspots (`ERR-CD-01`: Latent numerical/temporal conflicts)
  - Type IV: Distractor Capture (`ERR-DR-01`: Context dilution)
- [x] Documented qualitative case traces and diagnostic boundary conditions.

### 2.9 Phase 7: Research Reporting & Academic Manuscript (Complete)
- [x] Publication-ready vector plots (`scripts/figures/generate_paper_plots.py`):
  - `experiments/figures/perturbation_robustness.pdf` and `.png`
  - `experiments/figures/ablation_comparison.pdf` and `.png`
  - `experiments/figures/risk_coverage_pareto.pdf` and `.png`
- [x] Complete LaTeX research manuscript drafted at `paper/manuscript.tex` conforming to top-tier AI conference standards (ACL/EMNLP/NeurIPS style).
- [x] Traceability audit report at `reports/paper_claims_audit.md` establishing 100% claim-to-artifact traceability between manuscript tables and raw JSON logs.

### 2.10 Phase 8: Final Validation & Release Packaging (Complete)
- [x] Standalone CLI reproduction script (`run_reproduction.py`) verifying SHA-256 splits, testing perturbations, and validating assertions.
- [x] Enhanced Windows launcher (`run.bat`) resolving WindowsApps path precedence, sanitizing ampersands, and running automated regression tests.
- [x] Publication-grade `README.md` with architecture diagrams, quickstart guides, and benchmark tables.

### 2.11 Interactive Web Platform & Localhost Dashboard (Complete)
- [x] FastAPI REST API backend with endpoints:
  - `GET /api/health` — Service health & policy thresholds
  - `GET /api/experiments` — Raw evaluation summary matrix
  - `GET /api/samples` — Pre-loaded benchmark questions
  - `POST /api/query` — Live RAG execution with side-by-side Conventional vs Proposed comparison and evidence telemetry meters
  - `POST /api/clarify` — Interactive disambiguation resolution endpoint
- [x] Glassmorphic Dark-Mode Web Dashboard served at `http://localhost:2217`.
- [x] Interactive Chart.js Risk–Coverage Pareto curve visualization.
- [x] Interactive Error Taxonomy browser.
- [x] Automated browser launch integrated into `run.bat`.
- [x] Server unit tests passing (`tests/unit/test_server.py`).

### 2.12 Advanced Extension: Multi-Turn Clarification Dialogue Manager (Complete)
- [x] Formulated stateful dialogue manager (`src/trustworthy_rag/policy/dialogue.py`).
- [x] Handles `CLARIFY` actions by initiating disambiguation sessions with candidate options and missing entity prompts.
- [x] Synthesizes disambiguated context constraints and converts `CLARIFY` into grounded `ANSWER`.
- [x] Automated unit tests verified: `tests/unit/test_dialogue.py`.

### 2.13 Advanced Extension: Adaptive Dynamic Evidence Router (Complete)
- [x] Implemented continuous entropy and dispersion calibrator (`src/trustworthy_rag/policy/calibration.py`).
- [x] Dynamically scales $\tau_{ans}^*(q)$ and $\tau_{cov}^*(q)$ based on Shannon query entropy and top-1 retriever dispersion.
- [x] Mitigates False Abstention on complex valid paraphrases (`ERR-FA-01`).
- [x] Automated unit tests verified: `tests/unit/test_calibration.py`.

### 2.14 Advanced Extension: Multi-Lingual & Cross-Lingual Conflict Filter (Complete)
- [x] Implemented Unicode NFKC normalizer and diacritic stripper (`src/trustworthy_rag/data/multilingual.py`).
- [x] Multi-lingual stopword filtering across English, Spanish, French, and German.
- [x] Cross-lingual negation and refutation marker detection for multi-lingual corpora.
- [x] Automated unit tests verified: `tests/unit/test_multilingual.py`.

---

## 3. Atomic Task Completion Audit (TSK-001 through TSK-025)

Every task from `project-spec/task.md` has been completed and verified:

| Task ID | Task Description | Status | Verification Evidence |
| :--- | :--- | :---: | :--- |
| **TSK-001** | Scaffolding & Pydantic Data Models | `DONE` | `src/trustworthy_rag/schemas/models.py`, `tests/unit/test_schemas.py` |
| **TSK-002** | Configuration Loader & YAML Schema | `DONE` | `src/trustworthy_rag/config.py`, `tests/unit/test_config.py` |
| **TSK-003** | Dataset Ingestion & Chunker | `DONE` | `src/trustworthy_rag/data/`, `tests/unit/test_chunker.py` |
| **TSK-004** | Benchmark Freezing & SHA-256 Manifest | `DONE` | `scripts/freeze_datasets.py`, `data/splits/manifest_sha256.json` |
| **TSK-005** | Sparse BM25 Retrieval Engine | `DONE` | `src/trustworthy_rag/retrieval/bm25.py`, `tests/unit/test_bm25.py` |
| **TSK-006** | Dense Vector Retrieval Engine | `DONE` | `src/trustworthy_rag/retrieval/dense.py`, `tests/unit/test_dense.py` |
| **TSK-007** | Reciprocal Rank Fusion ($k_{rrf}=60$) | `DONE` | `src/trustworthy_rag/retrieval/fusion.py`, `tests/unit/test_fusion.py` |
| **TSK-008** | Context Builder with Offset Tracking | `DONE` | `src/trustworthy_rag/generation/context_builder.py`, `test_context_builder.py` |
| **TSK-009** | Baseline Answer Generation Engine | `DONE` | `src/trustworthy_rag/generation/generator.py` |
| **TSK-010** | Baseline End-to-End Execution | `DONE` | `scripts/run_baseline.py`, `EXP_BASELINE/predictions.jsonl` |
| **TSK-011** | Distractor Passage Injector | `DONE` | `src/trustworthy_rag/stress/distractor.py`, `test_perturbations.py` |
| **TSK-012** | Key Passage Removal Operator | `DONE` | `src/trustworthy_rag/stress/passage_removal.py`, `test_perturbations.py` |
| **TSK-013** | Near-Duplicate Passage Injector | `DONE` | `src/trustworthy_rag/stress/near_duplicate.py`, `test_perturbations.py` |
| **TSK-014** | Explicit Contradiction Injector | `DONE` | `src/trustworthy_rag/stress/contradiction_injection.py`, `test_perturbations.py` |
| **TSK-015** | Unified Perturbation Engine | `DONE` | `src/trustworthy_rag/stress/perturbation_engine.py` |
| **TSK-016** | Contradiction, Relevance, Coverage Analyzers | `DONE` | `src/trustworthy_rag/evidence/`, unit test suite |
| **TSK-017** | Citation Grounding Audit & Decision Engine | `DONE` | `src/trustworthy_rag/policy/decision.py`, `test_decision_policy.py` |
| **TSK-018** | Claim-Level Faithfulness Evaluator | `DONE` | `src/trustworthy_rag/evaluation/claims.py`, `test_claim_evaluator.py` |
| **TSK-019** | Risk–Coverage Compiler & AURCC | `DONE` | `src/trustworthy_rag/evaluation/risk_coverage.py`, `test_risk_coverage.py` |
| **TSK-020** | Mandatory Baseline Experiments | `DONE` | `experiments/evaluation_summary.json` (Baselines B1, B2, B3) |
| **TSK-021** | Proposed Policy & Ablation Matrix | `DONE` | `experiments/evaluation_summary.json` (Full + Ablations A1-A3) |
| **TSK-022** | Human Judge Validation ($\kappa \ge 0.70$) | `DONE` | `data/annotations/human_gold_sample.json`, `test_judge_calibration.py` |
| **TSK-023** | Systematic Failure Taxonomy Catalog | `DONE` | `reports/error_taxonomy.md` |
| **TSK-024** | Publication Figures & LaTeX Manuscript | `DONE` | `paper/manuscript.tex`, `experiments/figures/`, `paper_claims_audit.md` |
| **TSK-025** | Replication CLI & Release Packaging | `DONE` | `run_reproduction.py`, `run.bat`, `README.md` |

---

## 4. Empirical Benchmark Findings Delivered

Measured across the 100-query validation benchmark under controlled degradation (`experiments/evaluation_summary.json`):

| Method / System Variant | Clean Coverage | Distractor Coverage | Missing Passage Coverage | Contradiction Coverage | Selective Risk ($\mathcal{R}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conventional RAG (Unconditional)** | 1.00 | 1.00 | **1.00 *(100% Hallucinated)*** | **1.00 *(100% Conflict Blind)*** | 0.450 |
| **Retrieval-Only Score Gating** | 1.00 | 1.00 | 0.92 *(92% Hallucinated)* | 1.00 *(100% Conflict Blind)* | 0.380 |
| **Model Verbalized Confidence** | 1.00 | 1.00 | 0.00 | 0.00 | 0.220 |
| **Proposed Policy (Full Method)** | **0.66 – 0.73** | **0.66 – 0.73** | **0.00 *(100% Abstention)*** | **0.00 *(100% Intercepted)*** | **0.000** |
| *Ablation: Without Contradiction Analyzer* | 0.66 | 0.66 | 0.00 | **0.66 *(Vulnerable)*** | 0.220 |
| *Ablation: Without Coverage Analyzer* | 1.00 | 1.00 | **0.92 *(Vulnerable)*** | 0.00 | 0.250 |
| *Ablation: Without Relevance Normalizer* | 0.66 | 0.66 | 0.00 | 0.00 | 0.000 |

---

## 5. Deliberately Out of Scope (Scientific Boundaries)

The scientific scope and all extended capabilities are now complete. The following items represent intentional scientific boundaries for this controlled empirical benchmark:

1. **Live Web Crawling / Search Engine Indexing:**
   - *Intentionally Out of Scope:* The system operates strictly over cryptographically frozen, auditable offline benchmark splits (RAGTruth, Natural Questions / HotpotQA) to prevent data drift and guarantee scientific reproducibility.
2. **Cloud Cluster Deployment:**
   - *Intentionally Out of Scope:* Local containerization/cluster orchestration (e.g. AWS EKS / GCP Kubernetes) is beyond the scope of this research study. All execution is self-contained locally via `run.bat` or CLI (`python run_reproduction.py`).

---

## 6. How to Run, Test, and Verify

### A. One-Click Windows Launch (Interactive Web UI)
Double-click or run in terminal:
```cmd
run.bat
```
- Validates the 52 unit tests.
- Launches your default web browser to [**`http://localhost:2217`**](http://localhost:2217).
- Starts the FastAPI / Uvicorn server hosting the interactive playground and REST API docs at `http://localhost:2217/docs`.

### B. Run Automated Regression Test Suite
```bash
pytest tests
```
*Executes all 52 unit, integration, and server tests in ~1.2 seconds.*

### C. Run Headless Reproduction Suite
```bash
python run_reproduction.py
```
*Verifies cryptographic SHA-256 manifests, tests all 5 stress conditions, and displays the empirical replication table in stdout.*

### D. Re-Generate Paper Vector Figures
```bash
python scripts/figures/generate_paper_plots.py
```
*Outputs `perturbation_robustness.[pdf/png]`, `ablation_comparison.[pdf/png]`, and `risk_coverage_pareto.[pdf/png]` into `experiments/figures/`.*
