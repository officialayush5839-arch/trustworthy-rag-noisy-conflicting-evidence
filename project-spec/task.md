# Actionable Project Task Registry

**Project Title:** Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence  
**Document ID:** TASK-REG-001  
**Status:** Active Work Breakdown  
**Version:** 1.0.0  
**Related Documents:** [architecture.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/architecture.md), [goal.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/goal.md), [phases.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/phases.md), [prd.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/prd.md), [rules.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/rules.md), [memory.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/memory.md)

---

## 1. Task Governance and Status Rules

Tasks in this registry represent the atomic operational units of the project. Every task is assigned a strict lifecycle state:
- `TODO`: Specified, prerequisites satisfied, pending execution.
- `IN_PROGRESS`: Actively under development or test execution.
- `BLOCKED`: Cannot proceed until an identified dependency or technical decision is resolved.
- `DONE`: Implementation complete, unit/integration tests verified, output artifacts logged, acceptance criteria 100% satisfied.

> [!CAUTION]
> **Strict Completion Rule**: A task **MUST NOT** be marked `DONE` based solely on code being written. Passing tests, verified outputs, and logged evidence are strictly required prior to status transition.

---

## 2. Master Task Registry

### Subsystem 1: Environment, Configuration, and Foundation (Phase 0)

#### TSK-001: Initialize Python Package Structure and Dependency Manifests
- **Phase**: Phase 0
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: None
- **Affected Files**: `pyproject.toml`, `requirements.txt`, `src/trustworthy_rag/__init__.py`, `.gitignore`
- **Implementation Steps**:
  1. Author `pyproject.toml` targeting Python 3.11+.
  2. Pin core dependencies: `pydantic>=2.7.0`, `rank-bm25>=0.2.2`, `sentence-transformers>=3.0.0`, `transformers>=4.42.0`, `faiss-cpu>=1.8.0`, `pytest>=8.2.0`, `loguru>=0.7.2`.
  3. Author `.gitignore` excluding caches, checkpoints, datasets, and `.env`.
  4. Create directory structure: `src/trustworthy_rag/`, `tests/`, `configs/`, `data/`, `experiments/`.
- **Acceptance Criteria**: `pip install -e .` succeeds in a clean virtualenv; package is importable.
- **Required Tests**: Run `python -c "import trustworthy_rag"`.
- **Expected Artifacts**: Installed package and locked manifest.
- **Research Implications**: Establishes zero-drift execution environment.
- **Blockers**: None.

#### TSK-002: Implement Core Pydantic Schemas and Configuration Loaders
- **Phase**: Phase 0
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-001
- **Affected Files**: `src/trustworthy_rag/schemas/`, `src/trustworthy_rag/config.py`
- **Implementation Steps**:
  1. Define `Document`, `DocumentChunk`, `Question`, `RetrievedPassage`, `EvidenceQualityScore`, `DecisionResult`, `AtomicClaim`, `EvaluationResult` models.
  2. Implement YAML configuration parser validating against `ExperimentConfig` schema.
  3. Create default configuration template `configs/default.yaml`.
- **Acceptance Criteria**: Invalid schemas raise Pydantic validation errors; valid YAML loads cleanly.
- **Required Tests**: `tests/unit/test_schemas.py` testing serialization and edge-case validation.
- **Expected Artifacts**: Schema module and base config file.
- **Research Implications**: Enforces contract integrity across all experimental modules.
- **Blockers**: None.

---

### Subsystem 2: Dataset Preparation and Freezing (Phase 1)

#### TSK-003: Implement Benchmark Ingestion and Document Chunking Engine
- **Phase**: Phase 1
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-002
- **Affected Files**: `src/trustworthy_rag/data/ingestion.py`, `src/trustworthy_rag/data/chunker.py`
- **Implementation Steps**:
  1. Write ingestion parsers for RAGTruth and Natural Questions / HotpotQA JSONL formats.
  2. Implement sliding-window text chunker recording `start_char_offset`, `end_char_offset`, and token length.
  3. Generate deterministic chunk IDs: `{doc_id}_c{idx:04d}` with SHA-256 content hashes.
- **Acceptance Criteria**: Ingests test corpus without truncation; chunk offsets reconstruct source text.
- **Required Tests**: `tests/unit/test_chunker.py` asserting offset round-trip consistency.
- **Expected Artifacts**: Clean parsed document and chunk collections.
- **Research Implications**: Precision chunk IDs enable fine-grained citation tracking.
- **Blockers**: None.

#### TSK-004: Freeze and Hash Auditable Benchmark Splits
- **Phase**: Phase 1
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-003
- **Affected Files**: `scripts/freeze_datasets.py`, `data/splits/manifest_sha256.json`
- **Implementation Steps**:
  1. Partition datasets into frozen splits: Dev ($N=50$), Validation ($N=100$), Test ($N=250$).
  2. Compute SHA-256 hashes for each split file and write to `manifest_sha256.json`.
  3. Implement split integrity verifier checking hashes at runtime.
- **Acceptance Criteria**: Splits are completely disjoint; verifier passes with zero discrepancies.
- **Required Tests**: `tests/unit/test_splits.py` verifying split disjointness and hash integrity.
- **Expected Artifacts**: Frozen JSONL files and manifest in `data/frozen/`.
- **Research Implications**: Prevents data leakage between threshold tuning and final evaluation.
- **Blockers**: None.

---

### Subsystem 3: Storage and Retrieval Layer (Phase 2)

#### TSK-005: Implement Sparse BM25 Retrieval Engine
- **Phase**: Phase 2
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-003
- **Affected Files**: `src/trustworthy_rag/retrieval/bm25.py`
- **Implementation Steps**:
  1. Implement `BM25Retriever` wrapping `rank_bm25.BM25Okapi`.
  2. Tokenize text using consistent lower-casing and whitespace/punctuation splitting.
  3. Return top-$K$ candidates with raw BM25 scores and ranks.
- **Acceptance Criteria**: Returns ranked `RetrievedPassage` list sorted by score descending.
- **Required Tests**: `tests/unit/test_bm25.py` verifying known keyword matches on toy corpus.
- **Expected Artifacts**: BM25 module and persistence index loader.
- **Research Implications**: Fulfills mandatory sparse baseline requirement.
- **Blockers**: None.

#### TSK-006: Implement Dense Vector Retrieval Engine
- **Phase**: Phase 2
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-003
- **Affected Files**: `src/trustworthy_rag/retrieval/dense.py`
- **Implementation Steps**:
  1. Implement `DenseRetriever` using pinned model `BAAI/bge-small-en-v1.5`.
  2. Build FAISS FlatIP (inner product over normalized vectors) index.
  3. Query index and return cosine similarity scores in $[0.0, 1.0]$.
- **Acceptance Criteria**: Top-$K$ retrieval outputs normalized scores; index is deterministic.
- **Required Tests**: `tests/unit/test_dense.py` checking cosine similarity ranges and vector dimensions.
- **Expected Artifacts**: Dense retriever module and FAISS vector index builder.
- **Research Implications**: Fulfills mandatory dense baseline requirement.
- **Blockers**: None.

#### TSK-007: Implement Reciprocal Rank Fusion (RRF) and Score Normalizer
- **Phase**: Phase 2
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-005, TSK-006
- **Affected Files**: `src/trustworthy_rag/retrieval/fusion.py`
- **Implementation Steps**:
  1. Implement RRF algorithm: $RRF(d) = \sum_{m \in \{bm25, dense\}} \frac{1}{k_{rrf} + r_m(d)}$ with $k_{rrf}=60$.
  2. Implement min-max score normalizer for individual retriever scores.
  3. Output unified ranked list with fused scores.
- **Acceptance Criteria**: Merges disparate rank lists deterministically; ranks start at 1.
- **Required Tests**: `tests/unit/test_fusion.py` comparing output against manual math verification vectors.
- **Expected Artifacts**: Hybrid retrieval fusion module.
- **Research Implications**: Produces optimal candidate pools combining semantic and lexical recall.
- **Blockers**: None.

---

### Subsystem 4: Context Construction and Generation Baseline (Phase 2)

#### TSK-008: Implement Provenance-Anchored Context Builder
- **Phase**: Phase 2
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/generation/context_builder.py`
- **Implementation Steps**:
  1. Format top-$K$ passages with explicit chunk headers: `[Chunk ID: doc_xyz_c01]`.
  2. Track character offsets of each chunk within the assembled prompt context.
  3. Truncate context cleanly if token limits are exceeded, preserving highest-ranked chunks.
- **Acceptance Criteria**: Context retains exact chunk IDs and provenance mapping.
- **Required Tests**: `tests/unit/test_context_builder.py` asserting header presence and offset mapping.
- **Expected Artifacts**: Context builder module.
- **Research Implications**: Critical for chunk-level citation extraction and entailment auditing.
- **Blockers**: None.

#### TSK-009: Implement Baseline Generation Harness and Pinned Model Runners
- **Phase**: Phase 2
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-008
- **Affected Files**: `src/trustworthy_rag/generation/generator.py`, `configs/baseline_rag.yaml`
- **Implementation Steps**:
  1. Author prompt template enforcing chunk-level bracketed citations (`[doc_id_c001]`).
  2. Implement backend connector supporting local Hugging Face / vLLM (`Llama-3.1-8B-Instruct`) and pinned API.
  3. Implement generation pipeline running unconditional conventional RAG baseline.
- **Acceptance Criteria**: Executes end-to-end on test queries; returns generated text with latency and token metadata.
- **Required Tests**: `tests/integration/test_baseline_generation.py` on mock query.
- **Expected Artifacts**: Generator module, baseline config, baseline run logs.
- **Research Implications**: Establishes mandatory conventional RAG baseline for comparison.
- **Blockers**: None.

---

### Subsystem 5: Controlled Perturbation Stress Suite (Phase 3)

#### TSK-010: Implement Distractor Passage Injection Operator
- **Phase**: Phase 3
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/stress/distractor.py`
- **Implementation Steps**:
  1. Extract random non-relevant chunks from corpus matching query domain.
  2. Inject distractors into candidate pool at ratio $\alpha \in [0.2, 0.8]$.
  3. Record injected chunk IDs and seed in `PerturbationManifest`.
- **Acceptance Criteria**: Injected chunks are present in context; clean set remains unmodified.
- **Required Tests**: `tests/unit/test_distractor.py` verifying injection ratios and manifest records.
- **Expected Artifacts**: Distractor perturbation module.
- **Research Implications**: Evaluates susceptibility to prompt context dilution and distraction.
- **Blockers**: None.

#### TSK-011: Implement Key Passage Deletion Operator
- **Phase**: Phase 3
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/stress/passage_removal.py`
- **Implementation Steps**:
  1. Identify gold supporting passage for given query.
  2. Excise gold chunk from retrieved top-$K$ candidates.
  3. Record deleted chunk ID in manifest.
- **Acceptance Criteria**: Target chunk is completely absent from context; remaining candidates re-ranked.
- **Required Tests**: `tests/unit/test_removal.py` asserting gold chunk exclusion.
- **Expected Artifacts**: Passage removal module.
- **Research Implications**: Tests whether RAG hallucinates when essential supporting evidence is missing.
- **Blockers**: None.

#### TSK-012: Implement Near-Duplicate Passage Injection Operator
- **Phase**: Phase 3
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/stress/near_duplicate.py`
- **Implementation Steps**:
  1. Generate or retrieve paraphrased versions of top retrieved chunks ($\text{cosine similarity} \ge 0.85$).
  2. Inject duplicate chunks into context pool with distinct chunk IDs.
  3. Record duplicate pairings in manifest.
- **Acceptance Criteria**: Duplicate passages inserted with traceable manifests.
- **Required Tests**: `tests/unit/test_near_duplicate.py` checking similarity threshold of injected pairs.
- **Expected Artifacts**: Near-duplicate perturbation module.
- **Research Implications**: Assesses vulnerability to redundant evidence and attention crowding.
- **Blockers**: None.

#### TSK-013: Implement Explicit Contradiction Injection Operator
- **Phase**: Phase 3
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/stress/contradiction_injection.py`
- **Implementation Steps**:
  1. Mutate numbers, dates, or named entities in gold supporting passage to create direct conflict.
  2. Verify contradiction using DeBERTa cross-encoder ($P(\text{contradiction}) \ge 0.75$).
  3. Inject conflicting passage alongside original passage into retrieved context.
- **Acceptance Criteria**: Injected passage achieves verified NLI contradiction against gold chunk.
- **Required Tests**: `tests/unit/test_contradiction_injection.py` verifying NLI contradiction threshold.
- **Expected Artifacts**: Contradiction perturbation module.
- **Research Implications**: Direct stress-test for cross-passage contradiction detection.
- **Blockers**: None.

---

### Subsystem 6: Evidence Analysis and Policy Engine (Phase 4)

#### TSK-014: Implement Pairwise Cross-Passage Contradiction Analyzer
- **Phase**: Phase 4
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/evidence/contradiction.py`
- **Implementation Steps**:
  1. Load pinned NLI cross-encoder `cross-encoder/nli-deberta-v3-large`.
  2. Formulate pairwise premise-hypothesis pairs for top-$K$ passages ($K \le 8$).
  3. Compute pairwise contradiction probability matrix and aggregate $s_{con} = \max_{i < j} P(con_{ij})$.
- **Acceptance Criteria**: Correctly identifies contradictory passage pairs; outputs $K \times K$ matrix.
- **Required Tests**: `tests/unit/test_contradiction_analyzer.py` on synthetic contradicting pairs.
- **Expected Artifacts**: Contradiction analyzer module.
- **Research Implications**: Core mechanism to intercept conflicting evidence before generation.
- **Blockers**: None.

#### TSK-015: Implement Relevance and Evidence Coverage Analyzers
- **Phase**: Phase 4
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-007
- **Affected Files**: `src/trustworthy_rag/evidence/coverage.py`, `src/trustworthy_rag/evidence/relevance.py`
- **Implementation Steps**:
  1. Implement $s_{rel}$ computation from rank-decayed normalized retrieval scores.
  2. Implement $s_{cov}$ extracting named entities and predicate keywords from query and checking presence in context.
  3. Return typed `EvidenceQualityScore` record.
- **Acceptance Criteria**: $s_{rel}, s_{cov} \in [0.0, 1.0]$; zero coverage for disjoint query-context pairs.
- **Required Tests**: `tests/unit/test_coverage.py` on known entity overlap fixtures.
- **Expected Artifacts**: Coverage and relevance analyzer modules.
- **Research Implications**: Enables early abstention when evidence is irrelevant or partial.
- **Blockers**: None.

#### TSK-016: Implement Tri-State Decision Policy Engine
- **Phase**: Phase 4
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-014, TSK-015
- **Affected Files**: `src/trustworthy_rag/policy/decision.py`, `configs/policy_calibrated.yaml`
- **Implementation Steps**:
  1. Implement decision logic: `ANSWER`, `CLARIFY`, or `ABSTAIN` based on $(\tau_{ans}, \tau_{cov}, \tau_{clar}, \tau_{con})$.
  2. Calibrate thresholds strictly on the frozen validation split.
  3. Generate human-readable decision reasons and diagnostic telemetry.
- **Acceptance Criteria**: Deterministic decision output; full branch coverage under unit tests.
- **Required Tests**: `tests/unit/test_decision_policy.py` verifying all boundary truth-table conditions.
- **Expected Artifacts**: Decision policy engine and calibrated threshold configuration.
- **Research Implications**: Operationalizes the core research answer/abstain mechanism.
- **Blockers**: None.

#### TSK-017: Implement Post-Generation Citation Grounding Audit
- **Phase**: Phase 4
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-009, TSK-016
- **Affected Files**: `src/trustworthy_rag/evidence/citation.py`
- **Implementation Steps**:
  1. Parse generated answer text to extract chunk citations `[doc_id_c001]`.
  2. Decompose cited sentences into claims; evaluate premise-hypothesis NLI against cited chunk.
  3. Compute $s_{cit}$; trigger fallback abstention if $s_{cit} < \tau_{cit}$.
- **Acceptance Criteria**: Verifies inline citations; intercepts ungrounded draft answers.
- **Required Tests**: `tests/unit/test_citation_audit.py` testing grounded and ungrounded generation mocks.
- **Expected Artifacts**: Citation grounding audit module.
- **Research Implications**: Prevents model from fabricating citations or hallucinating despite good retrieval.
- **Blockers**: None.

---

### Subsystem 7: Systematic Evaluation and Ablation Matrix (Phase 5)

#### TSK-018: Implement Claim-Level Decomposition and Faithfulness Evaluator
- **Phase**: Phase 5
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-009, TSK-017
- **Affected Files**: `src/trustworthy_rag/evaluation/claims.py`
- **Implementation Steps**:
  1. Implement atomic factual claim splitter (rule-based sentence parser or pinned LLM claim extractor).
  2. Classify each claim as `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED` against context.
  3. Compute claim-level faithfulness ratio and citation precision.
- **Acceptance Criteria**: Accurately decomposes multi-sentence answers into verifiable atomic claims.
- **Required Tests**: `tests/unit/test_claim_evaluator.py` against reference annotated claim samples.
- **Expected Artifacts**: Claim decomposition and evaluation module.
- **Research Implications**: Primary fine-grained metric for RAG hallucination assessment.
- **Blockers**: None.

#### TSK-019: Implement Risk–Coverage Compiler and Pareto Curve Generator
- **Phase**: Phase 5
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-018
- **Affected Files**: `src/trustworthy_rag/evaluation/risk_coverage.py`
- **Implementation Steps**:
  1. Sweep decision thresholds across validation split to generate discrete operating points.
  2. Compute selective risk $\mathcal{R}$ and coverage $\mathcal{C}$ at each point.
  3. Compute Area Under Risk–Coverage Curve (AURCC) via trapezoidal integration.
  4. Compute 95% bootstrap confidence intervals across 1,000 resamples.
- **Acceptance Criteria**: Generates monotonic or step-wise Pareto curves with verified confidence bands.
- **Required Tests**: `tests/unit/test_risk_coverage.py` checking boundary values (coverage 0 to 1).
- **Expected Artifacts**: Risk–coverage compiler module and plotting utilities.
- **Research Implications**: Fulfills mandatory metric requirements for selective classification.
- **Blockers**: None.

#### TSK-020: Execute Mandatory Baseline Experiments
- **Phase**: Phase 5
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-009, TSK-019
- **Affected Files**: `experiments/runs/EXP_BASELINES/`, `configs/experiments/`
- **Implementation Steps**:
  1. Run Baseline B1: Conventional RAG (unconditional generation).
  2. Run Baseline B2: Retrieval-only thresholding (abstain if max retrieval score $< \tau_{ret}$).
  3. Run Baseline B3: Self-reported model verbalized confidence gating.
  4. Execute all baselines across clean and perturbed test splits across 5 random seeds.
- **Acceptance Criteria**: 100% of queries executed; logs, intermediate scores, and metrics persisted.
- **Required Tests**: Verify output JSONL schema integrity using `scripts/validate_run_logs.py`.
- **Expected Artifacts**: Complete baseline experiment records and summary tables.
- **Research Implications**: Authoritative control experiments required to evaluate $H_1$ and $H_2$.
- **Blockers**: None.

#### TSK-021: Execute Full Proposed Method and Signal Ablation Matrix
- **Phase**: Phase 5
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-016, TSK-020
- **Affected Files**: `experiments/runs/EXP_PROPOSED/`, `experiments/runs/EXP_ABLATIONS/`
- **Implementation Steps**:
  1. Run full Proposed Method ($\mathbf{s} = \langle s_{rel}, s_{cov}, s_{cit}, s_{con} \rangle$) across all test conditions and seeds.
  2. Execute Ablation A1: Without Contradiction Analyzer ($s_{con} = 0$).
  3. Execute Ablation A2: Without Coverage Analyzer ($s_{cov} = 0$).
  4. Execute Ablation A3: Without Citation Grounding ($s_{cit} = 0$).
  5. Execute Ablation A4: Without Relevance Normalizer ($s_{rel} = 0$).
- **Acceptance Criteria**: Full ablation matrix executed; metrics logged to `experiments/results_master.db`.
- **Required Tests**: Assert that all ablation runs share identical seeds and test subsets.
- **Expected Artifacts**: Comparative results database and ablation delta tables.
- **Research Implications**: Directly tests Hypothesis $H_3$ (additive value of evidence signals).
- **Blockers**: None.

#### TSK-022: Conduct Human Validation of Automated Judges
- **Phase**: Phase 5
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-018
- **Affected Files**: `data/annotations/human_gold_sample.json`, `reports/judge_calibration.md`
- **Implementation Steps**:
  1. Sample $N=100$ generated claims across clean and perturbed test sets.
  2. Conduct blind manual annotation of claim support (`SUPPORTED`, `CONTRADICTED`, `UNSUPPORTED`).
  3. Compute Cohen's $\kappa$ and confusion matrix between human labels and automated judge.
- **Acceptance Criteria**: Documented alignment report; verify $\kappa \ge 0.70$.
- **Required Tests**: Validation script comparing automated vs. human labels.
- **Expected Artifacts**: Human gold annotation dataset and calibration report.
- **Research Implications**: Guarantees scientific credibility of LLM/NLI automated evaluations.
- **Blockers**: None.

---

### Subsystem 8: Error Taxonomy, Reporting, and Release (Phases 6–8)

#### TSK-023: Build Systematic Failure Taxonomy and Qualitative Error Catalog
- **Phase**: Phase 6
- **Priority**: P1 | **Status**: `TODO`
- **Dependencies**: TSK-021
- **Affected Files**: `reports/error_taxonomy.md`
- **Implementation Steps**:
  1. Sample false positive answers (hallucinations that passed policy) and false abstentions.
  2. Categorize failure modes: entity confusion, premise truncation, over-strict thresholds, prompt defiance.
  3. Document boundary conditions and negative results in `memory.md`.
- **Acceptance Criteria**: Comprehensive categorized report with passage traces and qualitative analysis.
- **Required Tests**: Manual review of categorized cases.
- **Expected Artifacts**: Detailed Error Taxonomy Report.
- **Research Implications**: Provides vital qualitative insights for the paper's discussion section.
- **Blockers**: None.

#### TSK-024: Author Publication-Ready Figures, Tables, and LaTeX Manuscript Draft
- **Phase**: Phase 7
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-021, TSK-023
- **Affected Files**: `paper/manuscript.tex`, `scripts/figures/generate_paper_plots.py`
- **Implementation Steps**:
  1. Author Python scripts generating vector PDF/EPS plots (Risk–Coverage curves, ablation bars).
  2. Export LaTeX tables with 95% bootstrap confidence intervals.
  3. Draft full scientific paper: Abstract, Introduction, Related Work, Method, Experiments, Results, Error Taxonomy, Limitations.
  4. Perform claim-to-artifact audit verifying every quoted number against raw logs.
- **Acceptance Criteria**: Compilable LaTeX paper draft; 100% of claims traceable to logged artifacts.
- **Required Tests**: Run `python scripts/audit_paper_claims.py` asserting numerical agreement.
- **Expected Artifacts**: LaTeX source, compiled PDF, vector figures, and audit report.
- **Research Implications**: Delivers the primary research paper deliverable.
- **Blockers**: None.

#### TSK-025: Author Standalone Reproduction CLI and Release Package
- **Phase**: Phase 8
- **Priority**: P0 | **Status**: `TODO`
- **Dependencies**: TSK-024
- **Affected Files**: `run_reproduction.py`, `README.md`, `release_manifest.json`
- **Implementation Steps**:
  1. Author self-contained reproduction script `run_reproduction.py`.
  2. Author comprehensive `README.md` with installation, CLI commands, and reproduction steps.
  3. Validate clean reproduction in fresh virtualenv.
  4. Create release tag `v1.0.0`.
- **Acceptance Criteria**: Independent execution replicates core table within Monte Carlo variance.
- **Required Tests**: End-to-end fresh virtualenv reproduction test.
- **Expected Artifacts**: Validated release bundle and replication documentation.
- **Research Implications**: Ensures peer reviewers can independently reproduce all findings.
- **Blockers**: None.
