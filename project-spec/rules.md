# Project Operational Rules and Governance Standards

**Project Title:** Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence  
**Document ID:** RULES-SPEC-001  
**Status:** Mandatory Governance  
**Version:** 1.0.0  
**Related Documents:** [architecture.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/architecture.md), [goal.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/goal.md), [phases.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/phases.md), [prd.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/prd.md), [memory.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/memory.md), [task.md](file:///c:/Users/ARYAN%20-%20AYUSH/OneDrive/Desktop/Trustworthy%20Retrieval-Augmented%20Generation%20Under%20Noisy%20and%20Conflicting%20Evidence/project-spec/task.md)

---

## 1. Conformance and Terminology

The keywords **MUST**, **MUST NOT**, **REQUIRED**, **SHALL**, **SHALL NOT**, **SHOULD**, **SHOULD NOT**, **RECOMMENDED**, **MAY**, and **OPTIONAL** in this document are to be interpreted as described in RFC 2119.

Every contributor—whether a human researcher, software engineer, or autonomous AI coding agent—**MUST** strictly adhere to the operational rules set forth in this document. Violations of research integrity rules constitute immediate invalidation of associated experimental findings.

---

## 2. Research Integrity Rules

### 2.1 Scientific Veracity and Fabrication
- **R-001 (Zero Fabrication)**: Contributors **MUST NOT** fabricate, hallucinate, extrapolate, or falsify experimental numbers, metrics, p-values, sample counts, benchmark results, dataset passages, or bibliographic citations under any circumstance.
- **R-002 (Hypothesis vs. Finding Distinction)**: Contributors **MUST NOT** describe an unverified hypothesis (e.g., $H_1$ through $H_5$ in `goal.md`) as an empirical finding or fact until full experimental data has been generated, logged, and audited.
- **R-003 (Preservation of Negative Results)**: Negative results, failed hypotheses, and instances where the proposed method fails to outperform conventional baselines **MUST** be transparently preserved and reported in `memory.md` and the final manuscript. Negative results with rigorous methodology are scientifically valuable.

### 2.2 Data Integrity and Leakage Prevention
- **R-004 (Partition Isolation)**: Clean evaluation datasets and perturbed evaluation datasets **MUST** be maintained in distinct, immutable directories. They **MUST NOT** be mingled or cross-contaminated.
- **R-005 (Strict Anti-Leakage Tuning)**: Hyperparameters and decision policy thresholds $(\tau_{ans}, \tau_{cov}, \tau_{clar}, \tau_{con})$ **MUST NOT** be tuned or optimized on the frozen evaluation `test` split. All tuning **MUST** occur exclusively on the `dev` or `validation` splits. Reporting test-set-tuned results as unbiased performance is strictly forbidden.
- **R-006 (Dataset Provenance & Hashing)**: Every dataset split **MUST** have an accompanying SHA-256 cryptographic manifest. If a data file's hash deviates from the recorded manifest, experimental runs using that file **MUST** be halted immediately.

### 2.3 Evaluation and Judge Validation
- **R-007 (Human Calibration of Automated Judges)**: Any automated evaluation procedure relying on an LLM-as-a-judge or NLI cross-encoder **MUST** be validated against blind human annotations on a sample of at least $N=100$ items, reporting Cohen's $\kappa$ or Pearson correlation.
- **R-008 (Balanced Coverage Reporting)**: Selective risk or accuracy on answered queries **MUST NEVER** be reported in isolation without stating the corresponding answer coverage ($\mathcal{C}$). Comparisons between systems **MUST** be evaluated along continuous Risk–Coverage curves or at matched coverage levels.

---

## 3. Software Engineering Rules

### 3.1 Code Modification and Modularity
- **R-009 (Inspect Before Editing)**: Contributors **MUST** inspect existing files, class hierarchies, and tests before introducing modifications.
- **R-010 (Atomic and Modular Changes)**: Code changes **SHOULD** be small, modular, and accompanied by unit tests. Large monolithic commits that refactor multiple subsystems simultaneously **MUST NOT** be made without explicit justification.
- **R-011 (Contract Preservation)**: Public interfaces (e.g., `BaseRetriever`, `BaseScorer`, `BaseEvaluator`) **MUST NOT** be silently altered or renamed. Changes to component contracts require updating `architecture.md` and `prd.md`.
- **R-012 (Dependency Minimalism)**: Contributors **MUST NOT** introduce extraneous, heavy, or proprietary third-party libraries when the capability can be reliably implemented using the standard library, NumPy, PyTorch, or Pydantic.

### 3.2 Configuration and Secret Management
- **R-013 (Declarative Configuration)**: All experimental hyperparameters (model names, temperatures, thresholds, seeds, top-$K$) **MUST** be specified in version-controlled YAML configuration files. Magic numbers hard-coded into Python source code **MUST NOT** be permitted.
- **R-014 (Zero Credential Leakage)**: API keys, tokens, and credentials **MUST NOT** be committed to source control. All secrets **MUST** be loaded dynamically from environment variables or a local `.env` file that is explicitly excluded in `.gitignore`.

### 3.3 Error Handling and Data Safety
- **R-015 (Explicit Error Handling)**: Model inference timeouts, network disconnections, and malformed inputs **MUST** be handled explicitly with structured retries or logged fallbacks. The system **MUST NOT** fail silently or emit empty predictions without diagnostic alerts.
- **R-016 (Data Preservation)**: Contributors **MUST NOT** execute destructive commands that delete raw datasets, historical experiment logs, or versioned artifacts without an explicit, multi-step backup and user confirmation.

---

## 4. Autonomous AI Agent Rules

### 4.1 Specification Synchronization and Primacy
- **R-017 (Master Spec Primacy)**: Before commencing any non-trivial implementation or research task, an AI agent **MUST** review the seven master specification files in `project-spec/`. The problem statement and approved architectural decisions are strictly authoritative.
- **R-018 (No Unilateral Spec Mutation)**: An AI agent **MUST NOT** unilaterally alter the core research question, remove mandatory baseline requirements, or weaken acceptance criteria to make a task appear simpler or successful.
- **R-019 (Synchronized Documentation)**: Whenever an implementation detail or architectural boundary changes, the agent **MUST** update the affected files among the seven master documents simultaneously.

### 4.2 Task Execution and Verifiable Evidence
- **R-020 (Dependency Ordering)**: Agents **MUST** follow task dependencies specified in `phases.md` and `task.md`. Tasks **MUST NOT** be initiated before their prerequisites are completed and validated.
- **R-021 (Evidence Before Completion Assertion)**: An agent **MUST NOT** mark a task as `DONE` in `task.md` or declare success to the user unless it has executed the required verification commands (e.g., `pytest`, data hash checks, script execution) and observed passing exit codes.
- **R-022 (Transparent Status Reporting)**: In every milestone update, the agent **MUST** explicitly state:
  1. What files were created or modified.
  2. What tests were executed and their exact outcomes.
  3. What failures, anomalies, or negative results occurred.
  4. What decisions remain unresolved and require human clarification.

---

## 5. Model, Retrieval, and Scoring Rules

### 5.1 Versioning and Determinism
- **R-023 (Pinned Model Identifiers)**: Every model utilized in the pipeline (sparse retriever, dense embedding encoder, cross-encoder NLI, generative LLM) **MUST** have its exact identifier, revision/commit hash, and precision recorded in the run configuration.
- **R-024 (Seeded Determinism)**: All stochastic operations (random data splitting, perturbation sampling, distractor shuffling) **MUST** be seeded using a centralized seed controller. The seed **MUST** be recorded in the experiment manifest.

### 5.2 Retrieval and Attribution Integrity
- **R-025 (Provenance Anchoring)**: Every passage retrieved from corpus storage **MUST** retain its source document ID, chunk ID, character offsets, and retrieval score through all downstream transformations.
- **R-026 (Semantic Similarity vs. Factuality)**: Contributors and models **MUST NOT** treat high dense retrieval similarity or high BM25 scores as evidence of factual entailment. Retrieval measures lexical/semantic proximity; entailment measures logical truth under evidence.
- **R-027 (Uncalibrated Confidence Warning)**: LLM self-reported verbalized confidence (e.g., asking the model "How confident are you from 1 to 5?") **MUST NOT** be treated as a calibrated posterior probability without empirical calibration (e.g., Platt scaling or isotonic regression).

---

## 6. Testing, Verification, and Release Rules

### 6.1 Testing Standards
- **R-028 (Unit Test Coverage)**: Every scoring component, perturbation operator, rank fusion method, and policy decision branch **MUST** have dedicated unit tests in `tests/unit/`.
- **R-029 (Integration Pipelines)**: The complete pipeline from query ingestion to generation and claim evaluation **MUST** be tested end-to-end on a minimal synthetic benchmark fixture in `tests/integration/`.
- **R-030 (Schema Validation on Exports)**: All exported JSON/JSONL artifacts **MUST** be validated against their Pydantic schemas before writing to persistent storage.

### 6.2 Release and Reproduction
- **R-031 (Reproduction Script Integrity)**: The release **MUST** include a standalone, self-contained reproduction script (`run_reproduction.py`) capable of running the baseline, running the proposed policy, and generating comparison metrics in a fresh environment.
- **R-032 (Release Tagging)**: No code may be tagged as release `v1.0.0` until all Phase 8 exit criteria are satisfied and documented in `phases.md` and `memory.md`.
