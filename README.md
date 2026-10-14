# Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Status: Specification Complete](https://img.shields.io/badge/Status-Specification%20Complete-brightgreen.svg)]()

> A research-grade framework investigating evidence-aware answer/abstain policies to eliminate unsupported claims and hallucinations in Retrieval-Augmented Generation (RAG) under systematic evidence degradation and conflicting passages.

---

## 📌 Executive Overview

Retrieval-Augmented Generation (RAG) pipelines generate unsupported or contradictory statements when retrieved passages are irrelevant, incomplete, stale, or mutually inconsistent. 

This project investigates whether a lightweight **evidence-quality and contradiction-scoring mechanism**, combined with **calibrated abstention**, can improve answer faithfulness while preserving useful answer coverage.

### Key Highlights
- **Dual Retrieval Baseline**: Pure-Python BM25 and dense embeddings (`BAAI/bge-small-en-v1.5`) unified via Reciprocal Rank Fusion (RRF).
- **Four Stress-Test Perturbation Regimes**: Distractor injection, key supporting passage deletion, near-duplicate injection, and explicit factual contradiction.
- **Tri-State Decision Policy**: Explicitly decides between **`ANSWER`** (grounded with chunk citations), **`CLARIFY`** (resolves partial ambiguity), and **`ABSTAIN`** (calibrated refusal).
- **Risk–Coverage Analysis**: Computes empirical Pareto frontiers and evaluates selective risk at matched coverage levels.
- **Auditable Rigor**: Cryptographically frozen dataset splits (RAGTruth, Natural Questions / HotpotQA) with human-validated evaluation judges.

---

## 🏛️ Master Specification Framework

The engineering, scientific methodology, and execution roadmap are governed by seven synchronized master specification files located in [`project-spec/`](./project-spec/):

1. **[`architecture.md`](./project-spec/architecture.md)** — Comprehensive 17-component system architecture, Pydantic v2 data models, component contracts, and technical stack.
2. **[`goal.md`](./project-spec/goal.md)** — Core research question, 5 formal falsifiable hypotheses ($H_1$–$H_5$), evaluation metrics, and acceptance criteria.
3. **[`memory.md`](./project-spec/memory.md)** — Persistent project memory, decision logs (DEC-001 to DEC-007), experiment registry schemas, and state of truth.
4. **[`phases.md`](./project-spec/phases.md)** — 9-phase dependency roadmap (Phases 0–8) from specification to peer-reviewed release.
5. **[`prd.md`](./project-spec/prd.md)** — Product Requirements Document detailing 18 functional (FR-001–FR-018) and 10 non-functional (NFR-001–NFR-010) requirements.
6. **[`rules.md`](./project-spec/rules.md)** — Strict governance rules (RFC 2119) for research integrity, zero result fabrication, anti-leakage data tuning, and AI agent execution.
7. **[`task.md`](./project-spec/task.md)** — Actionable, atomic work breakdown of 25 verifiable tasks (TSK-001–TSK-025) across 8 subsystems.

---

## 🧪 Research Methodology

### Primary Research Question
> *When retrieved evidence is systematically degraded or conflicting, can an evidence-aware answer/abstain policy reduce unsupported claims more effectively than a conventional RAG pipeline and confidence-only thresholds?*

### Stress Perturbations
1. **Distractor Passages**: Irrelevant, non-answering context added at varying ratios ($\alpha \in [0.2, 0.8]$).
2. **Key Passage Deletion**: Gold supporting passage excised to test model abstention vs. confabulation.
3. **Near-Duplicate Passages**: Semantically redundant chunks inserted to test prompt dilution.
4. **Explicit Contradictions**: Key numerical or entity facts mutated to trigger pairwise cross-passage NLI conflict.

### Evidence Scoring Vector
$$\mathbf{s} = \langle s_{rel}, s_{cov}, s_{cit}, s_{con} \rangle$$
- $s_{rel}$: Normalized, rank-decayed retrieval relevance.
- $s_{cov}$: Entity and predicate query coverage in retrieved context.
- $s_{cit}$: Post-generation atomic claim citation grounding.
- $s_{con}$: Cross-passage pairwise contradiction probability via DeBERTa NLI.

---

## 🚀 Repository Structure

```text
.
├── project-spec/                # Seven Master Specification Documents
│   ├── architecture.md
│   ├── goal.md
│   ├── memory.md
│   ├── phases.md
│   ├── prd.md
│   ├── rules.md
│   └── task.md
├── src/                         # Source package (to be implemented in Phase 0-4)
│   └── trustworthy_rag/
├── tests/                       # Unit, integration, and regression suites
├── configs/                     # YAML experiment and policy configurations
├── data/                        # Frozen benchmark splits and manifests
├── experiments/                 # Structured JSONL run logs and databases
├── reports/                     # Error taxonomies and calibration reports
├── .gitignore
└── README.md
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
