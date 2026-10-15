# Project Goals, Research Hypotheses, and Success Criteria

**Project Title:** Trustworthy Retrieval-Augmented Generation Under Noisy and Conflicting Evidence  
**Document ID:** GOAL-SPEC-001  
**Status:** Approved Specification  
**Version:** 1.0.0  
**Related Documents:** [architecture.md](./architecture.md), [prd.md](./prd.md), [phases.md](./phases.md), [rules.md](./rules.md), [memory.md](./memory.md), [task.md](./task.md)

---

## 1. Central Research Objective and Question

### 1.1 Central Research Objective
To design, implement, and rigorously evaluate an evidence-aware decision policy for Retrieval-Augmented Generation (RAG) that inspects retrieval relevance, evidence coverage, citation support, and cross-passage contradiction to reliably choose between **Answering**, **Requesting Clarification**, and **Abstaining**, thereby maximizing answer faithfulness while maintaining useful query coverage under systematic evidence corruption.

### 1.2 Primary Research Question
> **When retrieved evidence is systematically degraded or conflicting, can an evidence-aware answer/abstain policy reduce unsupported claims more effectively than a conventional RAG pipeline and confidence-only thresholds?**

---

## 2. Formal Falsifiable Hypotheses

The following hypotheses represent scientific propositions to be experimentally tested, validated, or rejected. None of these hypotheses are assumed to be true prior to empirical verification.

### Hypothesis H1: Evidence-Aware Abstention vs. Conventional RAG
* **Statement:** Under controlled retrieval corruption (distractors, key passage deletion, near-duplicates, conflicting passages), an evidence-aware decision policy achieves a significantly lower claim-level hallucination rate on answered queries than an unconstrained conventional RAG baseline at statistically comparable answer coverage levels ($\alpha = 0.05$).
* **Falsification Condition:** If the claim-level hallucination rate of the proposed policy is equal to or greater than that of conventional RAG across tested perturbation severities, H1 is rejected.

### Hypothesis H2: Evidence-Aware Policy vs. Confidence-Only Thresholding
* **Statement:** A multi-signal evidence policy ($\mathbf{s} = \langle s_{rel}, s_{cov}, s_{cit}, s_{con} \rangle$) yields a superior empirical risk–coverage Pareto frontier (lower risk for equal coverage, or higher coverage for equal risk) compared to single-signal thresholds based solely on model self-reported verbalized confidence or top-$1$ raw retrieval scores.
* **Falsification Condition:** If the Area Under the Risk–Coverage Curve (AURCC) of the evidence-aware policy is not statistically lower than that of model confidence thresholding across evaluation datasets, H2 is rejected.

### Hypothesis H3: Additive Value of Individual Evidence Signals (Ablation)
* **Statement:** Each individual signal—specifically cross-passage contradiction detection ($s_{con}$) and post-generation citation support ($s_{cit}$)—provides a distinct, measurable reduction in risk under its corresponding perturbation regime (conflicting evidence and distractor injection, respectively).
* **Falsification Condition:** If removing $s_{con}$ or $s_{cit}$ from the decision policy produces no statistically significant change ($p > 0.05$) in claim faithfulness under conflicting or distractor-heavy evidence, the sub-hypothesis for that signal is rejected.

### Hypothesis H4: Risk–Coverage Trade-Off Monotonicity
* **Statement:** Tightening evidence-quality thresholds $(\tau_{ans}, \tau_{cov})$ results in a monotonic decrease in empirical risk among answered queries, accompanied by a predictable trade-off in answer coverage, enabling calibrated operational tuning.
* **Falsification Condition:** If tightening thresholds causes empirical risk to remain unchanged or increase (e.g., due to model over-conservatism discarding only easy factual queries), H4 is rejected.

### Hypothesis H5: Resilience Across Perturbation Typologies
* **Statement:** The proposed policy's relative risk reduction over the baseline is highest under explicit factual contradiction and key passage removal compared to random distractor insertion.
* **Falsification Condition:** If the policy exhibits no measurable risk reduction under contradiction injection or key passage removal, H5 is rejected.

---

## 3. Structured Research & Engineering Objectives

### 3.1 Primary Research Objectives
- **PRO-1:** Establish a reproducible, automated benchmark for evaluating RAG faithfulness under four controlled evidence corruption regimes.
- **PRO-2:** Formulate and calibrate a composite evidence-scoring function integrating relevance, entity coverage, contradiction probability, and citation entailment.
- **PRO-3:** Empirically quantify the risk–coverage trade-off across varying abstention thresholds and compare it against standard confidence-based gating.

### 3.2 Secondary Research Objectives
- **SRO-1:** Develop a fine-grained taxonomy of RAG failure modes (e.g., entity confusion, temporal mismatch, ungrounded extrapolation, contradictory synthesis).
- **SRO-2:** Investigate the feasibility and precision of targeted clarification prompts when evidence is partially incomplete versus completely absent.
- **SRO-3:** Compare cross-passage NLI against LLM-as-judge prompts for contradiction detection in terms of accuracy, inference latency, and computational cost.

### 3.3 Software Engineering Objectives
- **SEO-1:** Implement a modular, type-safe Python package adhering strictly to Pydantic v2 schemas and clean component boundaries.
- **SEO-2:** Build a pluggable retrieval layer supporting both sparse BM25 and dense embedding models with rank fusion.
- **SEO-3:** Maintain high automated test coverage (>85%) with deterministic unit, integration, and regression suites.

### 3.4 Reproducibility Objectives
- **REO-1:** Freeze and hash all dataset evaluation splits (RAGTruth, Natural Questions / HotpotQA subsets).
- **REO-2:** Pin all software dependencies, model weights, random seeds, and hyperparameter configurations in machine-readable manifests.
- **REO-3:** Ensure that any external researcher can replicate published results using a single CLI execution script (`run_reproduction.py`).

### 3.5 Paper-Readiness Objectives
- **PPO-1:** Generate publication-standard LaTeX result tables and vector graphics (PDF/EPS) illustrating risk–coverage curves, ablation deltas, and error distributions.
- **PPO-2:** Conduct human validation on a frozen sample ($N \ge 100$) of evaluation cases to calibrate and validate the automated LLM/NLI evaluation judges.
- **PPO-3:** Produce an auditable manuscript draft transparently reporting methodology, limitations, and both positive and negative experimental findings.

---

## 4. Measurable Acceptance Criteria Table

*Note: All numerical targets listed below represent **provisional engineering targets** established for directional project management. They are subject to refinement and approval based on baseline calibration in Phase 2.*

| Objective | Metric | Evaluation Method | Required Evidence | Provisional Target / Completion Condition |
| :--- | :--- | :--- | :--- | :--- |
| **AC-01: Baseline Pipeline** | End-to-end execution | Automated integration test on 100 benchmark queries | JSONL execution logs with zero unhandled exceptions | 100% completion of queries with BM25, Dense, and Generation logs |
| **AC-02: Stress Suite** | Perturbation accuracy & tracking | Deterministic unit tests across all 4 perturbation operators | Perturbation manifests matching seed-controlled mutation rules | 100% of generated test cases have verifiable manifests and clean/perturbed isolation |
| **AC-03: Faithfulness Improvement** | Claim-Level Faithfulness on answered queries | Automated NLI + Human-validated judge on perturbed test set | Comparative tables of Proposed Method vs. Conventional RAG | Statistically significant reduction in unsupported claims ($p < 0.05$) at equal coverage |
| **AC-04: Risk–Coverage Optimization** | Area Under Risk–Coverage Curve (AURCC) | Threshold sweep across validation split, tested on test split | Plotted Pareto curve comparing Proposed vs. Confidence-Only Gating | Proposed method achieves lower AURCC than verbalized confidence and retrieval score gating |
| **AC-05: Contradiction Defense** | Selective Abstention under Contradiction | Evaluation on Contradiction-Perturbed test split | Confusion matrix of Abstain vs. Answer under contradictory passages | Abstains on $\ge 80\%$ of cases where retrieved passages contain direct factual conflicts |
| **AC-06: Missing Passage Defense** | Selective Abstention under Key Passage Deletion | Evaluation on Key-Passage-Removed test split | Abstention rate compared to Clean split | Abstains on $\ge 85\%$ of cases where gold supporting evidence is completely excised |
| **AC-07: Human Judge Validation** | Inter-Annotator Agreement (Cohen's $\kappa$ / Pearson $r$) | Blind annotation of $N=100$ claims by human vs. automated evaluator | Correlation report and confusion matrix | $\kappa \ge 0.70$ between automated judge and human consensus on claim support |
| **AC-08: Reproducibility** | Full Pipeline Replicability | Independent clean run using recorded seeds and config | Exact match of data hashes and metric agreement within Monte Carlo variance | End-to-end reproduction script finishes with zero manual code modifications |

---

## 5. Evaluation Requirements and Pitfall Prevention

### 5.1 Formal Definitions of Core Metrics
1. **Answer Correctness ($Acc_{ans}$)**: Proportion of answered queries whose response matches reference gold answers (evaluated via token F1 / exact match / semantic judge).
2. **Claim-Level Faithfulness ($Faith_{claim}$)**: 
   $$Faith_{claim} = \frac{|\text{Atomic Claims Grounded in Evidence}|}{|\text{Total Atomic Claims in Answer}|}$$
3. **Selective Risk ($\mathcal{R}$)**: Error rate evaluated strictly among the subset of queries the model chooses to answer:
   $$\mathcal{R} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \mathcal{L}(y_i, \hat{y}_i)$$
   where $\mathcal{A}$ is the set of answered queries, and $\mathcal{L}$ is the claim unfaithfulness or answer error.
4. **Answer Coverage ($\mathcal{C}$)**: Proportion of total queries answered:
   $$\mathcal{C} = \frac{|\mathcal{A}|}{N_{total}}$$
5. **Citation Precision ($Prec_{cit}$)**: Fraction of emitted inline citations whose referenced chunk directly entails the associated claim.
6. **Retrieval Recall ($Rec_{ret}$)**: Fraction of gold reference passages present in the top-$K$ retrieved passages.
7. **Abstention Quality ($F1_{abst}$)**: Harmonic mean of precision and recall in abstaining on unanswerable/corrupted queries.
8. **Inference Latency & Cost**: Mean wall-clock time per query (seconds), prompt tokens, and completion tokens.

### 5.2 The "High Accuracy via Over-Abstention" Pitfall
> [!WARNING]
> Reporting accuracy or faithfulness solely on answered queries ($Acc_{ans}$) without reporting coverage ($\mathcal{C}$) creates a severe experimental vulnerability. A naive system that abstains on 99% of queries and answers only 1 trivial query with 100% accuracy appears artificially superior.

**Mandatory Experimental Safeguards:**
- Every quoted risk or faithfulness metric **must** be paired with its exact corresponding coverage level.
- All baseline comparisons must be evaluated either:
  1. Along continuous **Risk–Coverage curves** across the full range $\mathcal{C} \in [0.1, 1.0]$.
  2. At **fixed, matched coverage points** (e.g., comparing risk specifically at $\mathcal{C} = 0.50, 0.70, 0.90$).

---

## 6. Expected Project Deliverables

1. **Reproducible Software Package**: Complete Python source package (`src/`) containing retrievers, perturbation engines, evidence analyzers, decision policy, and evaluators.
2. **Deterministic Stress-Test Suite**: Reusable perturbation generators producing auditable test datasets with explicit cryptographic manifests.
3. **Frozen Benchmark Subsets**: Curated, preprocessed subsets of RAGTruth and Natural Questions / HotpotQA with documented license and provenance.
4. **Comprehensive Experiment Configs**: Version-controlled YAML configurations defining all baseline, perturbation, and ablation experiments.
5. **Traceable Results Database**: JSONL and SQLite database storing all raw generations, atomic claim breakdowns, and intermediate scoring vectors.
6. **Figure and Table Generation Pipeline**: Automated scripts generating publication-ready vector plots (Risk–Coverage, ablation bar charts, latency distributions).
7. **Error Taxonomy & Qualitative Failure Report**: Categorized inspection of false answers, premature abstentions, and judge discrepancies.
8. **Research Manuscript Draft**: Full scientific paper detailing methodology, experimental results, ablation analyses, limitations, and reproducible appendices.

---

## 7. Explicit Non-Goals (Out of Scope)

The following activities are strictly outside the initial project scope and must not distract from core scientific goals:
- **General-Purpose Conversational Chatbot UI**: Building complex web applications, chat bubbles, user session management, or multi-tenant authorization.
- **Novel Foundation Model Pre-training / Fine-Tuning**: Pre-training or fine-tuning underlying generative LLMs; the study uses fixed open-weights models or pinned APIs.
- **Universal Web Scraping / Live Internet Search**: Relying on unversioned live internet search results that compromise reproducibility.
- **Proprietary Commercial Framework Lock-In**: Heavy reliance on closed orchestration frameworks (e.g., proprietary vector SaaS) that prevent headless local testing.
- **Claims of Universal Truth Outside Context**: Attempting to resolve global real-world factual veracity beyond the ingested benchmark corpus.
