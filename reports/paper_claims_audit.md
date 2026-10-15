# Paper Claims and Artifact Traceability Audit

## 1. Overview
This audit document maps every quantitative statement, empirical metric, and figure presented in `paper/manuscript.tex` directly to its source experimental artifact, raw run log, and code implementation.

---

## 2. Traceability Matrix

| Section in Paper | Assertion / Claim / Number | Source Artifact | Code Reference | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Abstract / Intro** | "400 frozen, auditable questions partitioned into disjoint splits (Dev=50, Val=100, Test=250)" | `data/splits/manifest_sha256.json` | `scripts/freeze_datasets.py` | Verified |
| **Abstract / Intro** | "reducing false acceptance from 100% to 0.00%" | `experiments/evaluation_summary.json` | `src/trustworthy_rag/evaluation/evaluator.py` | Verified |
| **Abstract / Sec 4.3** | "Cohen's $\kappa \ge 0.70$ (specifically $\kappa = 0.742$)" | `data/annotations/human_gold_sample.json` | `tests/unit/test_judge_calibration.py` | Verified |
| **Section 2** | Hybrid BM25 + Dense RRF with $k_{rrf} = 60$ | `src/trustworthy_rag/retrieval/fusion.py` | `tests/unit/test_fusion.py` | Verified |
| **Section 3.2** | 4 stress operators: Distractor, Removal, Duplicate, Contradiction | `src/trustworthy_rag/stress/` | `tests/unit/test_perturbations.py` | Verified |
| **Table 1** | Baseline Unconditional Coverage: Clean 1.00, Missing 1.00, Conflict 1.00 | `experiments/evaluation_summary.json` (line 3-34) | `scripts/evaluate_experiments.py` | Verified |
| **Table 1** | Baseline Retrieval-Only Coverage: Missing 0.92, Conflict 1.00 | `experiments/evaluation_summary.json` (line 35-66) | `scripts/evaluate_experiments.py` | Verified |
| **Table 1** | Proposed Policy Coverage: Clean 0.66, Missing 0.00, Conflict 0.00 | `experiments/evaluation_summary.json` (line 99-130) | `scripts/evaluate_experiments.py` | Verified |
| **Figure 1** | Perturbation robustness plot across clean, distractor, near-dup, contradiction, removal | `experiments/figures/perturbation_robustness.[pdf/png]` | `scripts/figures/generate_paper_plots.py` | Verified |
| **Figure 2** | Ablation comparison under contradiction and passage removal | `experiments/figures/ablation_comparison.[pdf/png]` | `scripts/figures/generate_paper_plots.py` | Verified |

---

## 3. Verification Sign-Off
- **Automated Verification Script**: All numbers matched to JSON evaluation outputs.
- **Reproducibility Guarantee**: `experiments/evaluation_summary.json` can be regenerated deterministically via `python scripts/evaluate_experiments.py`.
- **Audit Conclusion**: Zero unverified assertions. All reported claims are backed by immutable artifacts.
