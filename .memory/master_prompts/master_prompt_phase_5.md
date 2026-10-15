# Master Prompt: Phase 5 — Systematic Evaluation and Ablation Matrix

**Phase:** Phase 5  
**Target:** Claim-Level Faithfulness Decomposition, Risk–Coverage Compiler, Comparative Baselines, Signal Ablation Matrix, and Human Judge Calibration  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Implement `src/trustworthy_rag/evaluation/claims.py`:
   - Decomposes generated sentences into verifiable atomic factual claims.
   - Evaluates factual entailment status: `SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED` against cited chunk context.
   - Computes claim faithfulness:
     $$Faith_{claim} = \frac{|\text{Supported Claims}|}{|\text{Total Generated Claims}|}$$
2. Implement `src/trustworthy_rag/evaluation/risk_coverage.py`:
   - Computes selective error risk $\mathcal{R}$ among answered queries across coverage sweep $\mathcal{C} \in [0.1, 1.0]$.
   - Computes Area Under Risk–Coverage Curve (AURCC) via trapezoidal integration.
   - Computes 95% bootstrap confidence intervals across random seeds.
3. Implement `src/trustworthy_rag/evaluation/evaluator.py`:
   - Orchestrates comparative baselines:
     - B1: Conventional RAG (unconditional).
     - B2: Retrieval-only threshold gating.
     - B3: Model verbalized confidence gating.
     - Proposed: Full multi-signal evidence policy ($\mathbf{s}$).
     - Ablations: A1 (no contradiction), A2 (no coverage), A3 (no citation audit), A4 (no relevance).
4. Implement `scripts/evaluate_experiments.py`:
   - Executes evaluation matrix on frozen validation split ($N=100$) and records results to `experiments/evaluation_summary.json`.
5. Human Judge Calibration:
   - Create gold calibration sample of 100 claims in `data/annotations/human_gold_sample.json`.
   - Compute Cohen's $\kappa$ metric proving $\kappa \ge 0.70$ between automated judge and human annotation.
6. Automated Tests:
   - `tests/unit/test_claim_evaluator.py`, `tests/unit/test_risk_coverage.py`, `tests/unit/test_judge_calibration.py`.

---

## 2. Verification Gates
- All evaluation unit tests pass.
- Experiment matrix executed and summary metrics logged.
- Human-judge agreement verifies Cohen's $\kappa \ge 0.70$.
