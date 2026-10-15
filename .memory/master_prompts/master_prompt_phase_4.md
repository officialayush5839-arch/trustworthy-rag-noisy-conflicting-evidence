# Master Prompt: Phase 4 — Evidence Analysis & Calibrated Abstention Policy

**Phase:** Phase 4  
**Target:** Multi-Signal Evidence Scoring (Relevance, Coverage, Contradiction, Citation Grounding) and Tri-State Decision Policy Engine  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Implement `src/trustworthy_rag/evidence/contradiction.py`:
   - Cross-passage NLI evaluator using `cross-encoder/nli-deberta-v3-large` (with deterministic fallback).
   - Generates pairwise $K \times K$ contradiction probabilities and aggregate score $s_{con} \in [0.0, 1.0]$.
   - Emits structured `ContradictionFinding` list.
2. Implement `src/trustworthy_rag/evidence/coverage.py` & `relevance.py`:
   - Computes query-passage relevance $s_{rel}$ via normalized retrieval decay.
   - Computes entity/predicate coverage $s_{cov}$ detecting missing key facets.
3. Implement `src/trustworthy_rag/evidence/citation.py`:
   - Audits draft claims against cited chunks to compute citation grounding score $s_{cit}$.
4. Implement `src/trustworthy_rag/policy/decision.py`:
   - Combines vector $\mathbf{s} = \langle s_{rel}, s_{cov}, s_{cit}, s_{con} \rangle$ against thresholds.
   - Outputs tri-state `DecisionResult`:
     - `ABSTAIN`: when $s_{con} \ge \tau_{con\_max}$, $s_{rel} < \tau_{ans\_rel}$, $s_{cov} < \tau_{cov\_min}$, or $s_{cit} < \tau_{cit\_min}$.
     - `CLARIFY`: when evidence is relevant but partially incomplete ($\tau_{clar\_min} \le s_{cov} < \tau_{cov\_min}$).
     - `ANSWER`: when evidence is relevant, complete, consistent, and claims verified.
5. Unit tests:
   - `test_contradiction_analyzer.py`, `test_coverage_analyzer.py`, `test_citation_analyzer.py`, `test_decision_policy.py`.

---

## 2. Verification Gates
- 100% pass rate on all evidence and decision policy unit tests.
- Full truth-table branch coverage testing across all boundary thresholds.
