# Master Prompt: Phase 3 — Controlled Perturbation Stress Suite

**Phase:** Phase 3  
**Target:** Four Controlled Evidence Corruption Operators, Stress Engine, and Manifest Audit  
**Ports:** Frontend: 2217 | Backend: 2218  

---

## 1. Objectives & Scope
1. Implement `src/trustworthy_rag/stress/distractor.py`:
   - Inject topically plausible non-answering chunks into retrieved candidate pools.
   - Severity $\alpha \in [0.2, 0.8]$ controlling distractor ratio.
   - Logs injected chunk IDs in `PerturbationMetadata`.
2. Implement `src/trustworthy_rag/stress/passage_removal.py`:
   - Identify gold supporting passage for a query and excise it from context.
   - Verifies complete removal while keeping remaining candidates re-ranked.
3. Implement `src/trustworthy_rag/stress/near_duplicate.py`:
   - Inject semantically redundant paraphrases of existing context chunks ($\ge 0.85$ semantic similarity).
   - Simulates prompt dilution and attention distraction.
4. Implement `src/trustworthy_rag/stress/contradiction_injection.py`:
   - Mutate numerical/entity facts in supporting chunks to generate direct factual contradiction.
   - Replaces/injects conflicting passage alongside original chunk with verified conflict marker.
5. Implement `src/trustworthy_rag/stress/perturbation_engine.py`:
   - Unified interface orchestrating all four perturbation regimes with random seed determinism.
6. Automated Tests:
   - `tests/unit/test_perturbations.py` validating each perturbation operator, manifest fidelity, and clean data immutability.

---

## 2. Verification Gates
- All perturbation unit tests pass with 100% success rate.
- Invariance tests confirm clean dataset files are never modified in-place.
