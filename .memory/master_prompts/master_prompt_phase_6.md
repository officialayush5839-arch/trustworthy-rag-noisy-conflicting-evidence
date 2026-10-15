# Master Prompt: Phase 6 — Error Analysis and Robustness

## Objective
Execute Phase 6 of the Trustworthy RAG project:
1. Conduct qualitative and quantitative error analysis across baseline and proposed methods under all perturbation scenarios (clean, distractor, passage removal, near duplicate, contradiction).
2. Characterize the taxonomy of failure modes:
   - False Positives (overconfident answers under corrupted or missing evidence)
   - False Abstentions (unnecessary refusals when evidence is adequate)
   - Contradiction Interception Successes and Blindspots
   - Threshold Boundary Conditions
3. Produce the comprehensive report `reports/error_taxonomy.md` with explicit qualitative case traces and diagnostic failure codes.
4. Record key scientific takeaways, limitations, and boundary conditions in `project-spec/memory.md`.
