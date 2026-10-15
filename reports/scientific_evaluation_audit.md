# Scientific Evaluation Audit & Issue Remediation Register

**Date:** 2026-10-15  
**Audit Scope:** Full codebase evaluation validity, metric semantics, baseline leakage, mock paths, and artifact provenance.  
**Branch:** `fix/scientific-evaluation-audit`  

---

## 1. Classification & Traceability Matrix

| ID | Issue | Location | Classification | Evidence & Impact | Remediation Applied |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0-01** | Abstention scored as perfect faithfulness | `src/trustworthy_rag/evaluation/evaluator.py:106` | **Confirmed from code** | When `not is_answered`, `claim_faithfulness` was set to `1.0`. Averaging over all queries falsely inflated faithfulness to 100%. | Set `claim_faithfulness = None` on abstentions. Explicitly separate answered-only faithfulness, coverage, and selective risk. |
| **P0-02** | Selective risk & AURCC results degenerate (all 0.0) | `src/trustworthy_rag/evaluation/risk_coverage.py:25-26`, `scripts/evaluate_experiments.py:102` | **Confirmed from code** | `err = 0.0` was assigned to abstentions, making error vectors identically zero. `compute_curve` also silently returned zeros on invalid/empty inputs instead of failing. | Enforce strict input validation (fail on empty/mismatch). Define selective risk $\mathcal{R}(\tau)$ strictly on answered cases. Unit test with hand-calculated vectors. |
| **P0-03** | Benchmark evaluator hardcodes mock generation | `src/trustworthy_rag/evaluation/evaluator.py:39` | **Confirmed from code** | `AnswerGenerator(backend="mock")` was hardcoded, ignoring `configs/default.yaml` setting `meta-llama/Llama-3.1-8B-Instruct`. | Make backend explicitly configurable. Pass `backend` parameter to evaluator. Require explicit `--allow-mock` flag for mock benchmark tests. |
| **P0-04** | Contradiction analyzer always runs mock heuristic | `src/trustworthy_rag/evidence/contradiction.py:66` | **Confirmed from code** | `analyze()` unconditionally calls `_compute_mock_contradiction()`, ignoring actual NLI cross-encoder models. | Separate heuristic and transformer NLI pipelines. If `use_mock_fallback=False`, fail closed if the transformer cannot be initialized. Label findings with inference engine. |
| **P0-05** | Claim faithfulness uses surface keyword overlap | `src/trustworthy_rag/evaluation/claims.py:74-88` | **Confirmed from code** | Support status is thresholded on token overlap ratio (`>= 0.70`). Fails on complex paraphrases and antonym traps. | Augment lexical overlap with semantic premise-hypothesis entailment checking. Validate against human gold sample. |
| **P0-06** | Citation precision/recall hardcoded | `src/trustworthy_rag/evaluation/evaluator.py:70-71, 126-127`, `src/trustworthy_rag/evidence/citation.py:30` | **Confirmed from code** | `citation_precision = 1.0 if gen.citations else 0.0` without checking whether cited chunk actually supports the claim assertion. | Implement claim-to-citation entailment check. Citation is valid only if cited chunk text supports the preceding sentence. |
| **P0-07** | Retrieval recall uses brittle chunk-ID substring matching | `src/trustworthy_rag/evaluation/evaluator.py:72, 109, 128` | **Confirmed from code** | `any(g in p.chunk_id ...)` checks substring containment rather than exact gold document/chunk ID equality. | Use exact set intersection against `question.gold_passage_ids`. |
| **P0-08** | Confidence baseline reads hidden perturbation parameter | `src/trustworthy_rag/evaluation/evaluator.py:81` | **Confirmed from code** | Baseline B3 sets confidence based on `if perturbation == PerturbationType.CONTRADICTION`, directly leaking ground truth test conditions! | Completely remove perturbation parameter inspection from baseline. Base confidence strictly on observable retrieval/generation score distribution. |
| **P0-09** | Citation gate evaluated before draft generation | `src/trustworthy_rag/server/app.py`, `src/trustworthy_rag/policy/decision.py` | **Confirmed from code** | $s_{cit}$ defaulted to 1.0 prior to answer generation. | Establish strict two-stage gating: Stage 1 (Pre-generation gating on $s_{rel}, s_{cov}, s_{con}$) -> Stage 2 (Post-generation citation grounding on draft answer). |
| **P1-01** | Query coverage is lexical token presence | `src/trustworthy_rag/evidence/coverage.py:29-44` | **Confirmed from code** | Evaluates token overlap of salient words. | Label metric as lexical query-term coverage; augment with entity extractor. |
| **P1-02** | Policy weights & thresholds need clear provenance | `configs/default.yaml`, `src/trustworthy_rag/policy/decision.py` | **Suspected from artifacts** | Fixed weights (0.35, 0.35, 0.30, -0.50) need calibration documentation on validation split. | Freeze calibration protocol strictly on `data/frozen/questions_val.jsonl`. |
| **P1-03** | Lack of per-query JSONL audit logs | `experiments/` | **Suspected from artifacts** | Only aggregate `evaluation_summary.json` was stored. | Emit per-query detailed JSONL logs with full provenance, model revisions, decision scores, and timing into `experiments/runs/canonical_eval/`. |
| **P1-04** | Perturbation manifest realism & labels | `src/trustworthy_rag/stress/` | **Confirmed from code** | Explicit markers (`conflict_`) could be exploited by heuristic models. | Remove trivial marker strings and test with naturalistic semantic contradictions. |
| **P1-06** | Confidence intervals missing in aggregate summary | `experiments/evaluation_summary.json` | **Confirmed from code** | Only point estimates stored. | Include 95% bootstrap confidence intervals in evaluation matrix. |
| **P1-07** | Discrepancies between README and evaluation summary | `README.md`, `experiments/evaluation_summary.json`, `paper/manuscript.tex` | **Confirmed from code** | Disparate coverage numbers (0.66 vs 0.70 vs 0.73). | Re-generate all documentation tables from one single canonical JSON evaluation artifact. |
| **P1-08** | Human annotation agreement calculation traceability | `data/annotations/human_gold_sample.json`, `paper/manuscript.tex` | **Confirmed from code** | Kappa calculation needs reproducible CLI script. | Author `scripts/compute_annotator_agreement.py` calculating Cohen's $\kappa$ with full class prevalence matrix. |
| **P1-11** | Absolute Windows paths in project-spec Markdown files | `project-spec/*.md` | **Confirmed from code** | `file:///C:/Users/ARYAN...` links break on non-Windows/portable checkouts. | Replace all absolute links with portable relative links (`./file.md`). |

---

## 2. Pre-Audit Output Preservation
Prior to applying code changes, the existing evaluation artifacts were archived non-destructively:
- `experiments/pre_audit_archive/evaluation_summary_pre_audit.json`
- `experiments/pre_audit_archive/predictions_pre_audit.jsonl`
