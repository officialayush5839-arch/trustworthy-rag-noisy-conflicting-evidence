# Active Execution Context

> **Note for Models:** DO NOT re-scan the entire repository. Read this file and the corresponding Master Prompt in `.memory/master_prompts/` to understand immediate objectives.

- **Project:** trustworthy-rag-noisy-conflicting-evidence
- **Target Ports:** Frontend: 2217 | Backend: 2218
- **Active Phase:** phase_4 (Phase 4: Evidence Analysis and Calibrated Abstention)
- **Phase Status:** IN_PROGRESS
- **Current Task:** TSK-014, TSK-015, TSK-016, TSK-017: Implement multi-signal evidence scoring engine (Relevance $s_{rel}$, Coverage $s_{cov}$, Contradiction $s_{con}$, Citation Grounding $s_{cit}$) and tri-state decision policy engine (`ANSWER`, `CLARIFY`, `ABSTAIN`).
- **Quota / Limit State:** NORMAL
- **Immediate Next Action:** Create `.memory/master_prompts/master_prompt_phase_4.md` and implement evidence analysis and decision policy modules.
