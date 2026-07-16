# Repro - Regression Language Models for Code (RegressLM)

## Pages

| Page |
| --- |
| [Claim 1 — unified multi-metric model](#/claim-1-unified-multi-metric-model) |
| [Claim 2 — APPS Spearman >0.9](#/claim-2-apps-spearman-0-9) |
| [Claim 3 — CodeNet 17 languages >0.5](#/claim-3-codenet-17-languages-0-5) |
| [Methods & environment](#/methods-environment) |
| [Negative controls & falsification](#/negative-controls-falsification) |
| [Conclusion](#/conclusion) |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_4689ffd8f4e4", "created_at": "2026-07-16T12:41:27+00:00", "title": "RegressLM (utTapVWtc7) — ICML 2026 reproduction"}
-->
Reproduction of *Regression Language Models for Code* (RegressLM) for the ICML 2026 Agent Reproduction Challenge.

Paper: arXiv 2509.26476 · OpenReview utTapVWtc7 · released checkpoint akhauriyash/RegressLM-gemma-s-RLM-table3 (181.5M) · released data akhauriyash/Code-Regression (7.5M rows).

Headline (Claim 2, APPS, 40-row local CPU validation): **Spearman ρ = 0.937** (claim >0.9 ✅; card reference 0.926). Pearson 0.920, permutation p=0.0005, shuffled-target control ρ=-0.205.

Key engineering finding: the checkpoint was exported with **transformers 4.53.2**; the package pin pulls 5.x which breaks this T5Gemma model's generate() (input-insensitive output). Pinning transformers==4.53.2 restores it.
