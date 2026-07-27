# Repro — Regression Language Models for Code (RegressLM)

**Current cumulative verdict: 6/10 forecast, unchanged from the live judge.**
Claims 1–3 are VERIFIED/HIGH. Claims 4 and 5 are BLOCKED/LOW after exact
four-route audits; no proxy or missing artifact is counted as a pass.

Start here:

1. [Release report and five-claim visibility matrix](../release-report/page.md)
2. [Current cumulative verifier](../current-cumulative-verification/page.md)
3. [Claim 1](../current-claim-1/page.md),
   [Claim 2](../current-claim-2/page.md),
   [Claim 3](../current-claim-3/page.md),
   [Claim 4](../claim-4-kendall-ranking/page.md),
   [Claim 5](../claim-5-ablation-scaling/page.md)
4. [Pinned environment](../../environment/pyproject.toml)
   and [complete uv lock](../../environment/uv.lock)


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_index_paper_scale_final", "created_at": "2026-07-18T23:08:23+00:00", "title": "Paper-scale evidence summary with row-level provenance"}
-->
This reproduction evaluates the released RegressLM checkpoint and released datasets at the author-card scale. The headline evidence is:

- ONNX validation accuracy: NASBench101 ρ=0.406599, ENAS ρ=0.249461, and NASNet ρ=0.206738, each at n=512 and eight draws per row;
- APPS memory: ρ=0.926807 at n=512 (claim threshold >0.9);
- KernelBook latency: ρ=0.535279 at n=512; and
- CodeNet memory: primary mean per-language ρ=0.529850 across 17 languages at n=200/language; an independent local run gives ρ=0.523403.

The accuracy bundle has 1,536 row-level predictions and 12,288 retained draws. Independent verification recomputes every median and headline correlation, tests a label-permutation null, and applies an input-shuffle falsification control. The deficient first claim was investigated through exactly 10 pre-registered evidence approaches, never an additional route.

Paper: https://arxiv.org/abs/2509.26476
Repository: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7
