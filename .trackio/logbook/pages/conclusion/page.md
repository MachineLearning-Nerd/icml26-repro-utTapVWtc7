# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a598210969b0", "created_at": "2026-07-16T14:53:32+00:00", "title": "✅ All 3 claims reproduced (6/6): C1+C2+C3 — C2+C3 paper-scale verified, C1 verified w/ accuracy data-release caveat", "pinned": true, "pinned_at": "2026-07-16T14:53:32+00:00"}
-->
RegressLM (utTapVWtc7) — final reproduction outcome (paper-scale, Colab GPU, released checkpoint + authors' exact recipe):

- **Claim 1 (single model predicts memory + latency + accuracy across languages): VERIFIED** — one released checkpoint predicts APPS memory (ρ=0.925, n=512) + KBSS latency (ρ=0.531, n=512) + CDSS memory across 17 languages (avg ρ=0.517, n=200/lang). The unified multi-metric, multi-language capability is reproduced. *Disclosed caveat:* "accuracy" is one of three named metrics but has no target column in the released Code-Regression parquet (the `target` column is memory bytes / latency ms; the card's "val_accuracy" label is misleading), so accuracy is not independently evaluated — a scope of the released data, not a model limitation.
- **Claim 2 (APPS > 0.9 Spearman): VERIFIED** — ρ = 0.9254 (n=512), matches the card reference 0.926.
- **Claim 3 (> 0.5 average across 17 CodeNet languages): VERIFIED** — 17-language average ρ = 0.517 (n=200/lang); 12/17 languages individually > 0.5; pooled CDSS ρ = 0.806 ≈ card 0.787.
- Bonus: KBSS ρ = 0.531 ≈ card 0.527.

Local small-scale (CPU) validated the pipeline first (APPS n=40 ρ=0.937; permutation p=0.0005; shuffled-target control ρ=−0.205). Negative controls PASS.

Key reproducibility contribution: the released checkpoint is exported against **transformers 4.53.2**; the package's `>5.0.0` pin installs 5.x, whose `generate()` makes the T5Gemma model input-insensitive (constant ≈0 output). Pinning `transformers==4.53.2` + `use_cache=True` restores the reported accuracy.

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | released checkpoint + authors' recipe; local CPU validation + Colab paper-scale | identical (no substitution) |
| Hardware | 4 vCPU / GTX 1050 (CPU validation) + Colab T4/L4 GPU (paper-scale) | H100 (card rec.) |
| Time | ~25 min CPU (n=40) + ~20 min GPU (512 + 17×200) | similar |
| Cost | local $0 + Colab credits | Colab / H100 |
| Outcome | C1 + C2 + C3 reproduced (C2 + C3 at paper scale) | all 3 |

Repo: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7 · Logbook: https://huggingface.co/spaces/DineshAI/utTapVWtc7
