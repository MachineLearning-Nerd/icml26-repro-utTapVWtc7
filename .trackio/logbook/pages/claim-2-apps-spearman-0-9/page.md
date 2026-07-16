# Claim 2 — APPS Spearman >0.9


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_10dfd98b7f6d", "created_at": "2026-07-16T12:41:28+00:00", "title": "✅ APPS Spearman = 0.937 (claim >0.9 reproduced)"}
-->
Claim: a 300M RLM (T5Gemma init) obtains >0.9 Spearman on APPS competitive-programming submissions.

Result (released checkpoint, authors' exact eval recipe, 40 APPS rows × 8 samples → median, CPU):
- **Spearman ρ = 0.937** — exceeds the >0.9 claim and matches the dataset-card reference (0.926).
- Pearson r = 0.920 (independent measure, agrees).
- 0/40 decode failures (nan_rate=0).
- Sample predictions track targets: tgt 13235 → pred 13144; 6036 → 5970; 5354 → 5455.

The regression target is memory BYTES (not accuracy — see Methods). Spearman is rank-based, so absolute scale is irrelevant; the rank order of predictions tracks ground truth.

Scale note: 40 rows is a local CPU validation of the pipeline (the box has no usable GPU). The authoritative paper-scale number (512 items) is produced by repro/colab/regresslm_table3.ipynb on Colab GPU. Result CSV: outputs/phaseA/apps_n40.csv.
