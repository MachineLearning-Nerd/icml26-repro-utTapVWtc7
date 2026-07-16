# Methodology — RegressLM Table-3 reproduction

## Goal
Reproduce the three official claims of *Regression Language Models for Code* (OpenReview
`utTapVWtc7`) using the **released checkpoint** (`akhauriyash/RegressLM-gemma-s-RLM-table3`,
181.5M params, T5Gemma-s backbone + IEEE-numeric decoder) and **released data**
(`akhauriyash/Code-Regression`, 7.5M rows) — no retraining, inference-only.

## Official claims (verbatim)
1. A single RLM predicts memory + latency + accuracy of code across multiple languages.
2. 300M RLM (T5Gemma init) obtains **>0.9 Spearman on APPS**.
3. Unified model obtains **>0.5 average Spearman across 17 CodeNet languages**.

## What the `target` column actually is
The dataset card calls the target `val_accuracy`, but inspecting the parquet shows `target`
is the **metric value**: memory **bytes** for APPS/CDSS (median ≈ 5.8 KB / 3 KB) and latency
**ms** for KBSS (median ≈ 0.026 ms). The authors' own reference snippet regresses `target`
and reports APPS=0.926, so we do the same. **Spearman is rank-based, so the absolute scale
is irrelevant** — only the rank order of predictions vs targets matters.

## Canonical numbers (dataset card results table)
| Space | This target | Reference ρ |
|---|---|---|
| APPS | memory bytes | **0.926** (claim 2: >0.9) |
| CDSS | memory bytes | 0.787 overall (claim 3: >0.5 avg across 17 langs) |
| KBSS | latency ms | 0.527 |

## Evaluation recipe (authors' published snippet, followed exactly)
- **Input prefix (required):** APPS/KBSS → `"{SPACE}\n{input}"`; CDSS → `"# CDSS\n# Language: {lang}\n{input}"`.
- `model.generate(do_sample=True, top_p=0.95, temperature=1.0, min_new_tokens=max_new_tokens=9, use_cache=True)`, 8 samples per input.
- Decode via `tokenizer.token_ids_to_floats(seq)[0]` (IEEE 9-token numeric decoder, 13-token vocab).
- Aggregate the 8 samples by **median**; score with `scipy.stats.spearmanr`.

## THE critical fix: transformers version
The checkpoint was exported with **transformers 4.53.2** (`config.transformers_version`).
The `regress-lm[extras]` pin `transformers>5.0.0` installs 5.x, whose rewritten seq2seq
`generate()` breaks this custom T5Gemma model: the encoder signal never reaches the decoder,
so the model emits a near-constant ≈0 output regardless of input (Spearman ≈ 0).

**Fix: `pip install transformers==4.53.2`** and use `use_cache=True`. Verified: greedy on 8
diverse APPS inputs then tracks targets (6036→5720, 13235→13137, …); sampling+median recovers
targets (6036→6026). This is the checkpoint's own export version, so it is the faithful choice.

## Two phases
- **Phase A (this box, CPU):** small-sample validation (≤ a few hundred rows) proving the
  pipeline produces the right rank correlation. Honest label: small-scale.
- **Phase B (Colab GPU):** full Table-3 scale — APPS 512 + KBSS 512 + 17 CodeNet languages
  × 8 samples → the authoritative ρ to compare against 0.926 / 0.787 / 0.527.
  (`repro/colab/regresslm_table3.ipynb`, which pins transformers==4.53.2.)

## Independent verification & negative controls (`repro/src/verify_independent.py`)
- Pearson r (different measure, must agree with Spearman).
- 95% bootstrap CI on Spearman.
- Permutation p-value vs H0 "predictions independent of targets".
- **False-positive control:** Spearman after shuffling predictions must be ≈ 0 (proves the
  metric is not trivially high).

## Deviations from the paper
- transformers pinned to 4.53.2 (checkpoint export version) instead of the package's >5.0.0
  (which breaks the model on current transformers). Inference path, recipe, and checkpoint
  are otherwise unchanged.
- Backend: local CPU (Phase A) / Colab GPU (Phase B). The model is the released checkpoint —
  no substitution.
