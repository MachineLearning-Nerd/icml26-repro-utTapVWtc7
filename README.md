# Repro — Regression Language Models for Code (RegressLM), ICML 2026

Reproduction of *Regression Language Models for Code* (RegressLM; Akhauri, Song et al.,
Google DeepMind/Cornell) for the
[ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `utTapVWtc7`.

RegressLM is a seq2seq model that **regresses numeric performance metrics** (memory, latency,
accuracy) directly from source code / problem text, via a T5Gemma encoder + a numeric decoder
that emits IEEE/P10 float tokens.

## Official claims (max 6 pts)
1. A single RLM simultaneously predicts **memory + latency + accuracy** across multiple languages.
2. 300M RLM (T5Gemma init) obtains **>0.9 Spearman on APPS** (Table 3 = **0.930**).
3. **>0.5 average Spearman across 17 CodeNet languages**.

## Artifacts (all released, public)
- Code: [google-deepmind/regress-lm](https://github.com/google-deepmind/regress-lm) — vendored unmodified in `upstream/` (commit `6c23ccb`).
- Checkpoint: [`akhauriyash/RegressLM-gemma-s-RLM-table3`](https://huggingface.co/akhauriyash/RegressLM-gemma-s-RLM-table3) (not gated).
- Data: [`akhauriyash/Code-Regression`](https://huggingface.co/datasets/akhauriyash/Code-Regression) (`data.parquet`, 5.6 GB).

## Reproduce

### ⚠️ Critical: pin transformers==4.53.2
The checkpoint was exported with **transformers 4.53.2** (`config.transformers_version`).
`regress-lm[extras]` pins `transformers>5.0.0`, but **5.x breaks this T5Gemma model**: its
rewritten seq2seq `generate()` never feeds the encoder signal to the decoder, so the model
emits a near-constant ≈0 output regardless of input (Spearman ≈ 0). Install 4.53.2 *after*
extras so it wins:

```bash
uv pip install "transformers==4.53.2"
```

### Phase A — local small-scale validation (CPU)
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e "./upstream[extras]" scipy pyarrow pandas pytest
uv pip install --python .venv/bin/python "transformers==4.53.2"   # MUST override >5.0.0

# 40 APPS rows × 8 samples → Spearman (≈0.937; claim >0.9; card ref 0.926)
.venv/bin/python repro/src/run_eval.py --space APPS --limit 40 --num_samples 8 \
    --batch_size 4 --device cpu --out outputs/phaseA/apps_n40.csv
# independent verification + false-positive controls
.venv/bin/python repro/src/verify_independent.py --inputs outputs/phaseA/apps_n40.csv \
    --out outputs/phaseA/independent_verification.json
```
On a CPU box the 40-row run takes ~20 min (the GTX 1050 is sm_61, unsupported by torch cu130 → CPU only).

### Phase B — full Table-3 scale (Colab GPU)
Open `repro/colab/regresslm_table3.ipynb` on a T4/L4/A100 Colab runtime. It pins
`transformers==4.53.2`, then runs APPS (512) + KBSS (512) + the 17 CodeNet languages × 8
samples (median aggregation) and prints Spearman vs Table 3 (APPS 0.926 / CDSS 0.787 / KBSS 0.527).

## Inference protocol (authors' dataset-card recipe)
The regression `target` is the **metric value** — memory bytes for APPS/CDSS, latency ms for
KBSS (the card's "val_accuracy" label is misleading; Spearman is rank-based so scale is irrelevant).
Input prefix is **required**: `"{SPACE}\n{input}"` for APPS/KBSS,
`"# CDSS\n# Language: {lang}\n{input}"` for CDSS. Then
`generate(do_sample=True, top_p=0.95, temperature=1.0, min=max_new_tokens=9, use_cache=True)`,
8 samples → `token_ids_to_floats`[0] → `np.nanmedian` → scipy `spearmanr`.

## Result (Phase A, small scale)
APPS Spearman **0.937** (claim >0.9 ✅) — Pearson 0.920, permutation p=0.0005,
shuffled-target control ρ=−0.205 (signal destroyed as required). Full-scale (512 + 17 langs)
via the Colab notebook. Logbook: https://huggingface.co/spaces/DineshAI/utTapVWtc7

## Layout
```
upstream/          vendored regress-lm (pinned 6c23ccb; re-clone per README, gitignored)
repro/src/         run_eval.py (filter → infer → Spearman), verify_independent.py,
                   inspect_data.py (cheap data probe)
repro/colab/       regresslm_table3.ipynb (full-scale Phase B)
repro/tests/       test_verify.py (4/4 pass)
outputs/phaseA/    apps_n40.csv (+.json), independent_verification.json
docs/              methodology.md
.trackio/          Trackio logbook → publishes to DineshAI/utTapVWtc7
STATUS.md          live resume state for the autonomous loop
```

See `STATUS.md` for current progress, and `icml-2026-reproduction-challenge/COORDINATION.md`
for the multi-session registry this paper is tracked in.
