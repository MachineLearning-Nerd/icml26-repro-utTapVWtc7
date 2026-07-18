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
- Accuracy data: [`akhauriyash/GraphArch-Regression`](https://huggingface.co/datasets/akhauriyash/GraphArch-Regression) (ONNX-readable graphs, `val_accuracy`).

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

### Phase B — full claim evidence (Colab GPU)
Open `repro/colab/regresslm_full_evidence_colab.ipynb` on a T4/L4/A100 runtime. It pins
`transformers==4.53.2`, runs 17 CodeNet languages × 200 rows × 8 samples, evaluates real
NASBench101 ONNX `val_accuracy`, records environment/model hashes and all raw draws, then
downloads a self-contained evidence ZIP.

## Inference protocol (authors' dataset-card recipe)
The regression `target` is the **metric value** — memory bytes for APPS/CDSS, latency ms for
KBSS (the card's "val_accuracy" label is misleading; Spearman is rank-based so scale is irrelevant).
Input prefix is **required**: `"{SPACE}\n{input}"` for APPS/KBSS,
`"# CDSS\n# Language: {lang}\n{input}"` for CDSS. Then
`generate(do_sample=True, top_p=0.95, temperature=1.0, min=max_new_tokens=9, use_cache=True)`,
8 samples → `token_ids_to_floats`[0] → `np.nanmedian` → scipy `spearmanr`.

## Results

- Claim 1 accuracy (same released checkpoint, n=512/space, eight draws/row):
  **NASBench101 ρ=0.406599**, **ENAS ρ=0.249461**, **NASNet ρ=0.206738**
  (card references 0.384/0.211/0.209). Mean ρ=0.287599; permutation p=0.000500;
  input-shuffle mean ρ=-0.013240. The retained evidence has 1,536 rows and
  12,288 raw draws.
- Claim 2: APPS **ρ=0.9268, n=512** (>0.9; card reference 0.926).
- Claim 3: CodeNet 17-language mean **ρ=0.529850, n=200/language**, stratified bootstrap
  95% CI **[0.502557, 0.554246]**, permutation p=0.000500. A second full local run gave
  **ρ=0.523403**.
- The Colab bundle retains 3,464 raw rows and 27,712 stochastic draws. The independent
  verifier recomputes all medians and statistics exactly.

Logbook: https://huggingface.co/spaces/DineshAI/utTapVWtc7

## Layout
```
upstream/          vendored regress-lm (pinned 6c23ccb; re-clone per README, gitignored)
repro/src/         evaluators plus independent bundle/CodeNet verification
repro/colab/       regresslm_full_evidence_colab.ipynb (one-click full evidence)
repro/tests/       verification tests (6/6 pass)
outputs/phaseA/    apps_n40.csv (+.json), independent_verification.json
docs/              methodology.md
.trackio/          Trackio logbook → publishes to DineshAI/utTapVWtc7
STATUS.md          live resume state for the autonomous loop
```

See `STATUS.md` for current progress, and `icml-2026-reproduction-challenge/COORDINATION.md`
for the multi-session registry this paper is tracked in.
