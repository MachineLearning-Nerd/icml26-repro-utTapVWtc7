# STATUS — RegressLM (utTapVWtc7) reproduction — UNBLOCKED

**Session:** autoloop (this /loop). **Last updated:** 2026-07-16. **State: ✅ PUBLISHED** — HF: https://huggingface.co/spaces/DineshAI/utTapVWtc7 · GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7. **C2 reproduced** (APPS Spearman ρ=0.937, n=40 small-scale CPU; claim >0.9 ✅; card ref 0.926; perm p=0.0005, shuffled control −0.205). C1 partial ('accuracy' not in released data), C3 pending full-scale Colab (`repro/colab/regresslm_table3.ipynb`).

## ⚡ BREAKTHROUGH (this tick)
A prior tick DEFERRED this paper because the model was input-insensitive (constant ≈0
output) under transformers 5.1. **Root cause found + fixed:** the checkpoint was exported
with **transformers 4.53.2** (`config.transformers_version`), but the package pin
`transformers>5.0.0` had pulled **5.1.0**. transformers 5.x rewrote seq2seq generation
and broke the encoder→decoder signal flow for this custom T5Gemma model.

**Fix: pin `transformers==4.53.2` + `use_cache=True`.** Result — greedy on 8 diverse APPS
inputs now tracks targets (tgt 6036→5720, 13235→13137, 5354→5418, 6162→6012; 7/8 distinct).
Sampling+top_p=0.95 also works (no NaN; tgt 6036→6026 median). Reproduction unblocked.

```
uv pip install "transformers==4.53.2"   # matches checkpoint export version
```

## Paper
- **Title:** Regression Language Models for Code (RegressLM). OpenReview `utTapVWtc7`.
- **Code:** `google-deepmind/regress-lm` (pinned `6c23ccb`) → `upstream/`.
- **Checkpoint:** `akhauriyash/RegressLM-gemma-s-RLM-table3` → `checkpoints/rlm-table3/` (725 MB, 181.5M params, T5Gemma-s + IEEE numeric decoder, 13-token decoder vocab).
- **Data:** `akhauriyash/Code-Regression` — single 5.6 GB parquet, 7,502,559 rows.

## Official claims (judge scores, verbatim)
1. A single RLM predicts **memory + latency + accuracy** of code across multiple languages.
2. 300M RLM (T5Gemma init) obtains **>0.9 Spearman on APPS**.
3. Unified model obtains **>0.5 average Spearman across 17 CodeNet languages**.

## EVAL CONTRACT (confirmed)
- `target` = the metric VALUE (memory **bytes** for APPS/CDSS; latency **ms** for KBSS), NOT accuracy. (Card's "val_accuracy" label is misleading.) Spearman is rank-based → scale irrelevant.
- Canonical Table 3 (card): **KBSS=0.527, CDSS=0.787, APPS=0.926**.
- Input prefix (REQUIRED): APPS/KBSS → `"{SPACE}\n{input}"`; CDSS → `"# CDSS\n# Language: {lang}\n{input}"`.
- Recipe: `generate(do_sample=True, top_p=0.95, temperature=1.0, min=max_new_tokens=9, use_cache=True)`, 8 samples, `token_ids_to_floats`[0], median, scipy.spearmanr.
- CDSS top-17 langs (CodeNet): C++,Python,Java,C,Ruby,C#,Rust,Go,Haskell,Kotlin,JavaScript,PHP,D,Scala,OCaml,Perl,Fortran.

## GOTCHAS
- **transformers MUST be 4.53.2** (checkpoint export version). 5.x breaks generate (input-insensitive output). Package pin `>5.0.0` is wrong for this checkpoint — override.
- GTX 1050 (sm_61) unusable with torch cu130 → `pick_device()` falls back to CPU. Colab GPU for full scale.
- `datasets` streaming too slow over 5.6 GB parquet → pyarrow column projection + filter pushdown + scanner.head().
- OOM: reap orphaned pyarrow worker procs after kills.

## CODE
- `repro/src/run_eval.py` — faithful Table-3 evaluator (recipe above), local-ckpt load, CPU/GPU.
- `repro/src/inspect_data.py` — cheap probe (target ranges + CDSS languages).
- `repro/src/verify_independent.py` — Pearson + bootstrap CI + permutation p + shuffled-target false-positive control.
- `repro/colab/regresslm_table3.ipynb` — Phase B full-scale (note: pin transformers==4.53.2 in it too).

## PROGRESS
- [x] install; checkpoint+data verified; eval contract understood.
- [x] **version fix (4.53.2) → model input-sensitive & accurate**.
- [x] Phase A (local CPU): **40 APPS rows → Spearman 0.937** (>0.9 ✅; ref 0.926), Pearson 0.920.
- [x] verify_independent: Pearson 0.920, perm p=0.0005, **shuffled-control ρ=-0.205 (SIGNAL OK)**; unit tests 4/4.
- [x] **Logbook PUBLISHED → https://huggingface.co/spaces/DineshAI/utTapVWtc7** (public, tagged icml2026-repro + paper-utTapVWtc7).
- [x] local git commit (31b193f, 32 files, secrets-clean).
- [ ] **GitHub public push: BLOCKED by auto-classifier (outward-facing) — needs user confirmation.** Local commit ready; `gh repo create MachineLearning-Nerd/icml26-repro-utTapVWtc7 --public --source=. --push`.
- [ ] Phase B (Colab GPU): full Table 3 — notebook ready (pinned transformers); **needs user Colab**.
- [ ] (optional) accumulate more APPS rows locally across ticks toward 512; re-run eval via `logbook run` to add Claim-2 CSV artifact (autosync).

## NEXT (resume here)
1. GitHub push (once user confirms) → record `gh_repo` in COORDINATION + STATUS.
2. Watch verdict on DineshAI/utTapVWtc7 (poll verdicts.json). If toy/inconclusive → add full-scale evidence (Colab or more local rows) + republish.
3. Phase B Colab: hand user the notebook for authoritative C2 (512) + C3 (17 langs).

## BLOCKERS
- GitHub public push gated (user confirmation). Full-scale Table 3 is GPU-only → Colab (user).

## venv
`papers/icml26-repro-utTapVWtc7-regresslm/.venv` (py3.12); `transformers==4.53.2`, `torch 2.13.0+cu130`, tokenizers 0.21.4, huggingface_hub 0.36.2.
