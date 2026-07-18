# STATUS — RegressLM (utTapVWtc7) reproduction — REVISED (resubmitted)

**Session:** perfect-score campaign. **Last updated:** 2026-07-18.
**State: Claim 1 paper-scale repair running with exactly 10 routes.** The latest
official verdict at Space SHA `40a78a2a039622516af008c75f074ce07cb44cf1`
is medium quality (5/6): Claims 2 and 3 are verified; Claim 1 remains `toy`
because the correct ONNX `val_accuracy` experiment covered only 64
NASBench101 rows. HF: https://huggingface.co/spaces/DineshAI/utTapVWtc7 ·
GitHub: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7.

## 2026-07-18 Claim 1 perfect-score repair

- Primary-source audit resolves “accuracy of code” to trained-network
  validation accuracy predicted from ONNX, not pass/fail accuracy of ordinary
  programs.
- `RLM-GemmaS-Code-v0` and the already-used Table-3 alias share the exact same
  725,864,700-byte weights (SHA-256 `7e9df429…`), so memory, latency, and
  accuracy are genuinely evaluated with one checkpoint.
- The Claim-1 ledger is frozen at exactly 10 routes—no route 11. Routes 6–8
  reproduce the author-card 512-row ONNX protocol on NASBench101, ENAS, and
  NASNet; routes 9–10 provide uncertainty/permutation and input-shuffle checks.
- Route 6 is currently running locally with resumable per-batch raw draws. The
  full released parquet is being acquired once for ENAS/NASNet, which are
  absent from Hugging Face's partial converted view.
- Thirteen scoped tests pass.

## 2026-07-17 repair result

- **Claim 1 accuracy gap closed:** released checkpoint on real `GraphArch-Regression` NASBench101 ONNX strings, `val_accuracy`, n=64, 8 draws/row: **Spearman 0.350603** (dataset-card reference 0.384).
- **Claim 3 full raw Colab evidence:** 17 languages × 200 rows × 8 draws: **mean per-language Spearman 0.529850**. Stratified bootstrap 95% CI **[0.502557, 0.554246]**, permutation p=0.000500, shuffled control 0.0352.
- **Independent full local corroboration:** 17 × 200 × 8, mean Spearman **0.523403**, permutation p=0.000500, shuffled control 0.0116.
- Colab evidence contains 3,464 raw rows and 27,712 raw draws. `verify_evidence_bundle.py` confirmed every prediction is the exact median of its 8 draws and recomputed every headline statistic. Bundle SHA-256: `b1d33e20922194cab7dfd20526cac13680f1ca16ce51ad7836a358feee5ebd1f`.
- Raw bundle: `outputs/colab/regresslm_evidence_bundle.zip`; audit: `outputs/colab/evidence_bundle_verification.json`; local raw rows: `outputs/codenet/full_gpu_n200.csv`.


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
- GTX 1050 (sm_61) requires torch 2.7.1+cu126; the cu130 build lacks sm_61 kernels. Colab T4 remains much faster.
- `datasets` streaming too slow over 5.6 GB parquet → pyarrow column projection + filter pushdown + scanner.head().
- OOM: reap orphaned pyarrow worker procs after kills.

## CODE
- `repro/src/run_eval.py` — faithful Table-3 evaluator (recipe above), local-ckpt load, CPU/GPU.
- `repro/src/inspect_data.py` — cheap probe (target ranges + CDSS languages).
- `repro/src/verify_independent.py` — Pearson + bootstrap CI + permutation p + shuffled-target false-positive control.
- `repro/src/verify_codenet.py` — language-stratified bootstrap/permutation verification.
- `repro/src/verify_evidence_bundle.py` — raw ZIP/draw/median/statistic audit.
- `repro/src/run_grapharch.py` — real ONNX accuracy evaluator.
- `repro/colab/regresslm_full_evidence_colab.ipynb` — one-click full CodeNet + ONNX accuracy evidence.

## PROGRESS
- [x] install; checkpoint+data verified; eval contract understood.
- [x] **version fix (4.53.2) → model input-sensitive & accurate**.
- [x] Phase A (local CPU): **40 APPS rows → Spearman 0.937** (>0.9 ✅; ref 0.926), Pearson 0.920.
- [x] verify_independent: Pearson 0.920, perm p=0.0005, **shuffled-control ρ=-0.205 (SIGNAL OK)**; unit tests 4/4.
- [x] **Logbook PUBLISHED → https://huggingface.co/spaces/DineshAI/utTapVWtc7** (public, tagged icml2026-repro + paper-utTapVWtc7).
- [x] local git commit (31b193f, 32 files, secrets-clean).
- [x] GitHub public repository and full raw evidence pushed.
- [x] Phase B Colab GPU: CodeNet 17×200 and NASBench101 accuracy n=64 complete with raw draws.
- [x] Independent local GPU: second CodeNet 17×200 run complete.
- [x] Repaired Claim 1, Claim 3, and conclusion logbook cells added and synced.
- [ ] (optional) accumulate more APPS rows locally across ticks toward 512; re-run eval via `logbook run` to add Claim-2 CSV artifact (autosync).

## NEXT (resume here)
1. Finish exactly routes 6–8 and execute the fixed routes 9–10—do not add more.
2. Independently verify 1,536 accuracy rows / 12,288 raw draws, sanitize and
   publish one pinned Trackio verdict, and pass the fail-closed publication gate.
3. Require an exact-SHA official 6/6 verdict, then continue to the next partial
   paper. The public judge's discovery loop is currently stuck because its Hub
   pagination query ignores `offset`; keep polling without calling stale 5/6
   perfect.

## BLOCKERS
- No evidence blocker. Official re-judging is temporarily blocked by the public
  judge service's discovery pagination bug.

## venv
`papers/icml26-repro-utTapVWtc7-regresslm/.venv` (py3.12); `transformers==4.53.2`, `torch 2.7.1+cu126` (local sm_61-compatible build).
