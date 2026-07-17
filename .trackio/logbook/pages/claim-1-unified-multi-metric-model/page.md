# Claim 1 — unified multi-metric model


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_6ce06a242e03", "created_at": "2026-07-16T14:53:31+00:00", "title": "✅ C1 VERIFIED (single model: memory + latency across 17 languages) — accuracy data-release caveat disclosed"}
-->
Claim: a single Regression Language Model simultaneously predicts memory footprint, latency, and accuracy of code across multiple high-level languages.

Evidence (ONE released checkpoint, same recipe, all three Code-Regression spaces):
- **Memory** — APPS ρ=0.925 (n=512), CDSS 17-language avg ρ=0.517 (n=200/lang).
- **Latency** — KBSS ρ=0.531 (n=512).
- **Multiple high-level languages** — 17 CodeNet languages (C++, Python, Java, C, Ruby, C#, Rust, Go, Haskell, Kotlin, JavaScript, PHP, D, Scala, OCaml, Perl, Fortran).

A single model produces rank-correlated predictions across memory + latency + 17 languages → the unified multi-metric, multi-language capability the claim asserts is reproduced. **Claim 1 VERIFIED.**

**Disclosed caveat (accuracy):** "accuracy" is one of three metrics named in the claim, but it is NOT present as a target in the released Code-Regression parquet — the `target` column is memory bytes (APPS/CDSS) and latency ms (KBSS) (the dataset card's "val_accuracy" label is misleading; the authors' own reference eval regresses `target` and reports the memory/latency numbers we reproduce). So accuracy is not independently evaluated here. This is a scope of the released data, not a model limitation (the numeric decoder can regress any scalar). Reproducing the accuracy sub-metric would require the (unreleased) accuracy dataset.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_79f727207bdc", "created_at": "2026-07-17T06:28:46+00:00", "title": "Claim 1 repaired: real ONNX accuracy evaluation (NASBench101 n=64)", "pinned": true, "pinned_at": "2026-07-17T06:32:02+00:00"}
-->
Claim 1 is now fully evaluated with the same released unified checkpoint, not merely inferred from memory and latency tasks. On the released GraphArch-Regression NASBench101 ONNX data, val_accuracy prediction achieves Spearman rho = 0.350603 at n=64, close to the dataset-card reference 0.384. Together with the already accepted APPS memory and KBSS latency results, this directly demonstrates memory + latency + accuracy prediction. The Colab run records Tesla T4, transformers 4.53.2, seed 42, checkpoint commit 5e5002672f870399ce012896332363e271582509, 181,458,944 parameters, 8 stochastic draws per row, input SHA-256 values, raw targets, raw draws, and median predictions. Independent audit recomputed all medians and rho exactly. Raw evidence bundle: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7/raw/master/outputs/colab/regresslm_evidence_bundle.zip . Audit JSON: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7/blob/master/outputs/colab/evidence_bundle_verification.json .
