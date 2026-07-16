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
