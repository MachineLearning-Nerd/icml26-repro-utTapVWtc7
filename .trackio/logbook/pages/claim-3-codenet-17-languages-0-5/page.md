# Claim 3 — CodeNet 17 languages >0.5


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_a83bdddce014", "created_at": "2026-07-16T12:42:10+00:00", "title": "CodeNet per-language Spearman (full scale via Colab)"}
-->
Claim: the unified model achieves >0.5 average Spearman across 17 CodeNet languages.

Data mapping confirmed: the CDSS space carries 33 languages; the top-17 by count are exactly the CodeNet set — C++, Python, Java, C, Ruby, C#, Rust, Go, Haskell, Kotlin, JavaScript, PHP, D, Scala, OCaml, Perl, Fortran. The card's overall CDSS reference ρ=0.787.

Reproduction status: per-language eval is GPU-bound at scale (CDSS rows × 17 langs × 8 samples). The Colab notebook (repro/colab/regresslm_table3.ipynb) runs it end-to-end: enumerate the 17 languages, fetch ~200 CDSS rows each, predict, report per-language ρ and the average. The notebook pins transformers==4.53.2.

Local CPU validation of a couple of CDSS languages is feasible next tick; the authoritative 17-language average comes from Colab.
