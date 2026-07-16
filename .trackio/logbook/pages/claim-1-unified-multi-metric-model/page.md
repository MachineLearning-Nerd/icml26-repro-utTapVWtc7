# Claim 1 — unified multi-metric model


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ff56c4458f77", "created_at": "2026-07-16T12:41:51+00:00", "title": "Single RLM across memory + latency + multiple languages"}
-->
Claim: a single Regression LM simultaneously predicts memory footprint, latency, and accuracy of code across multiple high-level languages.

Evidence framework: the released checkpoint is ONE model evaluated across all three spaces of Code-Regression — APPS (memory bytes), CDSS (memory bytes, 33 languages incl. the CodeNet 17), KBSS (latency ms). The single model produces rank-correlated predictions on each (APPS ρ=0.937 here; CDSS/KBSS measured by the Colab notebook). This demonstrates a unified model operating across metrics (memory/latency) and languages.

Honest caveat: the released dataset's target is memory (APPS/CDSS) + latency (KBSS); 'accuracy' is named in the claim but not present as a target in the released parquet, so accuracy is not independently reproduced from this release.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_44aadd7b3e30", "created_at": "2026-07-16T14:46:08+00:00", "title": "C1: single model across memory + latency + 17 languages (supported)"}
-->
Claim: a single Regression LM predicts memory + latency + accuracy across multiple languages.

Evidence: ONE released checkpoint (RegressLM-gemma-s-RLM-table3) is evaluated across all three Code-Regression spaces with the same recipe -- APPS (memory bytes, rho=0.925), KBSS (latency ms, rho=0.531), CDSS (memory bytes, 17 languages, avg rho=0.517). This demonstrates a unified model producing rank-correlated predictions across metrics (memory + latency) and 17 languages.

Honest caveat: the released dataset's target is memory (APPS/CDSS) + latency (KBSS); 'accuracy' is named in the claim but is not present as a target column in the released parquet, so accuracy is not independently reproduced from this release. The memory+latency+multi-language part of the claim is supported.
