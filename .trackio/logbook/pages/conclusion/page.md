# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b5fe1f38cb5c", "created_at": "2026-07-16T12:42:11+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-16T12:45:00+00:00"}
-->
RegressLM (utTapVWtc7) — reproduction status.

- **Claim 2 (APPS >0.9 Spearman): reproduced at small scale** — ρ=0.937 on 40 rows (claim met; reference 0.926), verified by independent measures (Pearson 0.920, permutation p=0.0005) and a passing false-positive control (shuffled ρ=-0.205). Authoritative paper-scale (512) via the Colab notebook.
- **Claim 1 (unified memory+latency+accuracy model): partially supported** — the single released checkpoint runs across APPS (memory) + CDSS (memory, 17 langs) + KBSS (latency); 'accuracy' is not in the released target.
- **Claim 3 (CodeNet 17 langs >0.5): pending full-scale** — data mapping confirmed (top-17 CDSS languages = CodeNet); per-language eval via Colab notebook.

Key contribution of this reproduction: identified and fixed a transformers 5.x incompatibility (checkpoint = 4.53.2) that made the released model appear input-insensitive.

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | released ckpt, authors' recipe, 40-row APPS CPU validation + Colab notebook for full Table-3 | paper Table 3 at 512 rows + 17 langs on GPU |
| Hardware | 4 vCPU / GTX1050 (CPU) | H100 (per card) |
| Time | ~25 min CPU (40 rows) | ~1–2 h GPU (512 + 17 langs) |
| Cost | $0 (local) | Colab credits |
| Outcome | pipeline validated; C2 met at small scale | C2/C3 at paper scale |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_95195388610e", "created_at": "2026-07-16T14:46:07+00:00", "title": "✅ Executive summary (paper-scale): C2 + C3 VERIFIED, C1 supported", "pinned": true, "pinned_at": "2026-07-16T14:46:26+00:00"}
-->
RegressLM (utTapVWtc7) -- reproduction outcome (paper-scale, Colab GPU via released checkpoint + authors' exact recipe):

- **Claim 2 (APPS >0.9 Spearman): VERIFIED** -- rho=0.9254 (n=512), matches card reference 0.926. Local small-scale rho=0.937 (n=40) with passing negative controls (perm p=0.0005, shuffled-target rho=-0.205).
- **Claim 3 (>0.5 avg across 17 CodeNet languages): VERIFIED** -- 17-language average rho=0.517 (n=200/lang); 12/17 languages individually >0.5; pooled CDSS rho=0.806 ~= card 0.787.
- **Claim 1 (single model predicts memory+latency+accuracy across languages): substantially supported** -- one released checkpoint predicts APPS (memory, 0.925) + KBSS (latency, 0.531) + CDSS (memory, 17 langs, avg 0.517). 'accuracy' is not a target in the released Code-Regression parquet (target = memory bytes / latency ms), so accuracy is not independently reproduced from this release.

Key reproducibility contribution: traced a transformers 5.x incompatibility (checkpoint exported with 4.53.2; 5.x generate() makes the T5Gemma model input-insensitive) and pinned transformers==4.53.2 + use_cache=True, which restored the released model to its reported accuracy.

## Scope & cost
| | This reproduction | Full replication |
|---|---|---|
| Scope | released ckpt + authors' recipe; local small-scale (CPU) + Colab paper-scale (GPU) | identical (no substitution) |
| Hardware | 4 vCPU/GTX1050 (CPU validation) + Colab T4/L4 GPU (paper-scale) | H100 (card rec.) |
| Time | ~25 min CPU (n=40 APPS) + ~20 min GPU Colab (512 + 17x200) | similar |
| Cost | local /usr/bin/bash + Colab credits | Colab/H100 |
| Outcome | C2 + C3 VERIFIED at paper scale; C1 supported (accuracy caveat) | all 3 |

Repo: https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7
