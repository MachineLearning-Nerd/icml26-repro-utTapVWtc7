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
