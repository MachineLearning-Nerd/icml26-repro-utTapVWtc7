# Release report — cumulative five-claim campaign

- **Previous live judged score:** `6/10`
- **Conservative projected score range after the proposed change:** `6–6/10`
- **Best-supported possible new score:** `6/10` — forecast only, not a judge result

The current total remains **6/10**. Claims 1–3 are VERIFIED/HIGH. Claims 4 and
5 changed from unaddressed/inconclusive to rigorously BLOCKED/LOW after four
materially different routes each. No forecast point is added for a BLOCKED
claim.

| Claim | Current points | Possible points | Confidence | Evidence status | Basis and remaining risk |
|---|---:|---:|---|---|---|
| 1 — unified memory, latency, accuracy | 2 | 2 | HIGH | VERIFIED | One released checkpoint supports all three prediction types; actual 181.5M parameters remain discrepant from the 300M label. |
| 2 — APPS and kernel latency | 2 | 2 | HIGH | VERIFIED | APPS ρ=0.926807 and KBSS ρ=0.535279, n=512 each. Full-scale row draws were not retained in the judged repository. |
| 3 — 17 CodeNet languages | 2 | 2 | HIGH | VERIFIED | Mean ρ=0.529850 at 200 rows/language; independent result 0.523403. |
| 4 — five-space Kendall ranking | 0 | 2 | LOW | BLOCKED | ENAS/NASNet target checkpoints and all five exact row manifests are unavailable; no valid counterexample found. |
| 5 — head ablation and scaling | 0 | 2 | LOW | BLOCKED | Exact head artifacts, splits, checkpoints, 600M RLM, and Table 6 row identities are unavailable; no valid counterexample found. |

## Evaluator-visible visibility matrix

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker | Control | Exact claim tested | Reviewer verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | [Claim 1](../current-claim-1/page.md) | [cumulative checker](../../code/verify_cumulative.py) | 1,536 rows/12,288 draws and three correlations summarized | [full ONNX CSV](../../evidence/cumulative/full_n512.csv) | [validation JSON](../../evidence/cumulative/claim1_validation.json) | permutation and input-shuffle values inline | unified released checkpoint across memory, latency, accuracy | VERIFIED / HIGH |
| 2 | [Claim 2](../current-claim-2/page.md) | [cumulative checker](../../code/verify_cumulative.py) | ρ and n inline | [paper-scale summary](../../evidence/cumulative/table3_results.json); [historical n40 rows](../../evidence/cumulative/apps_n40_canonical.csv) | cumulative hash/statistic checker | n40 permutation/shuffle control linked; full rows not retained | APPS 0.930 and latency 0.516 at n=512 | VERIFIED / HIGH |
| 3 | [Claim 3](../current-claim-3/page.md) | [cumulative checker](../../code/verify_cumulative.py) | all 17 correlations and means inline | [3,400-row CSV](../../evidence/cumulative/full_gpu_n200.csv) | [independent JSON](../../evidence/cumulative/codenet_independent_verification.json) | bootstrap/permutation values inline | average positive ranking across exactly 17 languages | VERIFIED / HIGH |
| 4 | [Claim 4](../claim-4-kendall-ranking/page.md) | [exact gate](../../code/claim4_exact_gate.py); [checker](../../code/verify_claim4_final.py) | paper table, diagnostics, routes, runtime inline | [audit JSON](../../evidence/claim4/raw_audit_output.json); [128 draws](../../evidence/claim4/raw_calibration_output.json) | independent Kendall pair counter | 200 shuffled targets fail acceptance | exact five-space target-specific Kendall τ-b and comparators | BLOCKED / LOW |
| 5 | [Claim 5](../claim-5-ablation-scaling/page.md) | [exact gate](../../code/claim5_exact_gate.py); [checker](../../code/verify_claim5_final.py) | Tables 5/6, artifact audit, routes, runtime inline | [raw audit JSON](../../evidence/claim5/raw_audit_output.json) | [independent JSON](../../evidence/claim5/independent_checker_output.json) | synthetic exact-artifact detector is detected and rejected | exact three-head ablation and same-settings 300M→600M comparison | BLOCKED / LOW |

## Release facts

- Judged Space: `DineshAI/utTapVWtc7`
- Previous HF Head and Judge Head:
  `19231479d69a31c8e01832c24e146c37eca9a5ba`
- Fixed command: `uv run --locked python repro/src/run_campaign.py`
- Environment: Python 3.12, repository-level `.venv`, `uv.lock`,
  `torch==2.7.1`, `transformers==4.53.2`
- Compute: CPU only; local for bounded one-core checks, Hugging Face
  `cpu-upgrade` for uncertain/longer CPU work; no GPU
- Winning experiment branch:
  `orx/record-claim-5-exact-audit-evidence` at
  `638553f84622313314a6153501c33d48bbc77b81`

The exact publication action, after manifest, subset, secret, link, notebook,
and blind-review gates pass, is a text-only update to the existing
`DineshAI/utTapVWtc7` Space. The same text paths plus the visual report and
notebook are then mirrored to GitHub `master`. The paper will be marked
awaiting judge; no score increase will be claimed before a live verdict.
