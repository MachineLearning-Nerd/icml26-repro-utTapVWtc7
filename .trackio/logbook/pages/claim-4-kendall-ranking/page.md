# Claim 4 — Kendall ranking and baseline comparison

**Current verdict: BLOCKED.** This is the current verifier and supersedes the
historical Spearman-only NAS diagnostic for purposes of Claim 4.

The exact paper contract is Kendall τ-b over five NDS search spaces:

| Method | NASNet | Amoeba | PNAS | ENAS | DARTS | Average |
|---|---:|---:|---:|---:|---:|---:|
| Arch2Vec | 0.209 | 0.107 | 0.184 | 0.224 | 0.333 | 0.212 |
| CATE | 0.150 | 0.160 | 0.217 | 0.236 | 0.425 | 0.238 |
| GNN | 0.364 | 0.376 | 0.444 | 0.438 | 0.523 | 0.429 |
| FLAN | 0.344 | 0.470 | 0.430 | 0.484 | 0.567 | 0.459 |
| RLM | 0.382 | 0.488 | 0.427 | 0.481 | 0.528 | 0.461 |

The protocol is 1,024 source examples, 16 target examples for NASNet,
Amoeba, PNAS, and ENAS, and 100 target examples for DARTS, followed by ranking
the remaining target search space. The target-shot identities must be excluded
from evaluation.

Public target-specific checkpoints are available for Amoeba, DARTS, and PNAS.
No public target-specific checkpoint is discoverable for ENAS or NASNet, and
none of the public checkpoint repositories contains a training/evaluation row
manifest or seeds. A unified base-model result is not accepted as a substitute.

The current executable audit uses the exact fixed command:

```bash
uv run --locked python repro/src/run_campaign.py
```

It reruns Claims 1–3, checks the public checkpoint revisions, recomputes the
Table 4 arithmetic, recomputes Kendall τ-b on retained ENAS/NASNet rows with
two independent implementations, and applies a shuffled-target control that
must fail the GNN threshold. Raw run output and CPU/runtime fields will be
inserted here after the first committed audit run.

Current code and contracts:

- `repro/src/verify_claim4_audit.py`
- `repro/src/check_claim4_audit.py`
- `.openresearch/artifacts/claim4_released_audit/claim_contract.json`
- `.openresearch/artifacts/claim4_released_audit/source_audit.md`
- `.openresearch/artifacts/claim4_released_audit/method.md`
- `.openresearch/artifacts/claim4_released_audit/EVAL.md`
- `.openresearch/artifacts/claim4_released_audit/limitations.md`
