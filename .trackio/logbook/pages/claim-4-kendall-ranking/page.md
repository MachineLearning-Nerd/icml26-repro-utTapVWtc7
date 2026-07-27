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
must fail the GNN threshold.

Audit run `b807025c-55e6-43d3-8c9f-7142caaccc48` at Git SHA
`bcedb13e09389703999d5bcc1735fc555a340e42` passed:

| Audit result | Observed |
|---|---:|
| Paper RLM average, recomputed | 0.4612 → 0.461 |
| ENAS unified-base τ-b | 0.172070, n=512 |
| NASNet unified-base τ-b | 0.138758, n=512 |
| Public target-specific checkpoints | 3/5 |
| Training/evaluation row manifests | 0 |
| Independent Kendall agreement | exact to 1e-12 |
| ENAS shuffled-target mean / max τ-b | 0.001141 / 0.072443 |
| NASNet shuffled-target mean / max τ-b | 0.005365 / 0.074279 |

Both shuffled controls remain far below the GNN threshold 0.429 and therefore
fail the claim acceptance test as intended. The ENAS and NASNet values above
use the unified base checkpoint and are diagnostics only; they are not the
paper's unavailable target-specific few-shot models.

Compute record: estimated one active core; Hugging Face `cpu-upgrade`;
64 CPUs visible to the job; verifier runtime 1.188 s; total run duration 37 s;
no GPU. The run record exposes no monetary-cost field.

Current code and contracts:

- `repro/src/verify_claim4_audit.py`
- `repro/src/check_claim4_audit.py`
- `.openresearch/artifacts/claim4_released_audit/claim_contract.json`
- `.openresearch/artifacts/claim4_released_audit/source_audit.md`
- `.openresearch/artifacts/claim4_released_audit/method.md`
- `.openresearch/artifacts/claim4_released_audit/EVAL.md`
- `.openresearch/artifacts/claim4_released_audit/limitations.md`
- `.openresearch/artifacts/claim4_released_audit/raw_audit_output.json`
- `.openresearch/artifacts/claim4_released_audit/independent_checker_output.json`
- `.openresearch/artifacts/claim4_released_audit/negative_control_output.json`
- `.openresearch/artifacts/claim4_released_audit/runtime.json`
