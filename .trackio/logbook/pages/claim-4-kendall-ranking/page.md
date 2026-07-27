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

Current evaluator-visible code and downloads:

- [exact scientific gate](../../code/claim4_exact_gate.py)
- [final independent checker](../../code/verify_claim4_final.py)
- [live public-artifact audit](../../code/verify_claim4_audit.py)
- [claim contract](../../evidence/claim4/claim_contract.json)
- [four-route record](../../evidence/claim4/four_routes.json)
- [raw audit output](../../evidence/claim4/raw_audit_output.json)
- [full calibration rows and 128 draws](../../evidence/claim4/raw_calibration_output.json)
- [released-audit independent output](../../evidence/claim4/released_audit_independent_checker_output.json)
- [calibration independent output](../../evidence/claim4/calibration_independent_checker_output.json)
- [negative control](../../evidence/claim4/negative_control_output.json)
- [method](../../evidence/claim4/method.md),
  [source audit](../../evidence/claim4/source_audit.md),
  [limitations](../../evidence/claim4/limitations.md)

## Four-route final assessment

Confidence remained **LOW** after three materially different routes, which
triggered the mandatory fourth, falsification-dedicated route:

| Route | Direct question | Result |
|---:|---|---|
| 1 | What exactly do Table 4 and the inherited FLAN protocol quantify? | Exact five-space arithmetic and protocol reconstructed; checkpoints/manifests incomplete. |
| 2 | Does correcting Spearman to Kendall on retained predictions answer the claim? | No. ENAS τ-b 0.172070 and NASNet τ-b 0.138758 use the unified base, not target-specific few-shot models. |
| 3 | Can a released target-specific checkpoint run end to end on CPU? | Yes. DARTS n=16 calibration passed, but is not the remaining full target domain. |
| 4 | Is there an assumption-satisfying counterexample to the exact five-space claim? | No valid counterexample: two target checkpoints and all five row manifests are absent. |

The exact claim verifier is deliberately fail-closed:

```bash
uv run --locked python repro/src/claim4_exact_gate.py
```

It exits 1 with `CLAIM4_EXACT_GATE_BLOCKED`. The cumulative fixed command
invokes an independent checker that requires this nonzero exit.

## Direct DARTS calibration — not paper-scale evidence

Run `6859632e-8d63-44d9-8e5b-97c5cb7fded8`, Git SHA
`7a489958ce0d307c491b740c3393c45fa0185418`, used the pinned target-specific
DARTS checkpoint and exact GraphArch release:

| Field | Value |
|---|---:|
| Rows / raw stochastic draws | 16 / 128 |
| Median prediction Kendall τ-b | 0.550000 |
| Independent pair-count τ-b | 0.550000 |
| Spearman ρ | 0.764706 |
| Shuffled-target mean τ-b (200 permutations) | -0.013583 |
| Setup / inference time | 50.250 s / 192.011 s |
| Linear 512-row inference estimate | 6,144.36 s |
| Estimated active / allocated CPUs | 8 / 64 |
| Total job duration | 5m12s |

Every draw, input hash, target, median, code hash, dataset hash, and timing
record is in
`.openresearch/artifacts/claim4_cpu_calibration/raw_calibration_output.json`.
The independent output is
`.openresearch/artifacts/claim4_cpu_calibration/independent_checker_output.json`,
and all route decisions are in
`.openresearch/artifacts/claim4_released_audit/four_routes.json`.

The calibration used a configuration-only compatibility patch because the
author checkpoint redundantly requests a gated base config despite serializing
the complete backbone. Author configuration SHA-256:
`5a72f125c3d1462f5394fdd0e376ee3ed3ac56b6f5a368585f527a59eab07313`;
compatibility configuration SHA-256:
`d3c6e7717cf86e279c0254b8c59b2643a2e88074f233a4e65eff95cfc08919e6`.
Model code and weights were unchanged.

The n=16 result is never used as evidence for the reported 0.461 average. It
cannot exclude unknown DARTS fine-tuning rows and covers neither the remaining
full DARTS domain nor all five spaces.

## Historical rejected baseline

The earlier Spearman-only NAS diagnostic is preserved for provenance but is
not the current verifier and must not be used to score Claim 4. The current
verifier is `repro/src/claim4_exact_gate.py`; the current evidence-integrity
checker is `repro/src/verify_claim4_final.py`.
