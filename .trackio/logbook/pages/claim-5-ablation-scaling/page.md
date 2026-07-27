# Claim 5 — regression-head ablation and model scaling

**Current verdict: BLOCKED (LOW confidence).** This is the current Claim 5
verifier. No previous generic head implementation or released-checkpoint
diagnostic should be interpreted as verification or falsification of Tables 5
or 6.

## Exact paper contract

The paper reports:

| Table 5 formulation | Training objective | Spearman ρ on 512 NASBench101 samples |
|---|---|---:|
| Standard regression head | MSE | 0.478 |
| Normalized regression head | MSE after per-dataset [0,1] normalization | 0.717 |
| Decoder head | Digit-token cross-entropy | 0.800 |

The Table 5 training domains are NASBench101, SNAS, OFA ResNet, OFA
ProxylessNAS, and OFA MobileNet. The standard and normalized alternatives are
described as encoder-only four-layer models; the decoder version uses two
encoder and two decoder layers.

| Table 6 model | Displayed size | Spearman ρ on 1,024 CodeNet samples |
|---|---:|---:|
| T5Gemma s-s prefix-LM | 300M | 0.744 |
| T5Gemma b-b prefix-LM | 600M | 0.782 |

The reported improvement is 0.038. The paper says the two sizes use exactly the
same settings and a smaller training subset of CodeNet, APPS, and KernelBook.

Source: ar5iv HTML retrieved 2026-07-27 with explicit User-Agent, SHA-256
`5947f4512cc86850a63409adf52af25ac1f40b15dcc797348fd8ae91a2740913`;
anchors `S6.SS2`, `S6.T5`, and `S6.T6`.

## Current executable evidence

The fixed command is:

```bash
uv run --locked python repro/src/run_campaign.py
```

The current integrity checker is `repro/src/verify_claim5_audit.py`. The exact
scientific gate is:

```bash
uv run --locked python repro/src/claim5_exact_gate.py
```

The exact gate deliberately exits 1 with `CLAIM5_EXACT_GATE_BLOCKED`. An
integrity-check PASS means that this BLOCKED conclusion is reproducible; it
does not mean the scientific claim passed.

The public evidence audit covers two pinned official repository snapshots, the
author's full public model listing, both named T5Gemma base-model metadata
records, and the cited normalized-regression primary implementation.

| Audit item | Directly observed | What it establishes |
|---|---|---|
| Official `regress-lm` paper-time and current trees | No dedicated Table 5/6 configuration, implementation, manifest, or exact-marker hit | Exact public experiment artifact not discoverable; not proof of nonexistence |
| Author public RLM/model listing | No Table 5 head checkpoint and no b-b/600M RLM | Exact public checkpoint not discoverable |
| T5Gemma s-s / b-b base metadata | 312,517,632 / 591,490,560 parameters; ratio 1.89266 | Paper's rounded size labels are consistent; performance is not tested |
| Table 6 arithmetic | 0.782 − 0.744 = 0.038 | Displayed arithmetic is internally consistent |
| Cited Qin et al. code | One-output sequence-classification regressor and per-dataset MinMaxScaler to [0,1] | Defensible normalization interpretation, not a Table 5 reproduction |
| Synthetic detector control | `configs/table5_regression_head.yaml` is detected | The absence detector responds to an artifact it should flag |

Both base-model repositories require manual license/access acceptance. That is
an access observation only. The released 181.5M RLM is not identified as either
exact Table 6 checkpoint and is therefore rejected as a counterexample.

## Four-route final assessment

Confidence remained **LOW** after three materially different routes, triggering
the mandatory falsification-dedicated fourth route:

| Route | Direct question | Result |
|---:|---|---|
| 1 | Are the exact Table 5/6 artifacts in the official public releases? | Values and domains reconstructed; exact implementations, checkpoints, splits, and 600M RLM not discoverable. |
| 2 | Does the cited primary implementation resolve normalized regression? | It confirms [0,1] per-dataset normalization and a scalar regressor, but not the paper's exact experiment. |
| 3 | Do model identities and public access support the scaling experiment? | Rounded base sizes and 0.038 arithmetic are consistent; the exact RLMs, data identities, and same-settings recipe are absent. |
| 4 | Is there an assumption-satisfying counterexample? | No. Missing access, generic implementations, base metadata, and the 181.5M release do not satisfy the exact experiment assumptions. |

No route produces raw predictions from all three exact Table 5 formulations or
both exact Table 6 RLMs on the specified rows. Missing artifacts are not
falsification. Claim 5 is therefore BLOCKED, forecasts 0/2 points, and needs:

- pinned Table 5 checkpoints or complete code/config for all three heads;
- exact Table 5 train/evaluation row IDs, seeds, and checkpoint selection;
- pinned Table 6 300M and 600M RLM checkpoints or complete training configs;
- exact Table 6 smaller training subset and 1,024 CodeNet evaluation row IDs.

## Visible code and evidence

- `repro/src/verify_claim5_audit.py`
- `repro/src/claim5_exact_gate.py`
- `.openresearch/artifacts/claim5_ablation_scaling/claim_contract.json`
- `.openresearch/artifacts/claim5_ablation_scaling/source_audit.md`
- `.openresearch/artifacts/claim5_ablation_scaling/method.md`
- `.openresearch/artifacts/claim5_ablation_scaling/four_routes.json`
- `.openresearch/artifacts/claim5_ablation_scaling/EVAL.md`
- `.openresearch/artifacts/claim5_ablation_scaling/limitations.md`

## Historical rejected baseline

The judged Space contained no Table 5 or Table 6 reproduction. Any earlier
generic released-checkpoint result remains preserved for provenance but is not
the current Claim 5 verifier. The current exact gate and public-artifact
integrity checker listed above supersede it.
