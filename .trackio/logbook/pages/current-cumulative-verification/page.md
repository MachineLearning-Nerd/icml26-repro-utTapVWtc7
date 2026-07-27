# Current cumulative verification

**Current campaign status: frozen 6/10 baseline.** This page supersedes no
accepted scientific evidence; it makes the current fail-closed regression
contract discoverable before new Claim 4/5 work begins.

Exact command:

```bash
uv run --locked python repro/src/run_campaign.py
```

Pinned environment: Python 3.12 from `.python-version`; all resolved packages
are in `uv.lock`; `transformers==4.53.2`; `torch==2.7.1`.

The committed verifier recomputes 1,536 ONNX prediction medians and all three
accuracy Spearman correlations, plus 17 separate CodeNet correlations and
their mean. It integrity-checks the paper-scale APPS/KBSS summary and evidence
bundle. Any missing row, mutated artifact, unexpected group, changed metric,
or bad hash raises and exits nonzero.

Raw expected output:

| Check | Result |
|---|---:|
| NASBench101 accuracy | ρ=0.406599, n=512 |
| ENAS accuracy | ρ=0.249461, n=512 |
| NASNet accuracy | ρ=0.206738, n=512 |
| APPS memory | ρ=0.926807, n=512 |
| KBSS latency | ρ=0.535279, n=512 |
| CodeNet | mean ρ=0.529850, 17×200 rows |

Scope limitation: this baseline independently rechecks retained rows and
provenance; it does not regenerate model predictions. The judged revision's
historical pages remain byte-for-byte preserved and reachable below.

Artifacts and executable source:

- `repro/src/run_campaign.py`
- `repro/src/verify_cumulative.py`
- `.openresearch/artifacts/cumulative_baseline/claim_contract.json`
- `.openresearch/artifacts/cumulative_baseline/expected_output.json`
- `.openresearch/artifacts/protected_judged_space_manifest.sha256`
