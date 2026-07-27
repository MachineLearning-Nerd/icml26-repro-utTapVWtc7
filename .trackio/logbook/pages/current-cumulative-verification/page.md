# Current cumulative verification

**Current campaign status: cumulative PASS; scientific forecast remains
6/10.** Claims 1–3 are VERIFIED/HIGH. Claims 4 and 5 are BLOCKED/LOW after four
routes each, including dedicated falsification attempts. This page is the
current regression-suite entrypoint and supersedes the pre-Claim-4/5 campaign
status.

Exact command:

```bash
uv run --locked python repro/src/run_campaign.py
```

| Claim | Verdict | Confidence | Current points | Current gate |
|---|---|---|---:|---|
| 1 | VERIFIED | HIGH | 2/2 | cumulative checker PASS |
| 2 | VERIFIED | HIGH | 2/2 | cumulative checker PASS |
| 3 | VERIFIED | HIGH | 2/2 | cumulative checker PASS |
| 4 | BLOCKED | LOW | 0/2 | exact gate exits 1; integrity checker PASS |
| 5 | BLOCKED | LOW | 0/2 | exact gate exits 1; integrity checker PASS |

Current code:
[fixed campaign](../../code/run_campaign.py),
[cumulative checker](../../code/verify_cumulative.py),
[Claim 4 final checker](../../code/verify_claim4_final.py),
and
[Claim 5 final checker](../../code/verify_claim5_final.py).

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
| CodeNet primary Colab bundle | mean ρ=0.529850, 17×200 rows, 27,200 draws |
| CodeNet independent CPU run | mean ρ=0.523403, 17×200 rows |

Scope limitation: this suite independently rechecks retained rows and
provenance; it does not regenerate model predictions. The judged revision's
historical content is preserved and remains reachable.

The frozen root's first fail-closed run exposed and rejected a provenance
mix-up between the two valid CodeNet tracks. The current verifier recomputes
each track separately.

Artifacts and executable source:

- `repro/src/run_campaign.py`
- `repro/src/verify_cumulative.py`
- `.openresearch/artifacts/cumulative_baseline/claim_contract.json`
- `.openresearch/artifacts/cumulative_baseline/expected_output.json`
- `.openresearch/artifacts/protected_judged_space_manifest.sha256`
