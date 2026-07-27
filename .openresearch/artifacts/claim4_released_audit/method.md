# Claim 4 released-artifact audit method

The fixed campaign entrypoint first reruns the accepted Claims 1–3 regression
suite. It then:

1. retrieves the public author model listing and each released target-specific
   checkpoint manifest with an explicit browser user agent;
2. pins the observed model revisions and checks for split, seed, trainer, or
   training-row manifests;
3. recomputes Kendall tau-b for the retained 512-row ENAS and NASNet unified
   base-checkpoint predictions;
4. recomputes each Kendall result with an independent O(n²) pair-count
   implementation;
5. recomputes the paper's five-space arithmetic and all named comparator
   inequalities; and
6. permutes targets 200 times with seed 20260926476 and confirms that the
   negative control cannot exceed the GNN acceptance threshold.

The retained unified checkpoint rows repair the previous judge's metric
criticism but do not substitute for the unavailable target-specific ENAS and
NASNet checkpoints. The audit therefore passes as an audit while the claim
verdict remains `BLOCKED`.

Exact command:

```bash
uv run --locked python repro/src/run_campaign.py
```

Pinned environment: Python 3.12, `transformers==4.53.2`,
`torch==2.7.1` from the CPU-only PyTorch index, and all complete resolutions
in `uv.lock`.
