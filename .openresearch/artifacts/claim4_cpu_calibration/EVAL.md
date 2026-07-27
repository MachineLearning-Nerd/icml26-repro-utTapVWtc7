# Calibration evaluator contract

The fixed command exits nonzero if:

- Claims 1–3 regress;
- the Claim 4 public-artifact audit changes unexpectedly;
- any of 16 DARTS predictions is non-finite;
- SciPy and independent Kendall tau-b differ by more than 1e-12; or
- the mean shuffled-target control reaches the 0.429 GNN threshold.

A pass establishes only a valid CPU throughput calibration. The scientific
gate is separate:

```bash
uv run --locked python repro/src/claim4_exact_gate.py
```

It must exit nonzero while the exact five-space evidence is incomplete.
