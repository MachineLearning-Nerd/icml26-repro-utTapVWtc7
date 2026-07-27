# Claim 4 evaluator contract

Run:

```bash
uv run --locked python repro/src/run_campaign.py
```

Success requires:

- the accepted Claims 1–3 cumulative verifier to pass;
- exact agreement between SciPy Kendall tau-b and the independent pair-count
  implementation;
- the Table 4 per-space arithmetic and comparator inequalities to match;
- all three public target-specific checkpoints to remain pinned to their
  audited revisions;
- no hidden substitution for the missing ENAS or NASNet checkpoint; and
- shuffled targets to fail the 0.429 GNN threshold.

Any discrepancy raises and exits nonzero. A successful audit does **not**
change the scientific Claim 4 verdict from `BLOCKED`; direct all-five-space
evidence remains required.

Observed run `b807025c-55e6-43d3-8c9f-7142caaccc48` at Git SHA
`bcedb13e09389703999d5bcc1735fc555a340e42` passed. The independent checker
recomputed the Table 4 average as 0.4612, verified four comparator
inequalities, matched both Kendall implementations exactly, and confirmed
that both shuffled-target controls fail acceptance.
