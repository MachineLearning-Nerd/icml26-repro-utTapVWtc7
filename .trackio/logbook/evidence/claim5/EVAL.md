# Claim 5 evaluation contract

## Scientific gate

`uv run --locked python repro/src/claim5_exact_gate.py` is the exact scientific
gate. It exits zero only for a recorded `VERIFIED` or `FALSIFIED` verdict. With
the current evidence it must exit 1 with `CLAIM5_EXACT_GATE_BLOCKED`.

## Evidence-integrity gate

`uv run --locked python repro/src/verify_claim5_final.py` must exit zero and
return `status=PASS`, `claim_verdict=BLOCKED`, `confidence=LOW`. The fixed
campaign also reruns `verify_claim5_audit.py` against the public sources. PASS
means the audit is internally consistent; it does not mean Claim 5 passes.

The integrity checker requires:

- exact Table 5 and Table 6 values and arithmetic;
- pinned official repository, model metadata, and cited implementation;
- no undisclosed substitution for the missing experimental artifacts;
- a working artifact-detector control;
- four materially different routes with the fourth dedicated to falsification;
- a nonzero exact scientific gate.

Any newly detected candidate artifact, changed pinned revision, missing route,
or scientific gate that incorrectly exits zero makes the integrity checker fail.
