#!/usr/bin/env python3
"""Independent final checker for the four-route Claim 4 evidence."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = (
    ROOT
    / ".openresearch/artifacts/claim4_cpu_calibration/raw_calibration_output.json"
)
ROUTES = ROOT / ".openresearch/artifacts/claim4_released_audit/four_routes.json"
EXACT_GATE = ROOT / "repro/src/claim4_exact_gate.py"


def manual_tau_b(xs: list[float], ys: list[float]) -> float:
    concordant = 0
    discordant = 0
    for left in range(len(xs) - 1):
        for right in range(left + 1, len(xs)):
            product = (
                (xs[left] > xs[right]) - (xs[left] < xs[right])
            ) * (
                (ys[left] > ys[right]) - (ys[left] < ys[right])
            )
            concordant += int(product > 0)
            discordant += int(product < 0)
    pairs = len(xs) * (len(xs) - 1) // 2
    tied_x = sum(n * (n - 1) // 2 for n in Counter(xs).values())
    tied_y = sum(n * (n - 1) // 2 for n in Counter(ys).values())
    return (concordant - discordant) / math.sqrt(
        (pairs - tied_x) * (pairs - tied_y)
    )


def verify() -> dict:
    calibration = json.loads(CALIBRATION.read_text())
    routes = json.loads(ROUTES.read_text())
    rows = calibration["raw_rows"]
    if calibration["status"] != "PASS":
        raise AssertionError("calibration execution did not pass")
    if calibration["paper_scale_evidence"] is not False:
        raise AssertionError("bounded calibration mislabeled as paper scale")
    if calibration["claim_verdict"] != "BLOCKED":
        raise AssertionError("calibration cannot decide Claim 4")
    if len(rows) != 16 or sum(len(row["draws"]) for row in rows) != 128:
        raise AssertionError("expected 16 rows and 128 raw draws")
    if len({row["identifier"] for row in rows}) != len(rows):
        raise AssertionError("duplicate calibration identifier")
    for row in rows:
        if not math.isclose(
            statistics.median(row["draws"]), row["prediction"], abs_tol=1e-12
        ):
            raise AssertionError(f"median mismatch: {row['identifier']}")
    tau = manual_tau_b(
        [row["target"] for row in rows],
        [row["prediction"] for row in rows],
    )
    if not math.isclose(
        tau, calibration["metrics"]["kendall_tau_b"], abs_tol=1e-12
    ):
        raise AssertionError("independent Kendall pair count disagrees")
    if calibration["negative_control"]["passes_acceptance"] is not False:
        raise AssertionError("shuffled-target control incorrectly passes")
    route_rows = routes["routes"]
    if [row["route"] for row in route_rows] != [1, 2, 3, 4]:
        raise AssertionError("four-route sequence is incomplete")
    if len({row["name"] for row in route_rows}) != 4:
        raise AssertionError("routes are not materially distinct")
    if route_rows[-1]["name"] != "Dedicated exact-claim falsification attempt":
        raise AssertionError("fourth route is not falsification-dedicated")
    if any(row["outcome"] != "BLOCKED" for row in route_rows):
        raise AssertionError("unsupported route outcome")
    if routes["confidence_after_three_routes"] != "LOW":
        raise AssertionError("mandatory fourth-route trigger not recorded")
    gate = subprocess.run(
        [sys.executable, str(EXACT_GATE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if gate.returncode == 0 or "CLAIM4_EXACT_GATE_BLOCKED" not in gate.stdout:
        raise AssertionError("exact claim gate did not fail closed")
    return {
        "status": "PASS",
        "claim_verdict": "BLOCKED",
        "confidence": "LOW",
        "routes_completed": len(route_rows),
        "fourth_route": "falsification attempted; no valid counterexample",
        "calibration": {
            "rows": len(rows),
            "raw_draws": 128,
            "manual_kendall_tau_b": tau,
            "inference_seconds": calibration["timing"]["inference_seconds"],
            "source_run_id": calibration["source_run_id"],
            "source_git_sha": calibration["source_git_sha"],
            "paper_scale_evidence": False,
        },
        "negative_control": calibration["negative_control"],
        "exact_claim_gate": {
            "exit_code": gate.returncode,
            "stdout": gate.stdout.strip(),
        },
        "raw_calibration_sha256": hashlib.sha256(
            CALIBRATION.read_bytes()
        ).hexdigest(),
        "unblockers": routes["unblockers"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
