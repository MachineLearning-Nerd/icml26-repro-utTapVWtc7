#!/usr/bin/env python3
"""Static independent checker for the recorded Claim 5 audit evidence."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "evidence/claim5"
RAW = ARTIFACT_DIR / "raw_audit_output.json"
ROUTES = ARTIFACT_DIR / "four_routes.json"
EXACT_GATE = ROOT / "code/claim5_exact_gate.py"


def verify() -> dict:
    raw = json.loads(RAW.read_text())
    routes = json.loads(ROUTES.read_text())
    if raw["status"] != "PASS":
        raise AssertionError("recorded audit execution did not pass")
    if raw["claim_verdict"] != "BLOCKED" or raw["confidence"] != "LOW":
        raise AssertionError("recorded Claim 5 assessment changed")
    table5 = raw["exact_claim_tested"]["table5"]
    if table5 != {
        "decoder_head": 0.8,
        "normalized_regression_head": 0.717,
        "standard_regression_head": 0.478,
    }:
        raise AssertionError("Table 5 contract mismatch")
    table6 = raw["exact_claim_tested"]["table6"]
    difference = table6["t5gemma_b_b_600m"] - table6["t5gemma_s_s_300m"]
    if not math.isclose(difference, 0.038, abs_tol=1e-12):
        raise AssertionError("Table 6 difference mismatch")
    bases = raw["base_models"]
    small = bases["google/t5gemma-s-s-prefixlm"]["safetensors_parameter_count"]
    big = bases["google/t5gemma-b-b-prefixlm"]["safetensors_parameter_count"]
    if (small, big) != (312_517_632, 591_490_560):
        raise AssertionError("base parameter counts mismatch")
    if any(
        row["exact_table_artifact_discovered"]
        for row in raw["official_repository_snapshots"].values()
    ):
        raise AssertionError("record says an exact official artifact was found")
    if raw["author_public_models"]["exact_claim_checkpoint_discovered"]:
        raise AssertionError("record says an exact public checkpoint was found")
    cited = raw["cited_normalized_head_reference"]
    if cited["sha256"] != (
        "e5b2e3d31177fe2fa9572cbf99bcf87ff06e5786d2f029a251734f7d7e7558f3"
    ) or not all(cited["features"].values()):
        raise AssertionError("cited implementation evidence mismatch")
    control = raw["negative_control"]
    if not control["observed_detection"] or control["passes_claim_acceptance"]:
        raise AssertionError("detector control behaved incorrectly")
    route_rows = routes["routes"]
    if [row["route"] for row in route_rows] != [1, 2, 3, 4]:
        raise AssertionError("four-route sequence incomplete")
    if len({row["name"] for row in route_rows}) != 4:
        raise AssertionError("routes are not distinct")
    if route_rows[-1]["name"] != "Dedicated exact-claim falsification attempt":
        raise AssertionError("fourth route is not falsification-dedicated")
    if any(row["outcome"] != "BLOCKED" for row in route_rows):
        raise AssertionError("unsupported route result")
    gate = subprocess.run(
        [sys.executable, str(EXACT_GATE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if gate.returncode == 0 or "CLAIM5_EXACT_GATE_BLOCKED" not in gate.stdout:
        raise AssertionError("exact claim gate did not fail closed")
    return {
        "status": "PASS",
        "claim_verdict": "BLOCKED",
        "confidence": "LOW",
        "source_run_id": raw["source_run_id"],
        "source_git_sha": raw["source_git_sha"],
        "table6_difference": difference,
        "base_parameter_ratio": big / small,
        "routes_completed": len(route_rows),
        "fourth_route": "falsification attempted; no valid counterexample",
        "negative_control": control,
        "exact_claim_gate": {
            "exit_code": gate.returncode,
            "stdout": gate.stdout.strip(),
        },
        "unblockers": routes["unblockers"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
