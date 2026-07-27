#!/usr/bin/env python3
"""Fail-closed gate for the exact scientific Claim 5 contract."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / ".openresearch/artifacts/claim5_ablation_scaling/four_routes.json"


def main() -> None:
    evidence = json.loads(ROUTES.read_text())
    verdict = evidence["final_verdict"]
    if verdict == "VERIFIED":
        print("CLAIM5_EXACT_GATE_VERIFIED")
        return
    if verdict == "FALSIFIED":
        print("CLAIM5_EXACT_GATE_FALSIFIED")
        return
    print(
        "CLAIM5_EXACT_GATE_BLOCKED missing exact Table 5 implementations, "
        "training splits, checkpoints, and the Table 6 600M RLM checkpoint"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
