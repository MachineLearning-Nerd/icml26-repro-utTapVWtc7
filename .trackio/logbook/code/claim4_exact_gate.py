#!/usr/bin/env python3
"""Fail-closed gate for the exact scientific Claim 4 contract."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUTES = ROOT / ".openresearch/artifacts/claim4_released_audit/four_routes.json"


def main() -> None:
    evidence = json.loads(ROUTES.read_text())
    if evidence["final_verdict"] == "VERIFIED":
        print("CLAIM4_EXACT_GATE_VERIFIED")
        return
    if evidence["final_verdict"] == "FALSIFIED":
        print("CLAIM4_EXACT_GATE_FALSIFIED")
        return
    print(
        "CLAIM4_EXACT_GATE_BLOCKED missing target-specific ENAS/NASNet "
        "checkpoints and five-space row manifests"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
