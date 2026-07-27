#!/usr/bin/env python3
"""Run every current standalone evaluator-facing evidence checker."""
from __future__ import annotations

import json

import verify_claim4_final
import verify_claim5_final
import verify_cumulative_space


def main() -> None:
    result = {
        "status": "PASS",
        "claims_1_to_3": verify_cumulative_space.verify(),
        "claim_4": verify_claim4_final.verify(),
        "claim_5": verify_claim5_final.verify(),
    }
    print("SPACE_RELEASE_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
