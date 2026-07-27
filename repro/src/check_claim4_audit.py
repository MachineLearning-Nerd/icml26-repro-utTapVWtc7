#!/usr/bin/env python3
"""Independent fail-closed checks for the Claim 4 released-artifact audit."""
from __future__ import annotations

import math


EXPECTED_RLM = {
    "NASNet": 0.382,
    "Amoeba": 0.488,
    "PNAS": 0.427,
    "ENAS": 0.481,
    "DARTS": 0.528,
}
EXPECTED_BASELINES = {
    "Arch2Vec": 0.212,
    "CATE": 0.238,
    "GNN": 0.429,
    "FLAN": 0.459,
}
EXPECTED_PUBLIC_TARGETS = {"Amoeba", "DARTS", "PNAS"}


def check(result: dict) -> dict:
    table = result["paper_table"]
    calculated_average = sum(EXPECTED_RLM.values()) / len(EXPECTED_RLM)
    if not math.isclose(calculated_average, 0.4612, abs_tol=1e-15):
        raise AssertionError("independent Table 4 RLM average changed")
    if not math.isclose(table["rlm_average_unrounded"], calculated_average, abs_tol=1e-15):
        raise AssertionError("reported RLM average is not the mean of five spaces")
    if round(calculated_average, 3) != 0.461:
        raise AssertionError("Table 4 rounded average is not 0.461")
    for name, expected in EXPECTED_BASELINES.items():
        observed = table["reported_average"][name]
        if not math.isclose(observed, expected, abs_tol=1e-15):
            raise AssertionError(f"{name} reported average changed")
    if not all(calculated_average > EXPECTED_BASELINES[name]
               for name in ("Arch2Vec", "CATE", "GNN", "FLAN")):
        raise AssertionError("reported RLM average does not exceed all named comparators")

    availability = result["public_artifacts"]
    if set(availability["released_target_specific"]) != EXPECTED_PUBLIC_TARGETS:
        raise AssertionError("public target-specific checkpoint coverage changed")
    if set(availability["missing_target_specific"]) != {"ENAS", "NASNet"}:
        raise AssertionError("missing target-specific checkpoint coverage changed")
    if availability["training_row_manifests_found"]:
        raise AssertionError("training-row manifest appeared; rerun the exact split audit")

    for row in result["retained_base_checkpoint_metric_reanalysis"]:
        if not math.isclose(row["scipy_kendall_tau_b"], row["manual_kendall_tau_b"],
                            abs_tol=1e-12):
            raise AssertionError(f"{row['space']}: independent Kendall implementations differ")
        control = row["shuffled_target_control"]
        if control["passes_gnn_threshold"]:
            raise AssertionError(f"{row['space']}: shuffled-target control unexpectedly passes")
        if control["threshold"] != EXPECTED_BASELINES["GNN"]:
            raise AssertionError("negative-control threshold changed")

    if result["claim_verdict"] != "BLOCKED":
        raise AssertionError("incomplete checkpoint/split evidence cannot receive a claim verdict")
    return {
        "status": "PASS",
        "table_average_recomputed": calculated_average,
        "comparator_inequalities_verified": 4,
        "kendall_implementations_agree": True,
        "negative_controls_fail_acceptance": True,
        "claim_verdict": "BLOCKED",
    }
