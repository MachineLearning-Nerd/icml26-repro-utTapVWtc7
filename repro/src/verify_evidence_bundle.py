#!/usr/bin/env python3
"""Audit the Colab evidence ZIP from raw draws through headline statistics."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections import defaultdict

import numpy as np
from scipy import stats


def read_csv(archive, name):
    return list(csv.DictReader(io.TextIOWrapper(archive.open(name))))


def verify_prediction_medians(rows):
    for row in rows:
        draws = np.asarray([float(row[f"draw_{index}"]) for index in range(8)])
        expected = float(np.nanmedian(draws))
        if not math.isclose(
                float(row["prediction"]), expected, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"row {row.get('i')} prediction is not draw median")


def spearman(rows):
    return float(stats.spearmanr(
        [float(row["target"]) for row in rows],
        [float(row["prediction"]) for row in rows],
    ).correlation)


def verify(path):
    bundle_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    duplicate_audit = {}
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise AssertionError(f"corrupt ZIP member: {bad_member}")
        summary = json.loads(archive.read("summary.json"))
        language_correlations = []
        total_rows = 0
        for expected in summary["claim_3_codenet"]["per_language"]:
            filename = expected["task"] + ".csv"
            rows = read_csv(archive, filename)
            if len(rows) != 200:
                raise AssertionError(f"{filename}: expected 200 rows, got {len(rows)}")
            if [int(row["i"]) for row in rows] != list(range(200)):
                raise AssertionError(f"{filename}: row indices are incomplete")
            verify_prediction_medians(rows)
            correlation = spearman(rows)
            if not math.isclose(
                    correlation, expected["spearman"], rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f"{filename}: stored Spearman does not recompute")
            language_correlations.append(correlation)
            total_rows += len(rows)

            by_hash = defaultdict(list)
            for row in rows:
                by_hash[row["input_sha256"]].append(float(row["target"]))
            duplicate_groups = {
                digest: targets for digest, targets in by_hash.items()
                if len(targets) > 1
            }
            if duplicate_groups:
                duplicate_audit[filename] = {
                    "groups": len(duplicate_groups),
                    "rows": sum(len(values) for values in duplicate_groups.values()),
                    "all_targets_identical": all(
                        len(set(values)) == 1 for values in duplicate_groups.values()),
                }

        average = float(np.mean(language_correlations))
        stored_average = float(
            summary["claim_3_codenet"]["average_spearman"])
        if not math.isclose(average, stored_average, rel_tol=0, abs_tol=1e-15):
            raise AssertionError("stored CodeNet average does not recompute")

        onnx_expected = summary["claim_1_onnx_accuracy"][0]
        onnx_rows = read_csv(archive, "onnx_nasbench101.csv")
        if len(onnx_rows) != 64:
            raise AssertionError(f"ONNX: expected 64 rows, got {len(onnx_rows)}")
        if [int(row["i"]) for row in onnx_rows] != list(range(64)):
            raise AssertionError("ONNX row indices are incomplete")
        verify_prediction_medians(onnx_rows)
        onnx_correlation = spearman(onnx_rows)
        if not math.isclose(
                onnx_correlation, onnx_expected["spearman"],
                rel_tol=0, abs_tol=1e-12):
            raise AssertionError("stored ONNX Spearman does not recompute")
        total_rows += len(onnx_rows)

    return {
        "status": "PASS",
        "bundle": path,
        "bundle_sha256": bundle_sha256,
        "environment": summary["environment"],
        "model": summary["model"],
        "raw_prediction_rows": total_rows,
        "raw_stochastic_draws": total_rows * 8,
        "all_predictions_equal_median_of_8_draws": True,
        "codenet": {
            "languages": len(language_correlations),
            "rows_per_language": 200,
            "average_spearman_recomputed": average,
            "claim_threshold": 0.5,
            "claim_pass": average > 0.5,
        },
        "onnx_nasbench101": {
            "rows": len(onnx_rows),
            "spearman_recomputed": onnx_correlation,
            "dataset_card_reference": 0.384,
        },
        "duplicate_input_audit": duplicate_audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    result = verify(args.bundle)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            handle.write(rendered)


if __name__ == "__main__":
    main()
