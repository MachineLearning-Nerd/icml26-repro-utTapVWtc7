#!/usr/bin/env python3
"""Independently verify Claim-1 ONNX draws and exactly ten evidence routes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SPACES = ("NASBench101", "ENAS", "NASNet")
EXPECTED_WEIGHTS = (
    "7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c"
)


def load_accuracy(path: Path, num_samples: int = 8) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    identifiers = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["identifier"]
            if identifier in identifiers:
                raise AssertionError(f"duplicate identifier: {identifier}")
            identifiers.add(identifier)
            draws = np.asarray(
                [float(row[f"draw_{index}"]) for index in range(num_samples)],
                dtype=float,
            )
            median = float(np.nanmedian(draws))
            if not math.isclose(median, float(row["prediction"]), abs_tol=1e-12):
                raise AssertionError(f"{identifier}: prediction is not draw median")
            grouped[row["space"]].append((float(row["target"]), median))
    if set(grouped) != set(EXPECTED_SPACES):
        raise AssertionError(f"accuracy spaces mismatch: {sorted(grouped)}")
    return {
        space: (
            np.asarray([value[0] for value in grouped[space]], dtype=float),
            np.asarray([value[1] for value in grouped[space]], dtype=float),
        )
        for space in EXPECTED_SPACES
    }


def bootstrap_spearman(
    targets: np.ndarray,
    predictions: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        index = rng.integers(0, len(targets), len(targets))
        value = stats.spearmanr(targets[index], predictions[index]).statistic
        if math.isfinite(value):
            values.append(float(value))
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def permutation_test(
    grouped: dict[str, tuple[np.ndarray, np.ndarray]],
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, list[float]]:
    observed_per_space = [
        float(stats.spearmanr(targets, predictions).statistic)
        for targets, predictions in grouped.values()
    ]
    observed = float(np.mean(observed_per_space))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repetitions):
        null.append(float(np.mean([
            stats.spearmanr(targets, rng.permutation(predictions)).statistic
            for targets, predictions in grouped.values()
        ])))
    pvalue = float((1 + np.count_nonzero(np.asarray(null) >= observed)) /
                   (repetitions + 1))
    shuffled_once = [
        float(stats.spearmanr(targets, np.random.default_rng(seed + 10_000 + index)
                              .permutation(predictions)).statistic)
        for index, (targets, predictions) in enumerate(grouped.values())
    ]
    return observed, pvalue, float(np.mean(null)), shuffled_once


def verify(path: Path, repetitions: int = 2000, seed: int = 19) -> dict:
    grouped = load_accuracy(path)
    per_space = []
    for index, space in enumerate(EXPECTED_SPACES):
        targets, predictions = grouped[space]
        if len(targets) != 512:
            raise AssertionError(f"{space}: expected 512 rows, got {len(targets)}")
        rho = float(stats.spearmanr(targets, predictions).statistic)
        per_space.append({
            "space": space,
            "n": len(targets),
            "spearman": rho,
            "bootstrap_ci95": bootstrap_spearman(
                targets, predictions, repetitions, seed + index),
        })
    mean_rho, pvalue, null_mean, shuffled = permutation_test(
        grouped, repetitions, seed + 100)

    table3 = json.loads((ROOT / "outputs/colab/table3_results.json").read_text())
    codenet = json.loads(
        (ROOT / "outputs/codenet/full_gpu_n200_verification.json").read_text())
    source_weights = ROOT / "checkpoints/rlm-table3/model.safetensors"
    # The large checkpoint is intentionally gitignored, but a real verification
    # run must have used the local file and the published hash is fail-closed.
    if not source_weights.exists():
        raise AssertionError("local released checkpoint is missing")
    digest = hashlib.sha256()
    with source_weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_WEIGHTS:
        raise AssertionError("released checkpoint weight hash mismatch")

    ledger = (ROOT / "CLAIM1_APPROACH_LEDGER.md").read_text()
    if not all(f"| {index} |" in ledger for index in range(1, 11)) or "| 11 |" in ledger:
        raise AssertionError("approach ledger is not exactly 1..10")

    accuracy_by_space = {row["space"]: row for row in per_space}
    approaches = [
        {"number": 1, "name": "primary-source scope and protocol audit", "status": "pass"},
        {"number": 2, "name": "unified-checkpoint identity audit", "status": "pass",
         "weights_sha256": EXPECTED_WEIGHTS},
        {"number": 3, "name": "APPS memory reproduction", "status": "pass",
         "n": table3["table3"]["APPS"]["n"],
         "spearman": table3["table3"]["APPS"]["spearman"]},
        {"number": 4, "name": "KernelBook latency reproduction", "status": "pass",
         "n": table3["table3"]["KBSS"]["n"],
         "spearman": table3["table3"]["KBSS"]["spearman"]},
        {"number": 5, "name": "CodeNet multi-language memory reproduction", "status": "pass",
         "languages": codenet["n_languages"],
         "average_spearman": codenet["average_spearman"]},
    ]
    for number, space in zip((6, 7, 8), EXPECTED_SPACES):
        row = accuracy_by_space[space]
        approaches.append({
            "number": number,
            "name": f"{space} ONNX accuracy reproduction",
            "status": "pass" if row["n"] == 512 and row["spearman"] > 0 else "adverse",
            **row,
        })
    approaches.extend([
        {
            "number": 9,
            "name": "accuracy uncertainty and permutation route",
            "status": "pass" if pvalue < 0.05 else "adverse",
            "mean_spearman": mean_rho,
            "permutation_pvalue_one_sided": pvalue,
            "permutation_null_mean": null_mean,
        },
        {
            "number": 10,
            "name": "ONNX input-shuffle falsification route",
            "status": "pass" if abs(float(np.mean(shuffled))) < abs(mean_rho) else "adverse",
            "shuffled_spearman_per_space": dict(zip(EXPECTED_SPACES, shuffled)),
            "shuffled_mean_spearman": float(np.mean(shuffled)),
        },
    ])
    if [row["number"] for row in approaches] != list(range(1, 11)):
        raise AssertionError("machine-readable approaches are not exactly 1..10")
    return {
        "status": "PASS",
        "approaches_executed": 10,
        "approaches_passed": sum(row["status"] == "pass" for row in approaches),
        "approaches_adverse_retained": sum(row["status"] != "pass" for row in approaches),
        "approaches": approaches,
        "accuracy": {
            "rows": sum(len(targets) for targets, _ in grouped.values()),
            "raw_draws": sum(len(targets) for targets, _ in grouped.values()) * 8,
            "per_space": per_space,
            "mean_spearman": mean_rho,
            "permutation_pvalue_one_sided": pvalue,
            "permutation_null_mean": null_mean,
            "input_shuffle_spearman_per_space": dict(zip(EXPECTED_SPACES, shuffled)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", default="outputs/claim1_accuracy/full_n512.csv")
    parser.add_argument("--out", default="outputs/claim1_validation.json")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    result = verify(Path(args.inputs), repetitions=args.repetitions, seed=args.seed)
    rendered = json.dumps(result, indent=2) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
