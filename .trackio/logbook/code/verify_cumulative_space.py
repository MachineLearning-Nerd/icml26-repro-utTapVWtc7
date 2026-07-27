#!/usr/bin/env python3
"""Standalone fail-closed checker for the Space's retained Claim 1–3 evidence.

This intentionally uses only Python's standard library and files reachable
from the Space entrypoint. It recomputes the row-level ONNX and independent
CodeNet correlations, and integrity-checks the retained APPS/KBSS summary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/cumulative"
EXPECTED_HASHES = {
    "full_n512.csv":
        "3016dfb245deb56b3d0d3d06f902471bc415f80196d61ecf4d49eb773f1edd76",
    "full_gpu_n200.csv":
        "7843f75c38b9ee0bf3f15d796f151808fc01a526c704f429049016d57538f390",
    "table3_results.json":
        "173dbce239ca51269d3b231dd851b7fb299dffc82d2b0eede75f61ae6a835dc2",
}
EXPECTED_ACCURACY = {
    "NASBench101": 0.4065993610622034,
    "ENAS": 0.24946117751369043,
    "NASNet": 0.20673752343866375,
}
EXPECTED_LANGUAGES = {
    "C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Haskell",
    "Kotlin", "JavaScript", "PHP", "D", "Scala", "OCaml", "Perl", "Fortran",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values: list[float]) -> list[float]:
    """Average ranks for ties, equivalent to scipy.stats.rankdata."""
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            output[index] = rank
        start = end
    return output


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in rx) * sum((y - my) ** 2 for y in ry)
    )
    if denominator == 0:
        raise AssertionError("undefined Spearman correlation")
    return numerator / denominator


def check_hashes() -> dict[str, str]:
    observed = {}
    for filename, expected in EXPECTED_HASHES.items():
        path = EVIDENCE / filename
        if not path.is_file():
            raise AssertionError(f"missing retained artifact: {filename}")
        observed[filename] = sha256(path)
        if observed[filename] != expected:
            raise AssertionError(f"SHA-256 mismatch: {filename}")
    return observed


def check_accuracy() -> dict:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    identifiers = set()
    with (EVIDENCE / "full_n512.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["identifier"]
            if identifier in identifiers:
                raise AssertionError(f"duplicate identifier: {identifier}")
            identifiers.add(identifier)
            draws = [float(row[f"draw_{index}"]) for index in range(8)]
            prediction = float(row["prediction"])
            if not math.isclose(
                prediction, statistics.median(draws), abs_tol=1e-12
            ):
                raise AssertionError(f"prediction is not median: {identifier}")
            grouped[row["space"]].append((float(row["target"]), prediction))
    if set(grouped) != set(EXPECTED_ACCURACY):
        raise AssertionError("accuracy spaces changed")
    observed = {}
    for space, expected in EXPECTED_ACCURACY.items():
        pairs = grouped[space]
        if len(pairs) != 512:
            raise AssertionError(f"{space}: expected 512 rows")
        xs, ys = map(list, zip(*pairs))
        rho = spearman(xs, ys)
        if not math.isclose(rho, expected, abs_tol=1e-12):
            raise AssertionError(f"{space}: Spearman changed: {rho}")
        observed[space] = rho
    return {"rows": len(identifiers), "raw_draws": len(identifiers) * 8,
            "spearman": observed}


def check_codenet() -> dict:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with (EVIDENCE / "full_gpu_n200.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["language"]].append(
                (float(row["y_true"]), float(row["y_pred"]))
            )
    if set(grouped) != EXPECTED_LANGUAGES:
        raise AssertionError("CodeNet language set changed")
    observed = {}
    for language, pairs in grouped.items():
        if len(pairs) != 200:
            raise AssertionError(f"{language}: expected 200 rows")
        xs, ys = map(list, zip(*pairs))
        observed[language] = spearman(xs, ys)
    average = statistics.fmean(observed.values())
    if not math.isclose(average, 0.5234034026121069, abs_tol=1e-12):
        raise AssertionError(f"CodeNet mean changed: {average}")
    return {"languages": len(observed), "rows_per_language": 200,
            "average_spearman": average, "per_language": observed}


def check_table3() -> dict:
    summary = json.loads((EVIDENCE / "table3_results.json").read_text())["table3"]
    apps, kbss = summary["APPS"], summary["KBSS"]
    if apps["n"] != 512 or not math.isclose(
        apps["spearman"], 0.9268067718469594, abs_tol=1e-15
    ):
        raise AssertionError("APPS retained summary changed")
    if kbss["n"] != 512 or not math.isclose(
        kbss["spearman"], 0.5352789637599933, abs_tol=1e-15
    ):
        raise AssertionError("KBSS retained summary changed")
    return {"APPS": apps, "KBSS": kbss}


def verify() -> dict:
    return {
        "status": "PASS",
        "scope": (
            "row-level recomputation for ONNX accuracy and independent CodeNet; "
            "hash and summary integrity for APPS/KBSS"
        ),
        "sha256": check_hashes(),
        "claim_1": check_accuracy(),
        "claim_2": check_table3(),
        "claim_3": check_codenet(),
        "limitation": (
            "The judged repository did not retain paper-scale APPS/KBSS rows; "
            "this checker cannot independently recompute those two correlations."
        ),
    }


if __name__ == "__main__":
    print("SPACE_CUMULATIVE_RESULT " + json.dumps(verify(), sort_keys=True))
