#!/usr/bin/env python3
"""Fail-closed recheck of the preserved, previously accepted Claim 1–3 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HASHES = {
    "outputs/claim1_accuracy/full_n512.csv":
        "3016dfb245deb56b3d0d3d06f902471bc415f80196d61ecf4d49eb773f1edd76",
    "outputs/codenet/full_gpu_n200.csv":
        "7843f75c38b9ee0bf3f15d796f151808fc01a526c704f429049016d57538f390",
    "outputs/colab/regresslm_evidence_bundle.zip":
        "b1d33e20922194cab7dfd20526cac13680f1ca16ce51ad7836a358feee5ebd1f",
    "outputs/colab/table3_results.json":
        "173dbce239ca51269d3b231dd851b7fb299dffc82d2b0eede75f61ae6a835dc2",
}
EXPECTED_ACCURACY = {
    "NASBench101": 0.4065993610622034,
    "ENAS": 0.24946117751369043,
    "NASNet": 0.20673752343866375,
}
EXPECTED_CODENET_LANGUAGES = {
    "C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Haskell",
    "Kotlin", "JavaScript", "PHP", "D", "Scala", "OCaml", "Perl", "Fortran",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes() -> dict[str, str]:
    observed = {}
    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"missing preserved artifact: {relative}")
        observed[relative] = sha256(path)
        if observed[relative] != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")
    return observed


def check_accuracy() -> dict:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    identifiers = set()
    path = ROOT / "outputs/claim1_accuracy/full_n512.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = row["identifier"]
            if identifier in identifiers:
                raise AssertionError(f"duplicate accuracy identifier: {identifier}")
            identifiers.add(identifier)
            draws = np.asarray([float(row[f"draw_{i}"]) for i in range(8)])
            prediction = float(row["prediction"])
            if not math.isclose(
                prediction, float(np.nanmedian(draws)), abs_tol=1e-12
            ):
                raise AssertionError(f"stored prediction is not draw median: {identifier}")
            grouped[row["space"]].append((float(row["target"]), prediction))
    if set(grouped) != set(EXPECTED_ACCURACY):
        raise AssertionError(f"accuracy spaces differ: {sorted(grouped)}")
    rows = []
    for space, expected in EXPECTED_ACCURACY.items():
        pairs = grouped[space]
        if len(pairs) != 512:
            raise AssertionError(f"{space}: expected 512 rows, got {len(pairs)}")
        target, prediction = map(np.asarray, zip(*pairs))
        rho = float(stats.spearmanr(target, prediction).statistic)
        if not math.isclose(rho, expected, abs_tol=1e-12):
            raise AssertionError(f"{space}: Spearman changed: {rho}")
        rows.append({"space": space, "n": len(pairs), "spearman": rho})
    return {
        "rows": len(identifiers),
        "raw_draws": len(identifiers) * 8,
        "per_space": rows,
        "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
    }


def check_codenet() -> dict:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    path = ROOT / "outputs/codenet/full_gpu_n200.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["language"]].append(
                (float(row["y_true"]), float(row["y_pred"]))
            )
    if set(grouped) != EXPECTED_CODENET_LANGUAGES:
        raise AssertionError(f"CodeNet languages differ: {sorted(grouped)}")
    per_language = {}
    for language, pairs in grouped.items():
        if len(pairs) != 200:
            raise AssertionError(f"{language}: expected 200 rows, got {len(pairs)}")
        target, prediction = map(np.asarray, zip(*pairs))
        per_language[language] = float(
            stats.spearmanr(target, prediction).statistic
        )
    average = float(np.mean(list(per_language.values())))
    if not math.isclose(average, 0.5298501742814378, abs_tol=1e-12):
        raise AssertionError(f"CodeNet mean Spearman changed: {average}")
    return {
        "languages": len(grouped),
        "rows_per_language": 200,
        "average_spearman": average,
        "per_language_spearman": per_language,
    }


def check_table3_summary() -> dict:
    summary = json.loads(
        (ROOT / "outputs/colab/table3_results.json").read_text()
    )["table3"]
    apps = summary["APPS"]
    kbss = summary["KBSS"]
    if apps["n"] != 512 or not math.isclose(
        apps["spearman"], 0.9268067718469594, abs_tol=1e-15
    ):
        raise AssertionError("APPS preserved summary changed")
    if kbss["n"] != 512 or not math.isclose(
        kbss["spearman"], 0.5352789637599933, abs_tol=1e-15
    ):
        raise AssertionError("KBSS preserved summary changed")
    return {"APPS": apps, "KBSS": kbss}


def check_bundle() -> dict:
    path = ROOT / "outputs/colab/regresslm_evidence_bundle.zip"
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise AssertionError(f"corrupt evidence ZIP member: {bad_member}")
        summary = json.loads(archive.read("summary.json"))
    return {
        "members": 19,
        "model_commit": summary["model"]["checkpoint_commit"],
        "parameters": summary["model"]["parameters"],
        "transformers": summary["environment"]["transformers"],
        "seed": summary["environment"]["seed"],
    }


def verify() -> dict:
    return {
        "status": "PASS",
        "scope": (
            "independent recomputation from preserved row-level data for ONNX "
            "accuracy and CodeNet; integrity and summary recheck for APPS/KBSS"
        ),
        "artifact_sha256": check_hashes(),
        "claim_1_accuracy": check_accuracy(),
        "claim_2_table3": check_table3_summary(),
        "claim_3_codenet": check_codenet(),
        "bundle": check_bundle(),
        "limitations": [
            "This baseline rechecks preserved evidence; it does not rerun model inference.",
            "Full APPS/KBSS row-level draws were not present in the judged repository.",
        ],
    }


def main() -> None:
    result = verify()
    print("CUMULATIVE_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
