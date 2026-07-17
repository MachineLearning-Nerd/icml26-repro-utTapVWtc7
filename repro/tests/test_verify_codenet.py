"""Tests for language-stratified Claim 3 verification."""
import csv
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import verify_codenet as verifier  # noqa: E402


def test_mean_is_per_language_not_pooled(tmp_path):
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "y_true", "y_pred"])
        for language, offset in [("A", 0), ("B", 1000)]:
            for value in range(20):
                prediction = value if language == "A" else -value
                writer.writerow([language, offset + value, prediction])
    grouped = verifier.load_by_language(path)
    average, per_language = verifier.mean_language_spearman(grouped)
    assert per_language == {"A": 1.0, "B": -1.0}
    assert np.isclose(average, 0.0)


def test_controls_detect_strong_stratified_signal(tmp_path):
    rng = np.random.default_rng(4)
    path = tmp_path / "raw.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "y_true", "y_pred"])
        for language in ["A", "B", "C"]:
            targets = rng.normal(size=80)
            predictions = targets + rng.normal(scale=0.1, size=80)
            writer.writerows(zip([language] * 80, targets, predictions))
    result = verifier.verify(path, repetitions=100, seed=9)
    assert result["average_spearman"] > 0.9
    assert result["bootstrap_ci95"][0] > 0.85
    assert result["permutation_pvalue_one_sided"] < 0.05
    assert abs(result["permutation_null_mean"]) < 0.1
