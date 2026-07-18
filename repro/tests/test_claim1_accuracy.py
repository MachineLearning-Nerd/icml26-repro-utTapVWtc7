"""Fail-closed tests for the exactly-ten-route Claim-1 repair."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
import run_claim1_accuracy as runner  # noqa: E402
import verify_claim1_accuracy as verifier  # noqa: E402


def test_ledger_has_exactly_routes_one_through_ten():
    text = (ROOT / "CLAIM1_APPROACH_LEDGER.md").read_text()
    assert all(f"| {index} |" in text for index in range(1, 11))
    assert "| 11 |" not in text
    assert sum(f"| {index} |" in text for index in range(1, 11)) == 10


def test_input_digest_is_stable_and_sensitive():
    assert runner.input_digest("abc") == runner.input_digest("abc")
    assert runner.input_digest("abc") != runner.input_digest("abd")


def test_progress_roundtrip_requires_draw_count(tmp_path):
    path = tmp_path / "progress.jsonl"
    runner.append_progress(path, [{
        "identifier": "x",
        "draws": [1.0, 2.0],
        "prediction": 1.5,
    }])
    loaded = runner.load_progress(path, 2)
    assert loaded["x"]["prediction"] == 1.5
    try:
        runner.load_progress(path, 3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incompatible draw counts must fail closed")


def test_summary_recomputes_rank_metrics():
    rows = [
        {"target": float(value), "prediction": float(value + (value % 2) * 0.1)}
        for value in range(1, 21)
    ]
    result = runner.summarize(rows)
    assert result["n"] == 20
    assert np.isclose(result["spearman"], 1.0)
    assert result["nan_rate"] == 0.0


def test_declared_model_weight_hash_is_full_sha256():
    assert len(runner.MODEL_WEIGHTS_SHA256) == 64
    int(runner.MODEL_WEIGHTS_SHA256, 16)


def test_accuracy_loader_recomputes_medians_and_rejects_duplicates(tmp_path):
    path = tmp_path / "accuracy.csv"
    fields = ["space", "identifier", "target"]
    fields += [f"draw_{index}" for index in range(8)] + ["prediction"]
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for space_index, space in enumerate(verifier.EXPECTED_SPACES):
            for index in range(4):
                value = float(space_index * 10 + index)
                row = {"space": space, "identifier": f"{space}-{index}",
                       "target": value, "prediction": value}
                row.update({f"draw_{draw}": value for draw in range(8)})
                writer.writerow(row)
    grouped = verifier.load_accuracy(path)
    assert set(grouped) == set(verifier.EXPECTED_SPACES)
    assert all(len(values[0]) == 4 for values in grouped.values())


def test_permutation_route_detects_three_space_signal():
    rng = np.random.default_rng(7)
    grouped = {}
    for space in verifier.EXPECTED_SPACES:
        targets = rng.normal(size=80)
        grouped[space] = (targets, targets + rng.normal(scale=0.1, size=80))
    observed, pvalue, null_mean, shuffled = verifier.permutation_test(
        grouped, repetitions=100, seed=8)
    assert observed > 0.9
    assert pvalue < 0.05
    assert abs(null_mean) < 0.1
    assert len(shuffled) == 3
