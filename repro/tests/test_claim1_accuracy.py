"""Fail-closed tests for the exactly-ten-route Claim-1 repair."""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))
import run_claim1_accuracy as runner  # noqa: E402
import verify_claim1_accuracy as verifier  # noqa: E402
import finalize_claim1_report as report  # noqa: E402


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
    assert len(runner.DATASET_PARQUET_SHA256) == 64
    int(runner.DATASET_PARQUET_SHA256, 16)


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


def test_report_finalizer_enforces_exactly_ten_and_one_pin(tmp_path):
    root = tmp_path
    pages = root / ".trackio/logbook/pages"
    for slug in (
        "claim-1-unified-multi-metric-model",
        "methods-environment",
        "conclusion",
        "claim-3-codenet-17-languages-0-5",
    ):
        path = pages / slug / "page.md"
        path.parent.mkdir(parents=True)
        path.write_text('<!-- trackio-cell\n{"pinned": true, "pinned_at": "old"}\n-->\n')
    (root / ".trackio/metadata.json").write_text(json.dumps({
        "tags": [], "private": True, "autosync": True,
        "local_path_artifacts": [{"abs_path": "/home/example"}],
    }))

    approaches = [
        {"number": 1, "name": "scope", "status": "pass",
         "paper_accuracy_target": "trained-network accuracy"},
        {"number": 2, "name": "identity", "status": "pass",
         "weights_sha256": "a" * 64, "critical_shared_files": 8},
        {"number": 3, "name": "memory", "status": "pass", "n": 512,
         "spearman": 0.92},
        {"number": 4, "name": "latency", "status": "pass", "n": 512,
         "spearman": 0.53},
        {"number": 5, "name": "languages", "status": "pass", "languages": 17,
         "average_spearman": 0.53},
    ]
    per_space = []
    for number, space, rho in zip((6, 7, 8), report.EXPECTED_SPACES, (0.40, 0.30, 0.20)):
        row = {"space": space, "n": 512, "spearman": rho,
               "bootstrap_ci95": [rho - 0.05, rho + 0.05]}
        per_space.append(row)
        approaches.append({"number": number, "name": f"{space} accuracy",
                           "status": "pass", **row})
    approaches.extend([
        {"number": 9, "name": "permutation", "status": "pass",
         "mean_spearman": 0.3, "permutation_pvalue_one_sided": 0.0005,
         "permutation_null_mean": 0.0},
        {"number": 10, "name": "input shuffle", "status": "pass",
         "shuffled_spearman_per_space": {space: 0.01 for space in report.EXPECTED_SPACES},
         "shuffled_mean_spearman": 0.01},
    ])
    outputs = root / "outputs"
    outputs.mkdir()
    (outputs / "claim1_validation.json").write_text(json.dumps({
        "status": "PASS", "approaches_executed": 10, "approaches": approaches,
        "accuracy": {"rows": 1536, "raw_draws": 12288, "per_space": per_space,
                     "mean_spearman": 0.3, "permutation_pvalue_one_sided": 0.0005},
    }))
    (outputs / "claim1_source_audit.json").write_text(json.dumps({
        "status": "PASS",
        "dataset": {"revision": "dataset-revision"},
        "model_aliases": {"local_weights_bytes": 725864700,
                          "local_weights_sha256": "a" * 64},
    }))

    report.finalize(root, "2026-07-18T00:00:00+00:00")
    page_text = "\n".join(path.read_text() for path in pages.glob("*/page.md"))
    assert page_text.count('"pinned": true') == 1
    assert "Exactly 10 evidence approaches" in page_text
    assert "12,288 raw stochastic draws" in page_text
    assert "Paper-scale evidence summary" in (pages / "index.md").read_text()
    assert "/home/" not in page_text
    metadata = json.loads((root / ".trackio/metadata.json").read_text())
    assert metadata["autosync"] is False
    assert "local_path_artifacts" not in metadata


def test_report_local_root_sanitizer_preserves_relative_suffix():
    root = pathlib.Path("/home/example/reproduction")
    text = "loading model from: /home/example/reproduction/checkpoints/model"
    assert report.remove_local_root(text, root) == "loading model from: checkpoints/model"
