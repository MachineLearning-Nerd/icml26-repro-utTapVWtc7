# Claim 1 — unified memory, latency, and accuracy model


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_claim1_exact10_final", "created_at": "2026-07-18T23:08:23+00:00", "title": "Claim 1 VERIFIED: three paper-scale ONNX accuracy spaces"}
-->
**Outcome: VERIFIED.** The same released checkpoint directly predicts:

- memory: APPS n=512, ρ=0.926807;
- latency: KernelBook/KBSS n=512, ρ=0.535279;
- memory across 17 CodeNet languages: n=200/language, primary mean ρ=0.529850 and independent local mean ρ=0.523403; and
- trained-network validation accuracy from ONNX in all three author-card spaces below.

| Accuracy space | Rows | Spearman ρ | Author-card ρ | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| NASBench101 | 512 | 0.406599 | 0.384 | [0.331937, 0.476317] |
| ENAS | 512 | 0.249461 | 0.211 | [0.159663, 0.329150] |
| NASNet | 512 | 0.206738 | 0.209 | [0.122055, 0.285412] |

The accuracy experiment contains **1,536 rows and 12,288 raw stochastic draws** (eight per row, median aggregation), with no reduced-scale substitution. The mean three-space Spearman is 0.287599; its one-sided label-permutation p-value is 0.0005.

The paper and author GraphArch card define this accuracy target as held-out neural-network `val_accuracy` predicted from serialized ONNX. It is not source-program pass/fail accuracy. The earlier n=64 NASBench101-only cell and the earlier claim that accuracy data were unavailable are superseded by this full released GraphArch evaluation.

## Exactly 10 evidence approaches

Exactly 10 approaches were executed for deficient Claim 1—no more and no fewer. Protocol/hash/metric checks inside a listed route are subchecks, not extra approaches.

| # | Approach | Status | Decisive result |
|---:|---|---|---|
| 1 | primary-source scope and protocol audit | PASS | Paper/card audit identifies trained-network `val_accuracy` from ONNX. |
| 2 | unified-checkpoint identity audit | PASS | Released aliases share weight SHA-256 `7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c`. |
| 3 | APPS memory reproduction | PASS | n=512; Spearman ρ=0.926807. |
| 4 | KernelBook latency reproduction | PASS | n=512; Spearman ρ=0.535279. |
| 5 | CodeNet multi-language memory reproduction | PASS | 17 languages; mean per-language Spearman ρ=0.523403. |
| 6 | NASBench101 ONNX accuracy reproduction | PASS | n=512; Spearman ρ=0.406599; bootstrap 95% CI [0.331937, 0.476317]. |
| 7 | ENAS ONNX accuracy reproduction | PASS | n=512; Spearman ρ=0.249461; bootstrap 95% CI [0.159663, 0.329150]. |
| 8 | NASNet ONNX accuracy reproduction | PASS | n=512; Spearman ρ=0.206738; bootstrap 95% CI [0.122055, 0.285412]. |
| 9 | accuracy uncertainty and permutation route | PASS | Mean ρ=0.287599; one-sided permutation p=0.0005. |
| 10 | ONNX input-shuffle falsification route | PASS | Input-identity shuffle: NASBench101=0.043387, ENAS=0.015207, NASNet=-0.098314; mean=-0.013240. |

Primary sources: [paper](https://arxiv.org/abs/2509.26476), [released model](https://huggingface.co/akhauriyash/RLM-GemmaS-Code-v0), [released GraphArch data](https://huggingface.co/datasets/akhauriyash/GraphArch-Regression), and [reproduction repository](https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7). Row-level evidence: [full CSV](https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7/blob/master/outputs/claim1_accuracy/full_n512.csv) and [independent validation JSON](https://github.com/MachineLearning-Nerd/icml26-repro-utTapVWtc7/blob/master/outputs/claim1_validation.json).


---
<!-- trackio-cell
{"type": "code", "id": "cell_b6c773dcd008", "created_at": "2026-07-18T23:08:46+00:00", "title": "Independent exactly-10 Claim-1 verifier (1,536 rows / 12,288 draws)", "command": ["env", "PYTHONPATH=repro/src", ".venv/bin/python", "repro/src/verify_claim1_accuracy.py", "--inputs", "outputs/claim1_accuracy/full_n512.csv", "--out", "outputs/claim1_validation.json"], "exit_code": 0, "duration_s": 9.957}
-->
````bash
$ env PYTHONPATH=repro/src .venv/bin/python repro/src/verify_claim1_accuracy.py --inputs outputs/claim1_accuracy/full_n512.csv --out outputs/claim1_validation.json
````

exit 0 · 10.0s


````python title=verify_claim1_accuracy.py
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
    source_audit = json.loads((ROOT / "outputs/claim1_source_audit.json").read_text())
    if source_audit["status"] != "PASS" or not source_audit["dataset"]["local_matches_hub"]:
        raise AssertionError("source provenance audit did not pass")
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
        {"number": 1, "name": "primary-source scope and protocol audit", "status": "pass",
         "paper_accuracy_target": source_audit["scope"]["paper_accuracy_target"]},
        {"number": 2, "name": "unified-checkpoint identity audit", "status": "pass",
         "weights_sha256": EXPECTED_WEIGHTS,
         "critical_shared_files": len(source_audit["model_aliases"]["critical_shared_files"])},
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

````


````json title=claim1_validation.json
{
  "status": "PASS",
  "approaches_executed": 10,
  "approaches_passed": 10,
  "approaches_adverse_retained": 0,
  "approaches": [
    {
      "number": 1,
      "name": "primary-source scope and protocol audit",
      "status": "pass",
      "paper_accuracy_target": "trained-neural-network validation accuracy from ONNX"
    },
    {
      "number": 2,
      "name": "unified-checkpoint identity audit",
      "status": "pass",
      "weights_sha256": "7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c",
      "critical_shared_files": 8
    },
    {
      "number": 3,
      "name": "APPS memory reproduction",
      "status": "pass",
      "n": 512,
      "spearman": 0.9268067718469594
    },
    {
      "number": 4,
      "name": "KernelBook latency reproduction",
      "status": "pass",
      "n": 512,
      "spearman": 0.5352789637599933
    },
    {
      "number": 5,
      "name": "CodeNet multi-language memory reproduction",
      "status": "pass",
      "languages": 17,
      "average_spearman": 0.5234034026121069
    },
    {
      "number": 6,
      "name": "NASBench101 ONNX accuracy reproduction",
      "status": "pass",
      "space": "NASBench101",
      "n": 512,
      "spearman": 0.4065993610622034,
      "bootstrap_ci95": [
        0.3319366099221084,
        0.4763170539865802
      ]
    },
    {
      "number": 7,
      "name": "ENAS ONNX accuracy reproduction",
      "status": "pass",
      "space": "ENAS",
      "n": 512,
      "spearman": 0.24946117751369043,
      "bootstrap_ci95": [
        0.15966279605884995,
        0.3291495921338195
      ]
    },
    {
      "number": 8,
      "name": "NASNet ONNX accuracy reproduction",
      "status": "pass",
      "space": "NASNet",
      "n": 512,
      "spearman": 0.20673752343866375,
      "bootstrap_ci95": [
        0.12205549844390134,
        0.28541191432251395
      ]
    },
    {
      "number": 9,
      "name": "accuracy uncertainty and permutation route",
      "status": "pass",
      "mean_spearman": 0.28759935400485254,
      "permutation_pvalue_one_sided": 0.0004997501249375312,
      "permutation_null_mean": 0.00016907574423914896
    },
    {
      "number": 10,
      "name": "ONNX input-shuffle falsification route",
      "status": "pass",
      "shuffled_spearman_per_space": {
        "NASBench101": 0.043386530390742174,
        "ENAS": 0.0152066140383861,
        "NASNet": -0.0983142578997609
      },
      "shuffled_mean_spearman": -0.013240371156877545
    }
  ],
  "accuracy": {
    "rows": 1536,
    "raw_draws": 12288,
    "per_space": [
      {
        "space": "NASBench101",
        "n": 512,
        "spearman": 0.4065993610622034,
        "bootstrap_ci95": [
          0.3319366099221084,
          0.4763170539865802
        ]
      },
      {
        "space": "ENAS",
        "n": 512,
        "spearman": 0.24946117751369043,
        "bootstrap_ci95": [
          0.15966279605884995,
          0.3291495921338195
        ]
      },
      {
        "space": "NASNet",
        "n": 512,
        "spearman": 0.20673752343866375,
        "bootstrap_ci95": [
          0.12205549844390134,
          0.28541191432251395
        ]
      }
    ],
    "mean_spearman": 0.28759935400485254,
    "permutation_pvalue_one_sided": 0.0004997501249375312,
    "permutation_null_mean": 0.00016907574423914896,
    "input_shuffle_spearman_per_space": {
      "NASBench101": 0.043386530390742174,
      "ENAS": 0.0152066140383861,
      "NASNet": -0.0983142578997609
    }
  }
}

````


````output
{
  "status": "PASS",
  "approaches_executed": 10,
  "approaches_passed": 10,
  "approaches_adverse_retained": 0,
  "approaches": [
    {
      "number": 1,
      "name": "primary-source scope and protocol audit",
      "status": "pass",
      "paper_accuracy_target": "trained-neural-network validation accuracy from ONNX"
    },
    {
      "number": 2,
      "name": "unified-checkpoint identity audit",
      "status": "pass",
      "weights_sha256": "7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c",
      "critical_shared_files": 8
    },
    {
      "number": 3,
      "name": "APPS memory reproduction",
      "status": "pass",
      "n": 512,
      "spearman": 0.9268067718469594
    },
    {
      "number": 4,
      "name": "KernelBook latency reproduction",
      "status": "pass",
      "n": 512,
      "spearman": 0.5352789637599933
    },
    {
      "number": 5,
      "name": "CodeNet multi-language memory reproduction",
      "status": "pass",
      "languages": 17,
      "average_spearman": 0.5234034026121069
    },
    {
      "number": 6,
      "name": "NASBench101 ONNX accuracy reproduction",
      "status": "pass",
      "space": "NASBench101",
      "n": 512,
      "spearman": 0.4065993610622034,
      "bootstrap_ci95": [
        0.3319366099221084,
        0.4763170539865802
      ]
    },
    {
      "number": 7,
      "name": "ENAS ONNX accuracy reproduction",
      "status": "pass",
      "space": "ENAS",
      "n": 512,
      "spearman": 0.24946117751369043,
      "bootstrap_ci95": [
        0.15966279605884995,
        0.3291495921338195
      ]
    },
    {
      "number": 8,
      "name": "NASNet ONNX accuracy reproduction",
      "status": "pass",
      "space": "NASNet",
      "n": 512,
      "spearman": 0.20673752343866375,
      "bootstrap_ci95": [
        0.12205549844390134,
        0.28541191432251395
      ]
    },
    {
      "number": 9,
      "name": "accuracy uncertainty and permutation route",
      "status": "pass",
      "mean_spearman": 0.28759935400485254,
      "permutation_pvalue_one_sided": 0.0004997501249375312,
      "permutation_null_mean": 0.00016907574423914896
    },
    {
      "number": 10,
      "name": "ONNX input-shuffle falsification route",
      "status": "pass",
      "shuffled_spearman_per_space": {
        "NASBench101": 0.043386530390742174,
        "ENAS": 0.0152066140383861,
        "NASNet": -0.0983142578997609
      },
      "shuffled_mean_spearman": -0.013240371156877545
    }
  ],
  "accuracy": {
    "rows": 1536,
    "raw_draws": 12288,
    "per_space": [
      {
        "space": "NASBench101",
        "n": 512,
        "spearman": 0.4065993610622034,
        "bootstrap_ci95": [
          0.3319366099221084,
          0.4763170539865802
        ]
      },
      {
        "space": "ENAS",
        "n": 512,
        "spearman": 0.24946117751369043,
        "bootstrap_ci95": [
          0.15966279605884995,
          0.3291495921338195
        ]
      },
      {
        "space": "NASNet",
        "n": 512,
        "spearman": 0.20673752343866375,
        "bootstrap_ci95": [
          0.12205549844390134,
          0.28541191432251395
        ]
      }
    ],
    "mean_spearman": 0.28759935400485254,
    "permutation_pvalue_one_sided": 0.0004997501249375312,
    "permutation_null_mean": 0.00016907574423914896,
    "input_shuffle_spearman_per_space": {
      "NASBench101": 0.043386530390742174,
      "ENAS": 0.0152066140383861,
      "NASNet": -0.0983142578997609
    }
  }
}

````
