#!/usr/bin/env python3
"""Audit the exact Table 4 contract and all publicly released RLM artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from scipy import stats

from check_claim4_audit import check


ROOT = Path(__file__).resolve().parents[2]
AUTHOR = "akhauriyash"
MODEL_PREFIX = f"{AUTHOR}/RLM-GemmaS-Code-"
EXPECTED_MODELS = {
    "Amoeba": ("akhauriyash/RLM-GemmaS-Code-Amoeba-v0",
               "2998b2c8cef6cbb899536e7451561ccfc9e5433f"),
    "DARTS": ("akhauriyash/RLM-GemmaS-Code-DARTS-v0",
              "7aedf5c3586c98bbadce5f09a97ce91a644d25cc"),
    "PNAS": ("akhauriyash/RLM-GemmaS-Code-PNAS-v0",
             "57c23247f6bbdc181fc29e5fd4c43a2a18e1fb23"),
}
PAPER_RLM = {
    "NASNet": 0.382,
    "Amoeba": 0.488,
    "PNAS": 0.427,
    "ENAS": 0.481,
    "DARTS": 0.528,
}
PAPER_AVERAGES = {
    "MLP": 0.052,
    "Arch2Vec": 0.212,
    "CATE": 0.238,
    "GNN": 0.429,
    "FLAN": 0.459,
    "RLM": 0.461,
}
USER_AGENT = "OpenResearch-Reproduction/1.0 paper-2509.26476"


def manual_kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """O(n^2) tau-b implementation independent of scipy.stats.kendalltau."""
    n = len(x)
    concordant = 0
    discordant = 0
    for left in range(n - 1):
        products = np.sign(x[left] - x[left + 1:]) * np.sign(y[left] - y[left + 1:])
        concordant += int(np.count_nonzero(products > 0))
        discordant += int(np.count_nonzero(products < 0))
    total_pairs = n * (n - 1) // 2
    tied_x = sum(count * (count - 1) // 2 for count in Counter(x.tolist()).values())
    tied_y = sum(count * (count - 1) // 2 for count in Counter(y.tolist()).values())
    denominator = math.sqrt((total_pairs - tied_x) * (total_pairs - tied_y))
    if denominator == 0:
        return math.nan
    return (concordant - discordant) / denominator


def retained_metric_reanalysis() -> list[dict]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    path = ROOT / "outputs/claim1_accuracy/full_n512.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["space"] in {"ENAS", "NASNet"}:
                grouped[row["space"]].append(
                    (float(row["target"]), float(row["prediction"]))
                )
    rng = np.random.default_rng(20260926476)
    output = []
    for space in ("ENAS", "NASNet"):
        pairs = grouped[space]
        if len(pairs) != 512:
            raise AssertionError(f"{space}: expected 512 retained rows")
        targets, predictions = map(np.asarray, zip(*pairs))
        scipy_tau = float(stats.kendalltau(targets, predictions, variant="b").statistic)
        manual_tau = float(manual_kendall_tau_b(targets, predictions))
        shuffled = []
        for _ in range(200):
            shuffled.append(float(stats.kendalltau(
                rng.permutation(targets), predictions, variant="b"
            ).statistic))
        output.append({
            "space": space,
            "checkpoint_role": "released unified base; not a substitute for the missing target-specific checkpoint",
            "n": len(pairs),
            "scipy_kendall_tau_b": scipy_tau,
            "manual_kendall_tau_b": manual_tau,
            "source_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shuffled_target_control": {
                "seed": 20260926476,
                "permutations": len(shuffled),
                "mean_tau_b": float(np.mean(shuffled)),
                "max_tau_b": float(np.max(shuffled)),
                "threshold": PAPER_AVERAGES["GNN"],
                "passes_gnn_threshold": bool(np.mean(shuffled) >= PAPER_AVERAGES["GNN"]),
            },
        })
    return output


def public_artifact_audit() -> dict:
    response = requests.get(
        "https://huggingface.co/api/models",
        params={"author": AUTHOR, "limit": 100, "full": "true"},
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    author_models = response.json()
    visible = {model["id"]: model for model in author_models}
    details = {}
    training_row_manifests = []
    for space, (model_id, expected_revision) in EXPECTED_MODELS.items():
        if model_id not in visible:
            raise AssertionError(f"expected public checkpoint missing: {model_id}")
        detail_response = requests.get(
            f"https://huggingface.co/api/models/{model_id}",
            params={"blobs": "true"},
            headers={"User-Agent": USER_AGENT},
            timeout=120,
        )
        detail_response.raise_for_status()
        detail = detail_response.json()
        if detail["sha"] != expected_revision:
            raise AssertionError(f"{model_id}: revision changed")
        siblings = detail["siblings"]
        filenames = sorted(item["rfilename"] for item in siblings)
        manifests = [
            name for name in filenames
            if any(fragment in name.lower()
                   for fragment in ("train", "split", "manifest", "seed", "trainer"))
        ]
        training_row_manifests.extend(f"{model_id}:{name}" for name in manifests)
        weight = next(item for item in siblings if item["rfilename"] == "model.safetensors")
        details[space] = {
            "model_id": model_id,
            "revision": detail["sha"],
            "model_safetensors_size": weight.get("size"),
            "model_safetensors_blob_id": weight.get("blobId"),
            "files": filenames,
            "training_row_manifest_files": manifests,
        }
    public_target_spaces = {
        space for space in ("NASNet", "Amoeba", "PNAS", "ENAS", "DARTS")
        if f"{MODEL_PREFIX}{space}-v0" in visible
    }
    return {
        "api_url": "https://huggingface.co/api/models?author=akhauriyash&limit=100&full=true",
        "user_agent": USER_AGENT,
        "author_listing_sha256": hashlib.sha256(response.content).hexdigest(),
        "released_target_specific": sorted(public_target_spaces),
        "missing_target_specific": sorted(
            {"NASNet", "Amoeba", "PNAS", "ENAS", "DARTS"} - public_target_spaces
        ),
        "training_row_manifests_found": training_row_manifests,
        "details": details,
    }


def verify() -> dict:
    result = {
        "status": "PASS",
        "claim_verdict": "BLOCKED",
        "exact_claim_tested": (
            "Table 4 Kendall tau-b over NASNet, Amoeba, PNAS, ENAS, and DARTS, "
            "with RLM average 0.461 compared with Arch2Vec 0.212, CATE 0.238, "
            "GNN 0.429, and FLAN 0.459"
        ),
        "paper_table": {
            "per_space_rlm": PAPER_RLM,
            "reported_average": PAPER_AVERAGES,
            "rlm_average_unrounded": float(np.mean(list(PAPER_RLM.values()))),
        },
        "protocol": {
            "metric": "Kendall tau-b",
            "source_training_examples": 1024,
            "target_examples": {
                "NASNet": 16,
                "Amoeba": 16,
                "PNAS": 16,
                "ENAS": 16,
                "DARTS": 100,
            },
            "evaluation_domain": "remaining target search-space architectures",
            "flan_trials": 3,
        },
        "public_artifacts": public_artifact_audit(),
        "retained_base_checkpoint_metric_reanalysis": retained_metric_reanalysis(),
        "blockers": [
            "No public ENAS target-specific RLM checkpoint was discoverable.",
            "No public NASNet target-specific RLM checkpoint was discoverable.",
            "The three public target-specific checkpoint repositories expose no training-row, evaluation-row, seed, or split manifest.",
            "The public RLM repository does not contain a Table 4 training/evaluation script or committed configuration.",
        ],
        "limitations": [
            "The retained ENAS and NASNet rows only correct the prior metric mismatch; their unified base checkpoint is not the missing few-shot model.",
            "Checkpoint availability is a public-artifact audit, not proof that a private checkpoint does not exist.",
            "No Table 4 points are forecast from this audit alone.",
        ],
    }
    result["independent_checker"] = check(result)
    return result


def main() -> None:
    result = verify()
    print("CLAIM4_AUDIT_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
