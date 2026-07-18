#!/usr/bin/env python3
"""Machine-readable provenance audit for Claim-1 routes 1 and 2."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import requests

from run_claim1_accuracy import DATASET_PARQUET_SHA256, MODEL_WEIGHTS_SHA256


ROOT = Path(__file__).resolve().parents[2]
MODEL_IDS = (
    "akhauriyash/RLM-GemmaS-Code-v0",
    "akhauriyash/RegressLM-gemma-s-RLM-table3",
)
CRITICAL_SHARED_FILES = (
    "model.safetensors",
    "modeling_regresslm.py",
    "tokenization_p10.py",
    "tokenizer_config.json",
    "generation_config.json",
    "ieee_vocab.json",
    "encoder_tokenizer/tokenizer.json",
    "encoder_tokenizer/tokenizer.model",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sibling_map(payload: dict) -> dict[str, dict]:
    return {item["rfilename"]: item for item in payload["siblings"]}


def blob_identity(item: dict) -> str:
    return (item.get("lfs") or {}).get("sha256") or item.get("blobId", "")


def fetch_api(kind: str, repo_id: str) -> dict:
    response = requests.get(
        f"https://huggingface.co/api/{kind}/{repo_id}",
        params={"blobs": "true"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    model_payloads = {repo_id: fetch_api("models", repo_id) for repo_id in MODEL_IDS}
    model_files = {repo_id: sibling_map(payload) for repo_id, payload in model_payloads.items()}
    identity = {}
    for filename in CRITICAL_SHARED_FILES:
        values = {
            repo_id: blob_identity(model_files[repo_id][filename])
            for repo_id in MODEL_IDS
        }
        identity[filename] = {
            "identities": values,
            "identical": len(set(values.values())) == 1,
        }
    if not all(value["identical"] for value in identity.values()):
        raise SystemExit("CLAIM1_SOURCE_AUDIT_FAIL: model aliases differ on inference files")

    weights = ROOT / "checkpoints/rlm-table3/model.safetensors"
    parquet = ROOT / ".trackio/cache/grapharch_full/data.parquet"
    local_weights_sha = sha256(weights)
    local_parquet_sha = sha256(parquet)
    if local_weights_sha != MODEL_WEIGHTS_SHA256:
        raise SystemExit("CLAIM1_SOURCE_AUDIT_FAIL: checkpoint hash mismatch")
    if local_parquet_sha != DATASET_PARQUET_SHA256:
        raise SystemExit("CLAIM1_SOURCE_AUDIT_FAIL: parquet hash mismatch")

    dataset = fetch_api("datasets", "akhauriyash/GraphArch-Regression")
    dataset_file = sibling_map(dataset)["data.parquet"]
    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "status": "PASS",
        "scope": {
            "paper_accuracy_target": "trained-neural-network validation accuracy from ONNX",
            "ordinary_program_accuracy_target": False,
            "source_audit": "CLAIM1_SOURCE_AUDIT.md",
        },
        "author_source": {
            "repo": "google-deepmind/regress-lm",
            "commit": upstream_commit,
            "expected_prefix": "6c23ccb",
            "matches_expected": upstream_commit.startswith("6c23ccb"),
        },
        "model_aliases": {
            "ids": list(MODEL_IDS),
            "revisions": {repo_id: model_payloads[repo_id]["sha"] for repo_id in MODEL_IDS},
            "critical_shared_files": identity,
            "local_weights_bytes": weights.stat().st_size,
            "local_weights_sha256": local_weights_sha,
        },
        "dataset": {
            "id": "akhauriyash/GraphArch-Regression",
            "revision": dataset["sha"],
            "local_parquet_bytes": parquet.stat().st_size,
            "local_parquet_sha256": local_parquet_sha,
            "hub_lfs_sha256": dataset_file["lfs"]["sha256"],
            "hub_lfs_bytes": dataset_file["lfs"]["size"],
            "local_matches_hub": (
                local_parquet_sha == dataset_file["lfs"]["sha256"]
                and parquet.stat().st_size == dataset_file["lfs"]["size"]
            ),
        },
    }
    if not result["author_source"]["matches_expected"] or not result["dataset"]["local_matches_hub"]:
        raise SystemExit("CLAIM1_SOURCE_AUDIT_FAIL: source revision mismatch")
    out = ROOT / "outputs/claim1_source_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
