#!/usr/bin/env python3
"""Independent public-artifact audit for the exact Table 5 and Table 6 claims."""
from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / ".openresearch/artifacts/claim5_ablation_scaling"
ROUTES = ARTIFACT_DIR / "four_routes.json"
EXACT_GATE = ROOT / "repro/src/claim5_exact_gate.py"
USER_AGENT = "OpenResearch-Reproduction/1.0 paper-2509.26476"
REGRESS_REPO = "google-deepmind/regress-lm"
REGRESS_REVISIONS = {
    "paper_time": "b36c45898c88f96bd9e3bd73a7d17c89f9d73f0b",
    "current_audited": "6c23ccb51ae9862d98af9faffc1286ca2149b12e",
}
QIN_REPO = "shiwenqin/transferrable-surrogates"
QIN_REVISION = "d087a70ec483fb8cee7536c99a0f9c363609eb05"
QIN_PATH = "lm_tuning/src/fine_tuning/bert_loo_tuning.py"
QIN_SHA256 = "e5b2e3d31177fe2fa9572cbf99bcf87ff06e5786d2f029a251734f7d7e7558f3"
BASE_MODELS = {
    "google/t5gemma-s-s-prefixlm": {
        "revision": "7b875178ee4c8a97a5df548972b10cfe200ee695",
        "parameters": 312_517_632,
    },
    "google/t5gemma-b-b-prefixlm": {
        "revision": "df5e1422c2a53948a57901c54bfcf397c92bfa1d",
        "parameters": 591_490_560,
    },
}
TABLE5 = {
    "standard_regression_head": 0.478,
    "normalized_regression_head": 0.717,
    "decoder_head": 0.800,
}
TABLE6 = {"t5gemma_s_s_300m": 0.744, "t5gemma_b_b_600m": 0.782}
EXACT_MARKERS = (
    "standard regression head",
    "normalized regression head",
    "automodelforsequenceclassification",
    "minmaxscaler",
    "t5gemma-b-b",
    "0.717",
    "0.782",
)


def get_json(url: str, **params: object) -> tuple[dict | list, bytes]:
    response = requests.get(
        url,
        params=params or None,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    return response.json(), response.content


def github_tree_audit(revision: str) -> dict:
    archive_url = (
        f"https://codeload.github.com/{REGRESS_REPO}/tar.gz/{revision}"
    )
    response = requests.get(
        archive_url,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    archive = tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz")
    members = [member for member in archive.getmembers() if member.isfile()]
    paths = sorted(member.name.split("/", 1)[1] for member in members)
    dedicated_paths = [
        path for path in paths
        if any(marker in path.lower() for marker in (
            "table5", "table_5", "table6", "table_6", "ablation",
            "regression_head", "regression-head", "scaling", "600m",
        ))
    ]
    marker_hits: dict[str, list[str]] = {marker: [] for marker in EXACT_MARKERS}
    inspected_text_files = 0
    for member in members:
        path = member.name.split("/", 1)[1]
        if not path.endswith(
            (".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".sh")
        ):
            continue
        extracted = archive.extractfile(member)
        if extracted is None:
            raise AssertionError(f"archive file could not be read: {path}")
        text = extracted.read().decode("utf-8", errors="replace").lower()
        inspected_text_files += 1
        for marker in EXACT_MARKERS:
            if marker in text:
                marker_hits[marker].append(path)
    exact_hits = {key: value for key, value in marker_hits.items() if value}
    return {
        "revision": revision,
        "archive_url": archive_url,
        "archive_sha256": hashlib.sha256(response.content).hexdigest(),
        "file_count": len(paths),
        "inspected_text_files": inspected_text_files,
        "dedicated_table_or_ablation_paths": dedicated_paths,
        "exact_marker_hits": exact_hits,
        "exact_table_artifact_discovered": bool(dedicated_paths or exact_hits),
    }


def author_model_audit() -> dict:
    listing, raw = get_json(
        "https://huggingface.co/api/models",
        author="akhauriyash",
        limit=1000,
        full="true",
    )
    if not isinstance(listing, list):
        raise AssertionError("unexpected Hugging Face author listing")
    ids = sorted(item["id"] for item in listing)
    relevant = [
        model_id for model_id in ids
        if any(token in model_id.lower() for token in ("regress", "rlm", "gemma"))
    ]
    table5_or_600m = [
        model_id for model_id in relevant
        if any(token in model_id.lower() for token in (
            "600m", "b-b", "table5", "ablation", "standard", "normalized",
        ))
    ]
    return {
        "api_url": (
            "https://huggingface.co/api/models?author=akhauriyash"
            "&limit=1000&full=true"
        ),
        "listing_sha256": hashlib.sha256(raw).hexdigest(),
        "relevant_public_models": relevant,
        "table5_or_600m_rlm_models": table5_or_600m,
        "exact_claim_checkpoint_discovered": bool(table5_or_600m),
    }


def base_model_audit() -> dict:
    output = {}
    for model_id, expected in BASE_MODELS.items():
        detail, raw = get_json(f"https://huggingface.co/api/models/{model_id}")
        if not isinstance(detail, dict):
            raise AssertionError(f"unexpected model response: {model_id}")
        if detail.get("sha") != expected["revision"]:
            raise AssertionError(f"{model_id}: revision changed")
        total = detail.get("safetensors", {}).get("total")
        if total != expected["parameters"]:
            raise AssertionError(f"{model_id}: parameter count changed")
        config_response = requests.get(
            f"https://huggingface.co/{model_id}/resolve/{expected['revision']}/config.json",
            headers={"User-Agent": USER_AGENT},
            timeout=120,
        )
        output[model_id] = {
            "revision": detail["sha"],
            "gated": detail.get("gated"),
            "safetensors_parameter_count": total,
            "metadata_sha256": hashlib.sha256(raw).hexdigest(),
            "unauthenticated_config_http_status": config_response.status_code,
            "config_publicly_downloadable_without_acceptance": (
                config_response.status_code == 200
            ),
        }
    return output


def qin_reference_audit() -> dict:
    url = (
        f"https://raw.githubusercontent.com/{QIN_REPO}/{QIN_REVISION}/{QIN_PATH}"
    )
    response = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=120
    )
    response.raise_for_status()
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != QIN_SHA256:
        raise AssertionError("pinned Qin reference implementation changed")
    text = response.text
    required = {
        "sequence_classification_regressor": (
            "AutoModelForSequenceClassification.from_pretrained" in text
        ),
        "one_output_label": "num_labels=1" in text,
        "regression_problem_type": 'problem_type="regression"' in text,
        "per_dataset_minmax_0_1": (
            "MinMaxScaler(feature_range=(0, 1))" in text
        ),
    }
    if not all(required.values()):
        raise AssertionError("cited normalized-regression implementation changed")
    return {
        "url": url,
        "sha256": digest,
        "features": required,
        "scope": (
            "Confirms the cited normalized target transformation and scalar "
            "sequence-classification regressor pattern; it is not the paper's "
            "missing Table 5 implementation, split, or checkpoint."
        ),
    }


def detector_negative_control() -> dict:
    synthetic_paths = ["configs/table5_regression_head.yaml"]
    detected = any(
        any(marker in path.lower() for marker in (
            "table5", "regression_head", "ablation", "600m"
        ))
        for path in synthetic_paths
    )
    if not detected:
        raise AssertionError("artifact detector missed synthetic exact artifact")
    return {
        "synthetic_paths": synthetic_paths,
        "expected_detection": True,
        "observed_detection": detected,
        "passes_claim_acceptance": False,
        "reason": "Synthetic file only tests detector sensitivity; it is not evidence.",
    }


def verify() -> dict:
    repository = {
        label: github_tree_audit(revision)
        for label, revision in REGRESS_REVISIONS.items()
    }
    if any(row["exact_table_artifact_discovered"] for row in repository.values()):
        raise AssertionError("manual review required: exact marker appeared upstream")
    author_models = author_model_audit()
    if author_models["exact_claim_checkpoint_discovered"]:
        raise AssertionError("manual review required: candidate Claim 5 model appeared")
    bases = base_model_audit()
    qin = qin_reference_audit()
    control = detector_negative_control()
    routes = json.loads(ROUTES.read_text())
    route_rows = routes["routes"]
    if [row["route"] for row in route_rows] != [1, 2, 3, 4]:
        raise AssertionError("four-route sequence is incomplete")
    if len({row["name"] for row in route_rows}) != 4:
        raise AssertionError("routes are not materially distinct")
    if route_rows[-1]["name"] != "Dedicated exact-claim falsification attempt":
        raise AssertionError("fourth route is not falsification-dedicated")
    if any(row["outcome"] != "BLOCKED" for row in route_rows):
        raise AssertionError("unsupported route outcome")
    if routes["confidence_after_three_routes"] != "LOW":
        raise AssertionError("mandatory fourth-route trigger not recorded")
    if routes["final_verdict"] != "BLOCKED":
        raise AssertionError("public audit cannot decide the exact claim")
    gate = subprocess.run(
        [sys.executable, str(EXACT_GATE)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if gate.returncode == 0 or "CLAIM5_EXACT_GATE_BLOCKED" not in gate.stdout:
        raise AssertionError("exact claim gate did not fail closed")
    improvement = TABLE6["t5gemma_b_b_600m"] - TABLE6["t5gemma_s_s_300m"]
    ratio = (
        BASE_MODELS["google/t5gemma-b-b-prefixlm"]["parameters"]
        / BASE_MODELS["google/t5gemma-s-s-prefixlm"]["parameters"]
    )
    if not math.isclose(improvement, 0.038, abs_tol=1e-12):
        raise AssertionError("Table 6 arithmetic mismatch")
    return {
        "status": "PASS",
        "claim_verdict": "BLOCKED",
        "confidence": "LOW",
        "exact_claim_tested": {
            "table5": TABLE5,
            "table6": TABLE6,
            "table6_difference": improvement,
        },
        "official_repository_snapshots": repository,
        "author_public_models": author_models,
        "base_models": bases,
        "base_parameter_ratio": ratio,
        "cited_normalized_head_reference": qin,
        "negative_control": control,
        "routes_completed": len(route_rows),
        "fourth_route": "falsification attempted; no valid counterexample",
        "exact_claim_gate": {
            "exit_code": gate.returncode,
            "stdout": gate.stdout.strip(),
        },
        "unblockers": routes["unblockers"],
        "limitations": [
            "Absence from audited public releases does not prove a private artifact does not exist.",
            "The public base-model sizes validate rounded 300M/600M labels, not the claimed correlation improvement.",
            "The released 181.5M RLM is not identified as either exact Table 6 checkpoint and is not a counterexample.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), sort_keys=True))
