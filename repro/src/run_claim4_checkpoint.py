#!/usr/bin/env python3
"""CPU-only released-checkpoint inference for Claim 4."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from scipy import stats
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from check_claim4_audit import EXPECTED_BASELINES
from run_claim1_accuracy import (
    DATASET,
    DATASET_PARQUET_SHA256,
    fetch_spaces,
)
from run_eval import decode_seq
from verify_claim4_audit import manual_kendall_tau_b


ROOT = Path(__file__).resolve().parents[2]
DATASET_REVISION = "c557392740094b539bbdb527d03e3a78e5b34a38"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def predict(rows: list[dict], tokenizer, model, cfg: dict, seed: int) -> tuple[list[dict], float]:
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["identifier"]))
    completed = []
    started = time.monotonic()
    for batch_index, start in enumerate(range(0, len(ordered), cfg["batch_size"])):
        chunk = ordered[start:start + cfg["batch_size"]]
        batch_seed = seed + batch_index
        torch.manual_seed(batch_seed)
        encoded = tokenizer(
            [row["input"] for row in chunk],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=cfg["max_length"],
        )
        n_out = int(model.config.num_tokens_per_obj * model.config.max_num_objs)
        generated = model.generate(
            **encoded,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            min_new_tokens=n_out,
            max_new_tokens=n_out,
            num_return_sequences=cfg["num_samples"],
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
            use_cache=True,
        ).reshape(len(chunk), cfg["num_samples"], -1)
        for row_index, source in enumerate(chunk):
            draws = []
            for draw_index in range(cfg["num_samples"]):
                try:
                    value = decode_seq(
                        tokenizer, generated[row_index, draw_index].tolist()
                    )
                except Exception:
                    value = math.nan
                draws.append(value)
            prediction = float(np.nanmedian(np.asarray(draws, dtype=float)))
            completed.append({
                "identifier": source["identifier"],
                "input_sha256": source["input_sha256"],
                "target": source["target"],
                "draws": draws,
                "prediction": prediction,
                "batch_seed": batch_seed,
            })
        elapsed = time.monotonic() - started
        print(
            f"CLAIM4_PROGRESS space={cfg['space']} rows={len(completed)}/{len(rows)} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
    by_identifier = {row["identifier"]: row for row in completed}
    return [by_identifier[row["identifier"]] for row in rows], time.monotonic() - started


def run(config: dict) -> dict:
    cfg = config["claim4_inference"]
    if cfg["evidence_role"] != "runtime calibration only":
        raise AssertionError("this node is calibration-only")
    torch.set_num_threads(config["compute"]["estimated_required_cores"])
    torch.set_num_interop_threads(1)
    setup_started = time.monotonic()
    parquet = Path(hf_hub_download(
        repo_id=DATASET,
        filename="data.parquet",
        repo_type="dataset",
        revision=DATASET_REVISION,
    ))
    parquet_sha256 = sha256(parquet)
    if parquet_sha256 != DATASET_PARQUET_SHA256:
        raise AssertionError(
            f"GraphArch Parquet hash mismatch: {parquet_sha256}"
        )
    rows = fetch_spaces(
        (cfg["space"],),
        cfg["limit"],
        ROOT / ".trackio/cache/claim4_grapharch",
        parquet,
    )[cfg["space"]]
    snapshot = Path(snapshot_download(
        repo_id=cfg["checkpoint"],
        revision=cfg["revision"],
    ))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).cpu().eval()
    weight_path = snapshot / "model.safetensors"
    setup_seconds = time.monotonic() - setup_started
    raw_rows, inference_seconds = predict(rows, tokenizer, model, cfg, config["seed"])
    targets = np.asarray([row["target"] for row in raw_rows], dtype=float)
    predictions = np.asarray([row["prediction"] for row in raw_rows], dtype=float)
    if not np.all(np.isfinite(predictions)):
        raise AssertionError("calibration produced non-finite predictions")
    scipy_tau = float(stats.kendalltau(targets, predictions, variant="b").statistic)
    manual_tau = float(manual_kendall_tau_b(targets, predictions))
    if not math.isclose(scipy_tau, manual_tau, abs_tol=1e-12):
        raise AssertionError("independent Kendall implementations disagree")
    rng = np.random.default_rng(20260926476)
    shuffled = [
        float(stats.kendalltau(rng.permutation(targets), predictions, variant="b").statistic)
        for _ in range(200)
    ]
    shuffled_mean = float(np.mean(shuffled))
    if shuffled_mean >= EXPECTED_BASELINES["GNN"]:
        raise AssertionError("shuffled-target calibration control passed")
    return {
        "status": "PASS",
        "claim_verdict": "BLOCKED",
        "evidence_role": cfg["evidence_role"],
        "paper_scale_evidence": False,
        "dataset": "akhauriyash/GraphArch-Regression",
        "dataset_revision": DATASET_REVISION,
        "dataset_parquet_sha256": parquet_sha256,
        "checkpoint": cfg["checkpoint"],
        "checkpoint_revision": cfg["revision"],
        "model_weights_sha256": sha256(weight_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": {
            "space": cfg["space"],
            "n": len(raw_rows),
            "num_samples": cfg["num_samples"],
            "batch_size": cfg["batch_size"],
            "max_length": cfg["max_length"],
            "seed": config["seed"],
            "aggregation": "median",
        },
        "metrics": {
            "kendall_tau_b": scipy_tau,
            "manual_kendall_tau_b": manual_tau,
            "spearman": float(stats.spearmanr(targets, predictions).statistic),
        },
        "negative_control": {
            "seed": 20260926476,
            "permutations": len(shuffled),
            "mean_shuffled_target_tau_b": shuffled_mean,
            "max_shuffled_target_tau_b": float(np.max(shuffled)),
            "gnn_threshold": EXPECTED_BASELINES["GNN"],
            "passes_acceptance": False,
        },
        "timing": {
            "setup_seconds": setup_seconds,
            "inference_seconds": inference_seconds,
            "seconds_per_row": inference_seconds / len(raw_rows),
            "linear_512_row_inference_estimate_seconds": (
                inference_seconds / len(raw_rows) * 512
            ),
        },
        "raw_rows": raw_rows,
        "limitations": [
            "This bounded run calibrates throughput and the end-to-end path only.",
            "Its sample size is not the Table 4 evaluation domain.",
            "It cannot verify or falsify Claim 4.",
        ],
    }
