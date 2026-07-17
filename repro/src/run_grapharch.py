#!/usr/bin/env python3
"""RegressLM Claim 1 accuracy evaluator on released ONNX architecture data.

Evaluates the released unified checkpoint on the first ``limit`` rows from each
requested GraphArch-Regression space. Rows are fetched from Hugging Face's
dataset-server filter endpoint, avoiding the 4.1 GB parquet download. The raw
target/prediction pairs and a machine-readable per-space summary are retained.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import requests
import torch
from scipy import stats

from run_eval import load_model, pick_device, predict


DEFAULT_SPACES = ["NASBench101"]


def fetch_space(space, limit):
    """Fetch exactly ``limit`` released rows using server-side filtering."""
    url = "https://datasets-server.huggingface.co/filter"
    rows = []
    offset = 0
    while len(rows) < limit:
        length = min(100, limit - len(rows))
        params = {
            "dataset": "akhauriyash/GraphArch-Regression",
            "config": "default",
            "split": "train",
            "where": f'"space" = {space!r}',
            "offset": offset,
            "length": length,
        }
        for attempt in range(6):
            response = requests.get(url, params=params, timeout=600)
            if response.status_code < 500:
                break
            print(f"  dataset-server attempt {attempt + 1}/6 returned "
                  f"{response.status_code}; retrying", flush=True)
            time.sleep(min(2 ** attempt, 20))
        response.raise_for_status()
        page = response.json().get("rows", [])
        if not page:
            break
        rows.extend(item["row"] for item in page)
        offset += len(page)
    if len(rows) != limit:
        raise RuntimeError(
            f"server index has {len(rows)}/{limit} {space} rows; the current HF "
            "converted view is partial. NASBench101 is fully accessible without "
            "downloading the 4.1 GB original parquet.")
    inputs = [f"{space}\n\n{row['input']}" for row in rows]
    targets = [float(row["val_accuracy"]) for row in rows]
    return inputs, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--out", default="outputs/grapharch/full_gpu_n64.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)
    tokenizer, model = load_model(device, dtype, args.model_path)
    n_out = int(getattr(model.config, "num_tokens_per_obj", 9)) * \
            int(getattr(model.config, "max_num_objs", 1))
    print(f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M n_out={n_out}",
          flush=True)

    raw_rows = []
    results = []
    spaces = [value.strip() for value in args.spaces.split(",") if value.strip()]
    for space_index, space in enumerate(spaces):
        inputs, targets = fetch_space(space, args.limit)
        print(f"[{space_index + 1}/{len(spaces)}] {space}: fetched {len(inputs)}; "
              f"target range={min(targets):.4f}..{max(targets):.4f}", flush=True)
        predictions = predict(
            tokenizer, model, inputs, args.num_samples, device, n_out,
            args.batch_size, dtype, max_length=args.max_length,
        )
        target_array = np.asarray(targets, dtype=float)
        prediction_array = np.asarray(predictions, dtype=float)
        valid = np.isfinite(target_array) & np.isfinite(prediction_array)
        rho = float(stats.spearmanr(
            target_array[valid], prediction_array[valid]).correlation)
        pearson = float(stats.pearsonr(
            target_array[valid], prediction_array[valid])[0])
        result = {
            "space": space,
            "n": int(valid.sum()),
            "spearman": rho,
            "pearson": pearson,
            "mse": float(np.mean(
                (target_array[valid] - prediction_array[valid]) ** 2)),
            "nan_rate": float(1 - valid.mean()),
        }
        results.append(result)
        raw_rows.extend((space, index, target, prediction)
                        for index, (target, prediction)
                        in enumerate(zip(targets, predictions)))
        print("RESULT", json.dumps(result), flush=True)

    finite_rhos = [r["spearman"] for r in results if math.isfinite(r["spearman"])]
    summary = {
        "dataset": "akhauriyash/GraphArch-Regression",
        "target": "val_accuracy",
        "limit_per_space": args.limit,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "average_spearman": float(np.mean(finite_rhos)),
        "per_space": results,
    }
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["space", "row_index", "y_true", "y_pred"])
        writer.writerows(raw_rows)
    with open(args.out.replace(".csv", ".json"), "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
