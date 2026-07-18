#!/usr/bin/env python3
"""Paper-scale Claim-1 ONNX accuracy reproduction with resumable raw draws.

The public GraphArch release has a single ``train`` split.  Its rows are
already shuffled by identifier, so this runner takes the first finite 512 rows
returned for each requested accuracy space.  Inputs are cached locally, while
the committed CSV retains identifiers, input hashes, targets, all stochastic
draws, and their medians.  A deterministic seed per length-bucketed batch makes
interrupted runs exactly resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import requests
import torch
from scipy import stats

from run_eval import decode_seq, load_model, pick_device


DATASET = "akhauriyash/GraphArch-Regression"
MODEL_ALIAS = "akhauriyash/RLM-GemmaS-Code-v0"
MODEL_WEIGHTS_SHA256 = (
    "7e9df42926babb54c4e47c14a8fd1daecdf54e382f62b07d63d6c7c5fa9f000c"
)
DEFAULT_SPACES = ("NASBench101", "ENAS", "NASNet")


def input_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_row(space: str, row: dict) -> dict | None:
    try:
        target = float(row["val_accuracy"])
        raw_input = str(row["input"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(target):
        return None
    prefixed = f"{space}\n\n{raw_input}"
    return {
        "identifier": str(row["identifier"]),
        "space": space,
        "input": prefixed,
        "input_sha256": input_digest(prefixed),
        "target": target,
    }


def _fetch_space_from_local_parquet(space: str, limit: int, parquet: Path) -> list[dict]:
    """Stream a filtered space from the full author parquet without materializing it."""
    dataset = ds.dataset(parquet, format="parquet")
    predicate = pa.compute.equal(pa.compute.field("space"), space)
    scanner = dataset.scanner(
        columns=["identifier", "space", "input", "val_accuracy"],
        filter=predicate,
        batch_size=128,
        use_threads=True,
    )
    rows = []
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            normalized = _normalize_row(space, row)
            if normalized is not None:
                rows.append(normalized)
            if len(rows) == limit:
                return rows
    return rows


def fetch_spaces(
    spaces: tuple[str, ...],
    limit: int,
    cache_dir: Path,
    full_parquet: Path | None,
) -> dict[str, list[dict]]:
    """Resolve cached spaces and extract all remaining full-parquet spaces in one scan."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    missing = []
    for space in spaces:
        path = cache_dir / f"{space}_n{limit}.jsonl"
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            if len(rows) == limit:
                result[space] = rows
                continue
        missing.append(space)
    if missing and full_parquet is not None and full_parquet.is_file():
        dataset = ds.dataset(full_parquet, format="parquet")
        predicate = pa.compute.is_in(
            pa.compute.field("space"), value_set=pa.array(missing))
        scanner = dataset.scanner(
            columns=["identifier", "space", "input", "val_accuracy"],
            filter=predicate,
            batch_size=128,
            use_threads=True,
        )
        collected = {space: [] for space in missing}
        for batch in scanner.to_batches():
            for row in batch.to_pylist():
                space = row["space"]
                if space not in collected or len(collected[space]) >= limit:
                    continue
                normalized = _normalize_row(space, row)
                if normalized is not None:
                    collected[space].append(normalized)
            if all(len(rows) == limit for rows in collected.values()):
                break
        for space, rows in collected.items():
            if len(rows) != limit:
                raise RuntimeError(f"expected {limit} finite {space} rows, found {len(rows)}")
            path = cache_dir / f"{space}_n{limit}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            print(f"  {space}: cached {len(rows)}/{limit} rows from full parquet", flush=True)
            result[space] = rows
        missing = []
    for space in missing:
        result[space] = fetch_space(space, limit, cache_dir, None)
    return {space: result[space] for space in spaces}


def fetch_space(
    space: str,
    limit: int,
    cache_dir: Path,
    full_parquet: Path | None = None,
) -> list[dict]:
    """Fetch and cache exactly ``limit`` released finite-accuracy rows."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{space}_n{limit}.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) == limit:
            return rows

    if full_parquet is not None and full_parquet.is_file():
        rows = _fetch_space_from_local_parquet(space, limit, full_parquet)
        if len(rows) != limit:
            raise RuntimeError(f"expected {limit} finite {space} rows, found {len(rows)}")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(f"  {space}: cached {len(rows)}/{limit} rows from full parquet", flush=True)
        return rows

    url = "https://datasets-server.huggingface.co/filter"
    rows: list[dict] = []
    offset = 0
    while len(rows) < limit:
        params = {
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "where": f'"space" = {space!r}',
            "offset": offset,
            "length": min(100, limit - len(rows)),
        }
        payload = None
        last_error = ""
        for attempt in range(12):
            try:
                response = requests.get(url, params=params, timeout=600)
                if response.status_code == 200:
                    candidate = response.json()
                    if not candidate.get("error"):
                        payload = candidate
                        break
                    last_error = str(candidate["error"])
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = repr(exc)
            delay = min(2 ** attempt, 30)
            print(f"  {space} fetch retry {attempt + 1}/12 in {delay}s: {last_error}",
                  flush=True)
            time.sleep(delay)
        if payload is None:
            raise RuntimeError(f"could not fetch {space} at offset {offset}: {last_error}")
        page = payload.get("rows", [])
        if not page:
            break
        offset += len(page)
        for item in page:
            normalized = _normalize_row(space, item["row"])
            if normalized is None:
                continue
            rows.append(normalized)
            if len(rows) == limit:
                break
        print(f"  {space}: cached {len(rows)}/{limit} rows", flush=True)
    if len(rows) != limit:
        raise RuntimeError(f"expected {limit} finite {space} rows, found {len(rows)}")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def load_progress(path: Path, num_samples: int) -> dict[str, dict]:
    if not path.exists():
        return {}
    progress: dict[str, dict] = {}
    for raw in path.read_text().splitlines():
        if not raw:
            continue
        row = json.loads(raw)
        if len(row.get("draws", [])) != num_samples:
            raise RuntimeError(f"{path}: incompatible draw count")
        progress[row["identifier"]] = row
    return progress


def append_progress(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@torch.inference_mode()
def run_space(
    *,
    space: str,
    rows: list[dict],
    tokenizer,
    model,
    device: str,
    n_out: int,
    num_samples: int,
    batch_size: int,
    max_length: int,
    seed: int,
    progress_path: Path,
) -> list[dict]:
    progress = load_progress(progress_path, num_samples)
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["identifier"]))
    total_batches = (len(ordered) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(ordered), batch_size)):
        chunk = ordered[start:start + batch_size]
        missing = [row for row in chunk if row["identifier"] not in progress]
        if not missing:
            continue
        if len(missing) != len(chunk):
            raise RuntimeError("partial batch found; deterministic resume requires whole batches")
        batch_seed = seed + DEFAULT_SPACES.index(space) * 100_000 + batch_index
        torch.manual_seed(batch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(batch_seed)
        enc = tokenizer(
            [row["input"] for row in chunk],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(device)
        outputs = model.generate(
            **enc,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            min_new_tokens=n_out,
            max_new_tokens=n_out,
            num_return_sequences=num_samples,
            pad_token_id=getattr(tokenizer, "pad_token_id", 0),
            use_cache=True,
        ).reshape(len(chunk), num_samples, -1)
        completed = []
        for row_index, source in enumerate(chunk):
            draws = []
            for draw_index in range(num_samples):
                try:
                    value = decode_seq(
                        tokenizer, outputs[row_index, draw_index].tolist())
                except Exception:
                    value = math.nan
                draws.append(value)
            completed.append({
                "identifier": source["identifier"],
                "space": space,
                "input_sha256": source["input_sha256"],
                "target": source["target"],
                "draws": draws,
                "prediction": float(np.nanmedian(np.asarray(draws, dtype=float))),
                "batch_seed": batch_seed,
            })
        append_progress(progress_path, completed)
        progress.update({row["identifier"]: row for row in completed})
        print(
            f"  {space}: batch {batch_index + 1}/{total_batches}; "
            f"{len(progress)}/{len(rows)} rows complete",
            flush=True,
        )
    return [progress[row["identifier"]] for row in rows]


def summarize(rows: list[dict]) -> dict:
    targets = np.asarray([row["target"] for row in rows], dtype=float)
    predictions = np.asarray([row["prediction"] for row in rows], dtype=float)
    valid = np.isfinite(targets) & np.isfinite(predictions)
    return {
        "n": int(valid.sum()),
        "spearman": float(stats.spearmanr(targets[valid], predictions[valid]).correlation),
        "pearson": float(stats.pearsonr(targets[valid], predictions[valid]).statistic),
        "mse": float(np.mean((targets[valid] - predictions[valid]) ** 2)),
        "nan_rate": float(1 - valid.mean()),
        "target_min": float(np.min(targets[valid])),
        "target_max": float(np.max(targets[valid])),
    }


def write_outputs(path: Path, all_rows: list[dict], summary: dict, num_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["space", "identifier", "input_sha256", "target"]
    fields += [f"draw_{index}" for index in range(num_samples)]
    fields += ["prediction", "batch_seed"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            rendered = {key: row[key] for key in fields if key in row}
            rendered.update({f"draw_{index}": value
                             for index, value in enumerate(row["draws"])})
            writer.writerow(rendered)
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spaces", default=",".join(DEFAULT_SPACES))
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--cache_dir", default=".trackio/cache/claim1_grapharch")
    parser.add_argument(
        "--full_parquet",
        default=".trackio/cache/grapharch_full/data.parquet",
        help="full author parquet fallback required for spaces absent from the partial viewer",
    )
    parser.add_argument("--progress_dir", default="outputs/claim1_accuracy/raw")
    parser.add_argument("--out", default="outputs/claim1_accuracy/full_n512.csv")
    parser.add_argument("--prepare_only", action="store_true")
    args = parser.parse_args()

    spaces = tuple(value.strip() for value in args.spaces.split(",") if value.strip())
    unknown = set(spaces) - set(DEFAULT_SPACES)
    if unknown:
        raise SystemExit(f"unsupported accuracy spaces: {sorted(unknown)}")
    cache_dir = Path(args.cache_dir)
    full_parquet = Path(args.full_parquet) if args.full_parquet else None
    datasets = fetch_spaces(spaces, args.limit, cache_dir, full_parquet)
    if args.prepare_only:
        print(json.dumps({space: len(rows) for space, rows in datasets.items()}))
        return

    device = pick_device(args.device)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer, model = load_model(device, dtype, args.model_path)
    n_out = int(model.config.num_tokens_per_obj * model.config.max_num_objs)
    print(
        f"device={device} dtype={dtype} params={sum(p.numel() for p in model.parameters())} "
        f"n_out={n_out}",
        flush=True,
    )

    all_rows = []
    per_space = []
    for space in spaces:
        completed = run_space(
            space=space,
            rows=datasets[space],
            tokenizer=tokenizer,
            model=model,
            device=device,
            n_out=n_out,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
            progress_path=Path(args.progress_dir) / f"{space}_n{args.limit}.jsonl",
        )
        all_rows.extend(completed)
        per_space.append({"space": space, **summarize(completed)})
        print("RESULT", json.dumps(per_space[-1]), flush=True)

    summary = {
        "dataset": DATASET,
        "dataset_revision": "c557392740094b539bbdb527d03e3a78e5b34a38",
        "model_alias": MODEL_ALIAS,
        "model_weights_sha256": MODEL_WEIGHTS_SHA256,
        "same_checkpoint_as_code_metric_runs": True,
        "protocol": {
            "limit_per_space": args.limit,
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "aggregation": "median",
        },
        "per_space": per_space,
        "average_spearman": float(np.mean([row["spearman"] for row in per_space])),
        "rows": len(all_rows),
        "raw_draws": len(all_rows) * args.num_samples,
    }
    write_outputs(Path(args.out), all_rows, summary, args.num_samples)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
