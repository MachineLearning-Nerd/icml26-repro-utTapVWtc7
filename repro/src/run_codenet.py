#!/usr/bin/env python3
"""RegressLM Claim 3 driver — average Spearman across the 17 CodeNet languages.

CodeNet's 17 languages are the top-17 by count in the CDSS space (confirmed by
inspect_data.py). This driver makes a SINGLE pass over CDSS, bucketing rows by
language until every language has `limit` rows (reading the huge CDSS partition
once, not once-per-language), then predicts each bucket with the authors' recipe
(8 samples -> median) and reports per-language Spearman + the average (claim 3: >0.5).

Reuses run_eval.load_model / predict / pick_device so the recipe is identical to
Claim 2. Output: outputs/codenet/per_lang.csv + .json.
"""
import argparse
import csv
import json
import math
import os
from ast import literal_eval

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
from scipy import stats

from run_eval import load_model, predict, pick_device, PARQUET

CODENET_17 = ["C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Haskell",
              "Kotlin", "JavaScript", "PHP", "D", "Scala", "OCaml", "Perl", "Fortran"]


def fetch_all_langs(langs, limit):
    """One pass over CDSS; bucket rows by language; stop when every lang has `limit`."""
    dataset = ds.dataset(PARQUET, format="parquet")
    flt = pa.compute.equal(pa.compute.field("space"), "CDSS")
    wanted = set(langs)
    buckets = {l: [] for l in langs}
    seen = 0
    scanner = dataset.scanner(columns=["input", "target", "metadata"], filter=flt, batch_size=512)
    for batch in scanner.to_batches():
        for inp, tgt, md in zip(batch.column("input"), batch.column("target"), batch.column("metadata")):
            seen += 1
            try:
                mdd = literal_eval(md.as_py()) if md.is_valid else {}
            except Exception:
                mdd = {}
            lang = mdd.get("language")
            if lang in wanted and len(buckets[lang]) < limit:
                if not tgt.is_valid or not inp.is_valid:
                    continue
                x = f"# CDSS\n# Language: {lang}\n{inp.as_py()}"
                buckets[lang].append((x, float(tgt.as_py())))
        full = sum(1 for l in wanted if len(buckets[l]) >= limit)
        if seen % 25000 == 0:
            print(f"  scanned {seen} CDSS rows; filled {full}/{len(wanted)} langs", flush=True)
        if full == len(wanted):
            break
    print(f"  fetched in one pass: scanned {seen} CDSS rows; "
          + ", ".join(f"{l}={len(buckets[l])}" for l in langs), flush=True)
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=25, help="rows per language")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--langs", default=",".join(CODENET_17))
    ap.add_argument("--device", default=None)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--out", default=os.path.join("outputs", "codenet", "per_lang.csv"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)
    tok, model = load_model(device, dtype, args.model_path)
    n_out = int(getattr(model.config, "num_tokens_per_obj", 9)) * \
            int(getattr(model.config, "max_num_objs", 1))
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | n_out={n_out}", flush=True)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    print(f"single-pass fetch over CDSS for {len(langs)} langs (limit={args.limit}) ...", flush=True)
    buckets = fetch_all_langs(langs, args.limit)

    rows, per_lang = [], []
    for li, lang in enumerate(langs):
        inputs_targets = buckets[lang]
        if len(inputs_targets) < 20:
            print(f"  [{li+1}/{len(langs)}] {lang}: skip (only {len(inputs_targets)} rows)", flush=True)
            continue
        inputs = [x for x, _ in inputs_targets]
        targets = [t for _, t in inputs_targets]
        yt = np.array(targets, dtype=float)
        yp = np.array(predict(tok, model, inputs, args.num_samples, device, n_out,
                               args.batch_size, dtype), dtype=float)
        v = np.isfinite(yt) & np.isfinite(yp)
        rho = float(stats.spearmanr(yt[v], yp[v]).correlation) if v.sum() > 2 else float("nan")
        per_lang.append(dict(language=lang, n=int(v.sum()), spearman=rho))
        rows.extend([(lang, yt[i], yp[i]) for i in range(len(yt))])
        print(f"  [{li+1}/{len(langs)}] {lang:14s} n={v.sum():4d} Spearman={rho:.3f}", flush=True)

    rhos = [p["spearman"] for p in per_lang if math.isfinite(p["spearman"])]
    avg = float(np.nanmean(rhos)) if rhos else float("nan")
    res = dict(num_samples=args.num_samples, limit=args.limit, n_langs=len(per_lang),
               avg_spearman=avg, claim_threshold=0.5, per_lang=per_lang)
    print("=" * 60)
    print(f"CDSS average Spearman across {len(per_lang)} CodeNet languages = {avg:.3f}  "
          f"(claim: >0.5)  [{'PASS' if avg > 0.5 else 'FAIL'}]")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["language", "y_true", "y_pred"])
        for lang, t, p in rows:
            w.writerow([lang, t, p])
    with open(args.out.replace(".csv", ".json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
