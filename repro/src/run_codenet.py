#!/usr/bin/env python3
"""RegressLM Claim 3 driver — average Spearman across the 17 CodeNet languages.

CodeNet's 17 languages are the top-17 by count in the CDSS space (confirmed by
inspect_data.py): C++, Python, Java, C, Ruby, C#, Rust, Go, Haskell, Kotlin,
JavaScript, PHP, D, Scala, OCaml, Perl, Fortran. For each, fetch `--limit` CDSS
rows filtered by metadata.language, predict (8 samples -> median, authors' recipe),
and report per-language Spearman + the average (claim 3: >0.5).

Reuses run_eval.load_model / fetch_rows / predict / decode_seq so the recipe is
identical to Claim 2. Output: outputs/codenet/per_lang.csv + .json.
"""
import argparse
import csv
import json
import math
import os

import numpy as np
from scipy import stats

from run_eval import load_model, predict, pick_device
import torch

CODENET_17 = ["C++", "Python", "Java", "C", "Ruby", "C#", "Rust", "Go", "Haskell",
              "Kotlin", "JavaScript", "PHP", "D", "Scala", "OCaml", "Perl", "Fortran"]


def fetch_lang(space, lang, limit):
    # local import to reuse the pyarrow fetch with language filter
    from run_eval import fetch_rows
    return fetch_rows(space, lang=lang, limit=limit)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=150, help="rows per language")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--langs", default=",".join(CODENET_17),
                    help="comma-sep languages (default = CodeNet 17)")
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
    rows, per_lang = [], []
    for li, lang in enumerate(langs):
        inputs, targets = fetch_lang("CDSS", lang, args.limit)
        if len(inputs) < 20:
            print(f"  [{li+1}/{len(langs)}] {lang}: skip (only {len(inputs)} rows)", flush=True)
            continue
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
