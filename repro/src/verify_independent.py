#!/usr/bin/env python3
"""Independent verification + negative controls for RegressLM eval results.

Given one or more ``*_*.csv`` files produced by ``run_eval.py`` (columns i,y_true,y_pred),
this re-scores each with methods INDEPENDENT of the headline Spearman, and runs
false-positive controls that MUST pass:

  * Pearson r  (different correlation measure — should agree with Spearman in sign/strength)
  * 95% bootstrap CI on Spearman (resample pairs with replacement)
  * permutation test p-value vs H0: "preds independent of targets"
  * FALSE-POSITIVE CONTROL: Spearman after shuffling preds == ~0 (the metric is not
    trivially high). If the shuffled rho is large, the metric is broken.

Honest reporting: never force-fit. A small-sample Phase A rho that is positive and
significant (p<0.05, shuffled ~0) validates the pipeline; the authoritative number
comes from the full-scale Phase B (Colab GPU).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy import stats


def load(path):
    yt, yp = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                t, p = float(row["y_true"]), float(row["y_pred"])
            except (ValueError, KeyError):
                continue
            if np.isfinite(t) and np.isfinite(p):
                yt.append(t); yp.append(p)
    return np.array(yt), np.array(yp)


def bootstrap_spearman(yt, yp, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    n_ = len(yt)
    if n_ < 4:
        return float("nan"), float("nan")
    rhos = []
    idx = rng.integers(0, n_, size=(n, n_))
    for r in idx:
        rhos.append(stats.spearmanr(yt[r], yp[r]).correlation)
    rhos = np.array(rhos)[np.isfinite(rhos)]
    return float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def permutation_pvalue(yt, yp, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = stats.spearmanr(yt, yp).correlation
    if not np.isfinite(obs):
        return float("nan"), obs
    cnt = 0
    for _ in range(n):
        perm = rng.permutation(len(yp))
        r = stats.spearmanr(yt, yp[perm]).correlation
        if np.isfinite(r) and abs(r) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1), obs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="result CSVs from run_eval.py (e.g. outputs/phaseA/apps_n40.csv)")
    ap.add_argument("--out", default=os.path.join("outputs", "independent_verification.json"))
    args = ap.parse_args()

    all_res = []
    for path in args.inputs:
        yt, yp = load(path)
        if len(yt) < 5:
            print(f"{path}: only {len(yt)} finite rows — skipping\n"); continue
        rho = float(stats.spearmanr(yt, yp).correlation)
        pr = float(stats.pearsonr(yt, yp)[0])
        ci_lo, ci_hi = bootstrap_spearman(yt, yp)
        pval, _ = permutation_pvalue(yt, yp)
        # false-positive control: shuffle preds, recompute rho
        rng = np.random.default_rng(0)
        shuf = stats.spearmanr(yt, yp[rng.permutation(len(yp))]).correlation
        res = dict(file=path, n=len(yt), spearman=rho, pearson=pr,
                   spearman_ci95=[ci_lo, ci_hi], perm_pvalue=pval,
                   control_shuffled_spearman=float(shuf))
        all_res.append(res)
        print("=" * 70)
        print(f"{os.path.basename(path)}  (n={len(yt)})")
        print(f"  Spearman rho = {rho:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Pearson r    = {pr:.3f}")
        print(f"  permutation p-value (H0: independent) = {pval:.4f}")
        print(f"  CONTROL shuffled Spearman (must be ~0) = {shuf:+.3f}")
        verdict = ("SIGNAL OK" if (rho > 0.3 and shuf < 0.3 and pval < 0.05)
                   else "weak / inconclusive at this sample size")
        print(f"  -> {verdict}")
        print("=" * 70)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(all_res, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
