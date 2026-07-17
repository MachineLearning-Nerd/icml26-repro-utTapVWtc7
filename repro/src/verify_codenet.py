#!/usr/bin/env python3
"""Independently verify Claim 3 from raw per-language prediction rows.

The claim is the *mean of 17 per-language Spearman correlations*, not a pooled
correlation. Bootstrap resampling and the permutation null are therefore also
performed independently within each language before averaging.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy import stats


def load_by_language(path):
    grouped = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                language = row["language"]
                target = float(row["y_true"])
                prediction = float(row["y_pred"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(target) and np.isfinite(prediction):
                grouped.setdefault(language, [[], []])
                grouped[language][0].append(target)
                grouped[language][1].append(prediction)
    return {
        language: (np.asarray(values[0]), np.asarray(values[1]))
        for language, values in grouped.items()
    }


def mean_language_spearman(grouped):
    per_language = {}
    for language, (targets, predictions) in grouped.items():
        per_language[language] = float(
            stats.spearmanr(targets, predictions).correlation)
    finite = [value for value in per_language.values() if np.isfinite(value)]
    return float(np.mean(finite)), per_language


def bootstrap_ci(grouped, repetitions=2000, seed=0):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        correlations = []
        for targets, predictions in grouped.values():
            indices = rng.integers(0, len(targets), size=len(targets))
            correlation = stats.spearmanr(
                targets[indices], predictions[indices]).correlation
            if np.isfinite(correlation):
                correlations.append(correlation)
        draws.append(np.mean(correlations))
    return [float(value) for value in np.percentile(draws, [2.5, 97.5])]


def permutation_test(grouped, observed, repetitions=2000, seed=1):
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(repetitions):
        correlations = []
        for targets, predictions in grouped.values():
            correlation = stats.spearmanr(
                targets, predictions[rng.permutation(len(predictions))]).correlation
            if np.isfinite(correlation):
                correlations.append(correlation)
        null.append(np.mean(correlations))
    null_array = np.asarray(null)
    pvalue = float((np.count_nonzero(null_array >= observed) + 1) /
                   (len(null_array) + 1))
    return pvalue, float(null_array[0]), float(np.mean(null_array)), \
        float(np.std(null_array))


def verify(path, repetitions=2000, seed=0):
    grouped = load_by_language(path)
    observed, per_language = mean_language_spearman(grouped)
    ci95 = bootstrap_ci(grouped, repetitions=repetitions, seed=seed)
    pvalue, shuffled_once, null_mean, null_std = permutation_test(
        grouped, observed, repetitions=repetitions, seed=seed + 1)
    return {
        "source_csv": path,
        "n_languages": len(grouped),
        "rows_per_language": {
            language: len(values[0]) for language, values in grouped.items()
        },
        "per_language_spearman": per_language,
        "average_spearman": observed,
        "bootstrap_ci95": ci95,
        "permutation_pvalue_one_sided": pvalue,
        "control_shuffled_average_once": shuffled_once,
        "permutation_null_mean": null_mean,
        "permutation_null_std": null_std,
        "claim_threshold": 0.5,
        "claim_pass": bool(observed > 0.5),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", default="outputs/codenet/independent_verification.json")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = verify(args.input, repetitions=args.repetitions, seed=args.seed)
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
