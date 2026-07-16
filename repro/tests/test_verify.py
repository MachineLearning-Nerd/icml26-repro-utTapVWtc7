"""Unit tests for verify_independent.py — correlation + false-positive control logic.

Run:  .venv/bin/python -m pytest repro/tests/test_verify.py -q
"""
import csv
import os
import tempfile

import numpy as np

# import the module under test from the sibling src dir
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import verify_independent as V  # noqa: E402


def _write_csv(path, yt, yp):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["i", "y_true", "y_pred"])
        for i, (t, p) in enumerate(zip(yt, yp)):
            w.writerow([i, t, p])


def test_load_roundtrip():
    yt = [1.0, 2.0, 3.0]; yp = [1.1, 2.2, 2.9]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.csv"); _write_csv(p, yt, yp)
        a, b = V.load(p)
        assert len(a) == 3 and np.allclose(a, yt) and np.allclose(b, yp)


def test_bootstrap_ci_on_strong_signal():
    rng = np.random.default_rng(0)
    yt = rng.uniform(0, 100, 200)
    yp = yt + rng.normal(0, 2, 200)  # near-perfect rank correlation
    lo, hi = V.bootstrap_spearman(yt, yp, n=400)
    assert lo > 0.9 and hi <= 1.0 and lo <= hi


def test_permutation_pvalue_rejects_independent():
    rng = np.random.default_rng(1)
    yt = rng.uniform(0, 100, 200)
    yp = rng.uniform(0, 100, 200)  # genuinely independent
    pval, obs = V.permutation_pvalue(yt, yp, n=400)
    # under H0 the observed |rho| should be small; p-value should NOT be tiny
    assert pval > 0.05
    assert abs(obs) < 0.3


def test_false_positive_control_via_shuffle():
    """The whole point: shuffling preds destroys any real signal -> rho ~ 0."""
    rng = np.random.default_rng(2)
    yt = rng.uniform(0, 100, 200)
    yp = yt + rng.normal(0, 1, 200)
    shuffled = stats_spearman(yt, yp[rng.permutation(len(yp))])
    assert abs(shuffled) < 0.3  # control must be near zero


def stats_spearman(a, b):
    from scipy import stats
    return float(stats.spearmanr(a, b).correlation)
