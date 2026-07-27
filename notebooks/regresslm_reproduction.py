import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Regression Language Models for Code — evidence-first reproduction

    **Live judged baseline: 6/10 · conservative forecast: 6–6/10.**

    This tutorial explains why three claims are verified and two remain
    blocked. All numbers are embedded from the recorded evidence; opening
    the notebook does **not** rerun model inference.
    """)
    return


@app.cell
def _():
    claims = [
        {"claim": 1, "status": "VERIFIED", "points": 2, "confidence": "HIGH"},
        {"claim": 2, "status": "VERIFIED", "points": 2, "confidence": "HIGH"},
        {"claim": 3, "status": "VERIFIED", "points": 2, "confidence": "HIGH"},
        {"claim": 4, "status": "BLOCKED", "points": 0, "confidence": "LOW"},
        {"claim": 5, "status": "BLOCKED", "points": 0, "confidence": "LOW"},
    ]
    return (claims,)


@app.cell
def _(claims, mo):
    mo.ui.table(claims, label="Claim scorecard")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How the released checkpoint is evaluated

    1. Prefix each input with its task or language identifier.
    2. Sample eight fixed-length numeric token sequences.
    3. Decode the digit tokens to floats and take the median.
    4. Compute Spearman rank correlation against the released target.
    5. Recompute with an independent checker and run shuffle controls.

    The environment is pinned by `uv.lock`, Python 3.12,
    `torch==2.7.1`, and `transformers==4.53.2`.
    """)
    return


@app.cell
def _():
    paper_vs_observed = [
        {"metric": "APPS memory", "paper": 0.930, "observed": 0.926807, "n": 512},
        {"metric": "KBSS latency", "paper": 0.516, "observed": 0.535279, "n": 512},
        {"metric": "CodeNet mean", "paper": None, "observed": 0.529850, "n": 3400},
    ]
    return (paper_vs_observed,)


@app.cell
def _(mo, paper_vs_observed):
    mo.vstack(
        [
            mo.md("## Accepted evidence"),
            mo.ui.table(paper_vs_observed),
            mo.md(
                """
                The CodeNet value is the mean of 17 per-language correlations
                at 200 rows per language. A separate CPU run gives 0.523403.
                The released checkpoint has 181,458,944 parameters, not the
                paper's rounded 300M label.
                """
            ),
        ]
    )
    return


@app.cell
def _():
    claim4 = {
        "paper_metric": "five-space mean Kendall tau-b",
        "paper_rlm": 0.461,
        "comparators": {"Arch2Vec": 0.212, "CATE": 0.238, "GNN": 0.429},
        "public_target_checkpoints": "Amoeba, PNAS, DARTS only",
        "missing": "ENAS/NASNet checkpoints and every row manifest",
        "verdict": "BLOCKED",
    }
    claim5 = {
        "paper_table5": {"standard": 0.478, "normalized": 0.717, "decoder": 0.800},
        "paper_table6": {"300M": 0.744, "600M": 0.782},
        "missing": "exact head code/splits/checkpoints, 600M RLM, row identities",
        "verdict": "BLOCKED",
    }
    return claim4, claim5


@app.cell
def _(claim4, claim5, mo):
    mo.vstack(
        [
            mo.md("## Why the remaining claims are blocked"),
            mo.md(f"**Claim 4:** `{claim4}`"),
            mo.md(f"**Claim 5:** `{claim5}`"),
            mo.callout(
                "BLOCKED is deliberate: a nearby model, a small calibration, "
                "missing access, or base-model metadata cannot verify or falsify "
                "the paper's exact experiment.",
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce the integrity suite

    ```bash
    uv sync --locked
    uv run --locked python repro/src/run_campaign.py
    ```

    The exact Claim 4 and Claim 5 scientific gates exit nonzero while their
    verdict is BLOCKED. The cumulative campaign passes only when those
    fail-closed exits, their four-route records, Claims 1–3, controls, and
    artifact hashes all agree.
    """)
    return


if __name__ == "__main__":
    app.run()
