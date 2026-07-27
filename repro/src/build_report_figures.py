#!/usr/bin/env python3
"""Build the four evidence figures used by the cumulative reproduction report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INK = "#172033"
MUTED = "#65758b"
BLUE = "#3977d4"
GREEN = "#1f9d68"
AMBER = "#e59b2f"
RED = "#cf4b45"
PALE = "#eef3f8"


def finish(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def scorecard(path: Path) -> None:
    claims = ["Unified model", "APPS / latency", "17 languages", "NAS ranking", "Ablation / scaling"]
    points = [2, 2, 2, 0, 0]
    colors = [GREEN, GREEN, GREEN, AMBER, AMBER]
    verdicts = ["VERIFIED", "VERIFIED", "VERIFIED", "BLOCKED", "BLOCKED"]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    y = np.arange(len(claims))
    ax.barh(y, [2] * 5, color=PALE, height=0.62)
    ax.barh(y, points, color=colors, height=0.62)
    for index, (point, verdict) in enumerate(zip(points, verdicts)):
        ax.text(2.08, index, f"{verdict} · {point}/2", va="center", color=colors[index], weight="bold")
    ax.set_yticks(y, [f"Claim {i + 1} — {name}" for i, name in enumerate(claims)])
    ax.set_xlim(0, 3.05)
    ax.set_xticks([0, 1, 2], ["0", "1", "2 points"])
    ax.invert_yaxis()
    ax.set_title(
        "The evidence remains an honest 6/10",
        loc="left",
        fontsize=19,
        weight="bold",
        color=INK,
        pad=28,
    )
    ax.text(
        0,
        1.015,
        "Three direct reproductions survive; two exact contracts remain artifact-blocked.",
        transform=ax.transAxes,
        color=MUTED,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    finish(fig, path)


def headline_correlations(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [0.8, 1.2]})
    labels = ["APPS memory", "KBSS latency"]
    paper = [0.930, 0.516]
    observed = [0.9268067718469594, 0.5352789637599933]
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, paper, width, label="Paper", color="#9bb8e8")
    axes[0].bar(x + width / 2, observed, width, label="Observed", color=BLUE)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Spearman ρ")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title("Headline metrics", loc="left", weight="bold", color=INK)
    for position, value in zip(x - width / 2, paper):
        axes[0].text(position, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    for position, value in zip(x + width / 2, observed):
        axes[0].text(position, value + 0.025, f"{value:.3f}", ha="center", fontsize=9, weight="bold")

    languages = ["C++", "C", "Go", "Python"]
    paper_lang = [0.748, 0.741, 0.670, 0.647]
    observed_lang = [0.7875703786, 0.7350963371, 0.7201143805, 0.6109206813]
    y = np.arange(len(languages))
    axes[1].hlines(y, paper_lang, observed_lang, color="#b5c1d0", linewidth=3)
    axes[1].scatter(paper_lang, y, color="#9bb8e8", s=75, label="Paper")
    axes[1].scatter(observed_lang, y, color=BLUE, s=75, label="Observed")
    axes[1].set_yticks(y, languages)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.55, 0.83)
    axes[1].set_xlabel("Spearman ρ")
    axes[1].set_title("Named CodeNet languages", loc="left", weight="bold", color=INK)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e1e8f0", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Released-checkpoint correlations track the paper at full evaluation scale", x=0.06, ha="left", fontsize=18, weight="bold", color=INK)
    finish(fig, path)


def claim4_gap(path: Path) -> None:
    methods = ["Arch2Vec", "CATE", "GNN", "FLAN", "RLM"]
    averages = [0.212, 0.238, 0.429, 0.459, 0.461]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    colors = ["#b5c1d0", "#b5c1d0", "#7da2d9", "#7da2d9", BLUE]
    bars = ax.bar(methods, averages, color=colors, width=0.68)
    for bar, value in zip(bars, averages):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", weight="bold" if value == 0.461 else "normal")
    ax.axhline(0.429, color=RED, linestyle="--", linewidth=1.3, label="GNN comparison threshold")
    ax.scatter([0.15, 0.62, 1.09], [0.138758, 0.172070, 0.55], marker="D", s=70, color=AMBER, zorder=4, label="Diagnostics — not Table 4 evidence")
    ax.text(0.15, 0.105, "NASNet\nunified base", ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(0.62, 0.205, "ENAS\nunified base", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(1.09, 0.535, "DARTS n=16\ncalibration", ha="center", va="top", fontsize=8, color=MUTED)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("Kendall τ-b")
    ax.set_title("Claim 4’s paper bars are known; the matching experiment is not reconstructable", loc="left", fontsize=17, weight="bold", color=INK)
    ax.text(0, 0.617, "Two target checkpoints and every train/evaluation row manifest are missing.", color=MUTED)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.91))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e1e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    finish(fig, path)


def claim5_gap(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.05, 1.15]})
    labels = ["Standard", "Normalized", "Decoder", "300M", "600M"]
    values = [0.478, 0.717, 0.800, 0.744, 0.782]
    colors = ["#b5c1d0", "#7da2d9", BLUE, "#8ec5b0", GREEN]
    bars = axes[0].bar(labels, values, color=colors, width=0.7)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    axes[0].set_ylim(0, 0.92)
    axes[0].set_ylabel("Paper-reported Spearman ρ")
    axes[0].set_title("Reported ablation and scaling", loc="left", weight="bold", color=INK)
    axes[0].text(0.02, 0.955, "Paper values only — no observed reproduction values", transform=axes[0].transAxes, color=RED, weight="bold")

    prerequisites = [
        "Base metadata",
        "Table 5 implementations",
        "Table 5 row split + seeds",
        "600M trained RLM",
        "Table 6 row identities",
    ]
    availability = [1, 0, 0, 0, 0]
    y = np.arange(len(prerequisites))
    axes[1].barh(y, [1] * len(y), color=PALE, height=0.58)
    axes[1].barh(y, availability, color=[GREEN if item else AMBER for item in availability], height=0.58)
    for index, item in enumerate(availability):
        axes[1].text(0.5, index, "PUBLIC" if item else "MISSING", ha="center", va="center", color=GREEN if item else AMBER, weight="bold")
    axes[1].set_yticks(y, prerequisites)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_title("Exact reproducibility prerequisites", loc="left", weight="bold", color=INK)
    for ax in axes:
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Claim 5 is blocked by experiment identity, not by an unfavorable proxy result", x=0.04, ha="left", fontsize=17, weight="bold", color=INK)
    fig.subplots_adjust(wspace=0.48, top=0.80)
    finish(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorecard(args.output_dir / "headline-scorecard.png")
    headline_correlations(args.output_dir / "verified-correlations.png")
    claim4_gap(args.output_dir / "claim4-evidence-gap.png")
    claim5_gap(args.output_dir / "claim5-evidence-gap.png")


if __name__ == "__main__":
    main()
