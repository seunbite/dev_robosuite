#!/usr/bin/env python3
"""Placeholder social-nav accuracy: 8 pairs × Human vs LLM."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT_PDF = ROOT / "figure_social_nav_bars.pdf"
OUT_PNG = ROOT / "figure_social_nav_bars.png"

BLUE = "#1F6FEB"
GRAY = "#8A8A8A"

PAIRS = (
    ("nod_yes", "shake_head_no"),
    ("beckon_come_here", "stop_palm_out"),
    ("point_you", "point_self"),
    ("arms_open_welcome", "cross_arms_defensive"),
    ("facepalm", "highfive_invite"),
    ("shrug_idk", "air_quotes"),
    ("tilt_head_curious", "rub_eye_tired"),
    ("depict_lion", "depict_tree"),
)

# Layout only — human slightly above LLM. Not reported rates.
HUMAN = np.array([88.0, 84.0, 90.0, 86.0, 79.0, 81.0, 85.0, 83.0])
LLM = np.array([61.0, 54.0, 68.0, 57.0, 47.0, 51.0, 59.0, 49.0])


def main() -> None:
    n = len(PAIRS)
    x = np.arange(n)
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.20, 2.24), dpi=180)
    ax.bar(
        x - w / 2,
        HUMAN,
        width=w,
        color=BLUE,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
        label="Human",
    )
    ax.bar(
        x + w / 2,
        LLM,
        width=w,
        color=GRAY,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
        label="LLM",
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Correct A/B choice (%)", fontsize=8.3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{a}\n{b}" for a, b in PAIRS],
        fontsize=8.5,
        fontweight="bold",
        linespacing=1.05,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.tick_params(axis="y", labelsize=8.0)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.axhline(50, color="#BBBBBB", ls="--", lw=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor="none", label="Human"),
            Patch(facecolor=GRAY, edgecolor="none", label="LLM"),
        ],
        loc="upper right",
        fontsize=8.3,
        frameon=False,
        ncol=2,
        handlelength=1.1,
        borderaxespad=0.15,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.40)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=220)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
