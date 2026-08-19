#!/usr/bin/env python3
"""Render Figure-1-style 3-D Pose-DTW distances to human-authored motion."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "data/results/evaluation/motion_similarity_3d"
OUT = ROOT / "pose_dtw_distance_to_human_authored_bars.png"
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
DISPLAY = ("Humanoid", "Tiago", "Single Arm", "Stretch", "Roomba")
METHODS = ("joint", "xyzrpy", "trajectory", "motion_program", "gt")
HUMAN_AUTHORED = METHODS.index("gt")


def main() -> None:
    summary = json.loads((ROOT / "matrix_summary.json").read_text(encoding="utf-8"))
    values = {
        method: [
            float(summary[platform]["pose_dtw"]["distance"][METHODS.index(method)][HUMAN_AUTHORED])
            for platform in PLATFORMS
        ]
        for method in ("joint", "xyzrpy", "trajectory", "motion_program")
    }

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    x = np.arange(len(PLATFORMS))
    width = 0.19
    styles = (
        ("xyzrpy", "Cartesian-Space", "#202020", "#202020", None, "white"),
        ("joint", "Joint-Space", "white", "#202020", None, "#202020"),
        ("motion_program", "Motion Program", "white", "#202020", "//", "#202020"),
        ("trajectory", "Trajectory Sketching", "#1F6FEB", "#1F6FEB", None, "white"),
    )
    for index, (method, label, face, edge, hatch, text_color) in enumerate(styles):
        method_values = values[method]
        bars = ax.bar(
            x + (index - 1.5) * width,
            method_values,
            width=width,
            color=face,
            edgecolor=edge,
            linewidth=1.3,
            hatch=hatch,
            label=label,
        )
        for bar, value in zip(bars, method_values):
            outside = value < 0.15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012 if outside else value - 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom" if outside else "top",
                rotation=90,
                fontsize=11.5,
                fontweight="normal",
                color=text_color,
            )

    ax.set_title("3D Pose-DTW Distance to Human-authored Script", fontsize=17, fontweight="normal", pad=12)
    ax.set_ylabel("Pose-DTW distance (lower is closer)", fontsize=13)
    ax.set_xticks(x, DISPLAY, fontsize=13)
    ax.set_ylim(0, max(max(method_values) for method_values in values.values()) * 1.16)
    ax.tick_params(axis="both", labelsize=13, width=1.0, length=5)
    ax.tick_params(axis="x", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#202020",
        fontsize=10.5,
        borderpad=0.55,
        handlelength=2.4,
        columnspacing=0.9,
        labelspacing=0.45,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
