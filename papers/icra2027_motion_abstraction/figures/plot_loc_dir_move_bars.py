#!/usr/bin/env python3
"""Fig. 4: provisional five-robot human-evaluation comparison."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
FEASIBILITY = PAPER / "embodiment_physical_difficulty_90.json"
OUT_PDF = ROOT / "figure_humanoid_attr_bars.pdf"
OUT_PNG = ROOT / "figure_humanoid_attr_bars.png"
OUT_DATA = ROOT / "figure_humanoid_attr_bars_data.json"

ROBOTS = ("Humanoid", "Tiago", "Single Arm", "Stretch", "Roomba")
METHODS = ("Ours (human)", "Ours (LLM)", "Joint-space", "Cartesian-space")

# Provisional layout values only. Replace atomically after the five-robot
# human study; the figure caption labels the complete panel [Not fixed].
PLACEHOLDER = {
    "Humanoid": (96.7, 63.3, 58.9, 55.6),
    "Tiago": (93.3, 60.0, 52.2, 48.9),
    "Single Arm": (83.0, 59.6, 46.8, 42.6),
    "Stretch": (81.1, 56.6, 43.4, 39.6),
    "Roomba": (72.7, 54.5, 45.5, 42.4),
}


def build_data() -> dict:
    feasibility = json.loads(FEASIBILITY.read_text())
    return {
        "status": "NOT_FIXED_PLACEHOLDER",
        "robots": {
            robot: {
                "feasible_n": feasibility["by_robot"][robot]["feasible_n"],
                "values": dict(zip(METHODS, PLACEHOLDER[robot])),
            }
            for robot in ROBOTS
        },
    }


def plot(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.65, 2.62), dpi=220)
    x = np.arange(len(ROBOTS), dtype=float)
    width = 0.19
    offsets = (np.arange(len(METHODS)) - 1.5) * width
    styles = (
        {"color": "#202020", "edgecolor": "#202020", "hatch": None},
        {"color": "#1F6FEB", "edgecolor": "#1F6FEB", "hatch": None},
        {"color": "white", "edgecolor": "#303030", "hatch": None},
        {"color": "white", "edgecolor": "#555555", "hatch": "////"},
    )

    for mi, method in enumerate(METHODS):
        vals = [data["robots"][robot]["values"][method] for robot in ROBOTS]
        bars = ax.bar(
            x + offsets[mi], vals, width=width * 0.92, label=method,
            linewidth=0.8, zorder=3, **styles[mi],
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 1.5,
                f"{value:.1f}", ha="center", va="bottom",
                fontsize=6.7, rotation=90,
            )

    labels = [
        f"{robot}\n(n={data['robots'][robot]['feasible_n']})"
        for robot in ROBOTS
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.7)
    ax.set_ylabel("Human recognition (%)", fontsize=6.7)
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="y", labelsize=6.7)
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.19), ncol=4,
        fontsize=6.7, frameon=False, handlelength=1.7, columnspacing=1.0,
    )
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main() -> None:
    data = build_data()
    OUT_DATA.write_text(json.dumps(data, indent=2) + "\n")
    plot(data)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_DATA}")


if __name__ == "__main__":
    main()
