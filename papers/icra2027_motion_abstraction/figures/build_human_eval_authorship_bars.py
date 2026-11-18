#!/usr/bin/env python3
"""Plot reviewed cue-recognition accuracy by robot and authoring method."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_JSON = (
    ROOT
    / "data/results/evaluation/embodiment_cue_id/"
    / "cueid_human_cue_identification_with_traj_baselines.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = OUTPUT_DIR / "figure_humanoid_attr_bars"

PLATFORMS = ["humanoid", "tiago", "single_arm", "stretch", "roomba"]
ROBOT_LABELS = ["Unitree G1", "TIAGo", "IIWA", "Stretch", "Roomba"]
N_PER_BAR = 30


def load_data() -> tuple[dict, list[dict], list[dict]]:
    source = json.loads(SOURCE_JSON.read_text())
    if source["platforms"] != PLATFORMS:
        raise ValueError(
            f"Unexpected platform order: {source['platforms']} (expected {PLATFORMS})"
        )

    human_rates = source["human_authored"]
    llm_rates = source["ours_gemini_2_5_pro"]

    human = []
    llm = []
    for platform, robot, human_rate, llm_rate in zip(
        PLATFORMS, ROBOT_LABELS, human_rates, llm_rates
    ):
        human.append(
            {
                "platform": platform,
                "robot": robot,
                "correct": round(human_rate * N_PER_BAR / 100),
                "total": N_PER_BAR,
                "accuracy_percent": human_rate,
            }
        )
        llm.append(
            {
                "platform": platform,
                "robot": robot,
                "correct": round(llm_rate * N_PER_BAR / 100),
                "total": N_PER_BAR,
                "accuracy_percent": llm_rate,
            }
        )

    payload = {
        "source": str(SOURCE_JSON.relative_to(ROOT)),
        "protocol": "Reviewed binary YES/NO recognition annotation",
        "cue_count_per_robot_and_author": N_PER_BAR,
        "aggregate": {"human_authored": human, "llm_authored": llm},
        "overall": {
            "human_authored": source["overall"]["human_authored"],
            "llm_authored": source["overall"]["ours_gemini_2_5_pro"],
        },
    }
    return payload, human, llm


def annotate_bars(ax, bars, rows) -> None:
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{row['accuracy_percent']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.2,
        )


def main() -> None:
    payload, human, llm = load_data()
    human_values = np.array([row["accuracy_percent"] for row in human])
    llm_values = np.array([row["accuracy_percent"] for row in llm])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(ROBOT_LABELS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.16, 2.75))

    human_bars = ax.bar(
        x - width / 2,
        human_values,
        width,
        label="Human-authored",
        color="#242424",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    llm_bars = ax.bar(
        x + width / 2,
        llm_values,
        width,
        label="LLM-authored",
        color="#2870E8",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    annotate_bars(ax, human_bars, human)
    annotate_bars(ax, llm_bars, llm)

    ax.set_ylabel("Recognition accuracy (%)")
    ax.set_xticks(x, ROBOT_LABELS)
    ax.set_ylim(0, 111)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(axis="both", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.8,
    )

    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_data.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
