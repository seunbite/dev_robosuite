#!/usr/bin/env python3
"""Score returned balanced 4AFC decks and plot the latest partial results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_DIR = Path(__file__).resolve().parent.parent
RESPONSES_PATH = PAPER_DIR / "human_eval_4afc_responses.json"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = OUTPUT_DIR / "figure_human_eval_4afc_partial_latest"
PLATFORMS = ["humanoid", "tiago", "single_arm", "stretch", "roomba"]
ROBOT_LABELS = ["Unitree G1", "TIAGo", "IIWA", "Stretch", "Roomba"]
AUTHOR_ORDER = ["human", "llm"]


def normalize_response(raw: str) -> str:
    value = "".join(raw.upper().split())
    if value and value[0].isdigit():
        value = "".join("ABCDE"[int(ch) - 1] for ch in value)
    if any(ch not in "ABCDE" for ch in value):
        raise ValueError(f"Unsupported response character in {raw!r}")
    return value


def main() -> None:
    response_source = json.loads(RESPONSES_PATH.read_text())
    answer_key_path = PAPER_DIR / response_source["answer_key"]
    answer_key = json.loads(answer_key_path.read_text())
    decks = {deck["participant_id"]: deck for deck in answer_key["decks"]}

    cells = defaultdict(lambda: {"correct": 0, "total": 0})
    participants = []
    trial_rows = []

    for participant_id, response_record in response_source["responses"].items():
        deck = decks[participant_id]
        response = normalize_response(response_record["response"])
        trials = deck["trials"]
        if len(response) != len(trials):
            raise ValueError(
                f"{participant_id}: received {len(response)} responses for "
                f"{len(trials)} trials"
            )
        if response_record["source_authorship"] != deck["source_authorship"]:
            raise ValueError(f"{participant_id}: authorship does not match answer key")

        participant_correct = 0
        by_robot = defaultdict(lambda: {"correct": 0, "total": 0})
        for selected, trial in zip(response, trials):
            correct = selected == trial["answer_letter"]
            participant_correct += int(correct)
            platform = trial["morphology_id"]
            authorship = deck["source_authorship"]
            cells[(authorship, platform)]["correct"] += int(correct)
            cells[(authorship, platform)]["total"] += 1
            by_robot[platform]["correct"] += int(correct)
            by_robot[platform]["total"] += 1
            trial_rows.append(
                {
                    "participant_id": participant_id,
                    "source_authorship": authorship,
                    "trial": trial["local_trial_number"],
                    "robot": platform,
                    "cue": trial["cue_id"],
                    "selected": selected,
                    "answer": trial["answer_letter"],
                    "correct": int(correct),
                }
            )

        participants.append(
            {
                "participant_id": participant_id,
                "source_authorship": deck["source_authorship"],
                "correct": participant_correct,
                "total": len(trials),
                "accuracy_percent": 100 * participant_correct / len(trials),
                "by_robot": dict(by_robot),
            }
        )

    aggregates = {}
    for authorship in AUTHOR_ORDER:
        aggregates[authorship] = []
        for platform, robot_label in zip(PLATFORMS, ROBOT_LABELS):
            row = cells[(authorship, platform)]
            accuracy = 100 * row["correct"] / row["total"] if row["total"] else None
            aggregates[authorship].append(
                {
                    "platform": platform,
                    "robot": robot_label,
                    **row,
                    "accuracy_percent": accuracy,
                }
            )

    payload = {
        "status": "partial; participant counts are currently unbalanced by authorship",
        "response_source": str(RESPONSES_PATH.relative_to(PAPER_DIR)),
        "answer_key": str(answer_key_path.relative_to(PAPER_DIR)),
        "participants": participants,
        "aggregate": aggregates,
    }
    OUTPUT_STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    with OUTPUT_STEM.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_rows[0].keys())
        writer.writeheader()
        writer.writerows(trial_rows)

    human_values = np.array(
        [row["accuracy_percent"] for row in aggregates["human"]], dtype=float
    )
    llm_values = np.array(
        [row["accuracy_percent"] for row in aggregates["llm"]], dtype=float
    )

    # Match the visual specification of figure_humanoid_attr_bars exactly.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12.7,
            "axes.labelsize": 13.2,
            "xtick.labelsize": 12.4,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 12.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    x = np.arange(len(ROBOT_LABELS))
    width = 0.30
    bar_offset = 0.16  # retain a narrow gap between the paired bars
    x_limits = (-0.38, len(ROBOT_LABELS) - 1 + 0.38)
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    human_bars = ax.bar(
        x - bar_offset,
        human_values,
        width,
        label="Human-authored",
        color="#242424",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    llm_bars = ax.bar(
        x + bar_offset,
        llm_values,
        width,
        label="LLM-authored",
        color="#2870E8",
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    wave_x = np.linspace(*x_limits, 800)
    wave_y = 25.0 + 0.75 * np.sin(
        np.linspace(0, 28 * 2 * np.pi, wave_x.size)
    )
    ax.plot(
        wave_x,
        wave_y,
        label="Random (25%)",
        color="#B8B8B8",
        linewidth=2.0,
        zorder=10,
    )
    for bars, values in ((human_bars, human_values), (llm_bars, llm_values)):
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=12.4,
            )

    ax.set_ylabel("Recognition accuracy (%)")
    ax.set_xticks(x, ROBOT_LABELS)
    ax.set_xlim(*x_limits)
    ax.set_ylim(0, 80)
    ax.set_yticks(np.arange(0, 81, 20))
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(axis="both", length=0)
    handles, labels = ax.get_legend_handles_labels()
    legend_items = dict(zip(labels, handles))
    legend_order = ["Human-authored", "LLM-authored", "Random (25%)"]
    ax.legend(
        [legend_items[label] for label in legend_order],
        legend_order,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.01),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="#777777",
        facecolor="white",
        framealpha=1.0,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.tight_layout(pad=0.6)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
