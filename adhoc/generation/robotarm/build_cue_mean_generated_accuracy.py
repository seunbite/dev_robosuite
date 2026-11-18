#!/usr/bin/env python3
"""Plot cue recognizability averaged across embodiments and generated methods."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


REPO = Path(__file__).resolve().parents[3]
SOURCE = (
    REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "all_methods_human_yesno_c1_30.xlsx"
)
OUT = SOURCE.parent / "cue_mean_generated_accuracy_sorted"
METHODS = (
    "Joint-Space",
    "Cartesian-Space",
    "Motion Program",
    "Trajectory Sketching",
)


def main() -> None:
    workbook = load_workbook(SOURCE, data_only=True, read_only=True)
    sheet = workbook.active
    headers = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
    columns = [headers.index(method) for method in METHODS]
    groups = {
        "All": {"humanoid", "tiago", "stretch", "single_arm", "roomba"},
        "Human-like": {"humanoid", "tiago"},
        "Non-human-like": {"stretch", "single_arm", "roomba"},
    }
    cues: dict[str, dict[str, list[str]]] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        cue = str(row[0])
        platform = str(row[1]).split("·", 1)[1].strip()
        values = [str(row[column]).upper() for column in columns]
        for group, platforms in groups.items():
            if platform in platforms:
                cues.setdefault(cue, {}).setdefault(group, []).extend(values)

    def accuracy(labels: list[str]) -> float:
        return 100.0 * sum(label == "YES" for label in labels) / len(labels)

    ranked = sorted(cues, key=lambda cue: (-accuracy(cues[cue]["All"]), cue))
    selected = ranked[:7] + ranked[-7:]
    scores = {group: [accuracy(cues[cue][group]) for cue in selected] for group in groups}
    labels = [
        "\n".join(word.title() for word in cue.split(". ", 1)[1].replace("_", " ").split())
        for cue in selected
    ]
    x = np.r_[np.arange(7), np.arange(7) + 8]
    width = 0.21
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 13,
        "legend.fontsize": 8.0,
        "axes.edgecolor": "#202020",
        "axes.linewidth": 1.0,
    })
    fig, ax = plt.subplots(figsize=(10.2, 2.85))

    styles = (
        ("All", "#1F6FEB"),
        ("Human-like", "#202020"),
        ("Non-human-like", "#E8E8E8"),
    )
    for index, (group, face) in enumerate(styles):
        bars = ax.bar(
            x + (index - 1) * width,
            scores[group],
            width,
            color=face,
            edgecolor="none",
            linewidth=0,
            label=group,
        )
        for bar, value in zip(bars, scores[group]):
            inside = value > 55
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 2.0 if inside else value + 2.0,
                f"{value:.0f}",
                ha="center",
                va="top" if inside else "bottom",
                fontsize=9,
                rotation=90,
                color="white" if inside and face in ("#1F6FEB", "#202020") else "#202020",
            )
    ax.text(7, 7, "~", ha="center", va="center", fontsize=26, color="#444444")
    ax.set_xticks(x, labels, rotation=0, ha="center", va="top", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#202020",
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        borderpad=0.30,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.55,
        labelspacing=0.25,
        fontsize=8.0,
    )
    fig.tight_layout(pad=0.28)
    paper_fig = REPO / "papers/icra2027_motion_abstraction/figures/figure16_cue_difficulty.png"
    for path in (OUT.with_suffix(".png"), OUT.with_suffix(".svg"), paper_fig):
        try:
            fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
            print(path)
        except OSError as exc:
            print(f"skip save {path} ({exc})")
    plt.close(fig)


if __name__ == "__main__":
    main()
