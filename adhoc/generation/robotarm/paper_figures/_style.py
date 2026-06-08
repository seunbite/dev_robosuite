"""Compact matplotlib style for paper figures (B&W-friendly)."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt

# 4 models × distinct marker / fill for grayscale print
MODEL_ORDER = ("Gm", "Qw32", "Qw7", "Qw3")

MODEL_STYLE: dict[str, dict] = {
    "Gm": {"marker": "o", "fillstyle": "full", "linestyle": "-", "mfc": "black", "mec": "black"},
    "Qw32": {"marker": "s", "fillstyle": "none", "linestyle": "-", "mfc": "white", "mec": "black"},
    "Qw7": {"marker": "^", "fillstyle": "full", "linestyle": "-", "mfc": "black", "mec": "white", "markeredgewidth": 1.2},
    "Qw3": {"marker": "D", "fillstyle": "none", "linestyle": "--", "mfc": "white", "mec": "black"},
}

POSE_TASKS = ("1", "2", "3")
MOTION_TASKS = ("7", "8", "9")

EXP_NAMES = {
    "1": "Gen",
    "2": "VLM",
    "3": "Text",
    "4": "Pair",
    "5": "M6",
    "6": "M12",
    "7": "Gen",
    "8": "VLM",
    "9": "Text",
    "10": "Pair",
}


@dataclass
class ModelSuite:
    label: str
    out_dir: str  # relative to repo


DEFAULT_SUITES = [
    ModelSuite("Gm", "data/results/verify/pilot90_gemini"),
    ModelSuite("Qw32", "data/results/verify/pilot90_qwen32b"),
    ModelSuite("Qw7", "data/results/verify/pilot90_qwen7b"),
    ModelSuite("Qw3", "data/results/verify/pilot90_qwen3b"),
]


def apply_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.02,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 5,
        }
    )


def save_fig(fig: plt.Figure, path, *, caption: str | None = None) -> None:
    path = str(path)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    if caption:
        cap_path = path.rsplit(".", 1)[0] + "_caption.txt"
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(caption.strip() + "\n")
    plt.close(fig)
