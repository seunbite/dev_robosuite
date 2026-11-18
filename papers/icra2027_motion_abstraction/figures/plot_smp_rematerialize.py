#!/usr/bin/env python3
"""Conceptual figure: STOP attributes rematerialized across morphologies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "fig_smp_rematerialize.png"


def box(ax, xy, w, h, text, *, fc="#f7f7f7", ec="#222", fs=9, bold=False, wrap=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, fontweight=weight,
        family="DejaVu Sans", wrap=True, linespacing=1.25,
    )


def arrow(ax, p0, p1):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color="#333",
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Cue
    box(ax, (3.5, 8.7), 3.0, 0.85, "Cue: STOP", fc="#e8f1fb", bold=True, fs=11)
    arrow(ax, (5.0, 8.7), (5.0, 7.95))

    # Attributes
    box(
        ax, (1.2, 6.55), 7.6, 1.35,
        "Recognition-critical attributes\n"
        "orientation: palm outward   ·   location: front of body   ·   movement: hold",
        fc="#fff8e8", fs=9,
    )
    arrow(ax, (5.0, 6.55), (5.0, 5.85))

    # Rematerialization header
    # Header + math rendered separately for cleaner type.
    box(ax, (1.8, 5.15), 6.4, 0.65, "", fc="#f0f0f0", bold=True, fs=9)
    ax.text(
        5.0, 5.475,
        r"Morphology-conditioned rematerialization  "
        r"$\phi_M(f)\rightarrow a_M$ or $\emptyset$",
        ha="center", va="center", fontsize=9, fontweight="bold",
        family="DejaVu Sans",
    )

    # Three bodies
    bodies = [
        (0.4, "Humanoid", "preserve / rematerialize\npalm outward\nhand in front + hold", "#eaf6ea"),
        (3.5, "IIWA", "rematerialize\ngripper outward\nEE in front + hold", "#eaf6ea"),
        (6.6, "Roomba", "unavailable → ∅\npalm orientation gone\nbase forward + hold", "#fdeceb"),
    ]
    for x, title, body, fc in bodies:
        box(ax, (x, 2.55), 2.95, 2.35,
            f"{title}\n────────────\n{body}", fc=fc, fs=8.2)

    for x in (1.85, 5.0, 8.05):
        arrow(ax, (5.0, 5.15), (x, 4.95))

    # Executed
    box(
        ax, (1.5, 0.55), 7.0, 1.35,
        "Executed motion A on body M\n"
        "edit named factors (misspec)  ·  add stages when factors are ∅ (underspec / context)",
        fc="#f7f7f7", fs=8.5,
    )
    for x in (1.85, 5.0, 8.05):
        arrow(ax, (x, 2.55), (5.0, 1.95))

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
