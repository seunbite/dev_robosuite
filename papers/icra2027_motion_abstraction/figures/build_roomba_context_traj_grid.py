#!/usr/bin/env python3
"""Human contextualization grid: path plots + Motion Code (no mid-motion renders).

In-place yaw (r≈0 spins / wiggles) collapses on a plain (x,y) plot, so we expand
those segments into a small Archimedean spiral and draw heading ticks so a
360°/720° spin is readable from start→end.
"""

from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
from embodiment_renderers import _roomba_states  # noqa: E402
from stretch_renderer import (  # noqa: E402
    _STRETCH_SCENE,
    _apply,
    _ee,
    _stretch_states,
)

PAPER = Path(__file__).resolve().parent
GT_JSON = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"

FPS = 12

# First three keep the strong contextualization examples; last two swap in
# richer staged programs (spin/hold/retreat scaffolds) instead of thin cores.
CELLS = [
    ("roomba", "highfive_invite", "Roomba · highfive"),
    ("stretch", "raise_fists_victory", "Stretch · victory"),
    ("roomba", "cover_yawn_sleepy", "Roomba · yawn"),
    ("roomba", "hands_up_surrender", "Roomba · surrender"),
    ("stretch", "shrug_idk", "Stretch · shrug"),
]


def _unwrap(yaws: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(yaws, dtype=float))


def _display_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    *,
    spiral_gain: float = 0.038,
    spiral_r0: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    if n == 0:
        return xs, ys
    yaws_u = _unwrap(yaws)
    out_x = np.empty(n)
    out_y = np.empty(n)
    out_x[0], out_y[0] = xs[0], ys[0]
    spin_yaw0: float | None = None
    spin_x = spin_y = 0.0
    for i in range(1, n):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dyaw = abs(yaws_u[i] - yaws_u[i - 1])
        if math.hypot(dx, dy) < 1.5e-4 and dyaw > 1e-3:
            if spin_yaw0 is None:
                spin_yaw0 = float(yaws_u[i - 1])
                spin_x, spin_y = float(xs[i - 1]), float(ys[i - 1])
            turns = abs(float(yaws_u[i] - spin_yaw0))
            rho = spiral_r0 + spiral_gain * turns
            ang = float(yaws_u[i])
            out_x[i] = spin_x + rho * math.cos(ang)
            out_y[i] = spin_y + rho * math.sin(ang)
        else:
            spin_yaw0 = None
            out_x[i], out_y[i] = xs[i], ys[i]
    return out_x, out_y


def _densify_se2(
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    *,
    max_dyaw_deg: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(xs) < 2:
        return xs, ys, yaws
    yaws_u = _unwrap(yaws)
    ox, oy, oyaw = [float(xs[0])], [float(ys[0])], [float(yaws_u[0])]
    max_dyaw = math.radians(max_dyaw_deg)
    for i in range(1, len(xs)):
        dyaw = float(yaws_u[i] - yaws_u[i - 1])
        steps = max(1, int(math.ceil(abs(dyaw) / max_dyaw)))
        for s in range(1, steps + 1):
            t = s / steps
            ox.append(float(xs[i - 1] + (xs[i] - xs[i - 1]) * t))
            oy.append(float(ys[i - 1] + (ys[i] - ys[i - 1]) * t))
            oyaw.append(float(yaws_u[i - 1] + dyaw * t))
    return np.asarray(ox), np.asarray(oy), np.asarray(oyaw)


def _densify_xy(
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    *,
    max_step: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample long translational steps so reverse strokes have enough samples."""
    if len(xs) < 2:
        return xs, ys, yaws
    yaws_u = _unwrap(yaws)
    ox, oy, oyaw = [float(xs[0])], [float(ys[0])], [float(yaws_u[0])]
    for i in range(1, len(xs)):
        dist = math.hypot(float(xs[i] - xs[i - 1]), float(ys[i] - ys[i - 1]))
        steps = max(1, int(math.ceil(dist / max_step)))
        for s in range(1, steps + 1):
            t = s / steps
            ox.append(float(xs[i - 1] + (xs[i] - xs[i - 1]) * t))
            oy.append(float(ys[i - 1] + (ys[i] - ys[i - 1]) * t))
            oyaw.append(float(yaws_u[i - 1] + (yaws_u[i] - yaws_u[i - 1]) * t))
    return np.asarray(ox), np.asarray(oy), np.asarray(oyaw)


def _fan_retraces(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    pitch: float = 0.038,
    reverse_cos: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Offset successive back-and-forth passes so overlapping strokes stay visible.

    When the path reverses along nearly the same corridor (common for Roomba
    oscillation / multi-arc yawns), a single polyline collapses to one stroke.
    Each reversal starts a new lap, shifted along the local normal.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    if n < 3:
        return xs, ys

    lap = np.zeros(n, dtype=int)
    cur = 0
    for i in range(2, n):
        v0x, v0y = xs[i - 1] - xs[i - 2], ys[i - 1] - ys[i - 2]
        v1x, v1y = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        n0 = math.hypot(v0x, v0y)
        n1 = math.hypot(v1x, v1y)
        # Ignore in-place / near-stationary samples (spin densification).
        if n0 > 0.008 and n1 > 0.008:
            cos = (v0x * v1x + v0y * v1y) / (n0 * n1)
            # Trigger on soft reversals too (gentle arc turnarounds).
            if cos < reverse_cos:
                cur += 1
        lap[i] = cur
    if n > 2:
        lap[1] = lap[2]

    out_x, out_y = xs.copy(), ys.copy()
    for i in range(1, n):
        if lap[i] == 0:
            continue
        dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        norm = math.hypot(dx, dy)
        if norm < 1e-9 and i >= 2:
            dx, dy = xs[i - 1] - xs[i - 2], ys[i - 1] - ys[i - 2]
            norm = math.hypot(dx, dy)
        if norm < 1e-9:
            px, py = 0.0, 1.0
        else:
            px, py = -dy / norm, dx / norm
        # Alternate side each lap; full pitch step so stacked strokes are obvious.
        side = 1.0 if (lap[i] % 2 == 1) else -1.0
        mag = lap[i] * pitch
        out_x[i] += side * mag * px
        out_y[i] += side * mag * py
    return out_x, out_y

def _colored_path(ax, xs, ys, yaws=None, *, cmap="viridis", lw=2.3):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if yaws is not None and len(xs) == len(yaws):
        xs, ys, yaws = _densify_se2(xs, ys, np.asarray(yaws, float))
        xs, ys, yaws = _densify_xy(xs, ys, np.asarray(yaws, float))
        # Fan retraces in workspace xy first; then expand in-place yaw to spirals.
        xs, ys = _fan_retraces(xs, ys)
        xs, ys = _display_xy(xs, ys, yaws)
    pts = np.column_stack([xs, ys])
    if len(pts) < 2:
        ax.plot(xs, ys, color="#2a6fbb", lw=lw)
        return xs, ys, yaws
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidths=lw, alpha=0.95)
    lc.set_array(np.linspace(0, 1, len(segs)))
    ax.add_collection(lc)
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=26, c="#111111", zorder=5)
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=34, c="#c0392b", zorder=5, marker="s")
    return xs, ys, yaws
def _heading_ticks(ax, xs, ys, yaws, *, every_deg: float = 30.0, scale: float = 0.05):
    if len(xs) < 2:
        return
    yaws_u = _unwrap(yaws)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(yaws_u)))])
    targets = np.arange(every_deg * math.pi / 180.0, cum[-1] + 1e-9, every_deg * math.pi / 180.0)
    if len(targets) == 0 and cum[-1] > 1e-3:
        targets = np.array([cum[-1]])
    for t in targets:
        i = int(np.searchsorted(cum, t, side="left"))
        i = min(max(i, 1), len(xs) - 1)
        ang = float(yaws_u[i])
        ax.annotate(
            "",
            xy=(xs[i] + scale * math.cos(ang), ys[i] + scale * math.sin(ang)),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color="#222222", lw=0.7, mutation_scale=8),
            zorder=6,
        )


def _roomba_traj(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = _roomba_states(row, fps=FPS)
    xs = np.array([s[0] for s in states], float)
    ys = np.array([s[1] for s in states], float)
    yaws = np.array([s[2] for s in states], float)
    return xs, ys, yaws


def _stretch_traj(row: dict, model, data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = _stretch_states(row, model, data, fps=FPS)
    xs, ys, yaws = [], [], []
    eex, eey = [], []
    for fr in frames:
        x, y, yaw = float(fr["x"]), float(fr["y"]), float(fr["yaw"])
        xs.append(x)
        ys.append(y)
        yaws.append(yaw)
        _apply(model, data, x=x, y=y, yaw=yaw, joints=fr["joints"])
        ee = _ee(model, data)
        eex.append(float(ee[0]))
        eey.append(float(ee[1]))
    return (
        np.asarray(xs),
        np.asarray(ys),
        np.asarray(yaws),
        np.asarray(eex),
        np.asarray(eey),
    )


def _pretty_code(src: str, width: int = 34) -> str:
    """Wrap Motion Code so staged + terms stay readable under each panel."""
    src = " ".join(src.replace("\n", " ").split())
    parts = [p.strip() for p in src.split("+")]
    lines: list[str] = []
    cur = ""
    for i, part in enumerate(parts):
        piece = part if i == 0 else "+ " + part
        if not cur:
            cur = piece
        elif len(cur) + 1 + len(piece) <= width:
            cur = cur + " " + piece
        else:
            lines.append(cur)
            cur = piece
    if cur:
        lines.append(cur)
    # Hard-wrap any leftover long token line.
    out: list[str] = []
    for ln in lines:
        out.extend(textwrap.wrap(ln, width=width, break_long_words=False, break_on_hyphens=False) or [ln])
    return "\n".join(out)


def _draw_code_box(ax, code: str, *, border: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box = FancyBboxPatch(
        (0.02, 0.06),
        0.96,
        0.88,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0,
        edgecolor=border,
        facecolor="#f7f9fc",
        mutation_aspect=0.4,
    )
    ax.add_patch(box)
    ax.text(
        0.06,
        0.92,
        "MOTION CODE",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        fontweight="bold",
        color="#333333",
        family="sans-serif",
    )
    ax.text(
        0.06,
        0.72,
        code,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.7,
        family="monospace",
        color="#111111",
        linespacing=1.25,
    )


def main() -> None:
    rows_all = json.loads(GT_JSON.read_text())
    by_cue = {e["cue"]: e for e in rows_all}

    model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
    data = mujoco.MjData(model)

    fig, axes = plt.subplots(2, 5, figsize=(11.2, 3.55), dpi=200, gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.subplots_adjust(left=0.03, right=0.99, top=0.90, bottom=0.04, wspace=0.14, hspace=0.18)

    for j, (plat, cue, title) in enumerate(CELLS):
        entry = by_cue[cue]
        row = dict(entry["platforms"][plat])
        row["idx"] = entry["idx"]
        row["cue"] = cue
        ax_tr, ax_code = axes[0, j], axes[1, j]
        border = "#4a7c59" if plat == "stretch" else "#2a6fbb"

        ax_tr.set_title(title, fontsize=7.4, pad=3)
        if plat == "roomba":
            xs, ys, yaws = _roomba_traj(row)
            dx, dy, dyaw = _colored_path(ax_tr, xs, ys, yaws)
            _heading_ticks(ax_tr, dx, dy, dyaw)
            ax_tr.set_xlabel("base + yaw", fontsize=6.5, labelpad=1)
        else:
            xs, ys, yaws, eex, eey = _stretch_traj(row, model, data)
            dx, dy, dyaw = _colored_path(ax_tr, xs, ys, yaws, cmap="viridis")
            _heading_ticks(ax_tr, dx, dy, dyaw)
            yaw_travel = float(np.abs(_unwrap(yaws)[-1] - _unwrap(yaws)[0]))
            ee_travel = float(np.ptp(eex) + np.ptp(eey))
            if yaw_travel < math.radians(120) and ee_travel > 0.04:
                eex_b = eex - eex[0] + xs[0]
                eey_b = eey - eey[0] + ys[0]
                ax_tr.plot(eex_b, eey_b, color="#e67e22", lw=1.35, alpha=0.9, ls="--", zorder=3)
                ax_tr.set_xlabel("base+yaw; EE dashed", fontsize=6.2, labelpad=1)
            else:
                ax_tr.set_xlabel("base + yaw", fontsize=6.5, labelpad=1)

        ax_tr.set_aspect("equal", adjustable="datalim")
        ax_tr.margins(0.22)
        ax_tr.set_xticks([])
        ax_tr.set_yticks([])
        for spine in ax_tr.spines.values():
            spine.set_color(border)
            spine.set_linewidth(0.9)

        src = row.get("source_code") or ""
        if not src:
            # Fallback: reconstruct a short readable chain from movements.
            bits = []
            for mv in row.get("movements", []):
                bits.append(mv.get("type", "?"))
            src = " + ".join(bits) if bits else "(missing source_code)"
        _draw_code_box(ax_code, _pretty_code(src), border=border)

        print(f"{plat:7s} {cue:22s} path={len(xs):3d} yawΔ={np.rad2deg(_unwrap(yaws)[-1]-_unwrap(yaws)[0]):7.1f}°")

    axes[0, 0].set_ylabel("path", fontsize=8)
    axes[1, 0].set_ylabel("code", fontsize=8)
    out = PAPER / "figure_roomba_context_traj.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.04)
    print("wrote", out, out.stat().st_size)


if __name__ == "__main__":
    main()
