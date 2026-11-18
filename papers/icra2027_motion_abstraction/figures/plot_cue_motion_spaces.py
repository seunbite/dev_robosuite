#!/usr/bin/env python3
"""Conceptual cue ↔ motion space figures (generation, recognition, context, morphology)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(7)

RECOG_CMAP = LinearSegmentedColormap.from_list(
    "recog",
    ["#ffffff", "#fff6c8", "#ffe07a", "#ffb347", "#f07a1a", "#d94801", "#9b1c1c", "#6b0f12"],
    N=256,
)
RECOG_CMAP.set_bad("#ffffff")
# Weaker field: yellow→orange only (ambiguous / hard to pin a meaning).
SOFT_RECOG_CMAP = LinearSegmentedColormap.from_list(
    "recog_soft",
    ["#ffffff", "#fff9db", "#ffe9a0", "#ffd060", "#f5a84a", "#e67e22"],
    N=256,
)
SOFT_RECOG_CMAP.set_bad("#ffffff")

CUE_FILL = "#d7ecf6"
CUE_EDGE = "#8fb4c8"
INK = "#222222"
MUTED = "#9aa7b2"
TW = 7.16
# ieeeconf column width; Fig. 3/4 are drawn 1:1 for \columnwidth.
COL = 3.39
FS_SPACE = 10.0
FS_CUE = 8.5
FS_NOTE = 8.5
FS_TITLE = 11.0
FS_CHANNEL = 8.5
DOT_BG = 22
DOT_HI = 78
# Unified column-figure text size (= panel (a)/(b) title).
COL_FS = 6.0
COL_FS_SPACE = COL_FS
COL_FS_CUE = COL_FS
COL_FS_TITLE = COL_FS
COL_FS_CHANNEL = COL_FS
COL_DOT_BG = 12
COL_DOT_HI = 28


def _circle_xy(n: int, radius: float = 0.62) -> np.ndarray:
    pts = []
    while len(pts) < n:
        p = RNG.uniform(-1, 1, size=2)
        if np.hypot(*p) < 0.92:
            pts.append(p * radius)
    return np.asarray(pts)


CUES = _circle_xy(30)


def mixture(xx: np.ndarray, yy: np.ndarray, means, sigmas, weights) -> np.ndarray:
    z = np.zeros_like(xx)
    for (mx, my), s, w in zip(means, sigmas, weights):
        z += w * np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2.0 * s**2))
    return z


def motion_field(kind: str, n: int = 420) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return X, Y, Z, radius. Z is a schematic recognizability field."""
    if kind == "humanoid":
        radius = 1.12
        means = [
            (-0.46, 0.42),
            (0.48, 0.36),
            (-0.10, -0.52),
            (0.58, -0.32),
            (-0.62, -0.12),
            (0.08, 0.68),
            (0.22, -0.08),
        ]
        sigmas = [0.13, 0.12, 0.13, 0.11, 0.12, 0.10, 0.12]
        weights = [1.20, 1.15, 0.72, 0.58, 0.50, 0.48, 0.42]
    else:
        radius = 0.74
        means = [(0.0, 0.0)]
        sigmas = [0.22]
        weights = [1.0]
    lim = radius + 0.05
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    xx, yy = np.meshgrid(xs, ys)
    zz = mixture(xx, yy, means, sigmas, weights)
    zz /= zz.max()
    if kind == "roomba":
        zz = np.clip(zz, 0, 1)  # single central core; keep contrast
    rr = np.hypot(xx, yy)
    zz = np.where(rr <= radius, zz, np.nan)
    return xx, yy, zz, radius


def draw_cue_space(
    ax,
    cx: float,
    cy: float,
    radius: float,
    highlights: list[dict],
    dim: np.ndarray | None = None,
    *,
    fs_label: float | None = None,
    fs_space: float | None = None,
    s_bg: float | None = None,
    s_hi: float | None = None,
    space_gap: float = 0.16,
    space_name: str = "meaning space",
):
    fs_label = FS_CUE if fs_label is None else fs_label
    fs_space = FS_SPACE if fs_space is None else fs_space
    s_bg = DOT_BG if s_bg is None else s_bg
    s_hi = DOT_HI if s_hi is None else s_hi
    ax.add_patch(
        Circle((cx, cy), radius, facecolor=CUE_FILL, edgecolor=CUE_EDGE, lw=1.5, zorder=3)
    )
    pts = CUES * (radius / 0.62)
    hi_idx = {h["idx"] for h in highlights}
    alpha = np.full(len(pts), 0.72)
    colors = np.full(len(pts), MUTED)
    if dim is not None:
        alpha = np.where(dim, 0.16, 0.85)
        colors = np.where(dim, "#c5ced4", MUTED)
    # Fewer background dots so labels stay readable after shrink.
    bg = [i for i in range(len(pts)) if i not in hi_idx and i % 2 == 0]
    ax.scatter(
        cx + pts[bg, 0],
        cy + pts[bg, 1],
        s=s_bg,
        c=colors[bg],
        alpha=alpha[bg],
        zorder=3,
        linewidths=0,
    )
    for h in highlights:
        px, py = pts[h["idx"]]
        ax.scatter(
            cx + px,
            cy + py,
            s=s_hi,
            c=h["color"],
            edgecolors="white",
            linewidths=1.4,
            zorder=5,
            alpha=h.get("alpha", 1.0),
        )
        # Prefer absolute data-space label placement (no leader line).
        if "xy" in h:
            ax.text(
                h["xy"][0],
                h["xy"][1],
                h["label"],
                ha=h.get("ha", "center"),
                va=h.get("va", "center"),
                fontsize=fs_label,
                color=h["color"],
                alpha=h.get("alpha", 1.0),
                zorder=9,
                clip_on=False,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.94),
                linespacing=0.95,
            )
        else:
            ax.annotate(
                h["label"],
                (cx + px, cy + py),
                xytext=h.get("text", (-16, 14)),
                textcoords="offset points",
                ha=h.get("ha", "right"),
                va=h.get("va", "bottom"),
                fontsize=fs_label,
                color=h["color"],
                alpha=h.get("alpha", 1.0),
                zorder=6,
                clip_on=False,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.94),
                linespacing=0.95,
            )
    ax.text(
        cx,
        cy - radius - space_gap,
        space_name,
        ha="center",
        va="top",
        fontsize=fs_space,
        color=INK,
        clip_on=False,
    )
    return pts


def draw_motion_space(
    ax,
    cx: float,
    cy: float,
    kind: str,
    peaks: list[tuple[float, float]],
    *,
    scale: float = 1.0,
    fs_space: float | None = None,
    space_gap: float = 0.16,
    space_name: str = "action space",
    soft: bool = False,
):
    fs_space = FS_SPACE if fs_space is None else fs_space
    xx, yy, zz, radius = motion_field(kind)
    xx, yy, radius = xx * scale, yy * scale, radius * scale
    if soft:
        zz = np.clip(0.55 * zz, 0, 1)
        cmap = SOFT_RECOG_CMAP
        contour_c, contour_a = "#c47a2a", 0.22
    else:
        cmap = RECOG_CMAP
        contour_c, contour_a = "#8a2a12", 0.32
    ax.contourf(cx + xx, cy + yy, zz, levels=16, cmap=cmap, vmin=0, vmax=1, zorder=1)
    ax.contour(
        cx + xx,
        cy + yy,
        zz,
        levels=8,
        colors=contour_c,
        linewidths=0.32,
        alpha=contour_a,
        zorder=2,
    )
    ax.add_patch(
        Circle((cx, cy), radius, facecolor="none", edgecolor="#6b4a32", lw=1.5, zorder=4)
    )
    for px, py in peaks:
        ax.scatter(
            cx + px * scale,
            cy + py * scale,
            s=28 if fs_space <= 8.5 else 64,
            facecolors="white",
            edgecolors="#333333",
            linewidths=0.9,
            zorder=6,
        )
    ax.text(
        cx,
        cy - radius - space_gap,
        space_name,
        ha="center",
        va="top",
        fontsize=fs_space,
        color=INK,
        clip_on=False,
    )
    return radius


def arrow(ax, p0, p1, color="#4a4a4a"):
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=10,
        lw=1.05,
        linestyle=(0, (4.0, 2.4)),
        color=color,
        zorder=7,
        connectionstyle="arc3,rad=0.04",
    )
    ax.add_patch(patch)


def panel_frame(ax, title: str, xlim, ylim, *, fontsize=None, pad=6, anchor: str = "C"):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # Equal aspect; size the figure to match so box does not leave large white gutters.
    ax.set_aspect("equal", adjustable="box", anchor=anchor)
    ax.axis("off")
    ax.set_title(
        title,
        fontsize=FS_TITLE if fontsize is None else fontsize,
        pad=pad,
        color=INK,
        fontweight="normal",
    )


def save(fig, stem: str, *, tight: bool = True):
    png_kw: dict = dict(dpi=300, facecolor="white")
    pdf_kw: dict = dict(facecolor="white")
    if tight:
        png_kw.update(bbox_inches="tight", pad_inches=0.01)
        pdf_kw.update(bbox_inches="tight", pad_inches=0.01)
    png_path = OUT / f"{stem}.png"
    pdf_path = OUT / f"{stem}.pdf"
    fig.savefig(png_path, **png_kw)
    fig.savefig(pdf_path, **pdf_kw)
    plt.close(fig)
    if tight:
        _trim_png_whitespace(png_path, pad=4)
    print("wrote", stem)


def _trim_png_whitespace(path, *, pad: int = 4, thresh: float = 0.98):
    """Crop near-white margins left by equal-aspect axes boxes."""
    import matplotlib.pyplot as plt

    im = plt.imread(path)
    if im.ndim == 3:
        gray = im[..., :3].mean(axis=-1)
        alpha = im[..., 3] if im.shape[-1] == 4 else None
    else:
        gray = im
        alpha = None
    content = gray < thresh
    if alpha is not None:
        content |= alpha < 0.99
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return
    r0, r1 = max(0, rows[0] - pad), min(im.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(im.shape[1], cols[-1] + 1 + pad)
    cropped = im[r0:r1, c0:c1]
    plt.imsave(path, cropped)


def _wrap_cue_label(name: str) -> str:
    if "_" not in name or len(name) <= 8:
        return name
    parts = name.split("_")
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return f"{'_'.join(parts[:-1])}\n{parts[-1]}"


def _pair_highlights(kind: str) -> tuple[list[dict], list[tuple[float, float]]]:
    if kind == "humanoid":
        hi = [
            {"idx": 4, "color": "#3b82c4", "label": "cover_yawn_sleepy"},
            {"idx": 18, "color": "#7b4ea3", "label": "rub_eye_tired"},
        ]
        peaks = [(-0.46, 0.42), (0.48, 0.36)]
        return hi, peaks
    hi = [
        {"idx": 7, "color": "#c0392b", "label": "shake_head_no"},
        {"idx": 18, "color": "#e67e22", "label": "rub_eye_tired"},
    ]
    peaks = [(-0.17, 0.05), (0.18, 0.03)]
    return hi, peaks


def draw_pair(ax, kind: str, *, generate: bool, fontsize: float = 15.0) -> None:
    """Identical geometry for generation and recognition; only arrow direction changes."""
    cue_r = 0.70
    if kind == "humanoid":
        # Extra left margin so large cue labels sit outside the meaning circle.
        cue_c, mot_c, xlim = (-1.38, 0.08), (1.18, 0.08), (-3.35, 2.55)
    else:
        cue_c, mot_c, xlim = (-1.32, 0.08), (1.02, 0.08), (-3.15, 2.25)
    panel_frame(ax, "", xlim, (-1.55, 1.42))
    hi, peaks = _pair_highlights(kind)
    # Labels entirely left of the circle; ha=right so the box's right edge meets the rim.
    y_offs = (0.28, -0.32)
    placed = []
    for h, y_off in zip(hi, y_offs):
        placed.append(
            {
                **h,
                "label": _wrap_cue_label(h["label"]),
                "xy": (cue_c[0] - cue_r - 0.04, cue_c[1] + y_off),
                "ha": "right",
                "va": "center",
            }
        )
    cue_pts = draw_cue_space(
        ax, cue_c[0], cue_c[1], cue_r, placed, fs_label=fontsize, fs_space=fontsize
    )
    draw_motion_space(ax, mot_c[0], mot_c[1], kind, peaks, fs_space=fontsize)
    if kind == "roomba" and not generate:
        origin = (mot_c[0], mot_c[1] + 0.02)
        for h in hi:
            p = cue_pts[h["idx"]]
            arrow(ax, origin, (cue_c[0] + p[0] + 0.10, cue_c[1] + p[1]))
        return
    for h, peak in zip(hi, peaks):
        p = cue_pts[h["idx"]]
        src = (cue_c[0] + p[0] + 0.08, cue_c[1] + p[1])
        dst = (mot_c[0] + peak[0] - 0.10, mot_c[1] + peak[1])
        if generate:
            arrow(ax, src, dst)
        else:
            arrow(ax, (dst[0] + 0.02, dst[1]), (src[0] + 0.02, src[1]))


def fig_generation():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.15))
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.99, top=0.90, bottom=0.10)
    draw_pair(axes[0], "humanoid", generate=True)
    draw_pair(axes[1], "roomba", generate=True)
    axes[0].set_title("Humanoid", fontsize=15, pad=8, color=INK)
    axes[1].set_title("Roomba", fontsize=15, pad=8, color=INK)
    save(fig, "fig_generation")


def fig_recognition():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.15))
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.99, top=0.90, bottom=0.10)
    draw_pair(axes[0], "humanoid", generate=False)
    draw_pair(axes[1], "roomba", generate=False)
    axes[0].set_title("Humanoid", fontsize=15, pad=8, color=INK)
    axes[1].set_title("Roomba", fontsize=15, pad=8, color=INK)
    save(fig, "fig_recognition")


def fig_spaces():
    """Duplicate the Humanoid morphology panel with opposite arrow directions."""
    style = _col_style()
    cue_c, mot_c, cue_r, mot_scale = _pair_centers("humanoid")
    mot_r = 1.12 * mot_scale
    xlim = (cue_c[0] - cue_r - 0.02, mot_c[0] + mot_r + 0.02)
    ylim = (-max(cue_r, mot_r) - 0.20, max(cue_r, mot_r) + 0.16)
    dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
    wspace, top, bottom = 0.05, 0.86, 0.01
    fig_h = COL * ((1.0 - wspace) / 2.0) * (dy / dx) / (top - bottom)
    fig, axes = plt.subplots(1, 2, figsize=(COL, fig_h), gridspec_kw={"wspace": wspace})
    fig.subplots_adjust(left=0.0, right=1.0, top=top, bottom=bottom, wspace=wspace)

    for ax, generate in zip(axes, (True, False)):
        title = (
            r"(a) Generation" + "\n" + r"$p(A\mid G,M)$"
            if generate
            else r"(b) Recognition" + "\n" + r"$\hat{G}=\arg\max_G p(U{=}G\mid A,M)$"
        )
        panel_frame(ax, title, xlim, ylim, fontsize=COL_FS, pad=0.5, anchor="C")
        highlights = _cue_labels_on_circle(
            cue_c,
            cue_r,
            "shake_head_no",
            "wave_hi",
            "#c0392b",
            "#2e86ab",
            7,
            11,
        )
        cue_pts = draw_cue_space(
            ax, cue_c[0], cue_c[1], cue_r, highlights, space_name="meaning", **style
        )
        draw_motion_space(
            ax,
            mot_c[0],
            mot_c[1],
            "humanoid",
            [(0.48, 0.36), (-0.46, 0.42)],
            scale=mot_scale,
            fs_space=COL_FS,
            space_gap=0.14,
            space_name="action",
        )
        mappings = (
            (11, (0.48, 0.36), "#2e86ab", "arm"),
            (7, (-0.46, 0.42), "#c0392b", "head"),
        )
        for idx, peak, color, channel in mappings:
            meaning = (cue_c[0] + cue_pts[idx][0] + 0.06, cue_c[1] + cue_pts[idx][1])
            action = (
                mot_c[0] + peak[0] * mot_scale,
                mot_c[1] + peak[1] * mot_scale,
            )
            arrow(ax, meaning, action, color=color) if generate else arrow(
                ax, action, meaning, color=color
            )
            _channel_label(ax, action[0], action[1] + 0.08, channel, color)

    save(fig, "fig_spaces")


def _context_chip(ax, x, y, text, *, w=1.55, h=0.48, fs=None):
    """Context utterance chip. `w`/`h` ignored — bbox auto-sizes to the text."""
    del w, h
    fs = COL_FS if fs is None else fs
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        color="#2c5770",
        zorder=11,
        clip_on=False,
        linespacing=1.2,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#eef6fb",
            edgecolor="#7aa0b5",
            lw=1.2,
        ),
    )


def _channel_label(ax, x, y, text, color, fs=None):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=COL_FS if fs is None else fs,
        color=color,
        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.9),
        zorder=8,
        clip_on=False,
    )


def _col_style() -> dict:
    return dict(
        fs_label=COL_FS,
        fs_space=COL_FS,
        s_bg=COL_DOT_BG,
        s_hi=COL_DOT_HI,
        space_gap=0.14,
    )


def _pair_centers(kind: str) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Side-by-side meaning / action circles; no overlap.

    meaning radius = 4/5 of prior column cue_r; Roomba action radius = 1/2 of prior.
    """
    cue_r = 1.12 * 0.8  # 4/5 of previous meaning radius
    # Humanoid action unchanged; Roomba action radius halved vs previous 1.45 scale.
    mot_scale = 1.00 if kind == "humanoid" else (1.45 * 0.5)
    mot_r = (1.12 if kind == "humanoid" else 0.74) * mot_scale
    gap = 0.26
    cue_c = (-(cue_r + gap / 2.0), 0.0)
    mot_c = (mot_r + gap / 2.0, 0.0)
    return cue_c, mot_c, cue_r, mot_scale


def _panel_limits(cue_c, mot_c, cue_r, mot_r, *, headroom: float = 0.22):
    # Labels / context chip sit on the circles — keep margins tight.
    return (
        cue_c[0] - cue_r - 0.02,
        mot_c[0] + mot_r + 0.02,
        min(cue_c[1] - cue_r, mot_c[1] - mot_r) - 0.22,
        max(cue_c[1] + cue_r, mot_c[1] + mot_r) + headroom,
    )


def _col_layout(fig, xlim, ylim, *, top: float = 0.90, bottom: float = 0.02, wspace: float = 0.06):
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    # Height from data aspect for two side-by-side panels (no forced min height).
    panel_frac = (1.0 - wspace) / 2.0
    fig_h = COL * panel_frac * (dy / dx) / max(top - bottom, 0.5)
    fig.set_size_inches(COL, fig_h)
    fig.subplots_adjust(wspace=wspace, left=0.0, right=1.0, top=top, bottom=bottom)


def _cue_labels_on_circle(cue_c, cue_r, top_label, bot_label, top_color, bot_color, top_idx, bot_idx):
    """Place cue names on the circle (top / lower), not to the left.

    Long cue ids are wrapped so large title-sized type fits inside the circle
    diameter without stealing left-side figure width.
    """

    def _wrap(name: str) -> str:
        if "_" not in name or len(name) <= 8:
            return name
        parts = name.split("_")
        if len(parts) == 2:
            return f"{parts[0]}\n{parts[1]}"
        # e.g. shake_head_no → shake_head / no ; rub_eye_tired → rub_eye / tired
        return f"{'_'.join(parts[:-1])}\n{parts[-1]}"

    return [
        {
            "idx": top_idx,
            "color": top_color,
            "label": _wrap(top_label),
            "xy": (cue_c[0], cue_c[1] + cue_r * 0.55),
            "ha": "center",
            "va": "center",
        },
        {
            "idx": bot_idx,
            "color": bot_color,
            "label": _wrap(bot_label),
            "xy": (cue_c[0], cue_c[1] - cue_r * 0.35),
            "ha": "center",
            "va": "center",
        },
    ]


def fig_contextualize():
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.7))
    style = _col_style()
    peaks = [(0.0, 0.0)]  # single central action peak
    cue_c, mot_c, cue_r, mot_scale = _pair_centers("roomba")
    mot_r = 0.74 * mot_scale
    # Orange shake / red rub; label boxes swapped (colors/arrows unchanged).
    r_hi = [
        {
            "idx": 7,
            "color": "#e67e22",
            "label": "shake_head\nno",
            "xy": (cue_c[0], cue_c[1] - cue_r * 0.35),
            "ha": "center",
            "va": "center",
        },
        {
            "idx": 18,
            "color": "#c0392b",
            "label": "rub_eye\ntired",
            "xy": (cue_c[0], cue_c[1] + cue_r * 0.42),
            "ha": "center",
            "va": "center",
        },
    ]
    xlim = _panel_limits(cue_c, mot_c, cue_r, mot_r, headroom=0.12)
    ylim = xlim[2:]
    xlim = xlim[:2]
    _col_layout(fig, xlim, ylim, top=0.94, bottom=0.01, wspace=0.04)

    ax = axes[0]
    panel_frame(ax, r"(a) Without $C$", xlim, ylim, fontsize=COL_FS, pad=0.5)
    cue_pts = draw_cue_space(ax, cue_c[0], cue_c[1], cue_r, r_hi, space_name="meaning", **style)
    draw_motion_space(
        ax,
        mot_c[0],
        mot_c[1],
        "roomba",
        peaks,
        scale=mot_scale,
        fs_space=COL_FS,
        space_gap=0.14,
        space_name="action",
        soft=True,
    )
    origin = (mot_c[0], mot_c[1])
    for h in r_hi:
        p = cue_pts[h["idx"]]
        arrow(ax, origin, (cue_c[0] + p[0] + 0.06, cue_c[1] + p[1]), color=h["color"])

    ax = axes[1]
    panel_frame(ax, r"(b) With $C$", xlim, ylim, fontsize=COL_FS, pad=0.5)
    # Only shake_head_no remains (orange); no faded competing label.
    dim = np.array([i not in {7} for i in range(30)])
    only_shake = [
        {
            "idx": 7,
            "color": "#e67e22",
            "label": "shake_head\nno",
            "xy": (cue_c[0], cue_c[1] - cue_r * 0.20),
            "ha": "center",
            "va": "center",
            "alpha": 1.0,
        },
    ]
    cue_pts = draw_cue_space(ax, cue_c[0], cue_c[1], cue_r, only_shake, dim=dim, space_name="meaning", **style)
    draw_motion_space(
        ax,
        mot_c[0],
        mot_c[1],
        "roomba",
        peaks,
        scale=mot_scale,
        fs_space=COL_FS,
        space_gap=0.14,
        space_name="action",
    )
    p = cue_pts[7]
    arrow(ax, (mot_c[0] + 0.08, mot_c[1]), (cue_c[0] + p[0] + 0.06, cue_c[1] + p[1]), color="#e67e22")
    # Context chip overlaps the upper meaning circle (where shake sat in (a)).
    _context_chip(
        ax,
        cue_c[0],
        cue_c[1] + cue_r * 0.55,
        'C: "Are you\nfully charged?"',
        fs=COL_FS,
    )

    save(fig, "fig_contextualize", tight=True)


def fig_morphology_contextualize():
    style = _col_style()

    layouts = {}
    for kind in ("humanoid", "roomba"):
        cue_c, mot_c, cue_r, mot_scale = _pair_centers(kind)
        mot_r = (1.12 if kind == "humanoid" else 0.74) * mot_scale
        x_lo = cue_c[0] - cue_r - 0.02
        x_hi = mot_c[0] + mot_r + 0.02
        layouts[kind] = dict(
            cue_c=cue_c,
            mot_c=mot_c,
            cue_r=cue_r,
            mot_scale=mot_scale,
            mot_r=mot_r,
            xlim=(x_lo, x_hi),
            dx=x_hi - x_lo,
        )

    # Shared vertical window; panel widths ∝ dx so data scale (and meaning size) match.
    cue_r0 = layouts["humanoid"]["cue_r"]
    r_max = max(cue_r0, max(L["mot_r"] for L in layouts.values()))
    ylim = (-r_max - 0.20, r_max + 0.16)
    dy = ylim[1] - ylim[0]
    dx_h, dx_r = layouts["humanoid"]["dx"], layouts["roomba"]["dx"]

    wspace = 0.04
    top, bottom = 0.92, 0.01
    fig_h = COL * ((1.0 - wspace) * dy / (dx_h + dx_r)) / max(top - bottom, 0.5)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(COL, fig_h),
        gridspec_kw={"width_ratios": [dx_h, dx_r], "wspace": wspace},
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=top, bottom=bottom, wspace=wspace)

    ax = axes[0]
    L = layouts["humanoid"]
    cue_c, mot_c, cue_r, mot_scale = L["cue_c"], L["mot_c"], L["cue_r"], L["mot_scale"]
    panel_frame(ax, r"(a) Humanoid", L["xlim"], ylim, fontsize=COL_FS, pad=0.5, anchor="C")
    h_hi = _cue_labels_on_circle(
        cue_c,
        cue_r,
        "shake_head_no",
        "wave_hi",
        "#c0392b",
        "#2e86ab",
        7,
        11,
    )
    cue_pts = draw_cue_space(ax, cue_c[0], cue_c[1], cue_r, h_hi, space_name="meaning", **style)
    peaks = [(0.48, 0.36), (-0.46, 0.42)]
    draw_motion_space(
        ax,
        mot_c[0],
        mot_c[1],
        "humanoid",
        peaks,
        scale=mot_scale,
        fs_space=COL_FS,
        space_gap=0.14,
        space_name="action",
    )
    p_wave = cue_pts[11]
    p_shake = cue_pts[7]
    arrow(
        ax,
        (mot_c[0] + 0.48 * mot_scale, mot_c[1] + 0.36 * mot_scale),
        (cue_c[0] + p_wave[0] + 0.06, cue_c[1] + p_wave[1]),
        color="#2e86ab",
    )
    arrow(
        ax,
        (mot_c[0] - 0.46 * mot_scale, mot_c[1] + 0.42 * mot_scale),
        (cue_c[0] + p_shake[0] + 0.06, cue_c[1] + p_shake[1]),
        color="#c0392b",
    )
    _channel_label(ax, mot_c[0] + 0.48 * mot_scale, mot_c[1] + 0.36 * mot_scale + 0.10, "arm", "#2e86ab")
    _channel_label(ax, mot_c[0] - 0.46 * mot_scale, mot_c[1] + 0.42 * mot_scale + 0.08, "head", "#c0392b")

    ax = axes[1]
    L = layouts["roomba"]
    cue_c, mot_c, cue_r, mot_scale = L["cue_c"], L["mot_c"], L["cue_r"], L["mot_scale"]
    panel_frame(ax, r"(b) Roomba", L["xlim"], ylim, fontsize=COL_FS, pad=0.5, anchor="C")
    r_hi = _cue_labels_on_circle(
        cue_c,
        cue_r,
        "shake_head_no",
        "wave_hi",
        "#c0392b",
        "#2e86ab",
        7,
        11,
    )
    cue_pts = draw_cue_space(ax, cue_c[0], cue_c[1], cue_r, r_hi, space_name="meaning", **style)
    peaks = [(0.0, 0.0)]
    draw_motion_space(
        ax,
        mot_c[0],
        mot_c[1],
        "roomba",
        peaks,
        scale=mot_scale,
        fs_space=COL_FS,
        space_gap=0.14,
        space_name="action",
        soft=True,
    )
    origin = (mot_c[0], mot_c[1])
    for h in r_hi:
        p = cue_pts[h["idx"]]
        arrow(ax, origin, (cue_c[0] + p[0] + 0.06, cue_c[1] + p[1]), color=h["color"])

    save(fig, "fig_morphology_contextualize", tight=True)


if __name__ == "__main__":
    import sys

    targets = sys.argv[1:] or ["all"]
    if "all" in targets:
        fig_generation()
        fig_recognition()
        fig_spaces()
        fig_contextualize()
        fig_morphology_contextualize()
    else:
        fns = {
            "generation": fig_generation,
            "recognition": fig_recognition,
            "spaces": fig_spaces,
            "contextualize": fig_contextualize,
            "morphology": fig_morphology_contextualize,
        }
        for t in targets:
            fns[t]()
