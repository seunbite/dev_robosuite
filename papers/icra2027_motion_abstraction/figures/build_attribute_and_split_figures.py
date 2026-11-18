#!/usr/bin/env python3
"""Build the three-attribute overview figure (Venn + cue table)."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

ROOT = Path(__file__).resolve().parent


def _trim_white(im: Image.Image, *, pad: int = 1) -> Image.Image:
    arr = np.asarray(im.convert("RGB"))
    ink = (arr < 250).any(axis=2)
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return im
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(im.width, int(xs.max()) + 1 + pad),
        min(im.height, int(ys.max()) + 1 + pad),
    )
    return im.crop(box)


def _render_table(fs: float, dpi: int) -> Image.Image:
    """Compact table; column widths follow content."""
    n_rows = 11  # header + 10
    # Tighter rows: font fills more of each cell (less top/bottom pad).
    row_in = (fs / 72.0) / 0.62
    fig_h = n_rows * row_in
    headers = ["#", "Cue", "1 Orientation", "2 Location", "3 Movement"]
    data = [
        ["1", "nod yes", "—", "—", "head down/up (z±)"],
        ["2", "gaze avoid", "gaze away", "—", "turn away (y±)"],
        ["3", "point self", "index inward", "at own chest", "—"],
        ["4", "stop", "palm outward", "chest height", "—"],
        ["5", "salute", "palm outward", "at forehead", "raise/hold (z+)"],
        ["6", "fist pump", "closed fist", "above shoulder", "short thrust (z±)"],
        ["7", "beckon", "palm up/in", "front", "pull inward (x−)"],
        ["8", "air quotes", "two fingers", "upper torso", "finger flex (z±)"],
        ["9", "big bowl", "open curve", "front, low", "out/down (y±,z−)"],
        ["10", "flying bird", "wing-like span", "arms lateral", "flap (z±)"],
    ]
    # Approximate content-driven relative widths (inches at this fontsize).
    cols = list(zip(*([headers] + data)))
    char_w = []
    for col in cols:
        char_w.append(max(len(str(c)) for c in col) + 2)
    # Slightly boost numeric / short cue cols floor.
    char_w[0] = max(char_w[0], 3)
    total = sum(char_w)
    col_widths = [w / total for w in char_w]
    # Physical width from character budget (tight padding).
    fig_w = (sum(char_w) * fs * 0.52) / 72.0
    fig_w = max(fig_w, fig_h * 0.95)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    ax.axis("off")
    table = ax.table(
        cellText=data,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fs)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B9B9B9")
        cell.set_linewidth(0.65)
        cell.PAD = 0.002
        if row == 0:
            cell.set_facecolor("#E9EEF2")
            cell.set_text_props(weight="bold", fontsize=fs)
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba).convert("RGB")


def _render_venn(fs: float, target_h_px: int, dpi: int) -> Image.Image:
    """Venn rendered at exactly target_h_px tall (matches table height)."""
    r_prev = 2.85 * (5.0 / 6.0)
    r = r_prev * (8.0 / 9.0)
    d = 1.48 * r
    R_c = d / math.sqrt(3)
    cx, cy = 0.0, 0.0
    angles = (math.radians(90), math.radians(210), math.radians(330))
    colors = ("#77B8D1", "#E5B7C3", "#E8BF84")
    labels = (
        ("1  Orientation", "reach, palm,\ngrip / handshape"),
        ("2  Location", "body-relative target\n(face, chest, overhead)"),
        ("3  Movement", "signed direction\nand path"),
    )
    centers = [(cx + R_c * math.cos(a), cy + R_c * math.sin(a)) for a in angles]

    top_y = max(y + r for _, y in centers)
    bot_y = min(y - r for _, y in centers)
    left_x = min(x - r for x, _ in centers)
    right_x = max(x + r for x, _ in centers)
    title_gap = 0.08 * r
    title_h = 0.28 * r
    content_top = top_y + title_gap + title_h
    content_bot = bot_y
    content_h = content_top - content_bot
    content_w = right_x - left_x
    pad = 0.02 * content_h
    mid_x = 0.5 * (left_x + right_x)

    # Figure size from content aspect, height locked to table.
    aspect = (content_w + 2 * pad) / (content_h + 2 * pad)
    fig_h_in = target_h_px / float(dpi)
    fig_w_in = fig_h_in * aspect
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=dpi, facecolor="white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(mid_x - 0.5 * (content_w + 2 * pad), mid_x + 0.5 * (content_w + 2 * pad))
    ax.set_ylim(content_bot - pad, content_top + pad)

    ax.text(
        cx,
        top_y + title_gap,
        "Recognition-critical attributes",
        ha="center",
        va="bottom",
        fontsize=fs,
        weight="bold",
    )

    # Exclusive-lobe placement: outward, still inside each disk.
    text_push = 0.34 * r
    for (x, y), ang, color, (title, subtitle) in zip(centers, angles, colors, labels):
        ax.add_patch(Circle((x, y), r, color=color, alpha=0.72, ec="#4A4A4A", lw=1.0))
        tx = x + text_push * math.cos(ang)
        ty = y + text_push * math.sin(ang)
        ax.text(
            tx,
            ty + 0.09 * r,
            title,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold",
        )
        ax.text(
            tx,
            ty - 0.20 * r,
            subtitle,
            ha="center",
            va="center",
            fontsize=fs,
            linespacing=1.05,
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    im = Image.fromarray(rgba).convert("RGB")
    # Force exact pixel height (canvas may be ±1).
    if im.height != target_h_px:
        new_w = max(1, int(round(im.width * target_h_px / im.height)))
        im = im.resize((new_w, target_h_px), Image.Resampling.LANCZOS)
    return im


def build_attribute_figure() -> None:
    dpi = 300
    # Table font stays compact; Venn ~2× larger.
    fs_table = 9.4
    fs_venn = fs_table * 2.0

    table = _trim_white(_render_table(fs_table, dpi), pad=1)
    # Venn at ~9/10 the table height, vertically centered.
    venn_h = max(1, int(round(table.height * 0.9)))
    venn = _trim_white(_render_venn(fs_venn, venn_h, dpi), pad=1)
    if venn.height != venn_h:
        new_w = max(1, int(round(venn.width * venn_h / venn.height)))
        venn = venn.resize((new_w, venn_h), Image.Resampling.LANCZOS)

    gap = 4  # tiny gutter between Venn and table
    out_im = Image.new("RGB", (venn.width + gap + table.width, table.height), "white")
    venn_y = (table.height - venn.height) // 2
    out_im.paste(venn, (0, venn_y))
    out_im.paste(table, (venn.width + gap, 0))
    out_im = _trim_white(out_im, pad=1)

    out = ROOT / "figure_critical_attributes.png"
    out_im.save(out, optimize=True)
    print(
        f"wrote {out} size={out_im.size} "
        f"fs_table={fs_table} fs_venn={fs_venn} "
        f"venn={venn.size} table={table.size}"
    )


if __name__ == "__main__":
    build_attribute_figure()
