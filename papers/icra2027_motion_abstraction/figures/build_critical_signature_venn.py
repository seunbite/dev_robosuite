#!/usr/bin/env python3
"""Build the column-width cue-signature Venn diagram used in the paper."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


OUT = Path(__file__).with_name("figure_critical_signature_venn.png")
DPI = 600
W, H = 2100, 1500
X_MIN, X_MAX = 0.0, 10.0
Y_MAX = 7.45
SX, SY = W / (X_MAX - X_MIN), H / Y_MAX

DEFAULT_COLORS = {"O": "#0F8F83", "L": "#7C4DCC", "M": "#64748B"}
BAR_CHART_COLORS = {"O": "#2870E8", "L": "#242424", "M": "#B8B8B8"}
COLORS = DEFAULT_COLORS.copy()
TEXT_COLORS = DEFAULT_COLORS.copy()

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
FONT_MONO_BOLD = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"


def font(pt, *, bold=False, italic=False, mono=False):
    path = FONT_MONO_BOLD if mono else FONT_BOLD if bold else FONT_ITALIC if italic else FONT_REG
    return ImageFont.truetype(path, round(pt * DPI / 72))


F_AXIS = font(7.3, bold=True)
F_REGION = font(6.3, bold=True)
F_CUE = font(7.1, mono=True)
F_DETAIL = font(5.8, bold=True)
F_EMPTY = font(5.2, italic=True)
DETAIL_GAP = 0.43


def xy(x, y):
    return round((x - X_MIN) * SX), round(H - y * SY)


def centered(draw, x, y, text, used_font, fill, *, anchor="mm"):
    draw.text(xy(x, y), text, font=used_font, fill=fill, anchor=anchor)


def colored_detail(draw, x, y, parts, *, align="center"):
    runs = []
    for i, (kind, text) in enumerate(parts):
        if i:
            runs.append((" / ", "#333333"))
        runs.append((text, TEXT_COLORS[kind]))
    widths = [draw.textlength(text, font=F_DETAIL) for text, _ in runs]
    x_px, _ = xy(x, y)
    total_width = sum(widths)
    if align == "right":
        cursor = x_px - total_width
    elif align == "left":
        cursor = x_px
    else:
        cursor = x_px - total_width / 2
    baseline = H - y * SY
    for (text, color), width in zip(runs, widths):
        draw.text((round(cursor), round(baseline)), text, font=F_DETAIL, fill=color, anchor="lm")
        cursor += width


def add_region(draw, x, y, title, items, *, item_gap=0.68):
    centered(draw, x, y, title, F_REGION, "#202020", anchor="ma")
    if not items:
        centered(draw, x, y - 0.32, "No annotated cue (0/90)", F_EMPTY, "#555555", anchor="ma")
        return
    for i, (cue, parts) in enumerate(items):
        cue_y = y - 0.31 - i * item_gap
        centered(draw, x, cue_y, cue, F_CUE, "#000000", anchor="ma")
        detail_offset = 0.59 if "\n" in cue else DETAIL_GAP
        colored_detail(draw, x, cue_y - detail_offset, parts)


def add_callout(draw, *, anchor, line_start, x, y, cue, parts):
    """Label a narrow intersection outside the ellipses with an arrow."""
    start_px = np.asarray(xy(*line_start), dtype=float)
    end_px = np.asarray(xy(*anchor), dtype=float)
    delta = end_px - start_px
    length = float(np.linalg.norm(delta))
    unit = delta / length
    perp = np.asarray((-unit[1], unit[0]))
    base = end_px - unit * 18.0
    draw.line([tuple(start_px), tuple(end_px)], fill="#4B5563", width=4)
    draw.polygon(
        [tuple(end_px), tuple(base + perp * 7.0), tuple(base - perp * 7.0)],
        fill="#4B5563",
    )
    centered(draw, x, y, cue, F_CUE, "#000000", anchor="ma")
    colored_detail(draw, x, y - DETAIL_GAP, parts)


def ellipse_bbox(cx, cy, width, height):
    left, top = xy(cx - width / 2, cy + height / 2)
    right, bottom = xy(cx + width / 2, cy - height / 2)
    return left, top, right, bottom


def _rgba(hex_color, alpha):
    color = hex_color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)) + (alpha,)


def main(*, out=OUT, bar_palette=False):
    global COLORS, TEXT_COLORS
    COLORS = BAR_CHART_COLORS.copy() if bar_palette else DEFAULT_COLORS.copy()
    TEXT_COLORS = COLORS.copy()
    if bar_palette:
        TEXT_COLORS["M"] = "#4A4A4A"
    image = Image.new("RGB", (W, H), "white")
    fill_layer = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    fill_draw = ImageDraw.Draw(fill_layer)
    ellipses = [
        ((4.05, 4.55), 6.0, 3.3, _rgba(COLORS["O"], 54), COLORS["O"]),
        ((5.95, 4.55), 6.0, 3.3, _rgba(COLORS["L"], 48), COLORS["L"]),
        ((5.0, 3.20), 5.8, 3.2, _rgba(COLORS["M"], 54), COLORS["M"]),
    ]
    for (cx, cy), width, height, fill, _ in ellipses:
        fill_draw.ellipse(ellipse_bbox(cx, cy, width, height), fill=fill)
    image = Image.alpha_composite(image.convert("RGBA"), fill_layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw every boundary after compositing the fills so no ellipse can cover
    # another ellipse's outline in an overlap region.
    for (cx, cy), width, height, _, edge in ellipses:
        draw.ellipse(ellipse_bbox(cx, cy, width, height), outline=edge, width=8)

    centered(draw, 2.30, 6.40, "ORIENTATION (O)", F_AXIS, TEXT_COLORS["O"])
    centered(draw, 7.70, 6.40, "LOCATION (L)", F_AXIS, TEXT_COLORS["L"])
    centered(draw, 4.05, 2.30, "MOVEMENT (M)", F_AXIS, TEXT_COLORS["M"])

    # One compact representative per partition keeps all eight regions legible
    # at IEEE column width while preserving the cue-to-region assignments.
    add_region(draw, 2.15, 5.32, "O only", [
        ("point_you", [("O", "index outward")]),
    ])
    add_region(draw, 7.85, 5.32, "L only", [
        ("facepalm", [("L", "hand at face")]),
    ])
    add_region(draw, 5.0, 5.68, "O + L", [
        ("stop_palm_out", [("O", "palm out"), ("L", "chest")]),
    ])
    centered(draw, 3.38, 3.47, "O + M", F_REGION, "#202020", anchor="ma")
    add_callout(
        draw,
        anchor=(3.30, 3.20),
        line_start=(1.95, 2.62),
        x=1.28,
        y=2.66,
        cue="fist_pump",
        parts=[("O", "fist"), ("M", "thrust")],
    )
    centered(draw, 6.62, 3.47, "L + M", F_REGION, "#202020", anchor="ma")
    add_callout(
        draw,
        anchor=(6.70, 3.20),
        line_start=(8.05, 2.62),
        x=8.64,
        y=2.66,
        cue="rub_eye_tired",
        parts=[("L", "at eye"), ("M", "rub")],
    )
    add_region(draw, 5.0, 4.62, "O + L + M", [
        ("air_quotes", [("O", "two fingers"), ("L", "upper torso"), ("M", "flex")]),
    ])
    add_region(draw, 6.25, 2.82, "M only", [
        ("nod_yes", [("M", "vertical rock")]),
    ])

    # Keep the complement explicit but outside the sets.  A horizontal label
    # prevents this compact block from adding unnecessary figure height.
    centered(draw, 3.35, 1.18, "None", F_AXIS, "#374151", anchor="lm")
    centered(draw, 4.25, 1.30, "hesitation_pause_hold", F_CUE, "#000000", anchor="lm")
    colored_detail(draw, 4.25, 0.91, [("M", "pause and hold")], align="left")

    # Tight crop makes the Venn itself occupy the full available column width.
    background = Image.new("RGB", image.size, "white")
    content_bbox = ImageChops.difference(image, background).getbbox()
    if content_bbox is not None:
        pad = 12
        image = image.crop((
            max(0, content_bbox[0] - pad),
            max(0, content_bbox[1] - pad),
            min(W, content_bbox[2] + pad),
            min(H, content_bbox[3] + pad),
        ))

    # Matplotlib writes the tightly framed raster without introducing axes or
    # a second layer of horizontal whitespace.
    plt.imsave(
        out,
        np.asarray(image),
        pil_kwargs={"dpi": (DPI, DPI), "optimize": True},
    )
    print(f"wrote {out} size={image.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-palette", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    main(out=args.out, bar_palette=args.bar_palette)
