#!/usr/bin/env python3
"""Fig.3: 3 morph × 6 cues with green/red attribute badges (no column headers)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
FIGURES = PAPER / "figures"
ODDKIN = PAPER / "synthetic_morphology/renders/cues30"
OUT = FIGURES / "figure_humanoid_attribute_cases.png"

sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
sys.path.insert(0, str(REPO / "adhoc/vlm_test"))

from build_analysis_qual_grids import gif_path, load_gif_frames  # noqa: E402
from build_fig1_gaze_rub_grid import (  # noqa: E402
    fill_cell,
    peak_pose_with_traj,
    track_from_first,
)
from testset_utils import _image_with_ee_path  # noqa: E402

CELL = 250
LABEL_W = 102
HEADER_H = 28
ROW_GAP = 6
PAD = 6
FONT_SIZE = 16
CAP_RGB = (30, 30, 30)
GREEN = (34, 140, 70)
RED = (200, 55, 55)

# Columns: attribute exemplars (orient close/far, loc close/far, move OK/fail).
CUES = {
    9: "point_you",
    12: "stop_palm_out",
    25: "facepalm",
    20: "hands_up_surrender",
    1: "nod_yes",
    5: "fist_pump",
}
CUE_ORDER = [9, 12, 25, 20, 1, 5]

# Top-right badge: (label, is_good)
BADGE = {
    9: ("Orient", True),
    12: ("Orient", False),
    25: ("Loc", True),
    20: ("Loc", False),
    1: ("Move", True),
    5: ("Move", False),
}

COL_BOTTOM = {
    9: "point you",
    12: "stop palm",
    25: "facepalm",
    20: "hands up",
    1: "nod yes",
    5: "fist pump",
}

ROWS = [
    ("humanoid", "Unitree G1"),
    ("tiago", "Tiago"),
    ("gcr_alpha1", "Paddleclaw"),
]

CELL_CAPS = {
    ("humanoid", 25): "Hand covers the face and holds.",
    ("humanoid", 20): "Both arms shoot high overhead.",
    ("humanoid", 9): "Arm aims forward at you.",
    ("humanoid", 12): "Palm rises near the head.",
    ("humanoid", 1): "Head nods down-up repeatedly.",
    ("humanoid", 5): "Arm swings out instead of pumping.",
    ("tiago", 25): "Hand comes to the face and holds.",
    ("tiago", 20): "Both arms lift high and freeze.",
    ("tiago", 9): "Arm points straight toward you.",
    ("tiago", 12): "Hand stops high, palm unclear.",
    ("tiago", 1): "Head nods down-up repeatedly.",
    ("tiago", 5): "Arm rises sideways, not a pump.",
    ("gcr_alpha1", 25): "Paddle covers the head face-on.",
    ("gcr_alpha1", 20): "Both branches lift high and hold.",
    ("gcr_alpha1", 9): "Claw aims forward at you.",
    ("gcr_alpha1", 12): "Paddle faces out as a stop.",
    ("gcr_alpha1", 1): "Head nods down-up, three times.",
    ("gcr_alpha1", 5): "Claw rises, then pumps upward.",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _centered(draw: ImageDraw.ImageDraw, box, text: str, font, fill="#202020") -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=1, align="center")
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=font,
        fill=fill,
        spacing=1,
        align="center",
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
            if len(lines) >= max_lines:
                cur = ""
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines or [""]


def _chip(
    base: Image.Image,
    xy: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int], ImageFont.ImageFont]],
    *,
    pad: int = 5,
) -> Image.Image:
    draw = ImageDraw.Draw(base)
    line_h = FONT_SIZE + 4
    tw = max(int(draw.textlength(text, font=font)) for text, _, font in lines)
    th = line_h * len(lines)
    x, y = xy
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(
        [x, y, x + tw + 2 * pad, y + th + 2 * pad],
        radius=5,
        fill=(255, 255, 255, 250),
        outline=(200, 200, 200, 255),
        width=1,
    )
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    d = ImageDraw.Draw(out)
    cy = y + pad
    for text, color, font in lines:
        d.text((x + pad, cy), text, fill=color, font=font)
        cy += line_h
    return out.convert("RGB")


def _badge(
    base: Image.Image,
    xy: tuple[int, int],
    label: str,
    *,
    good: bool,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """Green/red pill in the cell top-right: (Orient) / (Loc) / (Move)."""
    text = f"({label})"
    color = GREEN if good else RED
    draw = ImageDraw.Draw(base)
    pad_x, pad_y = 6, 3
    tw = int(draw.textlength(text, font=font))
    th = FONT_SIZE + 2
    x, y = xy
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(
        [x, y, x + tw + 2 * pad_x, y + th + 2 * pad_y],
        radius=6,
        fill=(*color, 235),
        outline=(255, 255, 255, 230),
        width=1,
    )
    out = Image.alpha_composite(base.convert("RGBA"), overlay)
    d = ImageDraw.Draw(out)
    d.text((x + pad_x, y + pad_y), text, fill=(255, 255, 255), font=font)
    return out.convert("RGB")


def _llm_gif(platform: str, idx: int) -> Path | None:
    return gif_path("motion_program", platform, idx, CUES, {})


def _oddkin_gif(idx: int) -> Path:
    return ODDKIN / "gcr_alpha1" / f"{CUES[idx]}.gif"


def _oddkin_frames(path: Path) -> list[np.ndarray]:
    raw = load_gif_frames(path)
    frames: list[np.ndarray] = []
    for frame in raw:
        height = frame.shape[0]
        top = int(0.12 * height)
        bottom = int(0.10 * height)
        frames.append(frame[top : height - bottom])
    return frames


def _oddkin_cell(path: Path, idx: int) -> Image.Image:
    frames = _oddkin_frames(path)
    pose_idx = min(len(frames) - 1, max(0, int(0.58 * (len(frames) - 1))))
    base = Image.fromarray(frames[pose_idx]).convert("RGB")
    points = track_from_first(frames, "oddkin", idx=idx)
    if len(points) >= 2:
        base = _image_with_ee_path(base, points)
    return fill_cell(base, CELL, platform="oddkin")


def _cell(platform: str, idx: int) -> Image.Image:
    if platform == "gcr_alpha1":
        path = _oddkin_gif(idx)
        if path.is_file():
            return _oddkin_cell(path, idx)
    else:
        path = _llm_gif(platform, idx)
        if path is not None and path.is_file():
            image = peak_pose_with_traj(path, platform, idx)
            return fill_cell(image, CELL, platform=platform)

    missing = Image.new("RGB", (CELL, CELL), "#f2f2f2")
    _centered(ImageDraw.Draw(missing), (0, 0, CELL, CELL), "missing", _font(16))
    print(f"missing: {platform} {idx} {CUES[idx]}")
    return missing


def build() -> None:
    width = LABEL_W + len(CUE_ORDER) * CELL
    height = HEADER_H + len(ROWS) * CELL + (len(ROWS) - 1) * ROW_GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    bot_font = _font(13, bold=True)
    row_font = _font(17, bold=True)
    cap_font = _font(FONT_SIZE, bold=False)
    badge_font = _font(13, bold=True)

    for col, idx in enumerate(CUE_ORDER):
        x0 = LABEL_W + col * CELL
        _centered(draw, (x0, 0, x0 + CELL, HEADER_H), COL_BOTTOM[idx], bot_font)

    for row, (platform, label) in enumerate(ROWS):
        y0 = HEADER_H + row * (CELL + ROW_GAP)
        _centered(draw, (0, y0, LABEL_W - 4, y0 + CELL), label.replace(" ", "\n"), row_font)
        for col, idx in enumerate(CUE_ORDER):
            x0 = LABEL_W + col * CELL
            canvas.paste(_cell(platform, idx), (x0, y0))
            badge_label, good = BADGE[idx]
            badge_probe = ImageDraw.Draw(canvas)
            tw = int(badge_probe.textlength(f"({badge_label})", font=badge_font))
            bx = x0 + CELL - tw - 2 * 6 - PAD
            by = y0 + PAD
            canvas = _badge(canvas, (bx, by), badge_label, good=good, font=badge_font)

            probe = ImageDraw.Draw(canvas)
            key = (platform, idx)
            if key not in CELL_CAPS:
                raise KeyError(f"missing caption for {key}")
            wrapped = _wrap(probe, CELL_CAPS[key], cap_font, CELL - 2 * PAD - 10, max_lines=2)
            lines = [(ln, CAP_RGB, cap_font) for ln in wrapped]
            tw = max(int(probe.textlength(t, font=f)) for t, _, f in lines)
            th = (FONT_SIZE + 4) * len(lines)
            chip_w = tw + 2 * PAD
            chip_h = th + 2 * PAD
            cx = x0 + (CELL - chip_w) // 2
            cy = y0 + CELL - chip_h - PAD
            canvas = _chip(canvas, (cx, cy), lines)

        if row < len(ROWS) - 1:
            draw = ImageDraw.Draw(canvas)
            draw.line(
                (0, y0 + CELL + ROW_GAP // 2, width, y0 + CELL + ROW_GAP // 2),
                fill="#d0d0d0",
            )

    canvas.save(OUT, optimize=True)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    build()
