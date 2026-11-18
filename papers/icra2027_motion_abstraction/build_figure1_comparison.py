#!/usr/bin/env python3
"""Rebuild Fig. 1 using the original 260 px cell / 38 px header style."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "papers/icra2027_motion_abstraction"
FIG = PAPER / "figures"
sys.path[:0] = [str(FIG), str(PAPER), str(ROOT / "adhoc/generation/robotarm")]

from build_attr_buildup_grid import (  # noqa: E402
    CELL,
    CUE_BLUE,
    HEADER_H,
    _draw_traj_on,
    _font,
    _load_frames,
    _to_cell,
)
from build_fig1_gaze_rub_grid import track_from_first  # noqa: E402
from build_analysis_qual_grids import crop_progress  # noqa: E402


PANELS = {
    "pat_stomach": {
        ("Human", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_synth60_motion60/humanoid/gif/emb_g32_pat_stomach.gif",
        ("Human", "Roomba"): ROOT / "data/results/render/figure1_human_authored_roomba/roomba/gif/emb_g32_pat_stomach.gif",
        ("LLM", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike/humanoid/gif/emb_g02_pat_stomach.gif",
        ("LLM", "Roomba"): ROOT / "data/results/render/embodiment_extension20_gemini/roomba/gif/emb_g32_pat_stomach.gif",
    },
    "beckon_come_here": {
        ("Human", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_gt/humanoid/gif/emb_g11_beckon_come_here.gif",
        ("Human", "Roomba"): ROOT / "data/results/render/embodiment_motion_code_gt/roomba/gif/emb_g11_beckon_come_here.gif",
        ("LLM", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/humanoid/gif/emb_g11_beckon_come_here.gif",
        ("LLM", "Roomba"): ROOT / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/roomba/gif/emb_g11_beckon_come_here.gif",
    },
    "depict_flying_bird": {
        ("Human", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_gt/humanoid/gif/emb_g30_depict_flying_bird.gif",
        ("Human", "Roomba"): ROOT / "data/results/render/embodiment_motion_code_gt/roomba/gif/emb_g30_depict_flying_bird.gif",
        ("LLM", "Humanoid"): ROOT / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/humanoid/gif/emb_g30_depict_flying_bird.gif",
        ("LLM", "Roomba"): ROOT / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/roomba/gif/emb_g30_depict_flying_bird.gif",
    },
}

CUE_IDX = {
    "pat_stomach": 32,
    "beckon_come_here": 11,
    "depict_flying_bird": 30,
}

ORDER = [
    ("Human", "Humanoid"),
    ("LLM", "Humanoid"),
    ("Human", "Roomba"),
    ("LLM", "Roomba"),
]

CAPTIONS = {
    ("pat_stomach", "Human", "Humanoid"): "Hand pats inward against the stomach four times.",
    ("pat_stomach", "LLM", "Humanoid"): "Hand makes three short in-and-out strokes at the stomach.",
    ("pat_stomach", "Human", "Roomba"): "Turns left, then makes four small side-to-side pats.",
    ("pat_stomach", "LLM", "Roomba"): "Base rocks through a shallow arc twice.",
    ("beckon_come_here", "Human", "Humanoid"): "Palm faces forward, then wrist curls inward three times.",
    ("beckon_come_here", "LLM", "Humanoid"): "Palm faces up, then hand draws inward twice.",
    ("beckon_come_here", "Human", "Roomba"): "Retreats, turns, then wags twice.",
    ("beckon_come_here", "LLM", "Roomba"): "Base repeatedly moves forward and back.",
    ("depict_flying_bird", "Human", "Humanoid"): "Arms extend sideways and flap four times.",
    ("depict_flying_bird", "LLM", "Humanoid"): "Mirrored arms flap up and down four times.",
    ("depict_flying_bird", "Human", "Roomba"): "Base circles, surges forward, then circles again.",
    ("depict_flying_bird", "LLM", "Roomba"): "Base alternates four half-circle turns.",
}

# Enlarge only the explanatory text in the white caption boxes. Keep the cue
# headings and Human/LLM tags at their original sizes.
CAPTION_TEXT_SCALE = 1.5
AUTHOR_FONT_SIZE = 20
CAPTION_FONT_SIZE = round(15 * CAPTION_TEXT_SCALE)
HEADER_FONT_SIZE = 22


def _first_frame(path: Path, platform: str) -> Image.Image:
    key = "humanoid" if platform == "Humanoid" else "roomba"
    frames = _load_frames(path, key)
    if key == "roomba":
        frames = crop_progress(frames)
    if not frames:
        raise ValueError(f"no frames: {path}")
    return _to_cell(Image.fromarray(np.asarray(frames[0])).convert("RGB"), key)


def _tracked_trajectories(path: Path, cue: str, body: str) -> list[list[tuple[int, int]]]:
    """Extract the rendered carrier path using the same tracker as Fig. 2."""
    platform = "humanoid" if body == "Humanoid" else "roomba"
    frames = _load_frames(path, platform)
    if platform == "roomba":
        frames = crop_progress(frames)
    if not frames:
        return []
    step = max(1, len(frames) // 36)
    cells = [
        np.asarray(_to_cell(Image.fromarray(np.asarray(frame)).convert("RGB"), platform))
        for frame in frames[::step]
    ]
    points = track_from_first(cells, platform, idx=CUE_IDX[cue])
    if len(points) < 2:
        return []
    trajectories = [points]
    if cue == "depict_flying_bird" and body == "Humanoid":
        trajectories.append([(CELL - x, y) for x, y in points])
    return trajectories


def _pretty_cue(cue: str) -> str:
    return cue.replace("_", " ").title()


def _regular_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_author_tag(img: Image.Image, author: str) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = _regular_font(AUTHOR_FONT_SIZE)
    label = author
    box = draw.textbbox((0, 0), label, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.rounded_rectangle((6, 6, 18 + w, 15 + h), radius=3,
                           fill=(255, 255, 255, 235), outline=(210, 210, 210, 255))
    draw.text((12, 7), label, fill=(20, 20, 20, 255), font=font)


def _wrap_caption(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _draw_caption_box(img: Image.Image, cue: str, author: str, body: str) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    font = _regular_font(CAPTION_FONT_SIZE)
    lines = _wrap_caption(draw, CAPTIONS[(cue, author, body)], font, CELL - 28)
    line_h = round(18 * CAPTION_TEXT_SCALE)
    box_h = len(lines) * line_h + round(14 * CAPTION_TEXT_SCALE)
    text_w = max(draw.textlength(line, font=font) for line in lines)
    x0 = max(5, int((CELL - text_w) / 2) - 8)
    x1, y1 = min(CELL - 5, int((CELL + text_w) / 2) + 8), CELL - 6
    y0 = y1 - box_h
    draw.rounded_rectangle((x0, y0, x1, y1), radius=4,
                           fill=(255, 255, 255, 247), outline=(205, 205, 205, 255), width=1)
    for i, line in enumerate(lines):
        draw.text((x0 + 8, y0 + 6 + i * line_h), line,
                  fill=(30, 30, 30, 255), font=font)


def _cell(cue: str, author: str, body: str) -> Image.Image:
    path = PANELS[cue][(author, body)]
    if path is None or not path.is_file():
        raise FileNotFoundError(path)
    img = _first_frame(path, body)
    # Keep humanoid panels as clean pose snapshots; the trajectories are
    # retained only on Roomba, where base motion is the communicative carrier.
    if body == "Roomba":
        for pts in _tracked_trajectories(path, cue, body):
            img = _draw_traj_on(img, pts)
    _draw_author_tag(img, author)
    _draw_caption_box(img, cue, author, body)
    return img


def _center_header(draw: ImageDraw.ImageDraw, text: str, x0: int, y0: int, width: int) -> None:
    font = _font(HEADER_FONT_SIZE)
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((x0 + (width-w)//2, y0 + max(2, (HEADER_H-h)//2-1)), text,
              fill=CUE_BLUE, font=font)


def _save(canvas: Image.Image, stem: str) -> None:
    canvas.save(FIG / f"{stem}.png")
    canvas.convert("RGB").save(FIG / f"{stem}.pdf", "PDF", resolution=220.0)


def four_across() -> None:
    row_h = HEADER_H + CELL
    canvas = Image.new("RGB", (4*CELL, 3*row_h), "white")
    draw = ImageDraw.Draw(canvas)
    for r, cue in enumerate(PANELS):
        y0 = r * row_h
        _center_header(draw, _pretty_cue(cue), 0, y0, 4 * CELL)
        for c, (author, body) in enumerate(ORDER):
            canvas.paste(_cell(cue, author, body),
                         (c*CELL, y0 + HEADER_H))
    _save(canvas, "figure1_human_llm_humanoid_roomba_4up")


def two_across() -> None:
    canvas = Image.new("RGB", (2*CELL, 6*(HEADER_H+CELL)), "white")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for cue in PANELS:
        for pair_start in (0, 2):
            header = _pretty_cue(cue) if pair_start == 0 else "Roomba"
            _center_header(draw, header, 0, y, 2 * CELL)
            for c in range(2):
                author, body = ORDER[pair_start+c]
                canvas.paste(_cell(cue, author, body),
                             (c*CELL, y+HEADER_H))
            y += HEADER_H + CELL
    _save(canvas, "figure1_human_llm_humanoid_roomba_2up")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    four_across()
    two_across()
