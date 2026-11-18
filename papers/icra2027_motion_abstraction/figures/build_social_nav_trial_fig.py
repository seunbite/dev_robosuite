#!/usr/bin/env python3
"""One social-nav trial: one GIF (visual) + scene/question (text)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = REPO / "papers/icra2027_motion_abstraction"
sys.path[:0] = [str(REPO), str(REPO / "adhoc/generation/robotarm"), str(REPO / "adhoc/vlm_test")]

from build_fig1_gaze_rub_grid import fill_cell, peak_pose_with_traj  # noqa: E402

EVAL = PAPER / "roomba_nav_pair_sample" / "roomba_nav_eval.json"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt/roomba/gif"
OUT_PNG = PAPER / "figures" / "figure_social_nav_trial.png"
OUT_PDF = PAPER / "figures" / "figure_social_nav_trial.pdf"

CELL = 300
PAD = 14
TEXT_W = 248
HEADER = 34
CUE_BLUE = (31, 111, 235)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def main() -> None:
    spec = json.loads(EVAL.read_text())
    pair = spec["pairs"][1]  # doorway: beckon (A)
    q = pair["question"]
    a = pair["a"]
    gif_a = GT_GIF / f"emb_g{int(a['idx']):02d}_{a['cue']}.gif"
    cell_a = fill_cell(peak_pose_with_traj(gif_a, "roomba", int(a["idx"])), CELL, platform="roomba")

    left_w = PAD + CELL + PAD
    width = left_w + TEXT_W + PAD
    title = _font(30, bold=True)
    body = _font(26)
    opt = _font(25)

    probe = Image.new("RGB", (width, 1400), "white")
    pdraw = ImageDraw.Draw(probe)
    ty = PAD
    ty += 40
    ty += 34 * len(_wrap(pdraw, pair["scene"], body, TEXT_W - 8))
    ty += 18
    for ln in _wrap(pdraw, q["prompt"], title, TEXT_W - 8):
        ty += 34
    ty += 8
    for i, choice in enumerate((q["opt1"], q["opt2"]), start=1):
        ty += 32 * len(_wrap(pdraw, f"{i}. {choice}", opt, TEXT_W - 8))
        ty += 10
    height = max(PAD + HEADER + CELL + PAD, ty + PAD)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((PAD, 8), "Motion A", fill=CUE_BLUE, font=title)
    canvas.paste(cell_a, (PAD, PAD + HEADER))

    tx = left_w
    ty = PAD
    draw.text((tx, ty), "Context (text)", fill=(20, 20, 20), font=title)
    ty += 40
    for ln in _wrap(draw, pair["scene"], body, TEXT_W - 8):
        draw.text((tx, ty), ln, fill=(40, 40, 40), font=body)
        ty += 34
    ty += 18
    for ln in _wrap(draw, q["prompt"], title, TEXT_W - 8):
        draw.text((tx, ty), ln, fill=(20, 20, 20), font=title)
        ty += 34
    ty += 8
    for i, choice in enumerate((q["opt1"], q["opt2"]), start=1):
        wrapped = _wrap(draw, f"{i}. {choice}", opt, TEXT_W - 8)
        for ln in wrapped:
            draw.text((tx, ty), ln, fill=(40, 40, 40), font=opt)
            ty += 32
        ty += 10

    canvas.save(OUT_PNG, optimize=True)
    canvas.convert("RGB").save(OUT_PDF, "PDF", resolution=180.0)
    print("wrote", OUT_PNG, canvas.size)


if __name__ == "__main__":
    main()
