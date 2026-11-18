#!/usr/bin/env python3
"""Fig. 7 (right): three rematerialization strategies in a single horizontal row.

One LLM-authored cue per strategy, each on the body that best illustrates it:
  Carrier transfer  -- shake_head_no       (Single Arm)
  Shape tracing     -- depict_big_bowl     (Stretch)
  Additive movement -- raise_fists_victory (Roomba)

Layout follows Fig. 5: the cue name sits inside the top-left of each cell and the
strategy label sits centered just above the cell. A white box at the bottom gives
a short plain-language caption. All frames come from the already-rendered Gemini
clips; nothing is re-simulated. Every label is plain black (the third cue name is
drawn on a small white box so it stays legible over Roomba's checkerboard floor).
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
sys.path.insert(0, str(REPO / "adhoc/vlm_test"))

from build_analysis_qual_grids import CELL, font_pair  # noqa: E402
from build_fig1_gaze_rub_grid import fill_cell, peak_pose_with_traj  # noqa: E402

LLM = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
OUT = REPO / "papers/icra2027_motion_abstraction/figures/figure7_rematerialization_grid.png"

# (strategy, cue, cue index, platform key, gif dir, caption, cue-on-white-box)
PANELS = [
    ("Carrier transfer", "shake_head_no", 2, "single_arm",
     LLM / "manipulator/gif/IIWA", "Same yaw, new carrier", False),
    ("Shape tracing", "depict_big_bowl", 27, "stretch",
     LLM / "stretch/gif", "Rim drawn as an EE arc", False),
    ("Additive movement", "raise_fists_victory", 22, "roomba",
     LLM / "roomba/gif", "Added as a 720° spin", True),
]

HEAD_H = 24          # single centered strategy label above each cell
CAP_H = 28           # white caption box inside each cell
FONT = 16            # single uniform label size (was 18)
BLACK = (17, 17, 17)


def _gif(dir_: Path, idx: int, cue: str) -> Path:
    return dir_ / f"emb_g{idx:02d}_{cue}.gif"


def _centered(draw, x0, width, y, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (width - (bbox[2] - bbox[0])) / 2, y), text, fill=fill, font=font)


def _cue_label(cell: Image.Image, text: str, font, on_box: bool) -> None:
    draw = ImageDraw.Draw(cell, "RGBA")
    x, y = 6, 5
    bbox = draw.textbbox((x, y), text, font=font)
    if on_box:
        draw.rectangle([bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                       fill=(255, 255, 255, 232))
    draw.text((x, y), text, fill=BLACK, font=font)


def _caption_box(cell: Image.Image, text: str, font) -> None:
    draw = ImageDraw.Draw(cell, "RGBA")
    w, h = cell.size
    top = h - CAP_H
    draw.rectangle([0, top, w, h], fill=(255, 255, 255, 232))
    draw.line([(0, top), (w, top)], fill=(120, 120, 120, 255), width=1)
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text(((w - (bbox[2] - bbox[0])) / 2,
               top + (CAP_H - (bbox[3] - bbox[1])) / 2 - bbox[1]),
              text, fill=BLACK, font=font)


def main() -> None:
    _, font = font_pair(24, FONT)

    n = len(PANELS)
    W, H = CELL * n, HEAD_H + CELL
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    for j, (strategy, cue, idx, plat, gdir, caption, on_box) in enumerate(PANELS):
        x0 = j * CELL
        _centered(draw, x0, CELL, 3, strategy, font, BLACK)

        g = _gif(gdir, idx, cue)
        if not g.is_file():
            cell = Image.new("RGB", (CELL, CELL), "#f2f2f2")
        else:
            img = peak_pose_with_traj(g, plat, idx)
            cell = fill_cell(img, CELL, platform=plat)
        _cue_label(cell, cue.replace("_", " "), font, on_box)
        _caption_box(cell, caption, font)
        canvas.paste(cell, (x0, HEAD_H))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    main()
