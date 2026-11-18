#!/usr/bin/env python3
"""Build a two-cue by eight-embodiment qualitative comparison grid."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "papers/icra2027_motion_abstraction"
OUT = PAPER / "figures/figure_physical_difficulty_grid.png"
SYNTH = PAPER / "synthetic_morphology/renders/cues30"

sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
sys.path.insert(0, str(PAPER / "figures"))

from build_fig1_gaze_rub_grid import fill_cell, peak_pose_with_traj  # noqa: E402
from build_morphology_cue_grids import _oddkin_cell  # noqa: E402

CELL_W = 202
CELL_H = CELL_W
ROW_LABEL_W = 160
HEADER_H = 42
ROW_GAP = 6
COL_GAP = 5
PAD = 6

PLATFORMS = [
    ("humanoid", "Humanoid", "humanoid/gif"),
    ("tiago", "Tiago", "tiago/gif"),
    ("single_arm", "Single Arm", "manipulator/gif/IIWA"),
    ("stretch", "Stretch", "stretch/gif"),
    ("roomba", "Roomba", "roomba/gif"),
    ("gcr_alpha1", "Paddleclaw", None),
    ("gcr_alpha0", "Trileg Crown", None),
    ("gcr_alpha05", "Spine Crab", None),
]

ROWS = [
    (11, "beckon_come_here"),
    (21, "stretch_arms_up"),
]

CAPTIONS = {
    ("humanoid", "beckon_come_here"): "Palm presents; fingers pull inward.",
    ("tiago", "beckon_come_here"): "Gripper presents, then draws inward.",
    ("single_arm", "beckon_come_here"): "One gripper repeats an inward pull.",
    ("stretch", "beckon_come_here"): "Arm extends, then retracts.",
    ("roomba", "beckon_come_here"): "Base approach substitutes for the pull.",
    ("gcr_alpha1", "beckon_come_here"): "Claw pitches inward repeatedly.",
    ("gcr_alpha0", "beckon_come_here"): "Raised front leg curls and advances.",
    ("gcr_alpha05", "beckon_come_here"): "Long pincer curls inward repeatedly.",
    ("humanoid", "stretch_arms_up"): "Both arms extend overhead.",
    ("tiago", "stretch_arms_up"): "Both grippers rise and hold.",
    ("single_arm", "stretch_arms_up"): "One chain substitutes for two arms.",
    ("stretch", "stretch_arms_up"): "Mast and arm extend upward.",
    ("roomba", "stretch_arms_up"): "No arm carrier; the cue collapses.",
    ("gcr_alpha1", "stretch_arms_up"): "Paddle and claw lift together.",
    ("gcr_alpha0", "stretch_arms_up"): "All three legs lift and hold.",
    ("gcr_alpha05", "stretch_arms_up"): "Spine rises; both pincers lift.",
}


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def cue_index() -> dict[str, int]:
    import json
    rows = json.loads((REPO / "data/seed/agent_fill/embodiment_short/cues.json").read_text())
    return {str(row["cue"]): int(row["idx"]) for row in rows}


def catalog_path(subdir: str, idx: int, cue: str) -> Path:
    path = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro" / subdir / f"emb_g{idx:02d}_{cue}.gif"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def cell(platform: str, subdir: str | None, idx: int, cue: str) -> Image.Image:
    if platform.startswith("gcr_"):
        image = _oddkin_cell(SYNTH / platform / f"{cue}.gif", idx, trajectory=True)
        square = image.resize((CELL_W, CELL_H), Image.Resampling.LANCZOS)
    else:
        image = peak_pose_with_traj(catalog_path(str(subdir), idx, cue), platform, idx)
        square = fill_cell(image, CELL_W, platform=platform)
    return square


def wrap_text(draw: ImageDraw.ImageDraw, text: str, used: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=used) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines[:2]


def chip(image: Image.Image, caption: str) -> Image.Image:
    out = image.convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cap_font = font(9)
    lines = wrap_text(draw, caption, cap_font, CELL_W - 4 * PAD)
    line_h = 12
    cap_h = len(lines) * line_h + 2 * PAD
    cap_y = CELL_H - cap_h - PAD
    draw.rounded_rectangle(
        (PAD, cap_y, CELL_W - PAD, CELL_H - PAD),
        4,
        fill=(255, 255, 255, 238),
        outline=(205, 205, 205, 255),
    )
    for i, line in enumerate(lines):
        draw.text((2 * PAD, cap_y + PAD + i * line_h), line, font=cap_font, fill=(25, 25, 25, 255))
    return Image.alpha_composite(out, overlay).convert("RGB")


def centered(draw: ImageDraw.ImageDraw, box, text: str, used_font, fill="#202020") -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=used_font, align="center", spacing=1)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text((x0 + (x1 - x0 - w) / 2, y0 + (y1 - y0 - h) / 2), text, font=used_font, fill=fill, align="center", spacing=1)


def main() -> None:
    width = ROW_LABEL_W + len(PLATFORMS) * CELL_W + (len(PLATFORMS) - 1) * COL_GAP
    height = HEADER_H + 2 * CELL_H + ROW_GAP
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(11, bold=True)
    row_font = font(11, bold=True)

    x = ROW_LABEL_W
    for _, label, _ in PLATFORMS:
        centered(draw, (x, 0, x + CELL_W, HEADER_H), label, header_font)
        x += CELL_W + COL_GAP

    for row_i, (idx, cue) in enumerate(ROWS):
        y = HEADER_H + row_i * (CELL_H + ROW_GAP)
        centered(draw, (0, y, ROW_LABEL_W - 8, y + CELL_H), f"{idx}\n{cue}", row_font)
        x = ROW_LABEL_W
        for platform, _, subdir in PLATFORMS:
            canvas.paste(chip(cell(platform, subdir, idx, cue), CAPTIONS[(platform, cue)]), (x, y))
            x += CELL_W
            if platform != PLATFORMS[-1][0]:
                draw = ImageDraw.Draw(canvas)
                draw.line((x + COL_GAP // 2, y + 7, x + COL_GAP // 2, y + CELL_H - 7), fill="#d5d5d5", width=1)
                x += COL_GAP

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    main()
