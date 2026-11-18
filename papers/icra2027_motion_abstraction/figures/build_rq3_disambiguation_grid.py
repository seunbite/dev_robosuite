#!/usr/bin/env python3
"""RQ3 figure: human contextualization (top) + LLM contrastive Roomba (bottom)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
sys.path.insert(0, str(REPO / "adhoc/generation/robotarm"))
from build_analysis_qual_grids import overlay_path_on_first_frame  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = REPO / "data/results/roomba_contrastive_contextualize"
GT_GIF = REPO / "data/results/render/embodiment_motion_code_gt"
OUT = ROOT / "figure_rq3_disambiguation.png"

CELL = 400
COL_GAP = 10
ROW_GAP = 14
LABEL_W = 110
PAD = 6
FONT_SIZE = 27  # prior 28, then -1

TRUE_RGB = (31, 111, 235)  # blue cue
FALSE_RGB = (30, 30, 30)  # black foil
NEQ_RGB = (40, 40, 40)
MORPH_RGB = (90, 90, 90)
ROW_RGB = (40, 40, 40)

# Human-authored contextualization on constrained morphologies.
HUMAN = [
    (
        "roomba",
        "highfive_invite",
        15,
        "Full spin, then advance and hold.",
    ),
    (
        "roomba",
        "cover_yawn_sleepy",
        7,
        "Slow three-stage yaw arc, then hold.",
    ),
    (
        "stretch",
        "shrug_idk",
        16,
        "Lift mast, bob shoulder, settle.",
    ),
    (
        "stretch",
        "hands_up_surrender",
        20,
        "Raise high, back away, hold.",
    ),
    (
        "single_arm",
        "cover_yawn_sleepy",
        7,
        "Rise, settle low, then hold.",
    ),
]

# LLM contrastive Roomba (target ≠ foil).
PAIRS = [
    ("highfive_invite", "hands_up_surrender", 15),
    ("hands_up_surrender", "highfive_invite", 20),
    ("raise_fists_victory", "fist_pump", 22),
    ("stretch_arms_up", "hands_up_surrender", 21),
    ("fist_pump", "raise_fists_victory", 5),
]

MORPH_DISPLAY = {
    "roomba": "Roomba",
    "stretch": "Stretch",
    "single_arm": "Arm",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
        if bold
        else (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for name in names:
        p = Path(name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _llm_gif(target: str, foil: str, idx: int) -> Path:
    return DATA / "motion_program/render" / f"emb_g{idx:02d}_{target}__vs_{foil}.gif"


def _human_gif(platform: str, cue: str, idx: int) -> Path:
    if platform == "single_arm":
        return GT_GIF / "manipulator/gif/IIWA" / f"emb_g{idx:02d}_{cue}.gif"
    return GT_GIF / platform / "gif" / f"emb_g{idx:02d}_{cue}.gif"


def _cell_image(path: Path, platform: str, cell: int) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", (cell, cell), (40, 40, 40))
    try:
        img, _ = overlay_path_on_first_frame(path, platform)
    except Exception:
        with Image.open(path) as im:
            n = getattr(im, "n_frames", 1)
            im.seek(min(n - 1, max(0, int(0.65 * (n - 1)))))
            img = im.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = max(0, (h - side) // 2 - side // 12)
    return img.crop((left, top, left + side, top + side)).resize(
        (cell, cell), Image.Resampling.LANCZOS
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 3) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                cur = ""
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if lines and len(words) > sum(len(l.split()) for l in lines):
        last = lines[-1]
        while draw.textlength(last + "…", font=font) > max_w and len(last) > 4:
            last = last[:-1].rstrip()
        lines[-1] = last + "…"
    return lines or [""]


def _chip(
    base: Image.Image,
    xy: tuple[int, int],
    segments_lines: list[list[tuple[str, tuple[int, int, int], ImageFont.ImageFont]]],
    *,
    pad: int = 5,
    line_h: int | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(base)
    lh = line_h or (FONT_SIZE + 6)
    line_widths = [
        sum(int(draw.textlength(t, font=f)) for t, _, f in segs) for segs in segments_lines
    ]
    tw = max(line_widths) if line_widths else 0
    th = lh * len(segments_lines)
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
    for segs in segments_lines:
        cx = x + pad
        for text, color, font in segs:
            d.text((cx, cy), text, fill=color, font=font)
            cx += int(d.textlength(text, font=font))
        cy += lh
    return out.convert("RGB")


def _cue_neq_segments(draw, target, foil, font_bold, font_reg, max_w):
    neq = " \u2260 "
    one_w = (
        draw.textlength(target, font=font_bold)
        + draw.textlength(neq, font=font_reg)
        + draw.textlength(foil, font=font_bold)
    )
    if one_w <= max_w:
        return [[
            (target, TRUE_RGB, font_bold),
            (neq, NEQ_RGB, font_reg),
            (foil, FALSE_RGB, font_bold),
        ]]
    return [
        [(target, TRUE_RGB, font_bold)],
        [("\u2260 ", NEQ_RGB, font_reg), (foil, FALSE_RGB, font_bold)],
    ]


def _human_title_segments(morph: str, cue: str, font_bold, font_reg):
    return [
        [(f"{morph} · ", MORPH_RGB, font_reg), (cue, TRUE_RGB, font_bold)],
    ]


def _paste_caption(canvas, probe, x, y0, cap_text, font_reg, line_h):
    cap_lines = _wrap(probe, cap_text, font_reg, CELL - 2 * PAD - 10, max_lines=3)
    segs = [[(ln, (30, 30, 30), font_reg)] for ln in cap_lines]
    tw = max((sum(int(probe.textlength(t, font=f)) for t, _, f in s) for s in segs), default=0)
    th = line_h * len(segs)
    chip_w = tw + 2 * 5
    chip_h = th + 2 * 5
    cx = x + (CELL - chip_w) // 2
    cy = y0 + CELL - chip_h - PAD
    return _chip(canvas, (cx, cy), segs, pad=5, line_h=line_h)


def _row_label(canvas, text: str, y0: int, font) -> None:
    draw = ImageDraw.Draw(canvas)
    # Vertical-ish stacked label in the left gutter.
    lines = text.split("\n")
    total_h = len(lines) * (FONT_SIZE + 4)
    cy = y0 + (CELL - total_h) // 2
    for ln in lines:
        tw = int(draw.textlength(ln, font=font))
        draw.text(((LABEL_W - tw) // 2, cy), ln, fill=ROW_RGB, font=font)
        cy += FONT_SIZE + 4


def main() -> None:
    captions = json.loads((DATA / "captions.json").read_text())
    cols = max(len(HUMAN), len(PAIRS))
    W = LABEL_W + cols * CELL + (cols - 1) * COL_GAP
    H = 2 * CELL + ROW_GAP

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    font_bold = _font(FONT_SIZE, bold=True)
    font_reg = _font(FONT_SIZE, bold=False)
    font_row = _font(FONT_SIZE, bold=True)
    probe = ImageDraw.Draw(canvas)
    line_h = FONT_SIZE + 6

    # --- Top row: human contextualization ---
    _row_label(canvas, "Human", 0, font_row)
    for j, (plat, cue, idx, cap) in enumerate(HUMAN):
        x = LABEL_W + j * (CELL + COL_GAP)
        canvas.paste(_cell_image(_human_gif(plat, cue, idx), plat, CELL), (x, 0))
        morph = MORPH_DISPLAY[plat]
        title = _human_title_segments(morph, cue, font_bold, font_reg)
        canvas = _chip(canvas, (x + PAD, PAD), title, pad=5, line_h=line_h)
        probe = ImageDraw.Draw(canvas)
        canvas = _paste_caption(canvas, probe, x, 0, cap, font_reg, line_h)
        probe = ImageDraw.Draw(canvas)

    # --- Bottom row: LLM contrastive ---
    y1 = CELL + ROW_GAP
    _row_label(canvas, "LLM", y1, font_row)
    for j, (target, foil, idx) in enumerate(PAIRS):
        x = LABEL_W + j * (CELL + COL_GAP)
        canvas.paste(_cell_image(_llm_gif(target, foil, idx), "roomba", CELL), (x, y1))
        cue_segs = _cue_neq_segments(
            probe, target, foil, font_bold, font_reg, CELL - 2 * PAD - 10
        )
        canvas = _chip(canvas, (x + PAD, y1 + PAD), cue_segs, pad=5, line_h=line_h)
        probe = ImageDraw.Draw(canvas)
        key = f"{target}|{foil}"
        cap = captions.get(key, {}).get("motion_program", "")
        canvas = _paste_caption(canvas, probe, x, y1, cap, font_reg, line_h)
        probe = ImageDraw.Draw(canvas)

    canvas.save(OUT, dpi=(300, 300))
    print(f"wrote {OUT} ({canvas.size})")
    for plat, cue, idx, _ in HUMAN:
        p = _human_gif(plat, cue, idx)
        print(f"  human {plat} {cue}: {'OK' if p.is_file() else 'MISSING'} {p}")


if __name__ == "__main__":
    main()
