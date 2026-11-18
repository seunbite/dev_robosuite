#!/usr/bin/env python3
"""RQ2 Oddkin case-study grid: few-shot cues × α={0, 0.5, 1.0}.

Layout mirrors figure_rq3_disambiguation (row gutters + chip captions).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
GIF_ROOT = ROOT.parent / "synthetic_morphology" / "renders" / "fewshot_cues"
OUT = ROOT / "figure_rq2_oddkin.png"

CELL = 360
COL_GAP = 10
ROW_GAP = 12
LABEL_W = 118
CAP_H = 52
PAD = 6
FONT_SIZE = 22

ROW_RGB = (40, 40, 40)
CUE_RGB = (31, 111, 235)
NOTE_RGB = (55, 55, 55)

ROWS = [
    ("gcr_alpha0", "α=0", "Trileg\nCrown"),
    ("gcr_alpha05", "α=0.5", "Spine\nCrab"),
    ("gcr_alpha1", "α=1.0", "Gesture\nTree"),
]

# (cue_id, short title, per-row rematerialize note)
COLS = [
    (
        "raising_hand_greeting",
        "raise hand",
        {
            "gcr_alpha0": "Present front leg, then advance.",
            "gcr_alpha05": "Raise long pincer; short stays still.",
            "gcr_alpha1": "Paddle up, palm toward viewer.",
        },
    ),
    (
        "wave_hi",
        "wave hi",
        {
            "gcr_alpha0": "Oscillate one hip (legs planted).",
            "gcr_alpha05": "Wag long pincer only.",
            "gcr_alpha1": "Wave paddle (claw still/low).",
        },
    ),
    (
        "bow_greeting",
        "bow",
        {
            "gcr_alpha0": "Approach, then retreat on base.",
            "gcr_alpha05": "Spine high → fold low → settle.",
            "gcr_alpha1": "Trunk/head pitch toward viewer.",
        },
    ),
    (
        "blow_kiss",
        "blow kiss",
        {
            "gcr_alpha0": "In-place yaw rock, then surge.",
            "gcr_alpha05": "Long tip close → front throw.",
            "gcr_alpha1": "Claw pinch → open + extend.",
        },
    ),
    (
        "circle_temple_crazy",
        "circle temple",
        {
            "gcr_alpha0": "Base arc (r>0), 540°.",
            "gcr_alpha05": "Collar spin in place (r=0).",
            "gcr_alpha1": "Claw arcs near head (xz).",
        },
    ),
]


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


def _gif_frame(path: Path, cell: int) -> Image.Image:
    if not path.exists():
        return Image.new("RGB", (cell, cell), (40, 40, 40))
    with Image.open(path) as im:
        n = getattr(im, "n_frames", 1)
        # mid–late frame often shows the peak pose better than f0
        im.seek(min(n - 1, max(0, int(0.55 * (n - 1)))))
        img = im.convert("RGB")
    # crop away burnt-in title bar
    w, h = img.size
    top = int(0.12 * h)
    img = img.crop((0, top, w, h))
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top2 = (h - side) // 2
    return img.crop((left, top2, left + side, top2 + side)).resize(
        (cell, cell), Image.Resampling.LANCZOS
    )


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int, max_lines: int = 2) -> list[str]:
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
    return lines or [""]


def _chip(base: Image.Image, xy: tuple[int, int], lines: list[tuple[str, tuple[int, int, int], ImageFont.ImageFont]]) -> Image.Image:
    draw = ImageDraw.Draw(base)
    pad = 5
    lh = FONT_SIZE + 4
    tw = max(int(draw.textlength(t, font=f)) for t, _, f in lines)
    th = lh * len(lines)
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
        cy += lh
    return out.convert("RGB")


def main():
    font_b = _font(FONT_SIZE, bold=True)
    font_r = _font(FONT_SIZE - 3)
    font_gutter = _font(FONT_SIZE, bold=True)

    n_cols = len(COLS)
    n_rows = len(ROWS)
    width = LABEL_W + n_cols * CELL + (n_cols - 1) * COL_GAP + 2 * PAD
    # header row for cue names
    header_h = 36
    height = header_h + n_rows * (CELL + CAP_H) + (n_rows - 1) * ROW_GAP + 2 * PAD

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # column headers
    for c, (cue, title, _) in enumerate(COLS):
        x = PAD + LABEL_W + c * (CELL + COL_GAP)
        label = title
        tw = draw.textlength(label, font=font_b)
        draw.text((x + (CELL - tw) / 2, PAD + 6), label, fill=CUE_RGB, font=font_b)

    for r, (morph, alpha, gutter) in enumerate(ROWS):
        y0 = PAD + header_h + r * (CELL + CAP_H + ROW_GAP)
        # gutter
        gy = y0 + CELL // 2 - 28
        for i, line in enumerate(gutter.split("\n")):
            tw = draw.textlength(line, font=font_gutter)
            draw.text((PAD + (LABEL_W - 8 - tw) / 2, gy + i * (FONT_SIZE + 4)), line, fill=ROW_RGB, font=font_gutter)
        # small alpha tag
        tw = draw.textlength(alpha, font=font_r)
        draw.text((PAD + (LABEL_W - 8 - tw) / 2, gy + 52), alpha, fill=(120, 120, 120), font=font_r)

        for c, (cue, title, notes) in enumerate(COLS):
            x = PAD + LABEL_W + c * (CELL + COL_GAP)
            gif = GIF_ROOT / morph / f"{cue}.gif"
            cell = _gif_frame(gif, CELL)
            canvas.paste(cell, (x, y0))

            # chip overlay (cue + rematerialize note), RQ3-style
            note_lines = _wrap(draw, notes[morph], font_r, CELL - 20, max_lines=2)
            chip_lines = [(title, CUE_RGB, font_b)]
            for nl in note_lines:
                chip_lines.append((nl, NOTE_RGB, font_r))
            chip_h = (FONT_SIZE + 4) * len(chip_lines) + 10
            canvas = _chip(canvas, (x + 6, y0 + CELL - chip_h - 4), chip_lines)
            draw = ImageDraw.Draw(canvas)

    canvas.save(OUT, optimize=True)
    print("wrote", OUT, canvas.size)


if __name__ == "__main__":
    main()
