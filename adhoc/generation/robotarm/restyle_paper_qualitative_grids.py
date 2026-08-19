#!/usr/bin/env python3
"""Apply the paper's shared five-column qualitative-grid visual grammar."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "data/results/evaluation/method_comparison/analysis_fail"
PAPER = REPO / "papers/icra2027_motion_abstraction/figures"
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def cue_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], cue: str, size: int = 18) -> None:
    fnt = font(size)
    x0, y0 = xy[0], xy[1]
    pad_x, pad_y = 6, 2
    bbox = draw.textbbox((x0 + pad_x, y0 + pad_y), cue, font=fnt)
    box = (x0, y0, bbox[2] + pad_x, bbox[3] + pad_y)
    draw.rounded_rectangle(box, radius=3, fill="white")
    draw.text((x0 + pad_x, y0 + pad_y), cue, font=fnt, fill="#202020")


def _cover_old_label(im: Image.Image, src: Image.Image, x0: int, y0: int, x1: int, y1: int) -> None:
    """Replace a two-line label with a 1px background strip sampled just to its right."""
    sx = min(x1 + 4, src.width - 1)
    strip = src.crop((sx, y0, sx + 1, y1))
    im.paste(strip.resize((max(1, x1 - x0), max(1, y1 - y0))), (x0, y0))


def judgment(draw: ImageDraw.ImageDraw, x: int, y: int, yes: bool) -> None:
    color = "#00a943" if yes else "#d7191c"
    dark = "#063d22" if yes else "#551414"
    draw.rounded_rectangle((x, y, x + 29, y + 25), radius=4, fill=dark, outline=color, width=2)
    draw.text((x + (6 if yes else 7), y + 1), "O" if yes else "X", font=font(20), fill="white")


def fig5() -> None:
    src = Image.open(SRC / "qual_grid_main_cross_embodiment.png").convert("RGB")
    im = src.copy()
    draw = ImageDraw.Draw(im)
    rows = [
        (47, 108, "air quotes"),
        (290, 101, "point self"),
        (534, 228, "raise index one moment"),
        (778, 155, "stretch arms up"),
        (1022, 110, "depict lion"),
    ]
    for y, old_width, cue in rows:
        _cover_old_label(im, src, 5, y, old_width, y + 49)
        cue_box(draw, (5, y, old_width, y + 22), cue, 17)
    out = SRC / "qual_grid_main_cross_embodiment_cue_only.png"
    im.save(out, optimize=True)
    im.save(PAPER / "figure5_cross_embodiment_qualitative.png", optimize=True)


def fig8() -> None:
    src = Image.open(SRC / "universal_pose_grid_example_rows.png").convert("RGB")
    # Remove the external row-name gutter; cue labels move inside the GT cells.
    im = src.crop((260, 0, src.width, src.height))
    draw = ImageDraw.Draw(im)
    row_tops = [40, 284, 528]
    row_bottoms = [282, 526, 768]
    cues = ["depict flying bird", "depict big bowl", "cross arms defensive"]
    labels = [
        [True, True, False, True, True],
        [True, True, True, True, False],
        [True, True, True, True, False],
    ]
    for top, bottom, cue, verdicts in zip(row_tops, row_bottoms, cues, labels):
        width = max(133, int(draw.textlength(cue, font=font(17))) + 12)
        cue_box(draw, (6, top + 7, width, top + 37), cue, 17)
        for col, yes in enumerate(verdicts):
            judgment(draw, col * 240 + 204, bottom - 32, yes)
    out = SRC / "universal_pose_grid_example_rows_fig5_style.png"
    im.save(out, optimize=True)
    im.save(PAPER / "figure14_universal_pose_grid.png", optimize=True)


if __name__ == "__main__":
    PAPER.mkdir(parents=True, exist_ok=True)
    fig5()
    fig8()
    print("restyled Fig. 5 and Fig. 8")
