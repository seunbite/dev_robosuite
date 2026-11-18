#!/usr/bin/env python3
"""Overlay hand-location judgment dots onto MuJoCo humanoid frames."""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

PAPER = Path(__file__).resolve().parent
OUT = PAPER / "cue_hand_location_dots"
JUDG = OUT / "hand_location_judgments.json"
GT_GIF = (
    Path("/Users/sb/Downloads/workspace/dev_robosuite")
    / "data/results/render/embodiment_motion_code_gt/humanoid/gif"
)
CANVAS = Path(
    "/Users/sb/.cursor/projects/"
    "Users-sb-Downloads-workspace-dev-robosuite-papers-icra2027-motion-abstraction/"
    "canvases/gesture-critical-factors-105.canvas.tsx"
)

# Body frame calibrated on Unitree G1 GT renders (480×480, free cam ~azimuth 145).
# nx=0.5 body midline; ny=0 head-top / overhead, ny=1 near feet.
CX = 241.0
Y_TOP = 105.0
Y_BOT = 365.0
HALF = 145.0  # half-width at nx=0 / 1 extremes

CELL = 240
COLS = 10
MARGIN = 8
DOT_R = 9
SEEKBAR = 16


def _font(size: int, bold: bool = False):
    names = (
        (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        )
        if bold
        else (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        )
    )
    for n in names:
        if Path(n).exists():
            return ImageFont.truetype(n, size=size)
    return ImageFont.load_default()


def load_rows() -> list[dict]:
    data = json.loads(JUDG.read_text())
    return list(data["rows"])


def paper30_gif_by_cue() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in GT_GIF.glob("emb_g*.gif"):
        # emb_g12_stop_palm_out.gif
        m = re.match(r"emb_g(\d+)_(.+)\.gif$", p.name)
        if m:
            out[m.group(2)] = p
    return out


def mid_frame(path: Path) -> Image.Image:
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
    return frames[int(len(frames) * 0.55)].copy()


def base_frame() -> Image.Image:
    p = GT_GIF / "emb_g01_nod_yes.gif"
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(p))]
    return frames[0].copy()


def crop_seekbar(im: Image.Image) -> Image.Image:
    w, h = im.size
    return im.crop((0, 0, w, max(1, h - SEEKBAR)))


def to_px(nx: float, ny: float, w: int, h: int) -> tuple[int, int]:
    # Mapping defined on full 480×480 including seekbar band.
    sx, sy = w / 480.0, (h + SEEKBAR) / 480.0
    x = (CX + (nx - 0.5) * 2.0 * HALF) * sx
    y = (Y_TOP + ny * (Y_BOT - Y_TOP)) * sy
    return int(round(x)), int(round(y))


def draw_dot(draw: ImageDraw.ImageDraw, x: int, y: int, *, none: bool = False) -> None:
    if none:
        r = 14
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(40, 120, 220), width=3)
        return
    r = DOT_R
    draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=(220, 40, 40),
        outline=(90, 10, 10),
        width=1,
    )
    draw.line([x - 3, y, x + 3, y], fill=(255, 255, 255), width=1)
    draw.line([x, y - 3, x, y + 3], fill=(255, 255, 255), width=1)


def render_cell(row: dict, bg: Image.Image) -> Image.Image:
    im = crop_seekbar(bg).copy()
    # letterbox to square cell content then pad label
    side = min(im.size)
    im = im.crop(((im.size[0] - side) // 2, 0, (im.size[0] + side) // 2, side))
    im = im.resize((CELL, CELL), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(im)

    # scale mapping into CELL (from 480 source before seek crop ≈464 height)
    # Recompute using CELL as display of full framed body.
    src_w, src_h = 480, 480 - SEEKBAR
    scale = CELL / src_w

    dots = row["dots"]
    if not dots:
        x = int((CX) * scale)
        y = int((Y_TOP + 0.12 * (Y_BOT - Y_TOP)) * (CELL / 480.0))
        draw_dot(draw, x, y, none=True)
    else:
        for d in dots:
            # map in original 480 space then scale to CELL (width-based)
            x0, y0 = to_px(d["x"], d["y"], 480, 480 - SEEKBAR)
            x = int(x0 * scale)
            y = int(y0 * (CELL / src_h))
            draw_dot(draw, x, y)

    # caption strip
    canvas = Image.new("RGB", (CELL, CELL + 36), (255, 255, 255))
    canvas.paste(im, (0, 0))
    d2 = ImageDraw.Draw(canvas)
    title = f"{row['index']}. {row['cue']}"
    d2.text((4, CELL + 2), title[:34], fill=(20, 20, 20), font=_font(11, bold=True))
    if dots:
        note = dots[0]["note"]
        if len(dots) > 1:
            note = f"{len(dots)}h · {note}"
    else:
        note = "no hand (head/torso)"
    d2.text((4, CELL + 18), note[:36], fill=(120, 40, 40), font=_font(9))
    d2.rectangle([0, 0, CELL - 1, CELL + 35], outline=(210, 210, 210))
    return canvas


def build_grid(cells: list[Image.Image], title: str) -> Image.Image:
    n = len(cells)
    cols = COLS
    rows = math.ceil(n / cols)
    cw, ch = cells[0].size
    W = cols * cw + (cols + 1) * MARGIN
    H = rows * ch + (rows + 1) * MARGIN + 44
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((MARGIN, 12), title, fill=(25, 25, 25), font=_font(15, bold=True))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = MARGIN + c * (cw + MARGIN)
        y = 44 + MARGIN + r * (ch + MARGIN)
        canvas.paste(cell, (x, y))
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    gifs = paper30_gif_by_cue()
    base = base_frame()
    cells_dir = OUT / "mujoco_cells"
    cells_dir.mkdir(exist_ok=True)

    cells: list[Image.Image] = []
    used_gt = 0
    for row in rows:
        cue = row["cue"]
        if cue in gifs:
            bg = mid_frame(gifs[cue])
            used_gt += 1
            src = "gt"
        else:
            bg = base
            src = "neutral"
        cell = render_cell(row, bg)
        cell.save(cells_dir / f"{row['index']:03d}_{cue}.png")
        cells.append(cell)
        row["mujoco_src"] = src
        row["mujoco_cell"] = f"mujoco_cells/{row['index']:03d}_{cue}.png"

    title = (
        "Hand-location judgments on MuJoCo G1  ·  red = target hand  ·  "
        "blue ring = no hand  ·  paper30 uses cue mid-frame, else neutral stand"
    )
    grid = build_grid(cells, title)
    grid_path = OUT / "hand_location_mujoco_grid_90.png"
    grid.save(grid_path)
    build_grid(cells[:45], title + "  (1–45)").save(OUT / "hand_location_mujoco_grid_90_p1.png")
    build_grid(cells[45:], title + "  (46–91)").save(OUT / "hand_location_mujoco_grid_90_p2.png")

    # refresh JSON with mujoco paths
    payload = {"n": len(rows), "used_gt_frames": used_gt, "rows": rows}
    JUDG.write_text(json.dumps(payload, indent=2) + "\n")

    cards = []
    for m in rows:
        kind = m["kind"]
        badge = {"none": "no hand", "unimanual": "1 hand", "bimanual": "2 hands"}[kind]
        notes = "; ".join(d["note"] for d in m["dots"]) or "head/torso only"
        src = m.get("mujoco_src", "")
        cards.append(
            f"""<div class="card">
  <img src="{html.escape(m['mujoco_cell'])}" alt="{html.escape(m['cue'])}"/>
  <div class="meta"><b>{m['index']}. {html.escape(m['cue'])}</b>
  <span class="badge {kind}">{badge}</span>
  <span class="src">{src}</span></div>
  <div class="loc">{html.escape(m['location'])}</div>
  <div class="note">{html.escape(notes)}</div>
</div>"""
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Cue hand-location dots on MuJoCo (90)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background:#f4f4f4; color:#222; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
.sub {{ color:#666; margin-bottom: 18px; max-width: 900px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }}
.card {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:8px; }}
.card img {{ width:100%; background:#eee; }}
.meta {{ display:flex; flex-wrap:wrap; gap:6px; align-items:baseline; margin-top:6px; font-size:12px; }}
.loc {{ font-size:11px; color:#555; margin-top:2px; }}
.note {{ font-size:11px; color:#a33; margin-top:2px; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:999px; background:#eee; }}
.badge.none {{ background:#dce9f8; color:#1a5a9a; }}
.badge.unimanual {{ background:#f8e0e0; color:#9a1a1a; }}
.badge.bimanual {{ background:#f3e0f8; color:#6a1a9a; }}
.src {{ font-size:10px; color:#888; }}
a {{ color:#246; }}
</style></head><body>
<h1>Hand-location judgments on MuJoCo G1 — 90 cues</h1>
<p class="sub">
Red dots = where the communicative hand(s) should be (front-view body frame on the render).
Blue ring = location axis not required.
Paper-30 cues use that cue’s GT mid-frame; the rest use a neutral standing frame.
<a href="hand_location_mujoco_grid_90.png">full grid</a> ·
<a href="hand_location_mujoco_grid_90_p1.png">page 1</a> ·
<a href="hand_location_mujoco_grid_90_p2.png">page 2</a>
· GT frames used: {used_gt}/90
</p>
<div class="grid">
{''.join(cards)}
</div>
</body></html>
"""
    html_path = OUT / "index_mujoco.html"
    html_path.write_text(html_doc)
    # also replace main index to mujoco version
    (OUT / "index.html").write_text(html_doc)
    print("wrote", grid_path)
    print("wrote", html_path)
    print("used_gt_frames", used_gt, "/", len(rows))


if __name__ == "__main__":
    main()
