#!/usr/bin/env python3
"""Guess hand EE location for ~90 cues and render front-view silhouette + dots."""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAPER = Path(__file__).resolve().parent
CANVAS = Path(
    "/Users/sb/.cursor/projects/"
    "Users-sb-Downloads-workspace-dev-robosuite-papers-icra2027-motion-abstraction/"
    "canvases/gesture-critical-factors-105.canvas.tsx"
)
OUT = PAPER / "cue_hand_location_dots"
CELL = 220
COLS = 10
MARGIN = 10
DOT_R = 9

# Front-view body coords in cell-normalized space (0..1 within body panel).
# Origin top-left. x: left→right of viewer; y: head→feet.
# Right hand of figure is on viewer's left (x smaller) unless noted.
#
# Each entry: list of dots as (x, y, note). Empty = no hand location required.


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


# My judgments: where the communicative hand(s) should sit for recognition.
HAND_DOTS: dict[str, list[tuple[float, float, str]]] = {
    # paper30
    "nod_yes": [],  # head
    "shake_head_no": [],
    "draw_circle_repeat": [(0.62, 0.40, "front mid, fingertip circle")],
    "gaze_shift_avoid": [],
    "fist_pump": [(0.58, 0.22, "above R shoulder")],
    "tilt_head_curious": [],
    "cover_yawn_sleepy": [(0.50, 0.18, "over mouth")],
    "rub_eye_tired": [(0.42, 0.14, "at one eye")],
    "point_you": [(0.55, 0.38, "front, toward addressee")],
    "point_self": [(0.50, 0.40, "own chest")],
    "beckon_come_here": [(0.55, 0.42, "front of torso")],
    "stop_palm_out": [(0.55, 0.34, "chest/shoulder height, out")],
    "salute": [(0.40, 0.13, "forehead edge")],
    "raise_index_one_moment": [(0.58, 0.28, "raised index ~shoulder+")],
    "highfive_invite": [(0.58, 0.30, "shoulder height, front")],
    "shrug_idk": [(0.28, 0.48, "L beside chest"), (0.72, 0.48, "R beside chest")],
    "flex_bicep": [(0.62, 0.30, "fist near R shoulder")],
    "air_quotes": [(0.40, 0.34, "L upper torso"), (0.60, 0.34, "R upper torso")],
    "arms_open_welcome": [(0.18, 0.42, "L wide"), (0.82, 0.42, "R wide")],
    "hands_up_surrender": [(0.32, 0.12, "L above head"), (0.68, 0.12, "R above head")],
    "stretch_arms_up": [(0.38, 0.06, "L overhead"), (0.62, 0.06, "R overhead")],
    "raise_fists_victory": [(0.36, 0.08, "L above head"), (0.64, 0.08, "R above head")],
    "cross_arms_defensive": [(0.42, 0.40, "across L"), (0.58, 0.40, "across R")],
    "palms_together_thank_you": [(0.50, 0.38, "chest center (both)")],
    "facepalm": [(0.50, 0.16, "covering face")],
    "depict_tree": [(0.30, 0.22, "L branch"), (0.70, 0.22, "R branch")],
    "depict_big_bowl": [(0.32, 0.52, "L bowl rim"), (0.68, 0.52, "R bowl rim")],
    "depict_lion": [(0.40, 0.36, "L claw front"), (0.60, 0.36, "R claw front")],
    "depict_small_object": [(0.55, 0.42, "pinch in front")],
    "depict_flying_bird": [(0.15, 0.38, "L wing"), (0.85, 0.38, "R wing")],
    # pilot100 retained
    "self_scratch_head_confused": [(0.38, 0.10, "side/top of head")],
    "pat_stomach": [(0.50, 0.54, "abdomen")],
    "fan_face_hot": [(0.68, 0.16, "beside face")],
    "tap_temple_think": [(0.36, 0.14, "temple")],
    "circle_temple_crazy": [(0.34, 0.14, "beside temple")],
    "lean_forward_interest": [],  # torso
    "cover_mouth_gasp": [(0.50, 0.18, "over mouth")],
    "cheers_toast": [(0.60, 0.28, "raised cup-height front")],
    "highfive_imitation": [(0.60, 0.28, "partner palm height")],
    "fistbump_invite": [(0.58, 0.40, "fist front of torso")],
    "pat_self_back": [(0.62, 0.30, "own upper back/shoulder")],
    "throat_cut_stop": [(0.50, 0.22, "throat height")],
    "visor_search": [(0.48, 0.12, "above eyes")],
    "yield_turn": [(0.55, 0.46, "hand lowers near torso")],
    "request_turn": [(0.55, 0.34, "slightly forward above chest")],
    "guiding_arm": [(0.78, 0.40, "out toward destination")],
    "protective_arm_out": [(0.78, 0.38, "lateral barrier")],
    "slow_down_request_palm_down": [(0.55, 0.40, "front, palm-down presses")],
    "firm_accept_forward_reach": [(0.62, 0.40, "full reach toward partner")],
    "shh_gesture": [(0.50, 0.18, "finger at mouth")],
    "scale_expand": [(0.30, 0.42, "L"), (0.70, 0.42, "R")],
    "hesitant_accept_half_reach": [(0.56, 0.42, "halfway forward")],
    "cancel_previous_offer_retract": [(0.48, 0.44, "retract toward body")],
    "conditional_accept_pause_then_reach": [(0.60, 0.40, "reach after pause")],
    "final_decision_down_press": [(0.55, 0.48, "down-press in front")],
    "resume_speaking_pull": [(0.52, 0.42, "pull inward")],
    "emphasize_point_downbeat": [(0.55, 0.42, "downbeat in front")],
    "prepare_action_raise_hold": [(0.58, 0.24, "arm held raised")],
    "exploration_reach": [(0.64, 0.40, "cautious forward")],
    "uncertainty_wave": [(0.70, 0.40, "loose wave beside")],
    "clarification_request_palm_tilt": [(0.55, 0.40, "front torso")],
    "speed_up_circular_cue": [(0.58, 0.38, "circular path front")],
    "polite_decline_small_push": [(0.58, 0.40, "small push front")],
    "unsure_response_hand_hover_stop": [(0.55, 0.40, "hover mid-front")],
    "continue_on_forward_roll": [(0.55, 0.40, "forward roll front")],
    "start_action_forward_snap": [(0.58, 0.40, "forward snap")],
    "hesitation_pause_hold": [(0.55, 0.42, "freeze mid-action (front)")],
    "interrupt_hard": [(0.55, 0.38, "sharp cut in workspace")],
    "interrupt_soft": [(0.55, 0.36, "soft stop raised a bit")],
    "self_hug": [(0.40, 0.42, "L wrap"), (0.60, 0.42, "R wrap")],
    "rub_arms_cold": [(0.35, 0.36, "on R upper arm"), (0.65, 0.36, "on L upper arm")],
    "dominance_high_open_hold": [(0.22, 0.18, "L high wide"), (0.78, 0.18, "R high wide")],
    "contain_space": [(0.38, 0.40, "L enclose"), (0.62, 0.40, "R enclose")],
    "expand_space": [(0.32, 0.40, "L"), (0.68, 0.40, "R")],
    "compress_space": [(0.42, 0.40, "L in"), (0.58, 0.40, "R in")],
    "balance_scale": [(0.28, 0.44, "L scale"), (0.72, 0.36, "R opposed height")],
    "push_pull_dual": [(0.38, 0.40, "push"), (0.62, 0.40, "pull")],
    "abandon_action_drop_turn": [(0.55, 0.55, "drop low then turn")],
    "escorting_gesture": [(0.78, 0.42, "beside path, forward")],
    "wait_for_me_backward_palm": [(0.38, 0.42, "palm back while body leaves")],
    "prevent_fall_support_reach": [(0.68, 0.38, "rapid reach to person")],
    "frustration_down_throw": [(0.32, 0.50, "L throw down"), (0.68, 0.50, "R throw down")],
    "submission_lowered_open_palm": [(0.38, 0.58, "L low"), (0.62, 0.58, "R low")],
    "apology_forward_lowered_hold": [(0.40, 0.52, "L low"), (0.60, 0.52, "R low")],
    "chin_flick_dismiss": [(0.50, 0.20, "under chin")],
    "cover_ears_too_loud": [(0.34, 0.15, "L ear"), (0.66, 0.15, "R ear")],
    "disagreement_cross_then_release": [(0.42, 0.40, "cross L"), (0.58, 0.40, "cross R")],
    "thumbs_up_approval": [(0.58, 0.38, "front torso")],
    "clap_applause": [(0.46, 0.38, "L clap"), (0.54, 0.38, "R clap")],
    "hand_over_heart_sincerity": [(0.48, 0.38, "over heart")],
}


def load_rows() -> list[dict]:
    text = CANVAS.read_text(encoding="utf-8")
    pat = re.compile(
        r'\{\s*index:\s*(\d+),\s*cue:\s*"([^"]+)",\s*location:\s*"([^"]*)",'
        r'\s*orientation:\s*"([^"]*)",\s*movement:\s*"([^"]*)"\s*\}'
    )
    return [
        {
            "index": int(m.group(1)),
            "cue": m.group(2),
            "location": m.group(3),
            "orientation": m.group(4),
            "movement": m.group(5),
        }
        for m in pat.finditer(text)
    ]


def draw_silhouette(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0

    def P(nx: float, ny: float) -> tuple[int, int]:
        return (int(x0 + nx * w), int(y0 + ny * h))

    stroke = (55, 55, 55)
    fill = (235, 235, 235)
    # head
    hx, hy = P(0.50, 0.12)
    hr = int(0.09 * min(w, h))
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], outline=stroke, width=2, fill=fill)
    # torso
    draw.polygon(
        [P(0.38, 0.24), P(0.62, 0.24), P(0.58, 0.58), P(0.42, 0.58)],
        outline=stroke,
        fill=fill,
    )
    # hips / legs hint
    draw.line([P(0.45, 0.58), P(0.42, 0.88)], fill=stroke, width=2)
    draw.line([P(0.55, 0.58), P(0.58, 0.88)], fill=stroke, width=2)
    # arms neutral (light) — dots override communicative EE
    draw.line([P(0.38, 0.28), P(0.22, 0.48)], fill=(170, 170, 170), width=2)
    draw.line([P(0.62, 0.28), P(0.78, 0.48)], fill=(170, 170, 170), width=2)


def draw_cell(row: dict, size: int = CELL) -> Image.Image:
    im = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    # border
    draw.rectangle([0, 0, size - 1, size - 1], outline=(210, 210, 210), width=1)

    title = f"{row['index']}. {row['cue']}"
    f_title = _font(11, bold=True)
    f_small = _font(9)
    draw.text((6, 4), title[:42], fill=(25, 25, 25), font=f_title)

    body_box = (18, 28, size - 18, size - 36)
    draw_silhouette(draw, body_box)
    x0, y0, x1, y1 = body_box
    bw, bh = x1 - x0, y1 - y0

    dots = HAND_DOTS.get(row["cue"])
    if dots is None:
        # fallback: mark unknown
        draw.text((6, size - 28), "NO JUDGMENT", fill=(180, 40, 40), font=f_small)
        return im

    if not dots:
        # head / non-hand cue
        cx = int(x0 + 0.50 * bw)
        cy = int(y0 + 0.12 * bh)
        r = 14
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(40, 110, 200), width=2)
        draw.text((6, size - 28), "no hand (head/torso cue)", fill=(40, 110, 200), font=f_small)
        return im

    for i, (nx, ny, _note) in enumerate(dots):
        cx = int(x0 + nx * bw)
        cy = int(y0 + ny * bh)
        # white ring + accent fill
        draw.ellipse(
            [cx - DOT_R - 2, cy - DOT_R - 2, cx + DOT_R + 2, cy + DOT_R + 2],
            fill=(255, 255, 255),
        )
        draw.ellipse(
            [cx - DOT_R, cy - DOT_R, cx + DOT_R, cy + DOT_R],
            fill=(220, 40, 40),
            outline=(90, 10, 10),
            width=1,
        )
        # crosshair tick
        draw.line([cx - 3, cy, cx + 3, cy], fill=(255, 255, 255), width=1)
        draw.line([cx, cy - 3, cx, cy + 3], fill=(255, 255, 255), width=1)

    note = dots[0][2]
    if len(dots) > 1:
        note = f"{len(dots)} hands · {note}"
    draw.text((6, size - 28), note[:40], fill=(90, 90, 90), font=f_small)
    return im


def build_grid(cells: list[Image.Image], cols: int = COLS) -> Image.Image:
    n = len(cells)
    rows = math.ceil(n / cols)
    W = cols * CELL + (cols + 1) * MARGIN
    H = rows * CELL + (rows + 1) * MARGIN + 48
    canvas = Image.new("RGB", (W, H), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, 12),
        "Hand-location judgments (red dots) · front view · blue ring = no hand required",
        fill=(30, 30, 30),
        font=_font(16, bold=True),
    )
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = MARGIN + c * (CELL + MARGIN)
        y = 48 + MARGIN + r * (CELL + MARGIN)
        canvas.paste(cell, (x, y))
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    missing = [r["cue"] for r in rows if r["cue"] not in HAND_DOTS]
    if missing:
        raise SystemExit(f"missing judgments for: {missing}")

    cells = []
    meta = []
    for row in rows:
        cell = draw_cell(row)
        cell_path = OUT / "cells" / f"{row['index']:03d}_{row['cue']}.png"
        cell_path.parent.mkdir(exist_ok=True)
        cell.save(cell_path)
        cells.append(cell)
        dots = HAND_DOTS[row["cue"]]
        meta.append(
            {
                **row,
                "dots": [{"x": x, "y": y, "note": n} for x, y, n in dots],
                "kind": "none" if not dots else ("bimanual" if len(dots) > 1 else "unimanual"),
                "cell": str(cell_path.relative_to(OUT)),
            }
        )

    grid = build_grid(cells)
    grid_path = OUT / "hand_location_grid_90.png"
    grid.save(grid_path)

    # also split into two pages for readability
    mid = 45
    for tag, chunk in (("p1", cells[:mid]), ("p2", cells[mid:])):
        build_grid(chunk).save(OUT / f"hand_location_grid_90_{tag}.png")

    (OUT / "hand_location_judgments.json").write_text(
        json.dumps({"n": len(meta), "rows": meta}, indent=2) + "\n"
    )

    # HTML gallery
    cards = []
    for m in meta:
        kind = m["kind"]
        badge = {
            "none": "no hand",
            "unimanual": "1 hand",
            "bimanual": "2 hands",
        }[kind]
        notes = "; ".join(d["note"] for d in m["dots"]) or "head/torso only"
        cards.append(
            f"""<div class="card">
  <img src="{html.escape(m['cell'])}" alt="{html.escape(m['cue'])}"/>
  <div class="meta"><b>{m['index']}. {html.escape(m['cue'])}</b>
  <span class="badge {kind}">{badge}</span></div>
  <div class="loc">{html.escape(m['location'])}</div>
  <div class="note">{html.escape(notes)}</div>
</div>"""
        )
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Cue hand-location dots (90)</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background:#f6f6f6; color:#222; }}
h1 {{ font-size: 22px; margin: 0 0 6px; }}
.sub {{ color:#666; margin-bottom: 18px; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.card {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:8px; }}
.card img {{ width:100%; image-rendering: auto; background:#fff; }}
.meta {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; margin-top:6px; font-size:12px; }}
.loc {{ font-size:11px; color:#555; margin-top:2px; }}
.note {{ font-size:11px; color:#a33; margin-top:2px; }}
.badge {{ font-size:10px; padding:1px 6px; border-radius:999px; background:#eee; }}
.badge.none {{ background:#dce9f8; color:#1a5a9a; }}
.badge.unimanual {{ background:#f8e0e0; color:#9a1a1a; }}
.badge.bimanual {{ background:#f3e0f8; color:#6a1a9a; }}
a {{ color:#246; }}
</style></head><body>
<h1>Hand-location judgments — 90 cues</h1>
<p class="sub">Red dots = where I think the communicative hand(s) should be (front view).
Blue ring on head = location axis not required (head/torso cue).
Source location text from the recognition-critical factors table.
<a href="hand_location_grid_90.png">full grid PNG</a> ·
<a href="hand_location_grid_90_p1.png">page 1</a> ·
<a href="hand_location_grid_90_p2.png">page 2</a></p>
<div class="grid">
{''.join(cards)}
</div>
</body></html>
"""
    html_path = OUT / "index.html"
    html_path.write_text(html_doc)
    print("wrote", grid_path)
    print("wrote", html_path)
    print("n", len(rows))
    print(
        "kinds",
        {
            k: sum(1 for m in meta if m["kind"] == k)
            for k in ("none", "unimanual", "bimanual")
        },
    )


if __name__ == "__main__":
    main()
