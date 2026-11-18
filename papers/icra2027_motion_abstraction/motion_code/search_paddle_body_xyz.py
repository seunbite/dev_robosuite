#!/usr/bin/env python3
"""Grid-search paddlebot xyz for body-anchored cues; montage gallery for picking.

Usage:
  python motion_code/search_paddle_body_xyz.py                # all BODY_GRIDS
  python motion_code/search_paddle_body_xyz.py --batch face   # face/head only
  python motion_code/search_paddle_body_xyz.py --cues pat_stomach,facepalm
  python motion_code/search_paddle_body_xyz.py --apply        # write selection.json → paddle_xyz overrides

Outputs:
  motion_code/bank/_paddle_xyz_iter/body_combo/
    {cue}.png, index.html, manifest.json, selection.json
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ODDKIN = PAPER / "synthetic_morphology"
sys.path[:0] = [str(MC), str(ODDKIN), str(PAPER), str(REPO)]

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import render_cues30 as rc  # noqa: E402
import render_paddlebot90 as pb  # noqa: E402

OUT_ROOT = MC / "bank/_paddle_xyz_iter/body_combo"
SELECTION = OUT_ROOT / "selection.json"
OVERRIDE_JSON = pb.OUT_CFG / "paddle_xyz_body_overrides.json"

# Body-anchored cues: WIDE workspace sweeps (not fine local grids).
# Convention: high x=forward, high y=in, high z=up. Extremes should look
# obviously different (near floor / overhead / tucked / outstretched).
FACE_GRIDS: dict[str, dict[str, list[int]]] = {
    # face/head band: still span chest→overhead and out→across-midline
    "cover_yawn_sleepy": {"x": [25, 55, 90], "y": [20, 55, 95], "z": [40, 70, 95]},
    "rub_eye_tired": {"x": [25, 55, 90], "y": [25, 60, 95], "z": [45, 75, 95]},
    "salute": {"x": [20, 50, 85], "y": [25, 60, 95], "z": [50, 75, 95]},
    "facepalm": {"x": [25, 55, 90], "y": [25, 60, 95], "z": [40, 70, 95]},
    "self_scratch_head_confused": {"x": [20, 50, 85], "y": [25, 60, 95], "z": [50, 75, 95]},
    "fan_face_hot": {"x": [25, 55, 90], "y": [15, 50, 90], "z": [40, 70, 95]},
    "tap_temple_think": {"x": [20, 50, 85], "y": [30, 65, 95], "z": [45, 75, 95]},
    "circle_temple_crazy": {"x": [20, 50, 85], "y": [30, 65, 95], "z": [45, 75, 95]},
    "cover_mouth_gasp": {"x": [25, 55, 90], "y": [20, 55, 95], "z": [35, 65, 90]},
    "shh_gesture": {"x": [30, 60, 95], "y": [15, 50, 90], "z": [35, 65, 90]},
    "visor_search": {"x": [20, 50, 85], "y": [15, 50, 90], "z": [50, 75, 95]},
    "chin_flick_dismiss": {"x": [25, 55, 90], "y": [15, 50, 90], "z": [30, 60, 85]},
    "cover_ears_too_loud": {"x": [25, 55, 90], "y": [30, 65, 95], "z": [45, 75, 95]},
}

TORSO_GRIDS: dict[str, dict[str, list[int]]] = {
    "point_self": {"x": [15, 50, 90], "y": [20, 55, 95], "z": [25, 55, 85]},
    "flex_bicep": {"x": [15, 50, 85], "y": [30, 65, 95], "z": [40, 70, 95]},
    "cross_arms_defensive": {"x": [25, 55, 90], "y": [30, 65, 95], "z": [25, 55, 85]},
    "palms_together_thank_you": {"x": [25, 55, 90], "y": [25, 60, 95], "z": [30, 60, 90]},
    "pat_stomach": {"x": [30, 60, 95], "y": [15, 50, 90], "z": [10, 40, 70]},
    "throat_cut_stop": {"x": [20, 50, 85], "y": [15, 50, 90], "z": [35, 60, 85]},
    "pat_self_back": {"x": [5, 35, 70], "y": [15, 50, 90], "z": [30, 60, 90]},
    "hand_over_heart_sincerity": {"x": [25, 55, 90], "y": [20, 55, 95], "z": [25, 55, 85]},
    "self_hug": {"x": [30, 60, 95], "y": [25, 60, 95], "z": [20, 50, 80]},
    "rub_arms_cold": {"x": [25, 55, 90], "y": [25, 60, 95], "z": [20, 50, 80]},
    "clap_applause": {"x": [30, 60, 95], "y": [25, 60, 95], "z": [30, 60, 90]},
}

BODY_GRIDS: dict[str, dict[str, list[int]]] = {**FACE_GRIDS, **TORSO_GRIDS}
BATCHES = {
    "face": FACE_GRIDS,
    "torso": TORSO_GRIDS,
    "all": BODY_GRIDS,
}


def _font(size: int = 13) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _candidates(axes: dict[str, list[int]]) -> list[tuple[str, dict[str, int]]]:
    out: list[tuple[str, dict[str, int]]] = []
    for i, (x, y, z) in enumerate(product(axes["x"], axes["y"], axes["z"])):
        out.append((f"#{i:02d}", {"x": int(x), "y": int(y), "z": int(z)}))
    return out


def _pose_joints(cue: str, xyz: dict[str, int], xyz_db: dict[str, Any]) -> dict[str, float]:
    """Workspace-driven pose for grid search.

    Dir silhouette is only a soft bias (35%); xyz owns most of the travel so
    wide grids look clearly different instead of tiny nudges around one preset.
    """
    _, meta = pb.build_paddle_code(cue, xyz_db)
    arm = meta.get("dir") or "front"
    mirror = bool(meta.get("mirror"))
    opposed = bool(meta.get("opposed"))
    palm = meta.get("palm_hv") or "flat"

    joints: dict[str, float] = {
        **pb._preset("claw_still"),
        **pb._preset("paddle_still"),
        **rc.A1["grip_open"],
    }
    # soft canvas-dir bias
    claw_key = f"claw_{arm}" if f"claw_{arm}" in pb.A1_DIRS else "claw_front"
    for k, v in pb._preset(claw_key).items():
        joints[k] = 0.35 * float(v) + 0.65 * float(joints.get(k, 0.0))

    if mirror or opposed:
        pad_arm = arm
        if opposed:
            pad_arm = {"front": "back", "back": "front", "in": "out", "out": "in", "up": "up", "down": "down"}.get(
                arm, "front"
            )
        pad_key = f"paddle_{pad_arm}" if f"paddle_{pad_arm}" in pb.A1_DIRS else "paddle_front"
        for k, v in pb._preset(pad_key).items():
            joints[k] = 0.35 * float(v) + 0.65 * float(joints.get(k, 0.0))
        if palm == "edge":
            joints["bl2"] = joints.get("bl2", 0.5) + 0.35
        else:
            joints["bl2"] = joints.get("bl2", 0.9) * 0.7
    else:
        joints.update(pb._preset("paddle_still"))

    if palm == "edge":
        joints["br3"] = joints.get("br3", 0.0) + 0.55
    pb._apply_xyz_offsets(joints, xyz, gain=1.15)
    return joints


def _render_still(
    model: mujoco.MjModel,
    renderer: mujoco.Renderer,
    cam: mujoco.MjvCamera,
    joints: dict[str, float],
    *,
    base_z: float,
) -> Image.Image:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    rc._set_base(data, model, 0.0, 0.0, 0.0, base_z)
    for jn, v in joints.items():
        rc._set_joint(data, model, jn, v)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return Image.fromarray(renderer.render()).convert("RGB")


def _tile(image: Image.Image, label: str, xyz: dict[str, int], *, current: bool = False) -> Image.Image:
    bar = 30
    out = Image.new("RGB", (image.width, image.height + bar), (255, 252, 240) if current else "white")
    out.paste(image, (0, bar))
    draw = ImageDraw.Draw(out)
    tag = f"{label} ({xyz['x']},{xyz['y']},{xyz['z']})"
    if current:
        tag = "CUR " + tag
    draw.text((5, 6), tag, fill=(180, 40, 20) if current else (20, 20, 20), font=_font(12))
    return out


def _montage(tiles: list[Image.Image], cols: int = 9) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    out = Image.new("RGB", (cols * w, rows * h), (235, 235, 235))
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % cols) * w, (i // cols) * h))
    return out


def _current_xyz(cue: str, xyz_db: dict[str, Any]) -> dict[str, int]:
    _, meta = pb.build_paddle_code(cue, xyz_db)
    xyz = meta.get("xyz") or {}
    return {a: int(xyz[a]) for a in ("x", "y", "z") if a in xyz}


def run_grids(wanted: list[str], *, force: bool = False) -> list[dict[str, Any]]:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    xyz_db = json.loads(pb.XYZ_JSON.read_text(encoding="utf-8"))
    idxs = {r["cue"]: int(r["index"]) for r in pb._RC90_ROWS}

    model = mujoco.MjModel.from_xml_path(str(ODDKIN / "xml" / "gcr_alpha1.xml"))
    renderer = mujoco.Renderer(model, height=rc.H, width=rc.W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.0, 0.0, 0.40)
    cam.distance = 2.05
    cam.azimuth = 180
    cam.elevation = -8
    base_z = 0.13

    rows: list[dict[str, Any]] = []
    try:
        for ci, cue in enumerate(wanted, 1):
            axes = BODY_GRIDS[cue]
            out_png = OUT_ROOT / f"{cue}.png"
            cands = _candidates(axes)
            cur = _current_xyz(cue, xyz_db)
            tiles: list[Image.Image] = []
            if out_png.is_file() and out_png.stat().st_size > 1000 and not force:
                print(f"[{ci}/{len(wanted)}] skip existing {cue}", flush=True)
            else:
                # leading tile: current paddle90 xyz
                if cur:
                    j = _pose_joints(cue, cur, xyz_db)
                    img = _render_still(model, renderer, cam, j, base_z=base_z)
                    tiles.append(_tile(img, "CUR", cur, current=True))
                for label, xyz in cands:
                    j = _pose_joints(cue, xyz, xyz_db)
                    img = _render_still(model, renderer, cam, j, base_z=base_z)
                    tiles.append(_tile(img, label, xyz))
                _montage(tiles, cols=9).save(out_png)
                print(f"[{ci}/{len(wanted)}] {cue} n={len(cands)} +cur → {out_png.name}", flush=True)

            loc = next((r.get("location") for r in pb._RC90_ROWS if r["cue"] == cue), "")
            pose = pb.CANVAS[cue]["pose"]
            rows.append(
                {
                    "idx": idxs.get(cue, 0),
                    "cue": cue,
                    "location": loc,
                    "pose": pose,
                    "current": cur,
                    "n": len(cands),
                    "candidates": [{"label": lab, **xyz} for lab, xyz in cands],
                }
            )
    finally:
        renderer.close()

    (OUT_ROOT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not SELECTION.is_file():
        SELECTION.write_text(
            json.dumps({r["cue"]: None for r in rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        sel = json.loads(SELECTION.read_text(encoding="utf-8"))
        for r in rows:
            sel.setdefault(r["cue"], None)
        SELECTION.write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")

    cards = []
    for row in sorted(rows, key=lambda e: e["idx"]):
        cur = row.get("current") or {}
        cur_s = f"cur=({cur.get('x')},{cur.get('y')},{cur.get('z')})" if cur else "cur=—"
        cards.append(
            "<article>"
            f"<h2>{html.escape(str(row['idx']) + '. ' + row['cue'])}</h2>"
            f"<p class='meta'>{html.escape(str(row.get('location') or ''))} · {html.escape(cur_s)}<br>"
            f"<code>{html.escape(row.get('pose') or '')}</code></p>"
            f"<img src='{html.escape(row['cue'] + '.png')}?v=3' alt='{html.escape(row['cue'])}'>"
            "<p class='hint'>Pick label <b>#NN</b> → write into <code>selection.json</code> "
            "as <code>{\"cue\": {\"x\":..,\"y\":..,\"z\":..}}</code> or "
            "<code>{\"cue\": \"#NN\"}</code>.</p></article>"
        )
    (OUT_ROOT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Paddle body XYZ grids</title>"
        "<style>body{font-family:system-ui;margin:16px;background:#e8eaef;color:#222}"
        "h1{font-size:22px;margin:0 0 8px}.sub{color:#555;max-width:1100px;line-height:1.4}"
        ".grid{display:grid;grid-template-columns:1fr;gap:16px}"
        "article{background:#fff;padding:12px;border-radius:10px;border:1px solid #ccd}"
        "img{width:100%;image-rendering:auto} h2{margin:0 0 4px;font-size:16px}"
        ".meta{font-size:12px;color:#444;margin:0 0 8px} code{font-size:11px}"
        ".hint{font-size:12px;color:#666;margin:8px 0 0}</style>"
        "<h1>Paddlebot · body-anchored XYZ grids (wide workspace)</h1>"
        "<p class='sub'>Each cue: <b>CUR</b> = current paddle90, then 3×3×3 candidates "
        "spanning near-full workspace (low↔high, tucked↔reach, out↔in). "
        "Dir is only a soft bias; xyz drives the pose. "
        "Write picks to <code>selection.json</code>, then "
        "<code>python motion_code/search_paddle_body_xyz.py --apply</code>.</p>"
        f"<div class='grid'>{''.join(cards)}</div>",
        encoding="utf-8",
    )
    print(OUT_ROOT / "index.html", flush=True)
    return rows


def apply_selection() -> Path:
    if not SELECTION.is_file():
        raise SystemExit(f"missing {SELECTION}")
    man = {r["cue"]: r for r in json.loads((OUT_ROOT / "manifest.json").read_text(encoding="utf-8"))}
    raw = json.loads(SELECTION.read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = {}
    for cue, val in raw.items():
        if val is None:
            continue
        if isinstance(val, dict) and all(k in val for k in ("x", "y", "z")):
            out[cue] = {"x": int(val["x"]), "y": int(val["y"]), "z": int(val["z"])}
            continue
        if isinstance(val, str) and val.startswith("#") and cue in man:
            lab = val.strip()
            hit = next((c for c in man[cue]["candidates"] if c["label"] == lab), None)
            if not hit:
                raise SystemExit(f"{cue}: unknown label {lab}")
            out[cue] = {"x": int(hit["x"]), "y": int(hit["y"]), "z": int(hit["z"])}
            continue
        raise SystemExit(f"{cue}: bad selection {val!r}")
    OVERRIDE_JSON.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_JSON.write_text(json.dumps({"platform": "gcr_alpha1", "cues": out}, indent=2), encoding="utf-8")
    print(f"[apply] {len(out)} cues → {OVERRIDE_JSON}", flush=True)
    return OVERRIDE_JSON


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=list(BATCHES), default="all")
    ap.add_argument("--cues", default="", help="comma-separated cue names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--apply", action="store_true", help="write selection.json → overrides")
    args = ap.parse_args()

    if args.apply:
        apply_selection()
        return

    if args.cues.strip():
        wanted = [c.strip() for c in args.cues.split(",") if c.strip()]
        for c in wanted:
            if c not in BODY_GRIDS:
                raise SystemExit(f"unknown cue (add to BODY_GRIDS): {c}")
    else:
        wanted = list(BATCHES[args.batch])
    run_grids(wanted, force=args.force)


if __name__ == "__main__":
    main()
