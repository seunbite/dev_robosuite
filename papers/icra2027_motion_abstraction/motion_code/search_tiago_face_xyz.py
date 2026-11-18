#!/usr/bin/env python3
"""Dense XYZ combo search for Tiago face/body-contact cues."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ADHOC = REPO / "adhoc/generation/robotarm"
TIAGO = REPO / "adhoc/generation/tiago"
sys.path[:0] = [str(MC), str(PAPER), str(REPO), str(ADHOC), str(TIAGO)]

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from correction_lib import _import_tiago_render  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from render_synth60_tiago_paddle import (  # noqa: E402
    CUES60,
    XYZ_JSON,
    build_tiago_source_code,
    cue_indices,
)

OUT_ROOT = MC / "bank/_tiago_xyz_iter/face_combo"

# Per-cue candidate grids aimed at face / contact sites (body %).
FACE_GRIDS: dict[str, dict[str, list[int]]] = {
    # head / temple / mouth band
    "self_scratch_head_confused": {"x": [68, 75, 82], "y": [72, 80, 88], "z": [72, 80, 88]},
    "fan_face_hot": {"x": [72, 80, 88], "y": [55, 65, 75], "z": [68, 76, 84]},
    "tap_temple_think": {"x": [65, 72, 80], "y": [75, 82, 90], "z": [72, 80, 88]},
    "circle_temple_crazy": {"x": [65, 72, 80], "y": [75, 82, 90], "z": [72, 80, 88]},
    "shh_gesture": {"x": [75, 82, 90], "y": [55, 65, 75], "z": [62, 70, 78]},
    "visor_search": {"x": [60, 70, 80], "y": [45, 55, 65], "z": [70, 80, 90]},
    # back / arms / clap contact
    "pat_self_back": {"x": [35, 45, 55], "y": [78, 88, 95], "z": [55, 65, 75]},
    "rub_arms_cold": {"x": [55, 65, 75], "y": [70, 80, 90], "z": [48, 56, 64]},
    "clap_applause": {"x": [70, 78, 85], "y": [68, 78, 88], "z": [55, 65, 75]},
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _with_xyz(db: dict[str, Any], cue: str, xyz: dict[str, int]) -> dict[str, Any]:
    out = deepcopy(db)
    (out.get("cues") or out)[cue]["tiago"] = dict(xyz)
    return out


def _render_pose(render_config, env, cue: str, xyz_db: dict[str, Any]) -> tuple[Image.Image, str]:
    code, _ = build_tiago_source_code(cue, xyz_db)
    frag = convert_code(code, platform="tiago")
    pose_steps = [s for s in (frag.get("movements") or []) if s.get("type") == "pose"]
    pose = deepcopy(pose_steps[0])
    pose["duration"] = 0.08
    frames = render_config(
        {"movements": [pose]},
        env=env,
        overlay_progress_bar=False,
        ignore_xyz=False,
    )
    return frames[-1].convert("RGB"), code


def _tile(image: Image.Image, label: str, xyz: dict[str, int]) -> Image.Image:
    bar = 28
    out = Image.new("RGB", (image.width, image.height + bar), "white")
    out.paste(image, (0, bar))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), f"{label} ({xyz['x']},{xyz['y']},{xyz['z']})", fill=(20, 20, 20), font=_font(13))
    return out


def _montage(tiles: list[Image.Image], cols: int = 5) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    out = Image.new("RGB", (cols * w, rows * h), (240, 240, 240))
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % cols) * w, (i // cols) * h))
    return out


def _candidates(axes: dict[str, list[int]]) -> list[tuple[str, dict[str, int]]]:
    xs, ys, zs = axes["x"], axes["y"], axes["z"]
    out: list[tuple[str, dict[str, int]]] = []
    for i, (x, y, z) in enumerate(product(xs, ys, zs)):
        xyz = {"x": int(x), "y": int(y), "z": int(z)}
        out.append((f"#{i:02d}", xyz))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", default="")
    args = ap.parse_args()
    wanted = [c.strip() for c in args.cues.split(",") if c.strip()] or list(FACE_GRIDS)
    xyz_db = json.loads(XYZ_JSON.read_text(encoding="utf-8"))
    idxs = cue_indices()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    _make_env, render_config, _ = _import_tiago_render()
    env = _make_env()
    rows: list[dict[str, Any]] = []
    try:
        for ci, cue in enumerate(wanted, 1):
            if cue not in FACE_GRIDS:
                raise KeyError(cue)
            tiles = []
            cands = _candidates(FACE_GRIDS[cue])
            for label, xyz in cands:
                img, code = _render_pose(render_config, env, cue, _with_xyz(xyz_db, cue, xyz))
                tiles.append(_tile(img, label, xyz))
            _montage(tiles).save(OUT_ROOT / f"{cue}.png")
            rows.append(
                {
                    "idx": int(idxs.get(cue, 0)),
                    "cue": cue,
                    "n": len(cands),
                    "candidates": [{"label": lab, **xyz} for lab, xyz in cands],
                }
            )
            print(f"[{ci}/{len(wanted)}] {cue} n={len(cands)}", flush=True)
    finally:
        env.close()
    (OUT_ROOT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cards = []
    for row in rows:
        cards.append(
            f"<article><h2>{html.escape(str(row['idx'])+'. '+row['cue'])}</h2>"
            f"<img src='{html.escape(row['cue']+'.png')}'></article>"
        )
    (OUT_ROOT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Tiago face XYZ combos</title>"
        "<style>body{font-family:system-ui;margin:16px;background:#eee}.grid{display:grid;"
        "grid-template-columns:1fr;gap:14px}article{background:#fff;padding:10px;border-radius:8px}"
        "img{width:100%}</style><h1>Face/contact XYZ combinations</h1>"
        f"<div class=grid>{''.join(cards)}</div>",
        encoding="utf-8",
    )
    print(OUT_ROOT / "index.html")


if __name__ == "__main__":
    main()
