#!/usr/bin/env python3
"""Single-axis Tiago XYZ refinement sweeps."""
from __future__ import annotations

import html
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
sys.path[:0] = [
    str(MC),
    str(PAPER),
    str(REPO),
    str(REPO / "adhoc/generation/robotarm"),
    str(REPO / "adhoc/generation/tiago"),
]

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from correction_lib import _import_tiago_render  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from render_synth60_tiago_paddle import (  # noqa: E402
    XYZ_JSON,
    build_tiago_source_code,
    cue_indices,
)

OUT = MC / "bank/_tiago_xyz_iter/axis_refine"

# cue → axis to sweep, value list (others held at current JSON center)
SWEEPS: dict[str, tuple[str, list[int]]] = {
    "self_scratch_head_confused": ("y", [68, 74, 80, 86, 92, 98]),
    "fan_face_hot": ("y", [55, 62, 70, 78, 86, 94]),
    "tap_temple_think": ("y", [70, 76, 82, 88, 94, 100]),
    "circle_temple_crazy": ("y", [70, 76, 82, 88, 94, 100]),
    "pat_self_back": ("x", [20, 28, 36, 44, 52, 60]),
    "visor_search": ("z", [70, 76, 82, 88, 94, 100]),
    "shh_gesture": ("z", [58, 64, 70, 76, 82, 88]),
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _center(db: dict[str, Any], cue: str) -> dict[str, int]:
    t = ((db.get("cues") or {}).get(cue) or {}).get("tiago") or {}
    return {a: int(round(float(t.get(a, 50)))) for a in ("x", "y", "z")}


def _with(db: dict[str, Any], cue: str, xyz: dict[str, int]) -> dict[str, Any]:
    out = deepcopy(db)
    (out.get("cues") or out)[cue]["tiago"] = dict(xyz)
    return out


def _render(render_config, env, cue: str, db: dict[str, Any]) -> Image.Image:
    code, _ = build_tiago_source_code(cue, db)
    frag = convert_code(code, platform="tiago")
    pose = deepcopy([s for s in frag["movements"] if s["type"] == "pose"][0])
    pose["duration"] = 0.08
    frames = render_config(
        {"movements": [pose]}, env=env, overlay_progress_bar=False, ignore_xyz=False
    )
    return frames[-1].convert("RGB")


def _tile(img: Image.Image, label: str) -> Image.Image:
    bar = 28
    out = Image.new("RGB", (img.width, img.height + bar), "white")
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((6, 6), label, fill=(15, 15, 18), font=_font(14))
    return out


def main() -> None:
    db = json.loads(XYZ_JSON.read_text(encoding="utf-8"))
    idxs = cue_indices()
    OUT.mkdir(parents=True, exist_ok=True)
    _make_env, render_config, _ = _import_tiago_render()
    env = _make_env()
    rows: list[dict[str, Any]] = []
    try:
        for cue, (axis, values) in SWEEPS.items():
            base = _center(db, cue)
            tiles = []
            cands = []
            for v in values:
                xyz = dict(base)
                xyz[axis] = max(0, min(100, int(v)))
                img = _render(render_config, env, cue, _with(db, cue, xyz))
                lab = f"{axis}={xyz[axis]}  ({xyz['x']},{xyz['y']},{xyz['z']})"
                tiles.append(_tile(img, lab))
                cands.append(dict(xyz))
            w, h = tiles[0].size
            mont = Image.new("RGB", (len(tiles) * w, h), (245, 245, 245))
            for i, t in enumerate(tiles):
                mont.paste(t, (i * w, 0))
            mont.save(OUT / f"{cue}.png")
            rows.append(
                {
                    "idx": int(idxs.get(cue, 0)),
                    "cue": cue,
                    "axis": axis,
                    "center": base,
                    "candidates": cands,
                }
            )
            print(f"[ok] {cue} {axis}={values}", flush=True)
    finally:
        env.close()
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    cards = "".join(
        f"<article><h2>{html.escape(str(r['idx'])+'. '+r['cue']+' · '+r['axis'])}</h2>"
        f"<img src='{html.escape(r['cue']+'.png')}'></article>"
        for r in rows
    )
    (OUT / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>axis refine</title>"
        "<style>body{font-family:system-ui;margin:16px;background:#eee}"
        "img{width:100%;background:#fff} article{margin:0 0 14px;padding:10px;"
        "background:#fff;border-radius:8px}</style>"
        f"<h1>Single-axis refine</h1>{cards}",
        encoding="utf-8",
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
