#!/usr/bin/env python3
"""Render XYZ-only Tiago pose neighborhoods for visual selection.

The comparison deliberately renders only the initial pose. Direction, gripper
orientation, left-arm mode, head, and torso are rebuilt through the production
Synth60 pipeline and audited to remain identical across candidates.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from copy import deepcopy
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

OUT_ROOT = MC / "bank/_tiago_xyz_iter"
AXES = ("x", "y", "z")


def _font(size: int = 15) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _non_xyz_signature(code: str) -> str:
    return re.sub(r"\b[xyz]=-?\d+(?:\.\d+)?\s*,?", "", code)


def _load_centers(path: Path | None, xyz_db: dict[str, Any]) -> dict[str, dict[str, int]]:
    centers: dict[str, dict[str, int]] = {}
    picked: dict[str, Any] = {}
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        picked = payload.get("selected") or payload
    for cue in CUES60:
        base = ((xyz_db.get("cues") or {}).get(cue) or {}).get("tiago") or {}
        override = picked.get(cue) or {}
        centers[cue] = {
            axis: int(round(float(override.get(axis, base.get(axis, 50))))) for axis in AXES
        }
    return centers


def _candidates(center: dict[str, int], delta: int) -> list[tuple[str, dict[str, int]]]:
    out = [("C", dict(center))]
    for axis in AXES:
        for sign, suffix in ((-1, "−"), (1, "+")):
            xyz = dict(center)
            xyz[axis] = max(0, min(100, xyz[axis] + sign * delta))
            out.append((f"{axis}{suffix}", xyz))
    return out


def _with_xyz(db: dict[str, Any], cue: str, xyz: dict[str, int]) -> dict[str, Any]:
    candidate_db = deepcopy(db)
    row = (candidate_db.get("cues") or candidate_db)[cue]
    row["tiago"] = dict(xyz)
    return candidate_db


def _render_pose(render_config, env, cue: str, xyz_db: dict[str, Any]) -> tuple[Image.Image, str]:
    code, _meta = build_tiago_source_code(cue, xyz_db)
    frag = convert_code(code, platform="tiago")
    pose_steps = [step for step in frag.get("movements") or [] if step.get("type") == "pose"]
    if not pose_steps:
        raise RuntimeError(f"no pose step for {cue}")
    pose = deepcopy(pose_steps[0])
    pose["duration"] = 0.1
    frames = render_config(
        {"movements": [pose]},
        env=env,
        overlay_progress_bar=False,
        ignore_xyz=False,
    )
    if not frames:
        raise RuntimeError(f"no frames for {cue}")
    return frames[-1].convert("RGB"), code


def _tile(image: Image.Image, label: str, xyz: dict[str, int]) -> Image.Image:
    bar = 31
    out = Image.new("RGB", (image.width, image.height + bar), "white")
    out.paste(image, (0, bar))
    draw = ImageDraw.Draw(out)
    draw.text(
        (7, 7),
        f"{label}: ({xyz['x']}, {xyz['y']}, {xyz['z']})",
        fill=(15, 15, 18),
        font=_font(15),
    )
    return out


def _montage(tiles: list[Image.Image]) -> Image.Image:
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    out = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for i, tile in enumerate(tiles):
        out.paste(tile, ((i % cols) * w, (i // cols) * h))
    return out


def _write_html(rows: list[dict[str, Any]], out_dir: Path, delta: int) -> None:
    cards = []
    for row in rows:
        cue = str(row["cue"])
        cards.append(
            "<article><h2>"
            + html.escape(f"{row['idx']}. {cue}")
            + "</h2><img loading='lazy' src='"
            + html.escape(f"{cue}.png")
            + "'><pre>"
            + html.escape(str(row["orientation_signature"]))
            + "</pre></article>"
        )
    page = f"""<!doctype html><meta charset='utf-8'>
<title>Tiago XYZ iteration ±{delta}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:20px;background:#eee;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:16px}}
article{{background:white;padding:12px;border:1px solid #ccc;border-radius:10px}}
h1{{font-size:22px}} h2{{font-size:15px;margin:0 0 8px}}
img{{width:100%;height:auto;display:block}} pre{{white-space:pre-wrap;font-size:10px}}
</style><h1>Tiago XYZ-only neighborhood · delta={delta}</h1>
<p>C=current; all other candidates change exactly one XYZ axis. Non-XYZ Motion Code is locked.</p>
<div class='grid'>{''.join(cards)}</div>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delta", type=int, default=6)
    ap.add_argument("--stage", default="coarse")
    ap.add_argument("--centers", type=Path)
    ap.add_argument("--cues", default="", help="Comma-separated cue names; default=all 60")
    args = ap.parse_args()

    xyz_db = json.loads(XYZ_JSON.read_text(encoding="utf-8"))
    centers = _load_centers(args.centers, xyz_db)
    wanted = [c.strip() for c in args.cues.split(",") if c.strip()] or list(CUES60)
    unknown = sorted(set(wanted) - set(CUES60))
    if unknown:
        raise KeyError(f"unknown cues: {unknown}")

    out_dir = OUT_ROOT / args.stage
    out_dir.mkdir(parents=True, exist_ok=True)
    idxs = cue_indices()
    _make_env, render_config, _mod = _import_tiago_render()
    env = _make_env()
    rows: list[dict[str, Any]] = []
    try:
        for ci, cue in enumerate(wanted, start=1):
            tiles: list[Image.Image] = []
            codes: list[str] = []
            candidates = _candidates(centers[cue], args.delta)
            for label, xyz in candidates:
                image, code = _render_pose(render_config, env, cue, _with_xyz(xyz_db, cue, xyz))
                tiles.append(_tile(image, label, xyz))
                codes.append(code)
            signatures = {_non_xyz_signature(code) for code in codes}
            if len(signatures) != 1:
                raise AssertionError(f"non-XYZ fields changed for {cue}")
            _montage(tiles).save(out_dir / f"{cue}.png")
            rows.append(
                {
                    "idx": int(idxs.get(cue, 0)),
                    "cue": cue,
                    "center": centers[cue],
                    "delta": args.delta,
                    "candidates": [{"label": label, **xyz} for label, xyz in candidates],
                    "orientation_signature": next(iter(signatures)),
                }
            )
            print(f"[{ci:02d}/{len(wanted):02d}] {cue}", flush=True)
    finally:
        env.close()
    (out_dir / "manifest.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_html(rows, out_dir, args.delta)
    print(out_dir / "index.html")


if __name__ == "__main__":
    main()
