#!/usr/bin/env python3
"""Demo: Paddlebot (gcr_alpha1) supports full 6-way reach on claw AND paddle.

Schema (motion_code_synthetic_continuum.yaml):
  claw/paddle ∈ {front, back, in, out, up, down} (+ still)

Renders a small hold gallery and writes:
  motion_code/paddle_6dir_gallery.html
  data/results/render/paddle_6dir_demo/gcr_alpha1/gif/
"""
from __future__ import annotations

import html as html_mod
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

MC = Path(__file__).resolve().parent
PAPER = MC.parent
REPO = PAPER.parent.parent
ODDKIN = PAPER / "synthetic_morphology"
sys.path[:0] = [str(ODDKIN), str(PAPER), str(REPO)]

import mujoco  # noqa: E402
import render_cues30 as rc  # noqa: E402

DIRS = ("front", "back", "in", "out", "up", "down")
OUT_GIF = REPO / "data/results/render/paddle_6dir_demo/gcr_alpha1/gif"
OUT_HTML = MC / "paddle_6dir_gallery.html"

# 6-way reach: ARM JOINTS ONLY (no trunk/body). World-frame opposites as far as
# the claw/paddle workspace allows from a fixed front-facing torso.
A1_DIRS: dict[str, dict[str, float]] = {
    # claw: front(+x)↔back(retract/−x), out(−y)↔in(+y), up(+z)↔down(−z)
    "claw_front": {"br0": 0.02, "br1": 0.32, "br2": 0.23, "br3": 0.0},
    "claw_back": {"br0": 0.03, "br1": -2.20, "br2": -1.02, "br3": 0.0},  # fold rearward; height-penalized
    "claw_in": {"br0": 0.99, "br1": -0.01, "br2": 0.43, "br3": 0.0},
    "claw_out": {"br0": -1.14, "br1": 0.00, "br2": 0.08, "br3": 0.0},  # straightened lateral
    "claw_up": {"br0": -0.20, "br1": -0.70, "br2": 0.25, "br3": 0.0},  # raised forward (not folded-back)
    "claw_down": {"br0": -0.40, "br1": 0.40, "br2": 0.48, "br3": 0.0},  # hang low near body
    "claw_still": {"br0": -0.55, "br1": 0.25, "br2": 0.15, "br3": 0.0},
    # paddle
    "paddle_front": {"bl0": 0.10, "bl1": -0.20, "bl2": 1.00},
    "paddle_back": {"bl0": 1.15, "bl1": -0.95, "bl2": 0.55},  # elbow retract; not pure out
    "paddle_in": {"bl0": -0.99, "bl1": 0.00, "bl2": 0.70},
    "paddle_out": {"bl0": 1.18, "bl1": 0.00, "bl2": 0.50},
    "paddle_up": {"bl0": 0.10, "bl1": -1.15, "bl2": 1.00},
    "paddle_down": {"bl0": 1.15, "bl1": 0.50, "bl2": 0.35},  # max hang (lowest tip z)
    "paddle_still": {"bl0": 0.55, "bl1": 0.30, "bl2": 0.35},
}


def _preset(key: str) -> dict:
    if key in A1_DIRS:
        return dict(A1_DIRS[key])
    if key in rc.A1:
        return dict(rc.A1[key])
    raise KeyError(key)


def _joints(arm: str, d: str) -> dict:
    """arm ∈ {claw, paddle}; other arm explicitly parked still."""
    j = {}
    if arm == "claw":
        j.update(_preset(f"claw_{d}"))
        j.update(_preset("paddle_still"))
        j.update(rc.A1["grip_open"])
    else:
        j.update(_preset(f"paddle_{d}"))
        j.update(_preset("claw_still"))
        j.update(rc.A1["grip_open"])
    return j


def main() -> None:
    OUT_GIF.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ODDKIN / "xml" / "gcr_alpha1.xml"))
    lookat, distance, base_z = (0.0, 0.0, 0.40), 2.05, 0.13

    cards: list[tuple[str, str, Path]] = []
    for arm in ("claw", "paddle"):
        for d in DIRS:
            code = f"pose(dir=front, torso=mid, {arm}={d}, {'grip=open, paddle=still' if arm == 'claw' else 'claw=still, palm=flat'})"
            name = f"{arm}_{d}"
            out = OUT_GIF / f"{name}.gif"
            plan = [
                {"op": "pose", "home": "1", "joints": _joints(arm, d), "n": 22},
            ]
            frames = rc.execute_plan(plan)
            rc.render_clip(
                model,
                frames,
                out,
                title=f"{arm} = {d}",
                subtitle=code,
                lookat=lookat,
                distance=distance,
                base_z=base_z,
            )
            cards.append((arm, d, out))
            print(f"[ok] {name} -> {out}", flush=True)

    # HTML
    rel = lambda p: os.path.relpath(p, OUT_HTML.parent)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Paddlebot 6-dir channels</title>",
        "<style>",
        "body{font-family:-apple-system,Helvetica,Arial,sans-serif;margin:20px;background:#f4f4f4;color:#222}",
        "h1{font-size:22px;margin:0 0 6px} .sub{color:#555;max-width:980px;line-height:1.45;margin-bottom:14px}",
        "h2{font-size:16px;margin:18px 0 8px}",
        ".grid{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px}",
        ".card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:10px}",
        ".card img{width:100%;background:#0d0d0d;border-radius:6px;aspect-ratio:1;object-fit:contain}",
        ".tok{font-family:ui-monospace,Menlo,monospace;font-size:11px;background:#f0f3f8;",
        "padding:6px 7px;border-radius:5px;margin-top:6px;word-break:break-word}",
        ".pill{display:inline-block;background:#e3f2fd;border:1px solid #64b5f6;color:#0d47a1;",
        "border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}",
        "</style></head><body>",
        "<h1>Paddlebot · 6-way reach channels</h1>",
        "<p class='sub'><span class='pill'>claw × 6</span><span class='pill'>paddle × 6</span>"
        "Arm joints only (no trunk). Same vocab as canvas <b>dir</b>: "
        "front · back · in · out · up · down.</p>",
    ]
    for arm in ("claw", "paddle"):
        parts.append(f"<h2>{arm}=dir (other arm still)</h2><div class='grid'>")
        for d in DIRS:
            out = OUT_GIF / f"{arm}_{d}.gif"
            code = (
                f"pose(dir=front, torso=mid, {arm}={d}, "
                + ("grip=open, paddle=still)" if arm == "claw" else "claw=still, palm=flat)")
            )
            parts.append(
                "<div class='card'>"
                f"<img src='{html_mod.escape(rel(out))}?v=6' alt='{arm}_{d}'>"
                f"<div class='tok'><b>{arm}={d}</b><br>{html_mod.escape(code)}</div>"
                "</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")
    print(f"[html] {OUT_HTML}", flush=True)


if __name__ == "__main__":
    main()
