#!/usr/bin/env python3
"""Expose the 5 × 5 embodiment few-shot inputs used by vector baselines."""
from __future__ import annotations

import html
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
OUT = REPO / "data/results/html/fewshot_examples_5x5_embodiment.html"
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")


def block(title: str, value: object) -> str:
    raw = html.escape(json.dumps(value, ensure_ascii=False, indent=2))
    return (
        f"<details><summary>{html.escape(title)}</summary>"
        f"<pre>{raw}</pre></details>"
    )


def stretch_cartesian(shot: dict) -> dict[str, object]:
    joint = shot["joint_angles"]
    names = joint["joint_names"]
    columns = {name: i for i, name in enumerate(names)}
    rows = joint["joints_rad"]
    return {
        "schema": "11D per frame = base(3) + head(2) + hand(6)",
        "base_xyyaw": [[row[columns[name]] for name in ("base_x", "base_y", "base_yaw")] for row in rows],
        "head_rp": [[row[columns[name]] for name in ("head_pan", "head_tilt")] for row in rows],
        "hand_xyzrpy": shot["xyzrpy"],
    }


def main() -> None:
    shots = json.loads(SOURCE.read_text(encoding="utf-8"))
    columns = []
    for platform in PLATFORMS:
        cards = []
        for shot in shots[platform]:
            joint = shot["joint_angles"]
            cards.append(
                "<article>"
                f"<h3>{html.escape(shot['shot_id'].split('__')[1])}</h3>"
                f"<p><b>Frames:</b> {len(joint['joints_rad'])} · "
                f"<b>Hz:</b> {shot.get('hz', 8)}</p>"
                f"<details><summary>Motion Program</summary><pre>{html.escape(shot['motion_code'])}</pre></details>"
                + block("Joint-Space input", joint)
                + block(
                    "Cartesian-Space input (base + head + hand, 11D)"
                    if platform == "stretch" else "Cartesian-Space input (xyzrpy)",
                    stretch_cartesian(shot) if platform == "stretch" else {
                    "link_names": shot.get("link_names", ["end_effector"]),
                    "xyzrpy": shot["xyzrpy"],
                    },
                )
                + "</article>"
            )
        columns.append(
            f"<section><h2>{html.escape(platform.replace('_', ' ').title())}</h2>"
            + "".join(cards)
            + "</section>"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>5 × 5 Embodiment Few-shot Inputs</title><style>
body{margin:22px;background:#0d1117;color:#e6edf3;font-family:ui-sans-serif,system-ui,sans-serif}h1{margin:0 0 6px}p.sub{color:#9da7b3}.grid{display:grid;grid-template-columns:repeat(5,minmax(330px,1fr));gap:12px;overflow-x:auto;align-items:start}section{min-width:330px}h2{margin:12px 0;font-size:17px}article{background:#161b22;border:1px solid #30363d;margin-bottom:10px;padding:12px}h3{font-size:14px;margin:0 0 7px}p{font-size:12px;color:#9da7b3}details{margin-top:7px;border:1px solid #30363d;padding:6px}summary{cursor:pointer;font-size:12px;font-weight:700}pre{white-space:pre-wrap;word-break:break-word;font-size:10px;line-height:1.45;color:#c9d1d9}
</style></head><body><h1>5 × 5 embodiment few-shot inputs</h1>
<p class='sub'>Exact five-shot examples used for Joint-Space and Cartesian-Space baseline prompts. Stretch Cartesian is 11D: base (x,y,yaw), head (pan,tilt), hand (x,y,z,roll,pitch,yaw).</p><div class='grid'>"""
        + "".join(columns)
        + "</div></body></html>",
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
