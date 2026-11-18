#!/usr/bin/env python3
"""Strategy-conditioned Motion Code regeneration for Stretch and Single Arm.

Same protocol as generate_roomba_strategy_fewshot.py: the three
rematerialization strategies (carrier transfer / shape tracing / additive
movement) are taught explicitly with one human-authored few-shot each, then one
inference per cue with no retry loop.  Only the strategy block and the body
schema differ between conditions, so the result is comparable to the
unconditioned programs of the same generator.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from google import genai

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from motion_code_to_config import convert_code  # noqa: E402

GLOSS = REPO / "papers/icra2027_motion_abstraction/gesture_critical_factors_90.json"
GT = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"

BODIES = {
    "stretch": {
        "out": REPO / "data/results/stretch_strategy_fewshot",
        "blurb": (
            "a Hello Robot Stretch: a mobile base, a telescoping vertical mast, a "
            "single arm that extends laterally from that mast, a two-finger gripper, "
            "and a pan/tilt head. It has no second arm and no hand."
        ),
        "grammar": (
            "- pose(dir=front|back|up|down|in|out, grip=horiz|vert, torso=low|mid|high, x=.., z=..)\n"
            "  places the gripper; x/z are body-percent in [0,100]. There is no lateral y target.\n"
            "- move(head|torso|wrist, tilt|pan|height|x|y|z, <signed deg or list>, rep=.., spd=.., hold=..)\n"
            "- path(ee|base, line|arc, ...) with dist/len or r/deg/sweep, plus spd/rep/hold.\n"
            "- Connect with + (sequential), & (concurrent with the next step), hold(t=..)."
        ),
        "shots": [
            ("Carrier transfer", "shake_head_no",
             "pose(dir=still, torso=mid) + move(head, pan, [-30, 30], rep=2, spd=2.2, hold=0.1)"),
            ("Shape tracing", "depict_big_bowl",
             "pose(dir=out, grip=horiz, torso=mid, z=70, x=30) + pose(dir=in, grip=horiz, torso=mid, z=55, x=20) "
             "+ path(ee, arc, plane=yz, r=0.18, sweep=-160, spd=1.0) + hold(t=1.2)"),
            ("Additive movement", "stop_palm_out",
             "pose(dir=front, grip=horiz, torso=high, z=80, x=25) + path(base, arc, r=0, deg=±12, rep=2, spd=2.0) + hold(t=1.5)"),
        ],
    },
    "single_arm": {
        "out": REPO / "data/results/single_arm_strategy_fewshot",
        "blurb": (
            "a KUKA IIWA: one 7-DoF arm on a fixed pedestal with a two-finger gripper. "
            "It has no base motion, no head, no torso, and no second arm, so every "
            "channel is coupled to the same chain."
        ),
        "grammar": (
            "- pose(dir=front|back|up|down|in|out, grip=horiz|vert|open|close, x=.., y=.., z=.., elbow=T|F)\n"
            "  places the end effector; x/y/z are body-percent in [0,100].\n"
            "- move(shoulder|elbow|wrist, x|y|z, <signed deg or list>, rep=.., spd=.., hold=..)\n"
            "- path(ee, line|arc, ...) with axis/dist or plane/r/sweep, plus spd/rep/hold.\n"
            "- Connect with + (sequential), & (concurrent with the next step), hold(t=..)."
        ),
        "shots": [
            ("Carrier transfer", "shake_head_no",
             "pose(dir=front, grip=vert, y=50, z=70) + move(wrist, y, ±30, rep=2, spd=2.8)"),
            ("Shape tracing", "depict_big_bowl",
             "pose(dir=out, grip=horiz, z=55, x=20) + path(ee, arc, plane=yz, r=0.25, sweep=-180, spd=1.5) + hold(t=1.5)"),
            ("Additive movement", "depict_tree",
             "pose(dir=up, grip=horiz, z=90) + move(elbow, y, +20, spd=0.5, hold=2.0)"),
        ],
    },
}


def _prompt(body: str, target: dict, ref: str) -> str:
    cfg = BODIES[body]
    shots = "\n".join(
        f"{i+1}. {s} -- {cue}:\n   {code}" for i, (s, cue, code) in enumerate(cfg["shots"])
    )
    return f"""You design expressive Motion Codes for {cfg['blurb']}

A gesture authored for a humanoid must be adapted to this body. When the native carrier is missing or reduced, re-express the gesture with one of three rematerialization strategies:

1. Carrier transfer -- keep the SAME signed stroke (its axis and its back-and-forth sign) but move it onto a carrier this body has.
2. Shape tracing -- redraw the handshape or the spatial envelope of the gesture as a larger path the end effector or base draws.
3. Additive movement -- add a rotation, translation, or lift whose magnitude conveys the gesture's intent when no stroke can be transferred directly.

Few-shot examples (one per strategy):
{shots}

Grammar for this body:
{cfg['grammar']}

Rules:
- Give a clear temporal story: preparation -> expressive proxy -> release/recovery.
- Make it recognizable on THIS body; avoid a generic reach-and-hold that would fit any cue.
- Return exactly ONE Motion Code expression, no prose, no markdown.

Target cue: {target['cue']}
Location: {target['location']}
Orientation: {target['orientation']}
Movement: {target['movement']}
Humanoid reference program: {ref}

Pick whichever strategy (or combination) best fits this cue, then output the Motion Code."""


def _clean(text: str) -> str:
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    code = (fenced.group(1) if fenced else text).strip()
    code = re.sub(r"^(?:motion[_ ]?code\s*:?)\s*", "", code, flags=re.I)
    return " ".join(line.strip() for line in code.splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=sorted(BODIES))
    ap.add_argument("--model", default="gemini-2.5-pro")
    args = ap.parse_args()

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY")

    gt_rows = json.loads(GT.read_text(encoding="utf-8"))
    gt_by_cue = {r["cue"]: r for r in gt_rows}
    order = [r["cue"] for r in gt_rows]

    cues = json.loads(GLOSS.read_text(encoding="utf-8"))
    cues = cues["cues"] if isinstance(cues, dict) and "cues" in cues else cues
    gloss = {c["cue"]: c for c in cues}

    client = genai.Client(api_key=key)
    out = BODIES[args.body]["out"]
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, cue in enumerate(order, 1):
        t = gloss.get(cue)
        if t is None:
            print(f"[{i:02d}] {cue}: NO GLOSS, skipped")
            continue
        ref = (gt_by_cue[cue]["platforms"]["humanoid"].get("source_code") or "").replace("\n", " ")
        resp = client.models.generate_content(
            model=args.model, contents=_prompt(args.body, t, ref)
        )
        raw = (resp.text or "").strip()
        code = _clean(raw)
        conv = convert_code(code, platform=args.body)
        rows.append({
            "idx": int(t.get("index", i)), "cue": cue, "motion_code": code,
            "parse_ok": bool(conv.get("parse_ok")), "raw": raw,
        })
        print(f"[{i:02d}] {cue}: parse={rows[-1]['parse_ok']}  ->  {code[:110]}")

    (out / "results_all30.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("wrote", out / "results_all30.json")


if __name__ == "__main__":
    main()
