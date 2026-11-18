#!/usr/bin/env python3
"""Regenerate Roomba Motion Codes with the three rematerialization strategies
(carrier transfer / shape tracing / additive movement) taught explicitly, using
the paper's Fig. 7 Roomba few-shots.

One inference per cue, no retry loop. Defaults to the four Fig. 6 cues.
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
from embodiment_renderers import render_roomba  # noqa: E402

GLOSS = REPO / "papers/icra2027_motion_abstraction/gesture_critical_factors_90.json"
OUT = REPO / "data/results/roomba_strategy_fewshot"

# Fig. 7 Roomba few-shots, one per strategy.
FEWSHOTS = [
    ("Carrier transfer", "shake_head_no",
     "pose(dir=front) + path(base, arc, r=0, deg=[35, -70, 35], rep=2, spd=2.8)"),
    ("Shape tracing", "depict_big_bowl",
     "pose(dir=front) + path(base, arc, r=0.35, deg=90, spd=1.5) + "
     "path(base, arc, r=-0.35, deg=90, spd=1.5) + path(base, arc, r=0, deg=180, spd=2.0)"),
    ("Additive movement", "raise_fists_victory",
     "pose(dir=front) + path(base, arc, r=0, deg=720, spd=4.0) + hold(t=1.5)"),
]

# Fig. 6 highlighted cues.
DEFAULT_IDXS = (9, 20, 21, 25)


def _prompt(target: dict) -> str:
    shots = "\n".join(
        f"{i+1}. {strategy} -- {cue}:\n   {code}"
        for i, (strategy, cue, code) in enumerate(FEWSHOTS)
    )
    return f"""You design expressive Motion Codes for a Roomba-style planar mobile base. It has NO arms and NO head: only a body that can translate and spin on the floor.

A gesture authored for a humanoid must be adapted to this body. When the native carrier (arm, hand, head) is gone, re-express the gesture with one of three rematerialization strategies:

1. Carrier transfer -- keep the SAME signed stroke (its axis and its back-and-forth sign), but move it onto the base: a head/arm oscillation becomes a fore-aft rock or an in-place yaw wobble.
2. Shape tracing -- redraw the handshape or the spatial envelope of the gesture as a larger path the base drives on the floor (arcs and lines that outline the shape).
3. Additive movement -- add a rotation, a fore-aft translation, or a spin whose magnitude/energy conveys the gesture's intent when no stroke can be transferred directly.

Few-shot examples (one per strategy):
{shots}

Roomba constraints:
- Use ONLY pose(dir=...), path(base, line, ...), path(base, arc, ...), and hold(t=...).
- +x is forward, +y is left. In-place yaw arcs (r=0) act as head/hand proxies.
- Give a clear temporal story: preparation -> expressive proxy -> release/recovery.
- Make it recognizable even without arms; avoid a generic single advance-and-hold.
- Return exactly ONE Motion Code expression, no prose, no markdown.

Target cue: {target['cue']}
Location: {target['location']}
Orientation: {target['orientation']}
Movement: {target['movement']}

Pick whichever strategy (or combination) best fits this cue, then output the Motion Code."""


def _clean(text: str) -> str:
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    code = (fenced.group(1) if fenced else text).strip()
    code = re.sub(r"^(?:motion[_ ]?code\s*:?)\s*", "", code, flags=re.I)
    return " ".join(line.strip() for line in code.splitlines() if line.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", default=",".join(map(str, DEFAULT_IDXS)))
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY")
    want = {int(x) for x in args.idx.split(",") if x.strip()}

    cues = json.loads(GLOSS.read_text())
    cues = cues["cues"] if isinstance(cues, dict) and "cues" in cues else cues
    targets = [c for c in cues if int(c.get("index", -1)) in want]
    targets.sort(key=lambda c: int(c["index"]))

    client = genai.Client(api_key=key)
    OUT.mkdir(parents=True, exist_ok=True)
    render_dir = OUT / "render"
    rows = []
    for t in targets:
        idx = int(t["index"])
        resp = client.models.generate_content(model=args.model, contents=_prompt(t))
        raw = (resp.text or "").strip()
        code = _clean(raw)
        conv = convert_code(code, platform="roomba")
        row = {"idx": idx, "cue": t["cue"], "motion_code": code,
               "parse_ok": bool(conv.get("parse_ok")), "raw": raw}
        if row["parse_ok"] and not args.no_render:
            try:
                render_roomba([{
                    "idx": idx, "cue": t["cue"], "description": t["cue"],
                    "movements": conv["movements"], "source_code": code,
                    "parse_ok": True, "platform": "roomba",
                }], render_dir, force=True, fps=8)
                gif = render_dir / f"emb_g{idx:02d}_{t['cue']}.gif"
                row["render_gif"] = str(gif) if gif.is_file() else None
            except Exception as exc:  # noqa: BLE001
                row["render_error"] = str(exc)
        elif not row["parse_ok"]:
            row["error"] = str(conv.get("error") or "parse failed")
        rows.append(row)
        print(f"[{idx:02d}] {t['cue']}: parse={row['parse_ok']}  ->  {code}")

    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
