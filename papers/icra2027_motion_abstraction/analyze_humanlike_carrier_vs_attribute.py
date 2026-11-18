#!/usr/bin/env python3
"""Human-like robots: does the LLM pick the right carrier yet miss the attribute?

For G1 and TIAGo, compares the carrier set (head / torso / arm / second arm /
base) of each LLM-authored Motion Program with the human-authored reference,
then reads the per-cue attribute agreement already produced by
score_humanoid_loc_dir_move.py (the per-cue tables in the *_loc_dir_move.md
reports). Among carrier-matched cues it counts orientation (reach or palm),
location (any axis >= LOC_THRESHOLD body-%), and movement (axis/sign) errors.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CFG = REPO / "data/results/motion_configs"
LOC_THRESHOLD = 20.0

BODIES = {
    "G1": {
        "platform": "humanoid",
        "gt": CFG / "embodiment_motion_code_gt_humanlike90/embodiment_motion_code_all.json",
        "pred": CFG / "embodiment_motion_code_gemini_humanlike90/embodiment_motion_code_all.json",
        "report": HERE / "humanoid_gt_vs_gemini_humanlike90_loc_dir_move.md",
        "palm_match": {0.0},
    },
    "TIAGo": {
        "platform": "tiago",
        "gt": CFG / "embodiment_motion_code_gt_tiago_humanlike90/embodiment_motion_code_all.json",
        "pred": CFG / "embodiment_motion_code_gemini_humanlike90/embodiment_motion_code_all.json",
        "report": HERE / "tiago_gt_vs_gemini_synth60_loc_dir_move.md",
        "palm_match": {0.0, 180.0},
    },
}
OUT_JSON = HERE / "humanlike_carrier_vs_attribute.json"
OUT_MD = HERE / "humanlike_carrier_vs_attribute.md"


def carriers(entry: dict) -> frozenset[str]:
    used = set()
    for step in entry.get("movements", []):
        params = step.get("parameters", {})
        if step["type"] == "pose":
            pose = params.get("pose", {})
            if pose.get("head") not in (None, "center"):
                used.add("head")
            if pose.get("torso_height") not in (None, "mid"):
                used.add("torso")
            if pose.get("arm_position") not in (None, "still"):
                used.add("arm")
            if pose.get("left_arm") not in (None, "still"):
                used.add("second_arm")
        elif step["type"] == "movement":
            joints = [j.get("joint") for j in params.get("movement", {}).get("joints", [])]
            joints += [params["joint"]] if "joint" in params else []
            for joint in joints:
                used.add({"head": "head", "torso": "torso"}.get(joint, "arm"))
        elif step["type"] == "path":
            mode = params.get("path", params).get("mode", "ee")
            used.add("base" if mode == "base" else "arm")
    return frozenset(used)


def by_cue(path: Path, platform: str) -> dict[str, dict]:
    return {r["cue"]: r["platforms"][platform] for r in json.loads(path.read_text())
            if platform in r.get("platforms", {})}


def num(cell: str) -> float | None:
    cell = cell.strip()
    return None if cell in {"—", "-", ""} else float(cell)


def report_rows(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 11 or not cells[0].isdigit():
            continue
        cue = cells[1]
        rows[cue] = {
            "dx": num(cells[2]), "dy": num(cells[3]), "dz": num(cells[4]),
            "reach": num(cells[6]), "palm": num(cells[7]), "move": cells[10],
        }
    return rows


def main() -> None:
    payload = {"loc_threshold_body_pct": LOC_THRESHOLD, "bodies": {}}
    md = ["# Human-like robots: carrier choice vs attribute errors", "",
          f"Location error = any axis |Δ| ≥ {LOC_THRESHOLD:.0f} body-%. Orientation error = reach or palm mismatch.", ""]
    for name, cfg in BODIES.items():
        gt, pred = by_cue(cfg["gt"], cfg["platform"]), by_cue(cfg["pred"], cfg["platform"])
        attrs = report_rows(cfg["report"])
        cues = [c for c in attrs if c in gt and c in pred]
        matched, details = [], []
        err = Counter()
        for cue in cues:
            same = carriers(gt[cue]) == carriers(pred[cue])
            a = attrs[cue]
            o_err = (a["reach"] not in (None, 0.0)) or (a["palm"] is not None and a["palm"] not in cfg["palm_match"])
            l_err = any(v is not None and v >= LOC_THRESHOLD for v in (a["dx"], a["dy"], a["dz"]))
            m_err = a["move"] == "✗"
            details.append({"cue": cue, "same_carrier": same, "gt_carriers": sorted(carriers(gt[cue])),
                            "llm_carriers": sorted(carriers(pred[cue])),
                            "orientation_error": o_err, "location_error": l_err, "movement_error": m_err})
            if same:
                matched.append(cue)
                err["O"] += o_err
                err["L"] += l_err
                err["M"] += m_err
                err["any"] += o_err or l_err or m_err
        n, k = len(cues), len(matched)
        summary = {"n_cues": n, "carrier_matched": k, "matched_with_any_attribute_error": err["any"],
                   "matched_orientation_error": err["O"], "matched_location_error": err["L"],
                   "matched_movement_error": err["M"]}
        payload["bodies"][name] = {"summary": summary, "details": details}
        md += [f"## {name}", "",
               f"- Cues scored: {n}; LLM uses the same carrier set as the human program: **{k}/{n}**",
               f"- Among those, at least one attribute error: **{err['any']}/{k}**"
               f" (orientation {err['O']}, location {err['L']}, movement {err['M']})", "",
               "| Cue | Same carrier | O err | L err | M err | GT carriers | LLM carriers |",
               "|---|:---:|:---:|:---:|:---:|---|---|"]
        for d in details:
            md.append(f"| `{d['cue']}` | {'Y' if d['same_carrier'] else 'N'} | {'x' if d['orientation_error'] else ''} | "
                      f"{'x' if d['location_error'] else ''} | {'x' if d['movement_error'] else ''} | "
                      f"{', '.join(d['gt_carriers'])} | {', '.join(d['llm_carriers'])} |")
        md.append("")
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    OUT_MD.write_text("\n".join(md) + "\n")
    print(json.dumps({b: v["summary"] for b, v in payload["bodies"].items()}, indent=2))


if __name__ == "__main__":
    main()
