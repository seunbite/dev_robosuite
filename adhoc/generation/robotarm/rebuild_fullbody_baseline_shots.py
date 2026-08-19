#!/usr/bin/env python3
"""Re-extract the current 5 ours shots as full-body joint/named-link trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from motion_code_to_config import convert_code
from run_embodiment_baseline_traj import EXTRACTORS, RENDERERS, HZ

SOURCE = REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
OUT = REPO / "data/seed/shots/embodiment_unified/shots_fullbody.yaml"
RENDER = REPO / "data/results/baseline_traj/fullbody_pilot10/fewshot_render"
# Tiago's legacy renderer registers a top-level ``legacy`` module; run IIWA first
# so its robotarm.legacy imports cannot be shadowed in the same interpreter.
PLATFORMS = ("single_arm", "humanoid", "tiago")


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    robots = {}
    report = []
    for platform in PLATFORMS:
        robots[platform] = {}
        for rank, entry in enumerate(source[platform], 1):
            parsed = convert_code(entry["motion_code"], platform=platform)
            cell = {"idx": rank, "cue": entry["cue"], "description": entry.get("notes", ""),
                    "movements": parsed.get("movements") or [], "source_code": entry["motion_code"],
                    "parse_ok": bool(parsed.get("parse_ok")), "platform": platform}
            art = EXTRACTORS[platform](cell, hz=HZ)
            if art.get("proxy"):
                raise RuntimeError(f"{platform}/{entry['cue']}: proxy extraction is forbidden")
            n = len(art["joints_rad"])
            if not n or n != len(art["xyzrpy"]):
                raise RuntimeError(f"{platform}/{entry['cue']}: empty or unaligned trajectories")
            joint_span = max(max(col)-min(col) for col in zip(*art["joints_rad"]))
            cart_span = max(max(col)-min(col) for col in zip(*art["xyzrpy"]))
            if joint_span < 1e-5 or cart_span < 1e-5:
                raise RuntimeError(f"{platform}/{entry['cue']}: static extraction")
            robots[platform][entry["cue"]] = {
                "motion_code": {"source_code": entry["motion_code"]},
                "joint_angles": {"hz": HZ, "joint_names": art["joint_names"], "values": art["joints_rad"]},
                "xyzrpy": {"hz": HZ, "link_names": art.get("link_names") or ["end_effector"],
                            "values": art["xyzrpy"]},
            }
            gif = RENDER / platform / f"{rank:02d}_{entry['cue']}_joint.gif"
            RENDERERS[platform](
                {"hz": HZ, "joint_names": art["joint_names"], "joints_rad": art["joints_rad"]},
                "joint", gif, art)
            report.append({"platform": platform, "cue": entry["cue"], "frames": n,
                           "joint_dim": len(art["joint_names"]),
                           "link_names": art.get("link_names") or ["end_effector"],
                           "joint_span": joint_span, "cartesian_span": cart_span,
                           "render": str(gif.relative_to(REPO))})
            print(f"[shot] {platform}/{entry['cue']} n={n} q={len(art['joint_names'])} links={len(art.get('link_names') or [1])}", flush=True)
    payload = {"meta": {"description": "Full-body pilot few-shots; real robot extraction only.",
                         "n_per_robot": 5, "platforms": list(PLATFORMS)}, "robots": robots}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    report_path = RENDER.parent / "fewshot_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(f"wrote {OUT}\nwrote {report_path}")


if __name__ == "__main__":
    main()
