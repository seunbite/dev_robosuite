#!/usr/bin/env python3
"""Build non-humanoid factor-foil examples; no model/API calls."""
from __future__ import annotations

import copy
import html
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "adhoc/generation/robotarm"
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embodiment_renderers import render_roomba, render_stretch  # noqa: E402
from motion_code_to_config import render_manipulator, render_tiago  # noqa: E402

SOURCE = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
OUT = Path(__file__).with_name("factor_selection_inputs") / "cross_embodiment"
PLATFORMS = ("tiago", "single_arm", "stretch", "roomba")
DISPLAY = {"tiago": "Tiago", "single_arm": "Single Arm", "stretch": "Stretch", "roomba": "Roomba"}
AXIS = {"x": "z", "z": "x", "y": "x", "pan": "tilt", "tilt": "pan", "roll": "pan"}
REACH_FOIL = {"front": "down", "back": "front", "up": "front", "down": "up", "in": "out", "out": "in"}


def neg(value):
    if isinstance(value, (int, float)):
        return -value
    if isinstance(value, list):
        return [neg(v) for v in value]
    return value


def mutate(movements, kind):
    out = copy.deepcopy(movements)
    # Prefer the communicative line stroke over an optional preparatory spin.
    ordered = [s for s in out if s.get("type") == "path" and (s.get("parameters", {}).get("path", {}).get("shape") == "line")]
    ordered += [s for s in out if s not in ordered]
    for step in ordered:
        if step.get("type") == "path":
            path = step.get("parameters", {}).get("path", {})
            if kind == "axis":
                if "axis" in path:
                    path["axis"] = AXIS.get(str(path["axis"]), "z")
                    return out
                for axis in ("x", "y", "z"):
                    if axis in path:
                        value = path.pop(axis)
                        path[AXIS[axis]] = value
                        return out
            else:
                for key in ("distance", "dist", "x", "y", "z", "degrees", "deg"):
                    if key in path:
                        path[key] = neg(path[key])
                        return out
        if step.get("type") == "movement":
            params = step.get("parameters", {})
            movement = params.get("movement", params)
            for joint in movement.get("joints", []):
                if kind == "axis" and "axis" in joint:
                    joint["axis"] = AXIS.get(str(joint["axis"]), "z")
                    return out
                if kind == "sign" and "degrees" in joint:
                    joint["degrees"] = neg(joint["degrees"])
                    return out
            for direction in movement.get("directions", []):
                degrees = direction.get("degrees", {})
                if degrees:
                    key = next(iter(degrees))
                    value = degrees.pop(key)
                    degrees[AXIS.get(key, "z") if kind == "axis" else key] = value if kind == "axis" else neg(value)
            if movement.get("directions"):
                return out
    raise ValueError(f"No {kind} target")


def salient_pose(movements):
    """Return the final pose, which is the communicative pose for high-five."""
    poses = [
        step.get("parameters", {}).get("pose")
        for step in movements
        if step.get("type") == "pose"
    ]
    poses = [pose for pose in poses if pose is not None]
    return poses[-1] if poses else None


def mutate_pose(movements, kind):
    """Change exactly one pose/orientation field in the salient pose."""
    out = copy.deepcopy(movements)
    pose = salient_pose(out)
    if pose is None:
        raise ValueError(f"No pose target for {kind}")
    if kind == "location":
        z = float(pose.get("z", 70))
        pose["z"] = max(0, z - 30) if z >= 30 else min(100, z + 30)
        return out, f"z: {z:g} → {pose['z']:g}"
    if kind == "reach":
        key = "arm_position" if "arm_position" in pose else "dir"
        value = str(pose.get(key, "front"))
        pose[key] = REACH_FOIL.get(value, "front")
        return out, f"reach: {value} → {pose[key]}"
    if kind == "palm":
        value = str(pose.get("gripper_orientation", "horizontal"))
        parts = value.split("|")
        form = "vertical" if parts[0] in ("horizontal", "horiz") else "horizontal"
        pose["gripper_orientation"] = form
        return out, f"palm form: {parts[0]} → {form}"
    raise ValueError(kind)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads(SOURCE.read_text())
    cue_row = next(row for row in inventory if row["cue"] == "highfive_invite")
    by_platform = {}
    for pi, platform in enumerate(PLATFORMS):
        base = copy.deepcopy(cue_row["platforms"][platform])
        rows = []
        candidate_specs = [("reference", "Reference")]
        if salient_pose(base["movements"]) is not None:
            candidate_specs.extend([
                ("wrong_pose_location", None),
                ("wrong_reach_orientation", None),
                ("wrong_palm_orientation", None),
            ])
        candidate_specs.extend([
            ("wrong_movement_axis", "Wrong movement axis"),
            ("wrong_movement_sign", "Wrong movement sign"),
        ])
        candidates = []
        for vi, (kind, label) in enumerate(candidate_specs):
            row = copy.deepcopy(base)
            row["idx"] = 301 + pi * 10 + vi
            row["cue"] = f"highfive_invite__{kind}_v3"
            if kind.startswith("wrong_pose_"):
                row["movements"], detail = mutate_pose(base["movements"], "location")
                label = f"Wrong pose/location ({detail})"
            elif kind.startswith("wrong_reach_"):
                row["movements"], detail = mutate_pose(base["movements"], "reach")
                label = f"Wrong reach orientation ({detail})"
            elif kind.startswith("wrong_palm_"):
                row["movements"], detail = mutate_pose(base["movements"], "palm")
                label = f"Wrong palm orientation ({detail})"
            elif kind.startswith("wrong_movement_"):
                row["movements"] = mutate(base["movements"], kind.removeprefix("wrong_movement_"))
            rows.append(row)
            candidates.append({"kind": kind, "label": label, "cue": row["cue"]})
        if platform == "tiago":
            paths = render_tiago(rows, out_root=OUT, force=False)
        elif platform == "single_arm":
            paths = render_manipulator(rows, out_root=OUT, force=False)
        elif platform == "stretch":
            paths = render_stretch(rows, OUT / "stretch" / "gif", force=False, fps=12)
        else:
            paths = render_roomba(rows, OUT / "roomba" / "gif", force=False, fps=12)
        by_platform[platform] = []
        for candidate in candidates:
            cue_key = candidate["cue"]
            by_platform[platform].append({
                "kind": candidate["kind"],
                "label": candidate["label"],
                "path": str(paths[cue_key].relative_to(OUT)),
            })

    sections = []
    for platform, candidates in by_platform.items():
        groups = []
        for title, prefix in (
            ("Pose and orientation", ("reference", "wrong_pose", "wrong_reach", "wrong_palm")),
            ("Movement", ("reference", "wrong_movement")),
        ):
            selected = [c for c in candidates if c["kind"].startswith(prefix)]
            if title == "Pose and orientation" and len(selected) == 1:
                groups.append('<h3>Pose and orientation</h3><p class="na">Not applicable: this embodiment exposes no pose/orientation channel.</p>')
                continue
            cards = "".join(
                f'<div class="candidate"><img src="{html.escape(c["path"])}"><code>{html.escape(c["label"])}</code></div>'
                for c in selected
            )
            groups.append(f'<h3>{title}</h3><div class="grid">{cards}</div>')
        sections.append(f'<section><h2>{DISPLAY[platform]}</h2>{"".join(groups)}</section>')
    page = f'''<!doctype html><meta charset="utf-8"><title>Cross-embodiment factor inputs</title>
<style>body{{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f3f4f6;color:#171717;margin:0}}main{{max-width:1400px;margin:auto;padding:28px}}section{{background:#fff;border:1px solid #d4d4d8;border-radius:10px;padding:20px;margin:18px 0}}h3{{margin:22px 0 8px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}.candidate{{border:1px solid #d4d4d8;background:#fafafa;padding:7px}}img{{width:100%;aspect-ratio:1.1;object-fit:cover;display:block}}code{{display:block;padding:7px;font-weight:700;white-space:normal}}.na{{padding:14px;background:#f4f4f5;color:#52525b;border-left:4px solid #a1a1aa}}</style>
<main><h1>Cross-embodiment input examples</h1><p>Target: <code>highfive_invite</code>. Pose/location, reach orientation, palm orientation, movement axis, and movement sign are changed one at a time. Roomba has no independent pose/orientation channel, so those foils are explicitly marked not applicable. These are rendered inputs only; no VLM/API evaluation was run.</p>{"".join(sections)}</main>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
