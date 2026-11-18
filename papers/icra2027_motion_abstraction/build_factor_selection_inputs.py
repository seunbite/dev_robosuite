#!/usr/bin/env python3
"""Render factor-wise G1 candidate inputs for review; makes no API calls."""
from __future__ import annotations

import copy
import html
import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageSequence

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "adhoc/generation/robotarm"
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embodiment_renderers import _G1_CAT_PALM_AXIS, render_unitree_g1  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402

SOURCE = REPO / "data/results/evaluation/embodiment_cue_id/results_gt_olm_negatives_pilot20/negatives.json"
OUT = Path(__file__).with_name("factor_selection_inputs")
CUES = (
    "salute", "highfive_invite", "stop_palm_out", "cover_yawn_sleepy",
    "cover_ears_too_loud", "shh_gesture",
)
DIRECTIONS = ("front", "back", "up", "down", "in", "out")
AXIS_FOIL = {"x": "z", "z": "x", "y": "x", "pan": "tilt", "tilt": "pan", "roll": "pan"}


def first_pose(movements):
    for step in movements:
        if step.get("type") == "pose":
            return step["parameters"]["pose"]
    raise ValueError("motion has no pose")


def split_grip(raw: str) -> tuple[str, str | None]:
    parts = str(raw or "vertical").lower().replace("horiz", "horizontal").replace("vert", "vertical").split("|")
    hv = next((p for p in parts if p in ("horizontal", "vertical")), "vertical")
    palm = next((p for p in parts if p.startswith("palm")), None)
    return hv, palm


def pose_only(base, updates, cue):
    pose = copy.deepcopy(first_pose(base))
    pose.update(updates)
    return {"cue": cue, "movements": [{"type": "pose", "duration": 0.2, "parameters": {"pose": pose}}]}


def negate_number(value):
    if isinstance(value, (int, float)):
        return -value
    if isinstance(value, list):
        return [negate_number(v) for v in value]
    return value


def spaced_three(gt: float) -> list[float]:
    """Three in-range values, exactly 30 points apart, including GT."""
    if gt < 30:
        start = gt
    elif gt > 70:
        start = gt - 60
    else:
        start = gt - 30
    return [start, start + 30, start + 60]


def movement_variant(base, kind):
    row = copy.deepcopy(base)
    for step in row:
        if step.get("type") == "path":
            p = step["parameters"]["path"]
            if kind == "axis":
                p["axis"] = AXIS_FOIL.get(str(p.get("axis", "x")), "z")
            elif "distance" in p:
                p["distance"] = negate_number(p["distance"])
            else:
                continue
            return row, True
        if step.get("type") == "movement":
            joints = step["parameters"]["movement"].get("joints", [])
            for joint in joints:
                if kind == "axis" and "axis" in joint:
                    joint["axis"] = AXIS_FOIL.get(str(joint["axis"]), "z")
                    return row, True
                if kind == "sign" and "degrees" in joint:
                    joint["degrees"] = negate_number(joint["degrees"])
                    return row, True
    return row, False


def representative_png(gif: Path, png: Path):
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(Image.open(gif))]
    frames[len(frames) // 2].save(png)


def card(label, src, *, gt=False):
    badge = '<span class="gt">GT</span>' if gt else ""
    return f'<div class="candidate">{badge}<img src="{html.escape(src)}"><code>{html.escape(label)}</code></div>'


def main():
    OUT.mkdir(exist_ok=True)
    media = OUT / "media"
    media.mkdir(exist_ok=True)
    rows = json.loads(SOURCE.read_text())
    render_rows = []
    spec = []
    serial = 0
    for cue in CUES:
        source = next(r for r in rows if r["platform"] == "humanoid" and r["cue"] == cue)
        cfg = convert_code(source["gt_code"], platform="humanoid")
        movements = cfg["movements"]
        pose = first_pose(movements)
        direction = str(pose.get("arm_position", "front"))
        hv, palm = split_grip(pose.get("gripper_orientation", "vertical"))
        xyz = {axis: float(pose.get(axis, 50)) for axis in "xyz"}
        item = {"cue": cue, "gt": {"direction": direction, "hv": hv, "palm": palm, **xyz}, "groups": {}}

        reach = []
        for value in DIRECTIONS:
            serial += 1
            ident = f"r{serial:03d}_{cue}_reach_{value}"
            render_rows.append({"idx": serial, **pose_only(movements, {"arm_position": value, "gripper_orientation": hv}, ident)})
            reach.append({"id": ident, "label": f"{value}, {hv}", "gt": value == direction})
        item["groups"]["Reach (6-way)"] = reach

        palm_rows = []
        for orient in ("horizontal", "vertical"):
            palms = sorted(set(_G1_CAT_PALM_AXIS[f"{direction}|{orient}"].values()))
            for palm_value in palms:
                serial += 1
                ident = f"r{serial:03d}_{cue}_palm_{orient}_{palm_value}"
                grip = f"{orient}|{palm_value}"
                render_rows.append({"idx": serial, **pose_only(movements, {"gripper_orientation": grip}, ident)})
                palm_rows.append({"id": ident, "label": f"{orient}, {palm_value}", "gt": orient == hv and palm_value == palm})
        item["groups"]["Palm (4-way)"] = palm_rows

        for axis in ("x", "z"):
            vals = []
            for value in spaced_three(xyz[axis]):
                serial += 1
                ident = f"r{serial:03d}_{cue}_loc_{axis}_{int(value)}"
                render_rows.append({"idx": serial, **pose_only(movements, {axis: value}, ident)})
                vals.append({"id": ident, "label": f"{axis}={int(value)}", "gt": value == xyz[axis]})
            item["groups"][f"Location {axis} (30-point steps)"] = vals

        radius = abs(xyz["y"] - 50)
        radii = sorted(set(max(0, min(50, radius + delta)) for delta in (-30, 0, 30)))
        y_rows = []
        for rad in radii:
            members = [50] if rad == 0 else [50 - rad, 50 + rad]
            pair = []
            for value in members:
                serial += 1
                ident = f"r{serial:03d}_{cue}_loc_y_{int(value)}"
                render_rows.append({"idx": serial, **pose_only(movements, {"y": value}, ident)})
                pair.append(ident)
            y_rows.append({"ids": pair, "label": f"|y-50|={int(rad)}" + ("" if rad == 0 else " (mirror pair)"), "gt": rad == radius})
        item["groups"]["Location y (mirror-equivalent)"] = y_rows

        motion = []
        base_row = {"idx": serial + 1, "cue": f"r{serial+1:03d}_{cue}_move_gt", "movements": movements}
        serial += 1; render_rows.append(base_row)
        motion.append({"id": base_row["cue"], "label": "reference movement", "gt": True})
        for kind in ("axis", "sign"):
            mutated, ok = movement_variant(movements, kind)
            if ok:
                serial += 1
                ident = f"r{serial:03d}_{cue}_move_wrong_{kind}"
                render_rows.append({"idx": serial, "cue": ident, "movements": mutated})
                motion.append({"id": ident, "label": f"wrong {kind}", "gt": False})
            else:
                motion.append({"id": None, "label": f"wrong {kind}: not applicable", "gt": False})
        item["groups"]["Movement (reference + isolated foils)"] = motion
        spec.append(item)

    rendered = render_unitree_g1(render_rows, media, force=False, fps=8)
    for row in render_rows:
        gif = rendered[row["cue"]]
        alias_gif = media / f'{row["cue"]}.gif'
        alias_png = media / f'{row["cue"]}.png'
        if not alias_gif.exists():
            shutil.copyfile(gif, alias_gif)
        representative_png(gif, alias_png)

    sections = []
    for item in spec:
        groups = []
        for name, candidates in item["groups"].items():
            cells = []
            for candidate in candidates:
                if "ids" in candidate:
                    imgs = "".join(f'<img src="media/{ident}.png">' for ident in candidate["ids"])
                    cells.append(f'<div class="candidate pair">{("<span class=gt>GT</span>" if candidate["gt"] else "")}{imgs}<code>{html.escape(candidate["label"])}</code></div>')
                elif candidate["id"] is None:
                    cells.append(f'<div class="candidate na"><code>{html.escape(candidate["label"])}</code></div>')
                else:
                    ext = ".gif" if name.startswith("Movement") else ".png"
                    cells.append(card(candidate["label"], f'media/{candidate["id"]}{ext}', gt=candidate["gt"]))
            groups.append(f'<h3>{html.escape(name)}</h3><div class="grid">{"".join(cells)}</div>')
        gt = item["gt"]
        sections.append(f'<section><h2>{html.escape(item["cue"])}</h2><p class="meta">GT: reach={gt["direction"]}, form={gt["hv"]}, palm={gt["palm"] or "unspecified"}, location=({gt["x"]:.0f}, {gt["y"]:.0f}, {gt["z"]:.0f})</p>{"".join(groups)}</section>')

    page = f'''<!doctype html><meta charset="utf-8"><title>Factor-wise candidate inputs</title>
<style>
body{{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f3f4f6;color:#171717}}main{{max-width:1500px;margin:auto;padding:28px}}section{{background:white;padding:22px;margin:20px 0;border:1px solid #d4d4d8;border-radius:10px}}h1,h2,h3{{margin:.2em 0 .5em}}h3{{font-size:15px;margin-top:22px}}.meta,.note{{color:#52525b}}.note{{background:#fff7ed;border-left:4px solid #f97316;padding:12px 14px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}}.candidate{{position:relative;border:1px solid #d4d4d8;padding:6px;background:#fafafa;min-height:80px}}.candidate img{{display:block;width:100%;aspect-ratio:1;object-fit:cover}}.candidate.pair{{display:grid;grid-template-columns:1fr 1fr}}.candidate.pair code{{grid-column:1/-1}}code{{display:block;padding:6px 2px;white-space:normal}}.gt{{position:absolute;top:10px;left:10px;background:#111;color:#fff;padding:2px 6px;font-weight:700;z-index:2}}.na{{display:flex;align-items:center;justify-content:center;color:#71717a}} 
</style><main><h1>Factor-wise VLM input review — Humanoid</h1>
<p><a href="cross_embodiment/index.html"><b>Open Tiago, Single Arm, Stretch, and Roomba input examples →</b></a></p>
<p class="note"><b>No API calls were made.</b> Reach keeps the GT horizontal/vertical form and varies six directions. Palm uses four category-valid combinations at fixed reach. Location x/z use ±30-point steps; y uses distance from the body midline and treats left/right mirrors as one candidate. Movement changes axis and sign separately. A static motion has no sign, so that foil is marked not applicable rather than inventing an event.</p>
{"".join(sections)}</main>'''
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'index.html'} with {len(render_rows)} rendered candidates")


if __name__ == "__main__":
    main()
