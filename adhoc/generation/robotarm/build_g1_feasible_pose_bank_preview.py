#!/usr/bin/env python3
"""Prototype an offline G1 pose bank and review xyz-GT nearest lookups.

This is deliberately a diagnostic artifact, not the final production bank:
it expands each existing semantic seed locally, rejects joint-limit,
semantic-drift, and torso/self-collision failures, then keeps a diverse
Cartesian subset. Runtime preview motions pin the selected qpos and therefore
perform no xyz IK for the initial pose.
"""
from __future__ import annotations

import copy
import html
import json
import math
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO))

from embodiment_renderers import (  # noqa: E402
    _G1_EE_AXIS_SPAN_NEG_M,
    _G1_EE_AXIS_SPAN_POS_M,
    _g1_apply_palm_orientation,
    _g1_home,
    _g1_mirror_offset_sign,
    _g1_pose_target,
    _joint_qpos_map,
    _make_g1_scene_xml,
    render_unitree_g1,
)

GT_MASTER = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt/"
    "embodiment_motion_code_all.json"
)
GOOGLE_BANK = _REPO / "data/seed/tiago/mobile_pose_bank_729.json"
MANIP_BANK = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
OUT_ROOT = _REPO / "data/results/visualize/g1_feasible_pose_bank_preview"
OUT_BANK = OUT_ROOT / "g1_feasible_pose_bank_prototype.json"
OUT_HTML = _REPO / "data/results/html/g1_feasible_pose_bank_preview.html"
CANONICAL_CATEGORIES = {
    (direction, grip, False)
    for direction in ("front", "back", "in", "out", "up", "down")
    for grip in ("horizontal", "vertical")
}

ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
PERTURB = {
    "right_shoulder_pitch_joint": (-0.55, -0.28, 0.0, 0.28, 0.55),
    "right_shoulder_roll_joint": (-0.42, -0.21, 0.0, 0.21, 0.42),
    "right_shoulder_yaw_joint": (-0.40, 0.0, 0.40),
    "right_elbow_joint": (-0.45, -0.22, 0.0, 0.22, 0.45),
}


def _category(pose: dict[str, Any]) -> tuple[str, str, bool]:
    return (
        str(pose.get("arm_position", "front")).lower(),
        str(pose.get("gripper_orientation", "vertical")).lower(),
        str(pose.get("left_arm", "still")).lower() == "mirror",
    )


def _first_pose(row: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for index, step in enumerate(row.get("movements") or []):
        if step.get("type") == "pose":
            return index, (step.get("parameters") or {}).get("pose") or {}
    return None


def _xyz_gt_rows(master: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in master:
        row = entry["platforms"]["humanoid"]
        found = _first_pose(row)
        if found is None:
            continue
        pose_index, pose = found
        axes = {a: float(pose[a]) for a in ("x", "y", "z") if isinstance(pose.get(a), (int, float))}
        if not axes:
            continue
        rows.append(
            {
                "idx": int(entry["idx"]),
                "cue": str(entry["cue"]),
                "row": row,
                "pose_index": pose_index,
                "pose": pose,
                "target": axes,
                "category": _category(pose),
            }
        )
    return rows


def _pct(delta_m: float, axis: str) -> float:
    from embodiment_renderers import _g1_delta_m_to_pct

    return _g1_delta_m_to_pct(float(delta_m), axis)


def _body_id(model: mujoco.MjModel, name: str) -> int | None:
    try:
        return int(model.body(name).id)
    except Exception:
        return None


def _forbidden_arm_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Reject right-arm contact with torso/head/pelvis or the opposite arm."""
    for ci in range(data.ncon):
        contact = data.contact[ci]
        b1 = int(model.geom_bodyid[int(contact.geom1)])
        b2 = int(model.geom_bodyid[int(contact.geom2)])
        n1 = model.body(b1).name or ""
        n2 = model.body(b2).name or ""
        pair = (n1.lower(), n2.lower())
        right = next((name for name in pair if name.startswith("right_") and any(
            token in name for token in ("shoulder", "elbow", "wrist", "rubber")
        )), None)
        if right is None:
            continue
        other = pair[1] if pair[0] == right else pair[0]
        if other.startswith(("torso", "pelvis", "head", "left_")):
            return True
    return False


def _joint_margin_ok(model: mujoco.MjModel, q: np.ndarray, joint_map: dict[str, int]) -> bool:
    # Shoulder/elbow margin is the main naturalness guard. Explicit world-palm
    # categories can legitimately consume most of a wrist's range on G1, so
    # wrists are checked against hard limits by clipping but not rejected here.
    for name in ARM_JOINTS[:4]:
        if name not in joint_map:
            continue
        joint = model.joint(name)
        lo, hi = float(joint.range[0]), float(joint.range[1])
        margin = 0.015 * (hi - lo)
        value = float(q[joint_map[name]])
        if value <= lo + margin or value >= hi - margin:
            return False
    return True


def _clip_joint(model: mujoco.MjModel, q: np.ndarray, joint_map: dict[str, int], name: str) -> None:
    joint = model.joint(name)
    lo, hi = float(joint.range[0]), float(joint.range[1])
    q[joint_map[name]] = float(np.clip(q[joint_map[name]], lo + 1e-3, hi - 1e-3))


def _remirror(q: np.ndarray, joint_map: dict[str, int]) -> None:
    for right in ARM_JOINTS:
        left = right.replace("right_", "left_")
        if right in joint_map and left in joint_map:
            q[joint_map[left]] = _g1_mirror_offset_sign(right) * float(q[joint_map[right]])


def _palm_desired(grip: str) -> np.ndarray | None:
    return {
        "palmup": np.array([0.0, 0.0, 1.0]),
        "palmdown": np.array([0.0, 0.0, -1.0]),
        "palmout": np.array([0.0, -1.0, 0.0]),
        "palmin": np.array([0.0, 1.0, 0.0]),
    }.get(grip)


def _fps_xyz(entries: list[dict[str, Any]], limit: int = 64) -> list[dict[str, Any]]:
    if len(entries) <= limit:
        return entries
    xyz = np.asarray([[e["x"], e["y"], e["z"]] for e in entries], dtype=float) / 100.0
    center = xyz.mean(axis=0)
    selected = [int(np.argmin(np.linalg.norm(xyz - center, axis=1)))]
    min_dist = np.linalg.norm(xyz - xyz[selected[0]], axis=1)
    while len(selected) < limit:
        index = int(np.argmax(min_dist))
        selected.append(index)
        min_dist = np.minimum(min_dist, np.linalg.norm(xyz - xyz[index], axis=1))
    return [entries[i] for i in selected]


def _build_bank(
    model: mujoco.MjModel,
    categories: set[tuple[str, str, bool]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
    data = mujoco.MjData(model)
    joint_map = _joint_qpos_map(model)
    torso_id = _body_id(model, "torso_link")
    shoulder_id = _body_id(model, "right_shoulder_pitch_link")
    ee_id = _body_id(model, "right_wrist_yaw_link")
    assert torso_id is not None and shoulder_id is not None and ee_id is not None
    result: dict[str, list[dict[str, Any]]] = {}
    stats: dict[str, dict[str, int]] = {}

    for direction, grip, mirror in sorted(categories):
        key = f"{direction}|{grip}|{'mirror' if mirror else 'single'}"
        pose = {
            "arm_position": direction,
            "gripper_orientation": grip,
            **({"left_arm": "mirror"} if mirror else {}),
        }
        seed = _g1_pose_target(_g1_home(model, joint_map), joint_map, pose)
        seed = _g1_apply_palm_orientation(seed, model, joint_map, pose)
        if mirror:
            _remirror(seed, joint_map)
        data.qpos[:] = seed
        mujoco.mj_forward(model, data)
        seed_vec = np.asarray(data.xpos[ee_id] - data.xpos[shoulder_id], dtype=float)
        seed_vec /= max(float(np.linalg.norm(seed_vec)), 1e-9)
        seed_palm = np.asarray(data.xmat[ee_id], dtype=float).reshape(3, 3)[:, 2].copy()
        desired = _palm_desired(grip)

        counters = Counter()
        valid: list[dict[str, Any]] = []
        names = tuple(PERTURB)
        for offsets in product(*(PERTURB[name] for name in names)):
            counters["sampled"] += 1
            q = seed.copy()
            for name, offset in zip(names, offsets):
                q[joint_map[name]] = float(q[joint_map[name]]) + float(offset)
                _clip_joint(model, q, joint_map, name)
            if mirror:
                _remirror(q, joint_map)
            if not _joint_margin_ok(model, q, joint_map):
                counters["joint_margin"] += 1
                continue
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            if _forbidden_arm_contact(model, data):
                counters["collision"] += 1
                continue
            vec = np.asarray(data.xpos[ee_id] - data.xpos[shoulder_id], dtype=float)
            vec /= max(float(np.linalg.norm(vec)), 1e-9)
            if float(np.dot(vec, seed_vec)) < 0.78:
                counters["direction_drift"] += 1
                continue
            palm = np.asarray(data.xmat[ee_id], dtype=float).reshape(3, 3)[:, 2]
            palm_ref = desired if desired is not None else seed_palm
            if float(np.dot(palm, palm_ref)) < 0.80:
                counters["grip_drift"] += 1
                continue
            torso = np.asarray(data.xpos[torso_id], dtype=float)
            ee = np.asarray(data.xpos[ee_id], dtype=float)
            valid.append(
                {
                    "x": round(_pct(float(ee[0] - torso[0]), "x"), 3),
                    "y": round(_pct(float(ee[1] - torso[1]), "y"), 3),
                    "z": round(_pct(float(ee[2] - torso[2]), "z"), 3),
                    "qpos": [round(float(v), 8) for v in q],
                }
            )
        diverse = _fps_xyz(valid, limit=64)
        result[key] = diverse
        stats[key] = {
            **{name: int(value) for name, value in counters.items()},
            "valid_before_diversity": len(valid),
            "bank_count": len(diverse),
        }
    return result, stats


def _distance(candidate: dict[str, Any], target: dict[str, float]) -> float:
    return math.sqrt(sum((float(candidate[axis]) - value) ** 2 for axis, value in target.items()))


def _pin_row(item: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(item["row"])
    step = row["movements"][item["pose_index"]]
    pose = (step.setdefault("parameters", {})).setdefault("pose", {})
    for axis in ("x", "y", "z"):
        pose.pop(axis, None)
    pose["pinned_qpos"] = candidate["qpos"]
    row["description"] = f"Offline feasible-bank nearest pose for {item['cue']}"
    return row


def _render_top9_grid(
    model: mujoco.MjModel,
    candidates: list[dict[str, Any]],
    target: dict[str, float],
    out_path: Path,
) -> None:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=300, width=300)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.85)
    camera.distance = 2.7
    camera.azimuth = 145
    camera.elevation = -8
    thumbs: list[Image.Image] = []
    for rank, candidate in enumerate(candidates[:9], start=1):
        data.qpos[:] = np.asarray(candidate["qpos"], dtype=float)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        label = (
            f"#{rank} xyz=({candidate['x']:.0f},{candidate['y']:.0f},{candidate['z']:.0f}) "
            f"d={_distance(candidate, target):.1f}"
        )
        draw.rectangle((0, 0, image.width, 26), fill=(0, 0, 0))
        draw.text((6, 6), label, fill=(255, 255, 255))
        thumbs.append(image)
    renderer.close()
    canvas = Image.new("RGB", (900, 900), (245, 245, 245))
    for index, image in enumerate(thumbs):
        canvas.paste(image, ((index % 3) * 300, (index // 3) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _render_category_grid(
    model: mujoco.MjModel,
    candidates: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Render a Google-style 5×5 review sheet for one G1 bank category."""
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=240, width=240)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.85)
    camera.distance = 2.7
    camera.azimuth = 145
    camera.elevation = -8
    canvas = Image.new("RGB", (1200, 1200), (245, 245, 245))
    for index, candidate in enumerate(candidates[:25]):
        data.qpos[:] = np.asarray(candidate["qpos"], dtype=float)
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        image = Image.fromarray(renderer.render()).convert("RGB")
        draw = ImageDraw.Draw(image)
        label = f"#{index + 1} ({candidate['x']:.0f},{candidate['y']:.0f},{candidate['z']:.0f})"
        draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        canvas.paste(image, ((index % 5) * 240, (index // 5) * 240))
    renderer.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _existing_counts() -> dict[str, Any]:
    google = json.loads(GOOGLE_BANK.read_text(encoding="utf-8"))["poses"]
    google_counts = Counter((row["dir"], row["orient"]) for row in google)
    manip_rows = [
        json.loads(line)
        for line in MANIP_BANK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    iiwa_counts = Counter(
        (row["dir"], row["gripper_orientation"])
        for row in manip_rows
        if row.get("robot") == "IIWA"
    )
    return {
        "google": {"total": len(google), "groups": {"|".join(k): v for k, v in sorted(google_counts.items())}},
        "iiwa": {
            "total_all_robots": len(manip_rows),
            "total_iiwa": sum(iiwa_counts.values()),
            "groups": {"|".join(k): v for k, v in sorted(iiwa_counts.items())},
        },
    }


def _rel(path: Path) -> str:
    return os.path.relpath(path, OUT_HTML.parent)


def _build_html(
    items: list[dict[str, Any]],
    bank: dict[str, list[dict[str, Any]]],
    stats: dict[str, dict[str, int]],
    existing: dict[str, Any],
) -> None:
    category_rows = []
    category_sheets = []
    for key in sorted(bank):
        stat = stats[key]
        category_rows.append(
            f"<tr><td>{html.escape(key)}</td><td>{stat.get('sampled', 0)}</td>"
            f"<td>{stat.get('valid_before_diversity', 0)}</td><td>{stat['bank_count']}</td></tr>"
        )
        sheet = OUT_ROOT / "category_sheets" / f"{key.replace('|', '_')}.jpg"
        category_sheets.append(
            f"<section><h2>{html.escape(key)}</h2><p>{stat['bank_count']} retained candidates · "
            f"first 25 shown · labels are body-centered xyz percentiles</p>"
            f"<img src='{html.escape(_rel(sheet))}'></section>"
        )
    google_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in existing["google"]["groups"].items()
    )
    iiwa_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value}</td></tr>"
        for key, value in existing["iiwa"]["groups"].items()
    )
    cue_sections = []
    for item in items:
        idx, cue = item["idx"], item["cue"]
        key = "|".join(
            (item["category"][0], item["category"][1], "mirror" if item["category"][2] else "single")
        )
        candidates = sorted(bank.get(key, []), key=lambda row: _distance(row, item["target"]))
        selected = candidates[0] if candidates else None
        target = ", ".join(f"{axis}={value:g}" for axis, value in item["target"].items())
        selected_text = (
            f"xyz=({selected['x']:.1f}, {selected['y']:.1f}, {selected['z']:.1f}), "
            f"distance={_distance(selected, item['target']):.2f}"
            if selected
            else "no candidate"
        )
        current_gif = (
            _REPO
            / "data/results/render/embodiment_motion_code_gt/humanoid/gif"
            / f"emb_g{idx:02d}_{cue}.gif"
        )
        bank_gif = OUT_ROOT / "gif" / f"emb_g{idx:02d}_{cue}.gif"
        grid = OUT_ROOT / "top9" / f"g{idx:02d}_{cue}.png"
        cue_sections.append(
            f"<section><h2>{idx:02d}. {html.escape(cue)}</h2>"
            f"<p><b>category:</b> {html.escape(key)} · <b>target:</b> {target}<br>"
            f"<b>nearest offline candidate:</b> {html.escape(selected_text)}</p>"
            f"<div class='compare'><div><h3>Current category → xyz IK</h3>"
            f"<img src='{html.escape(_rel(current_gif))}'></div>"
            f"<div><h3>Nearest feasible bank qpos → same remaining motion</h3>"
            f"<img src='{html.escape(_rel(bank_gif))}'></div>"
            f"<div><h3>Top-9 feasible candidates</h3>"
            f"<img src='{html.escape(_rel(grid))}'></div></div></section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>G1 feasible pose-bank prototype</title><style>
body{{font-family:system-ui;margin:20px;background:#11151b;color:#e9edf2}}
table{{border-collapse:collapse;background:#181e27}}th,td{{border:1px solid #394250;padding:6px 9px}}
.inventory{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}}
section{{border-top:1px solid #394250;margin-top:24px;padding-top:14px}}
.compare{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
img{{width:100%;height:auto;background:#000}}h3{{font-size:14px;color:#bac5d1}}
code{{color:#9bc6ff}}.note{{max-width:1100px;color:#bac5d1}}
</style></head><body>
<h1>G1 feasible pose-bank prototype · latest Human GT</h1>
<p class="note">Diagnostic prototype. G1 currently has no stored pose bank: one analytic semantic seed is
translated by runtime IK. This page builds local collision/semantic-valid candidates offline and pins the
nearest qpos, so the initial pose uses no runtime xyz IK. Production should replace the local seed prior
with retargeted human-motion or learned pose-prior samples.</p>
<div class="inventory"><div><h2>Tiago existing bank</h2>
<p>{existing['google']['total']} total · 12 dir×orient groups</p><table><tr><th>group</th><th>N</th></tr>{google_rows}</table></div>
<div><h2>IIWA existing bank</h2><p>{existing['iiwa']['total_iiwa']} IIWA ·
{existing['iiwa']['total_all_robots']} across 7 arms</p><table><tr><th>group</th><th>N</th></tr>{iiwa_rows}</table></div>
<div><h2>G1 prototype bank</h2><p>Local semantic expansion → validity filters → xyz farthest-point subset</p>
<table><tr><th>category</th><th>sampled</th><th>valid</th><th>kept</th></tr>{''.join(category_rows)}</table></div></div>
<h1>G1 category review sheets</h1><div class="compare">{''.join(category_sheets)}</div>
<h1>25 xyz-specified Humanoid GT examples</h1>{''.join(cue_sections)}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")


def main() -> None:
    master = json.loads(GT_MASTER.read_text(encoding="utf-8"))
    items = _xyz_gt_rows(master)
    scene = _make_g1_scene_xml()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        categories = CANONICAL_CATEGORIES | {item["category"] for item in items}
        bank, stats = _build_bank(model, categories)
        payload = {
            "definition": {
                "prototype_only": True,
                "runtime_xyz_ik": False,
                "sampling": "local deterministic semantic-seed perturbations",
                "filters": ["joint margin", "arm/torso collision", "direction cosine", "palm cosine"],
                "diversity": "xyz farthest-point sampling, max 64/category",
            },
            "stats": stats,
            "categories": bank,
        }
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        OUT_BANK.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        for key, candidates in bank.items():
            _render_category_grid(
                model,
                candidates,
                OUT_ROOT / "category_sheets" / f"{key.replace('|', '_')}.jpg",
            )

        preview_rows = []
        for item in items:
            key = "|".join(
                (item["category"][0], item["category"][1], "mirror" if item["category"][2] else "single")
            )
            ranked = sorted(bank.get(key, []), key=lambda row: _distance(row, item["target"]))
            if not ranked:
                continue
            preview_rows.append(_pin_row(item, ranked[0]))
            _render_top9_grid(
                model,
                ranked,
                item["target"],
                OUT_ROOT / "top9" / f"g{item['idx']:02d}_{item['cue']}.png",
            )
        render_unitree_g1(preview_rows, OUT_ROOT / "gif", force=True)
        _build_html(items, bank, stats, _existing_counts())
        print(f"bank: {OUT_BANK}")
        print(f"html: {OUT_HTML}")
        print(f"xyz GT: {len(items)}; rendered: {len(preview_rows)}")
    finally:
        scene.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
