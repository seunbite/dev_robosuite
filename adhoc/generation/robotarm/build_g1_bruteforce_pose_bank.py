#!/usr/bin/env python3
"""Build and review a 90-degree Unitree G1 right-arm pose lattice.

The output is deliberately a review candidate, not a production seed.  Every
valid {-90, 0, +90} degree joint combination is classified into one of six
world-space directions and two legacy wrist-roll orientations.  Hard-filtered
and retained poses are both rendered so filtering can be audited.
"""
from __future__ import annotations

import html
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from build_g1_feasible_pose_bank_preview import (  # noqa: E402
    _forbidden_arm_contact,
    _pct,
)
from embodiment_renderers import (  # noqa: E402
    _g1_home,
    _joint_qpos_map,
    _make_g1_scene_xml,
)

ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
LATTICE_DEG = (-90.0, 0.0, 90.0)
ELBOW_BEND_THRESHOLD_DEG = 15.0
MIN_SHOULDER_WRIST_DISTANCE_M = 0.20  # hard-filter collapsed_arm (was 0.16)
MIN_EE_X_PCT = 10.0  # hard-filter: palm x percentile too close to / behind torso
OUT_ROOT = REPO / "data/results/visualize/g1_pose_bank_90deg_review"
OUT_JSON = OUT_ROOT / "g1_pose_bank_90deg_candidates.json"
OUT_HTML = REPO / "data/results/html/g1_pose_bank_90deg_review.html"
OUT_SELECTED = REPO / "data/seed/unitree_g1/g1_feasible_poses_metadata.json"
# After correcting inverted H/V classifier: former horizontal IDs are now vertical and vice versa.
FEASIBLE_SELECTION = {
    "front|horizontal": {1135, 1063, 273, 554, 311},
    "front|vertical": {1036, 276, 740, 292, 545, 302},
    "back|horizontal": {1151},
    "back|vertical": {1140, 31},
    "in|horizontal": {1087, 618, 231},
    "in|vertical": {1084, 4, 275, 544},
    "out|horizontal": {60, 76, 543, 712, 1148, 1201},
    "out|vertical": {1139, 1196, 546},
    "up|horizontal": {1150, 1144, 524, 221},
    "up|vertical": {1147, 1092, 1082, 643, 703, 230, 177},
    "down|horizontal": {1152, 1146, 689, 555},
    "down|vertical": {1149, 1088, 542},
}


def _joint_values(model: mujoco.MjModel, name: str) -> list[float]:
    joint = model.joint(name)
    lo, hi = np.rad2deg(np.asarray(joint.range, dtype=float))
    return [value for value in LATTICE_DEG if lo - 1e-6 <= value <= hi + 1e-6]


def _classify_direction(ee_direction: np.ndarray) -> str:
    """Classify wrist→palm EE direction by its dominant world axis."""
    vector = np.asarray(ee_direction, dtype=float)
    axis = int(np.argmax(np.abs(vector)))
    if axis == 0:
        return "front" if vector[0] >= 0 else "back"
    if axis == 1:
        return "in" if vector[1] >= 0 else "out"
    return "up" if vector[2] >= 0 else "down"


def _classify_grip(hand_span: np.ndarray, direction: str) -> str:
    """Classify hand-span as horizontal/vertical (visual match to Google pad opening).

    Note: G1 wrist Y-axis polarity required inverting the Google-style threshold so
    that thumb-up / blade-hand reads as vertical, palm-flat span as horizontal.
    """
    span = np.asarray(hand_span, dtype=float)
    if direction in ("up", "down"):
        projected = float(np.linalg.norm(np.array([0.0, span[1], span[2]])))
    else:
        projected = float(np.linalg.norm(np.array([span[0], span[1], 0.0])))
    # Inverted vs unit-projection ≥ √0.5 → horizontal (was visually swapped on G1).
    return "vertical" if projected >= math.sqrt(0.5) else "horizontal"


def _evaluate(model: mujoco.MjModel) -> list[dict[str, Any]]:
    data = mujoco.MjData(model)
    joint_map = _joint_qpos_map(model)
    home = _g1_home(model, joint_map)
    torso_id = int(model.body("torso_link").id)
    shoulder_id = int(model.body("right_shoulder_pitch_link").id)
    elbow_id = int(model.body("right_elbow_link").id)
    wrist_id = int(model.body("right_wrist_yaw_link").id)
    palm_site_id = int(model.site("right_palm").id)
    grids = [_joint_values(model, name) for name in ARM_JOINTS]
    rows: list[dict[str, Any]] = []

    for pose_id, values in enumerate(product(*grids), start=1):
        q = home.copy()
        joint_deg = dict(zip(ARM_JOINTS, values))
        for name, value in joint_deg.items():
            q[joint_map[name]] = math.radians(value)
        data.qpos[:] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        torso = np.asarray(data.xpos[torso_id], dtype=float)
        shoulder = np.asarray(data.xpos[shoulder_id], dtype=float)
        elbow = np.asarray(data.xpos[elbow_id], dtype=float)
        wrist = np.asarray(data.xpos[wrist_id], dtype=float)
        palm_site = np.asarray(data.site_xpos[palm_site_id], dtype=float)
        reasons: list[str] = []
        if _forbidden_arm_contact(model, data):
            reasons.append("arm_body_collision")
        if float(np.linalg.norm(wrist - shoulder)) < MIN_SHOULDER_WRIST_DISTANCE_M:
            reasons.append("collapsed_arm")

        wrist_frame = np.asarray(data.xmat[wrist_id], dtype=float).reshape(3, 3)
        ee_direction = palm_site - wrist
        direction = _classify_direction(ee_direction)
        hand_span = wrist_frame[:, 1]
        grip = _classify_grip(hand_span, direction)
        palm = wrist_frame[:, 2]
        x_pct = round(_pct(float(palm_site[0] - torso[0]), "x"), 3)
        y_pct = round(_pct(float(palm_site[1] - torso[1]), "y"), 3)
        z_pct = round(_pct(float(palm_site[2] - torso[2]), "z"), 3)
        if x_pct < MIN_EE_X_PCT:
            reasons.append("ee_x_below_10")
        rows.append(
            {
                "pose_id": pose_id,
                "direction": direction,
                "gripper_orientation": grip,
                "elbow_bended": joint_deg["right_elbow_joint"] >= ELBOW_BEND_THRESHOLD_DEG,
                "joint_deg": joint_deg,
                "qpos": [round(float(v), 8) for v in q],
                "x": x_pct,
                "y": y_pct,
                "z": z_pct,
                "elbow_xyz_world": [round(float(v), 5) for v in elbow],
                "palm_normal_world": [round(float(v), 5) for v in palm],
                "ee_direction_world": [round(float(v), 5) for v in ee_direction],
                "hand_span_world": [round(float(v), 5) for v in hand_span],
                "hard_filter_pass": not reasons,
                "hard_filter_reasons": reasons,
            }
        )
    return rows


XYZ_DEDUP_BIN = 1  # int(xyz) Chebyshev ≤ this ⇒ same location neighborhood


def _palm_direction_key(row: dict[str, Any]) -> str:
    """Coarse palm-facing bucket from world palm normal (6-way dominant axis)."""
    palm = np.asarray(row.get("palm_normal_world", (0.0, 0.0, 1.0)), dtype=float)
    norm = float(np.linalg.norm(palm)) + 1e-12
    palm = palm / norm
    axis = int(np.argmax(np.abs(palm)))
    sign = "+" if palm[axis] >= 0.0 else "-"
    return f"{'xyz'[axis]}{sign}"


def _xyz_int_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(round(float(row["x"]))),
        int(round(float(row["y"]))),
        int(round(float(row["z"]))),
    )


def _xyz_chebyshev(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def apply_xyz_palm_dedupe(
    rows: list[dict[str, Any]],
    *,
    bin_size: int = XYZ_DEDUP_BIN,
) -> list[dict[str, Any]]:
    """Drop near-duplicate XYZ poses unless palm direction differs (max 2 / neighborhood).

    XYZ is rounded to int. Two poses are in the same neighborhood when Chebyshev
    distance on that int grid is ≤ ``bin_size`` (default 1 ⇒ differ-by-1 counts as
    same place). Within a neighborhood keep at most one pose per palm bucket and
    at most two palms total. Mutates rows with ``xyz_palm_dedupe_pass``.
    """
    hard_pass = [row for row in rows if row.get("hard_filter_pass")]
    hard_pass_sorted = sorted(hard_pass, key=lambda r: int(r["pose_id"]))

    kept: list[tuple[tuple[int, int, int], str, int]] = []  # xyz, palm, pose_id
    kept_ids: set[int] = set()
    for row in hard_pass_sorted:
        xyz = _xyz_int_key(row)
        palm = _palm_direction_key(row)
        pid = int(row["pose_id"])
        neighbors = [item for item in kept if _xyz_chebyshev(xyz, item[0]) <= bin_size]
        neighbor_palms = {item[1] for item in neighbors}
        if palm in neighbor_palms:
            continue
        if len(neighbor_palms) >= 2:
            continue
        kept.append((xyz, palm, pid))
        kept_ids.add(pid)

    for row in rows:
        if not row.get("hard_filter_pass"):
            row["xyz_palm_dedupe_pass"] = False
            row["xyz_palm_dedupe_reason"] = "hard_filter_fail"
            continue
        pid = int(row["pose_id"])
        if pid in kept_ids:
            row["xyz_palm_dedupe_pass"] = True
            row["xyz_palm_dedupe_reason"] = ""
        else:
            row["xyz_palm_dedupe_pass"] = False
            row["xyz_palm_dedupe_reason"] = "same_xyz_bin_same_palm"

    return [row for row in rows if row.get("xyz_palm_dedupe_pass")]


def _select_xy_grid_balanced(rows: list[dict[str, Any]], grid_side: int = 5) -> list[dict[str, Any]]:
    """Google-style 5×5 task-space coverage sample for visual selection."""
    if len(rows) <= grid_side * grid_side:
        return list(rows)
    x_values = [float(row["x"]) for row in rows]
    y_values = [float(row["y"]) for row in rows]
    z_mid = 0.5 * (
        min(float(row["z"]) for row in rows) + max(float(row["z"]) for row in rows)
    )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for tx in np.linspace(min(x_values), max(x_values), grid_side):
        for ty in np.linspace(min(y_values), max(y_values), grid_side):
            remaining = [row for row in rows if int(row["pose_id"]) not in used]
            if not remaining:
                break
            best = min(
                remaining,
                key=lambda row: (
                    (float(row["x"]) - tx) ** 2
                    + (float(row["y"]) - ty) ** 2
                    + 0.15 * (float(row["z"]) - z_mid) ** 2,
                    int(row["pose_id"]),
                ),
            )
            selected.append(best)
            used.add(int(best["pose_id"]))
    return selected


def _render_pages(
    model: mujoco.MjModel,
    rows: list[dict[str, Any]],
    category: str,
    verdict: str,
) -> list[Path]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=180, width=180)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.85)
    camera.distance = 2.7
    camera.azimuth = 145
    camera.elevation = -8
    page_paths: list[Path] = []
    page_size = 30
    for page_index in range(0, len(rows), page_size):
        page_rows = rows[page_index : page_index + page_size]
        canvas = Image.new("RGB", (1080, 900), (240, 240, 240))
        for index, row in enumerate(page_rows):
            data.qpos[:] = np.asarray(row["qpos"], dtype=float)
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            image = Image.fromarray(renderer.render()).convert("RGB")
            draw = ImageDraw.Draw(image)
            elbow = "T" if row["elbow_bended"] else "F"
            reason = ",".join(row["hard_filter_reasons"]) or "pass"
            label = f"#{row['pose_id']} e={elbow} {reason}"
            draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
            draw.text((4, 4), label, fill=(255, 255, 255))
            if row.get("feasible_selected"):
                draw.rectangle(
                    (3, 3, image.width - 4, image.height - 4),
                    outline=(255, 0, 0),
                    width=7,
                )
            canvas.paste(image, ((index % 6) * 180, (index // 6) * 180))
        page_number = page_index // page_size + 1
        path = OUT_ROOT / "sheets" / category / verdict / f"page_{page_number:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=88)
        page_paths.append(path)
    renderer.close()
    return page_paths


def _relative(path: Path) -> str:
    return str(Path("..") / path.relative_to(OUT_HTML.parent.parent)).replace("\\", "/")


def _build_html(model: mujoco.MjModel, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['direction']}|{row['gripper_orientation']}"].append(row)

    sections = []
    summary_rows = []
    order = [
        f"{direction}|{grip}"
        for direction in ("front", "back", "in", "out", "up", "down")
        for grip in ("horizontal", "vertical")
    ]
    for category in order:
        category_rows = grouped.get(category, [])
        accepted = [row for row in category_rows if row["hard_filter_pass"]]
        rejected = [row for row in category_rows if not row["hard_filter_pass"]]
        # Soft filter after hard-filter: unique int(xyz), keep up to 2 if palm differs.
        apply_xyz_palm_dedupe(category_rows)
        deduped = [row for row in accepted if row.get("xyz_palm_dedupe_pass")]
        review_candidates = _select_xy_grid_balanced(deduped)
        review_ids = {int(row["pose_id"]) for row in review_candidates}
        selected_ids = FEASIBLE_SELECTION[category]
        for row in category_rows:
            row["google_grid_candidate"] = int(row["pose_id"]) in review_ids
            row["feasible_selected"] = int(row["pose_id"]) in selected_ids
        selected = [row for row in accepted if row["feasible_selected"]]
        if {int(row["pose_id"]) for row in selected} != selected_ids:
            raise RuntimeError(f"Invalid feasible selection for {category}")
        reasons = Counter(
            reason for row in rejected for reason in row["hard_filter_reasons"]
        )
        summary_rows.append(
            f"<tr><td><a href='#{category}'>{category}</a></td><td>{len(category_rows)}</td>"
            f"<td>{len(accepted)}</td><td>{len(rejected)}</td></tr>"
        )
        accepted_pages = _render_pages(model, accepted, category.replace("|", "_"), "accepted")
        rejected_pages = _render_pages(model, rejected, category.replace("|", "_"), "rejected")
        review_pages = _render_pages(
            model, review_candidates, category.replace("|", "_"), "google_grid_candidates"
        )
        selected_pages = _render_pages(
            model, selected, category.replace("|", "_"), "feasible_selected"
        )

        def images(paths: list[Path]) -> str:
            return "".join(
                f"<img loading='lazy' src='{html.escape(_relative(path))}'>" for path in paths
            ) or "<p>None</p>"

        sections.append(
            f"<section id='{category}'><h2>{category}</h2>"
            f"<p>{len(category_rows)} classified · {len(accepted)} retained · "
            f"{len(rejected)} filtered · {len(selected)} feasible selected · "
            f"reasons: {html.escape(str(dict(reasons)))}</p>"
            f"<details open><summary>Selected feasible set ({len(selected)}, red border)</summary>"
            f"<div class='sheets'>{images(selected_pages)}</div></details>"
            f"<details open><summary>Google-style diverse review set ({len(review_candidates)})</summary>"
            f"<div class='sheets'>{images(review_pages)}</div></details>"
            f"<details open><summary>Retained ({len(accepted)})</summary>"
            f"<div class='sheets'>{images(accepted_pages)}</div></details>"
            f"<details><summary>Filtered ({len(rejected)})</summary>"
            f"<div class='sheets'>{images(rejected_pages)}</div></details></section>"
        )

    retained = sum(bool(row["hard_filter_pass"]) for row in rows)
    selected_total = sum(len(ids) for ids in FEASIBLE_SELECTION.values())
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>G1 90-degree pose-bank review</title><style>
body{{font-family:system-ui;margin:20px;background:#11151b;color:#e9edf2}}
a{{color:#9bc6ff}}table{{border-collapse:collapse;background:#181e27}}
th,td{{border:1px solid #394250;padding:6px 9px}}section{{border-top:1px solid #394250;margin-top:28px}}
.sheets{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
img{{width:100%;height:auto;background:#000}}summary{{cursor:pointer;padding:10px 0;font-weight:700}}
.note{{max-width:1100px;color:#bac5d1}}
</style></head><body><h1>Unitree G1 · 90° right-arm pose lattice</h1>
<p class="note">Review candidate only. Mirror is not a category. The left arm can copy the accepted
right-arm pose with sign-correct mirroring. Every legal −90°/0°/+90° combination is shown exactly once.
Direction is classified from the dominant wrist→palm EE axis, matching Tiago's wrist→fingertips
rule. Horizontal/vertical uses the projected hand-span axis; palm normal is stored separately.
e=T/F denotes <code>elbow_bended</code>. Red borders mark the visually selected feasible subset,
using the same per-category counts as Tiago's 47-pose plausible set.</p>
<p><b>{len(rows)}</b> total · <b>{retained}</b> retained ·
<b>{len(rows) - retained}</b> hard-filtered · <b>{selected_total}</b> feasible selected</p>
<table><tr><th>category</th><th>all</th><th>retained</th><th>filtered</th></tr>
{''.join(summary_rows)}</table>{''.join(sections)}</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")


def main() -> None:
    scene = _make_g1_scene_xml()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        rows = _evaluate(model)
        payload = {
            "definition": {
                "prototype_only": True,
                "mirror_is_category": False,
                "joint_lattice_deg": list(LATTICE_DEG),
                "directions": ["front", "back", "in", "out", "up", "down"],
                "gripper_orientations": ["horizontal", "vertical"],
                "elbow_bended_threshold_deg": ELBOW_BEND_THRESHOLD_DEG,
                "hard_filters": [
                    "arm_body_collision",
                    "collapsed_arm",
                    "ee_x_below_10",
                ],
                "min_shoulder_wrist_distance_m": MIN_SHOULDER_WRIST_DISTANCE_M,
                "min_ee_x_pct": MIN_EE_X_PCT,
                "soft_filters": [
                    "xyz_chebyshev_bin1_dedupe_keep_distinct_palm_max2",
                ],
                "xyz_dedup_bin": XYZ_DEDUP_BIN,
            },
            "stats": {
                "total": len(rows),
                "retained": sum(bool(row["hard_filter_pass"]) for row in rows),
                "filtered": sum(not row["hard_filter_pass"] for row in rows),
            },
            "poses": rows,
        }
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        apply_xyz_palm_dedupe(rows)
        payload["stats"]["xyz_palm_dedupe_kept"] = sum(
            bool(row.get("xyz_palm_dedupe_pass")) for row in rows
        )
        _build_html(model, rows)
        payload["stats"]["feasible_selected"] = sum(
            bool(row.get("feasible_selected")) for row in rows
        )
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        selected_payload = {
            "definition": {
                "source": str(OUT_JSON.relative_to(REPO)),
                "selection": "hard-filtered then visually selected from Google-style 5x5 coverage sheets",
                "mirror_is_category": False,
                "xyz_reference": "right_palm site relative to torso_link",
            },
            "poses": [row for row in rows if row.get("feasible_selected")],
        }
        OUT_SELECTED.parent.mkdir(parents=True, exist_ok=True)
        OUT_SELECTED.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")
        print(f"json: {OUT_JSON}")
        print(f"selected: {OUT_SELECTED}")
        print(f"html: {OUT_HTML}")
        print(json.dumps(payload["stats"], indent=2))
    finally:
        scene.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
