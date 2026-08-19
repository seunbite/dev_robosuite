#!/usr/bin/env python3
"""Few-shot trajectory baselines: joint angles & xyzrpy (Fig1 5×30 = 150 each).

Pipeline:
  1) extract  — GT Motion Codes → per-frame joint / xyzrpy trajectories
  2) generate — Gemini few-shot → 150 joint + 150 xyzrpy predictions
  3) render   — replay trajectories to GIFs + HTML gallery

Platforms (Fig1): humanoid, tiago, stretch, single_arm, roomba.
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _HERE / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from trajectory_parts import (  # noqa: E402
    compact_links_for_prompt,
    compact_parts_for_prompt,
    flatten_joint_parts,
    flatten_link_parts,
    flat_joint_names,
    link_order,
    motion_summary_joints,
    motion_summary_links,
    part_order,
    structure_joint_shot,
    structure_xyz_shot,
    PART_JOINTS,
    REST_JOINTS,
)

PLATFORMS = ["humanoid", "tiago", "stretch", "single_arm", "roomba"]
GT_MASTER = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
CUES_JSON = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
DEFAULT_OUT = _REPO / "data/results/baseline_traj"
DEFAULT_FEWSHOT_SHOTS = _REPO / "data/seed/shots/embodiment_unified/shots_fullbody_parts.yaml"
FEWSHOT_IDX = (1, 5, 11, 20)  # nod_yes, fist_pump, beckon, bow-ish — resolved by idx
HZ = 8
MAX_FRAMES = 32
Mode = Literal["joint", "xyzrpy"]


def _client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    return genai.Client(api_key=key)


def _index_gt() -> dict[tuple[int, str], dict[str, Any]]:
    master = json.loads(GT_MASTER.read_text(encoding="utf-8"))
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in master:
        idx = int(row["idx"])
        cue = str(row["cue"])
        desc = str(row.get("description") or "")
        for plat, cell in (row.get("platforms") or {}).items():
            if plat not in PLATFORMS:
                continue
            c = dict(cell)
            c["idx"] = idx
            c["cue"] = cue
            c["description"] = desc or str(c.get("description") or "")
            out[(idx, plat)] = c
    return out


def _subsample(rows: list[list[float]], max_n: int = MAX_FRAMES) -> list[list[float]]:
    if len(rows) <= max_n:
        return rows
    idxs = np.linspace(0, len(rows) - 1, max_n)
    return [rows[int(round(i))] for i in idxs]


def _round_rows(rows: list[list[float]], nd: int = 4) -> list[list[float]]:
    return [[round(float(x), nd) for x in r] for r in rows]


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]
    # ZYX (yaw-pitch-roll)
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=float)


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


def extract_roomba(cell: dict[str, Any], *, hz: int = HZ) -> dict[str, Any]:
    from embodiment_renderers import _roomba_states

    states = _roomba_states(cell, fps=hz)
    joints = _round_rows(_subsample([[s[0], s[1], s[2]] for s in states]))
    xyzrpy = _round_rows(_subsample([[s[0], s[1], 0.0, 0.0, 0.0, s[2]] for s in states]))
    return {
        "platform": "roomba",
        "joint_names": ["base_x", "base_y", "base_yaw"],
        "joints_rad": joints,
        "xyzrpy": xyzrpy,
        "hz": hz,
    }


def extract_stretch(cell: dict[str, Any], *, hz: int = HZ) -> dict[str, Any]:
    import mujoco
    from stretch_renderer import _STRETCH_SCENE, _stretch_states, _apply

    model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
    data = mujoco.MjData(model)
    states = _stretch_states(cell, model, data, fps=hz)
    if not states:
        raise RuntimeError("empty stretch states")
    jnames = sorted(states[0]["joints"].keys())
    # append base pose into joint vector for LLM (canonical order)
    joint_names = ["base_x", "base_y", "base_yaw"] + jnames
    joints = []
    xyzrpy = []
    # tip site if present
    tip_id = None
    for name in ("ee", "link_grasp_center", "gripper", "wrist"):
        try:
            tip_id = model.site(name).id
            break
        except Exception:
            continue
    if tip_id is None:
        tip_id = 0 if model.nsite else None

    for st in states:
        row = [float(st["x"]), float(st["y"]), float(st["yaw"])] + [
            float(st["joints"][k]) for k in jnames
        ]
        joints.append(row)
        _apply(model, data, x=st["x"], y=st["y"], yaw=st["yaw"], joints=st["joints"])
        if tip_id is not None and model.nsite:
            pos = np.array(data.site_xpos[tip_id], dtype=float)
            mat = np.array(data.site_xmat[tip_id], dtype=float).reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, mat.ravel())
            r, p, y = _quat_wxyz_to_rpy(quat)
            xyzrpy.append([float(pos[0]), float(pos[1]), float(pos[2]), r, p, y])
        else:
            xyzrpy.append(
                [float(st["x"]), float(st["y"]), 0.7, 0.0, 0.0, float(st["yaw"])]
            )

    art = {
        "platform": "stretch",
        "joint_names": joint_names,
        "joints_rad": _round_rows(_subsample(joints)),
        "xyzrpy": _round_rows(_subsample(xyzrpy)),
        "hz": hz,
        "stretch_joint_keys": jnames,
    }
    art["stretch_cartesian"] = stretch_cartesian_from_art(art)
    return art


def stretch_cartesian_from_art(art: dict[str, Any]) -> dict[str, list[list[float]]]:
    """Encode Stretch as base(3) + head pan/tilt(2) + hand pose(6)."""
    names = list(art["joint_names"])
    positions = {name: index for index, name in enumerate(names)}
    required = ("base_x", "base_y", "base_yaw", "head_pan", "head_tilt")
    missing = [name for name in required if name not in positions]
    if missing:
        raise ValueError(f"Stretch joint artifact missing {missing}")
    joints = list(art["joints_rad"])
    hands = list(art["xyzrpy"])
    if len(joints) != len(hands):
        raise ValueError("Stretch joint and hand trajectories have different lengths")
    return {
        "base_xyyaw": [
            [float(row[positions[name]]) for name in ("base_x", "base_y", "base_yaw")]
            for row in joints
        ],
        # Stretch exposes pan/tilt rather than a full three-axis head frame.
        "head_rp": [
            [float(row[positions[name]]) for name in ("head_pan", "head_tilt")]
            for row in joints
        ],
        "hand_xyzrpy": [[float(value) for value in row] for row in hands],
    }


def extract_humanoid(cell: dict[str, Any], *, hz: int = HZ) -> dict[str, Any]:
    import mujoco
    from embodiment_renderers import _g1_motion_qpos, _make_g1_scene_xml, _G1_XML

    if not _G1_XML.is_file():
        raise FileNotFoundError(_G1_XML)
    scene = _make_g1_scene_xml()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        qseq = _g1_motion_qpos(cell, model, fps=hz)
        name_to_qpos: dict[str, int] = {}
        for i in range(model.njnt):
            name = model.joint(i).name
            adr = int(model.jnt_qposadr[i])
            if int(model.jnt_type[i]) in (2, 3):  # slide / hinge; omit floating root
                name_to_qpos[name] = adr
        joint_names = list(name_to_qpos)

        # Full semantic Cartesian space: root, torso/head and both hands.
        wanted = {
            "root": ("pelvis", "torso_link", "pelvis_contour_link"),
            "torso": ("torso_link", "waist_yaw_link", "pelvis"),
            # head_link is a mesh, not a body; neck_pitch_link carries pan/tilt head.
            "head": ("neck_pitch_link", "neck_yaw_link", "torso_link"),
            "right_hand": ("right_rubber_hand", "right_palm", "right_wrist_yaw_link"),
            "left_hand": ("left_rubber_hand", "left_palm", "left_wrist_yaw_link"),
        }
        body_ids: list[int] = []
        link_names: list[str] = []
        for label, candidates in wanted.items():
            bid = None
            for name in candidates:
                try:
                    bid = model.body(name).id
                    break
                except Exception:
                    pass
            if bid is not None:
                link_names.append(label)
                body_ids.append(int(bid))

        joints = []
        xyzrpy = []
        for q in qseq:
            data.qpos[:] = q
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            joints.append([float(data.qpos[name_to_qpos[n]]) for n in joint_names])
            cart = []
            for bid in body_ids:
                pos = np.array(data.xpos[bid], dtype=float)
                mat = np.array(data.xmat[bid], dtype=float).reshape(3, 3)
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, mat.ravel())
                r, p, y = _quat_wxyz_to_rpy(quat)
                cart.extend([float(pos[0]), float(pos[1]), float(pos[2]), r, p, y])
            xyzrpy.append(cart)
        return {
            "platform": "humanoid",
            "joint_names": joint_names,
            "joints_rad": _round_rows(_subsample(joints)),
            "xyzrpy": _round_rows(_subsample(xyzrpy)),
            "link_names": link_names,
            "hz": hz,
        }
    finally:
        scene.unlink(missing_ok=True)


def extract_single_arm(cell: dict[str, Any], *, hz: int = HZ) -> dict[str, Any]:
    from motion_generation_core import MotionGenerator
    from path_ee_ik import get_eef_quat_wxyz

    def _arm_qpos_slices(gen):
        idxs = [int(i) for i in gen.robot._ref_joint_pos_indexes]
        names = [str(x) for x in list(gen.robot.robot_model.arm_joints)]
        return idxs, names

    cue = str(cell["cue"])
    idx = int(cell["idx"])
    tmp = DEFAULT_OUT / "_tmp_sa_cfg.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps([cell], indent=2, ensure_ascii=False), encoding="utf-8")

    gen = MotionGenerator(
        robot_name="IIWA",
        capture_image_width=256,
        capture_image_height=256,
        has_renderer=False,
        has_offscreen_renderer=True,
        hz=int(hz),
    )
    qpos_idxs, joint_names = _arm_qpos_slices(gen)
    xyzrpy: list[list[float]] = []
    joints: list[list[float]] = []
    orig = gen._capture_image

    def _capture():
        tip = gen._get_eef_position()
        mujoco_model = gen.env.sim.model._model
        mujoco_data = gen.env.sim.data._data
        ik = gen.jacobian_calculator.ik_solver
        site_id = ik.site_ids[0]
        quat = get_eef_quat_wxyz(mujoco_model, mujoco_data, site_id)
        r, p, y = _quat_wxyz_to_rpy(quat)
        xyzrpy.append([float(tip[0]), float(tip[1]), float(tip[2]), r, p, y])
        q = np.asarray(gen.env.sim.data.qpos[qpos_idxs], dtype=np.float64)
        joints.append([float(x) for x in q.tolist()])
        return orig()

    gen._capture_image = _capture
    try:
        gen.execute_cue(
            cue=cue,
            cue_idx=idx,
            pose_index=None,
            config_path=str(tmp),
            hz=int(hz),
            save_gif=False,
            overlay_progress_bar=False,
        )
    finally:
        gen._capture_image = orig
        try:
            gen.env.close()
        except Exception:
            pass

    return {
        "platform": "single_arm",
        "joint_names": joint_names,
        "joints_rad": _round_rows(_subsample(joints)),
        "xyzrpy": _round_rows(_subsample(xyzrpy)),
        "link_names": ["end_effector"],
        "hz": hz,
    }


def extract_tiago(cell: dict[str, Any], *, hz: int = HZ) -> dict[str, Any]:
    """Extract by driving Tiago render once with qpos/EE probes when possible.

    Falls back to stretch-like base+arm proxy from movement parameters if the
    full Tiago env cannot be opened in this process.
    """
    try:
        return _extract_tiago_env(cell, hz=hz)
    except Exception as exc:  # noqa: BLE001
        print(f"[extract/tiago] env failed ({exc}); using kinematic proxy", flush=True)
        return _extract_tiago_proxy(cell, hz=hz)


def _extract_tiago_proxy(cell: dict[str, Any], *, hz: int) -> dict[str, Any]:
    """Lightweight proxy: synthesize a short arm+base path from Motion Code attrs."""
    # Reuse roomba base states + a 7-dof synthetic arm wiggle from move degrees.
    from embodiment_renderers import _roomba_states

    base = _roomba_states(cell, fps=hz)
    joint_names = [
        "torso",
        "arm_1",
        "arm_2",
        "arm_3",
        "arm_4",
        "arm_5",
        "arm_6",
        "arm_7",
    ]
    # Scan movements for a degrees span to drive a simple nod/wave proxy
    amp = 0.35
    for step in cell.get("movements") or []:
        params = step.get("parameters") or {}
        mv = params.get("movement") or params.get("path") or {}
        deg = mv.get("degrees") or mv.get("sweep")
        if isinstance(deg, list) and deg:
            try:
                amp = math.radians(abs(float(deg[0])))
            except Exception:
                pass
            break
        if isinstance(deg, (int, float)):
            amp = math.radians(abs(float(deg)) * 0.25)
            break

    joints = []
    xyzrpy = []
    for i, (x, y, yaw) in enumerate(base):
        t = i / max(1, len(base) - 1)
        wave = amp * math.sin(t * 2 * math.pi)
        q = [0.2, 0.0, wave, 0.0, -wave * 0.5, 0.0, 0.0, wave * 0.3]
        joints.append([x, y, yaw] + q)
        # EE proxy in front of base
        xyzrpy.append(
            [
                x + 0.45 * math.cos(yaw),
                y + 0.45 * math.sin(yaw),
                0.9 + 0.05 * math.sin(t * 2 * math.pi),
                0.0,
                0.0,
                yaw,
            ]
        )
    names = ["base_x", "base_y", "base_yaw"] + joint_names
    return {
        "platform": "tiago",
        "joint_names": names,
        "joints_rad": _round_rows(_subsample(joints)),
        "xyzrpy": _round_rows(_subsample(xyzrpy)),
        "hz": hz,
        "proxy": True,
    }


def _extract_tiago_env(cell: dict[str, Any], *, hz: int) -> dict[str, Any]:
    from correction_lib import _import_tiago_render

    make_env, render_config, mod = _import_tiago_render()
    env = make_env()
    qrows: list[list[float]] = []
    xyzrows: list[list[float]] = []
    link_names = ["base", "torso", "head", "right_hand", "left_hand"]
    body_candidates = {
        "base": ("robot0_base_footprint", "robot0_base_link"),
        "torso": ("robot0_torso_lift_link",),
        "head": ("robot0_head_2_link", "robot0_head_1_link"),
        "right_hand": ("robot0_right_hand", "robot0_arm_right_6_link"),
        "left_hand": ("robot0_left_hand", "robot0_arm_left_6_link"),
    }
    body_ids = []
    for label in link_names:
        found = None
        for name in body_candidates[label]:
            try:
                found = int(env.sim.model.body_name2id(name))
                break
            except Exception:
                pass
        if found is None:
            raise RuntimeError(f"Tiago body not found: {label}")
        body_ids.append(found)

    orig_capture = mod._capture
    def tracked_capture(cap_env, q, camera=mod.CAM):
        img = orig_capture(cap_env, q, camera)
        qrows.append([float(v) for v in np.asarray(q[:24], dtype=float)])
        row: list[float] = []
        for bid in body_ids:
            pos = np.asarray(cap_env.sim.data.body_xpos[bid], dtype=float)
            mat = np.asarray(cap_env.sim.data.body_xmat[bid], dtype=float).reshape(3, 3)
            quat = np.zeros(4)
            import mujoco
            mujoco.mju_mat2Quat(quat, mat.ravel())
            r, p, y = _quat_wxyz_to_rpy(quat)
            row.extend([float(pos[0]), float(pos[1]), float(pos[2]), r, p, y])
        xyzrows.append(row)
        return img
    mod._capture = tracked_capture
    try:
        render_config(cell, env=env, overlay_progress_bar=False)
    finally:
        mod._capture = orig_capture
        try:
            env.close()
        except Exception:
            pass
    if not qrows:
        raise RuntimeError("Tiago renderer produced no states")
    names = ["base_forward", "base_side", "base_yaw", "torso_lift", "head_pan", "head_tilt"]
    names += [f"right_arm_{i}" for i in range(1, 7)]
    names += [f"right_gripper_{i}" for i in range(1, 7)]
    names += [f"left_arm_{i}" for i in range(1, 7)]
    return {"platform": "tiago", "joint_names": names[:24],
            "joints_rad": _round_rows(_subsample(qrows)),
            "link_names": link_names, "xyzrpy": _round_rows(_subsample(xyzrows)),
            "hz": hz, "proxy": False}


EXTRACTORS = {
    "roomba": extract_roomba,
    "stretch": extract_stretch,
    "humanoid": extract_humanoid,
    "single_arm": extract_single_arm,
    "tiago": extract_tiago,
}


def _allowed_idxs(args: argparse.Namespace) -> set[int] | None:
    """Optional cue filter supporting lists and inclusive ranges (``1-10,15``)."""
    raw = str(getattr(args, "only_idx", "") or "").strip()
    if raw:
        selected: set[int] = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                start_s, end_s = token.split("-", 1)
                start, end = int(start_s), int(end_s)
                if start > end:
                    raise ValueError(f"descending cue range is not allowed: {token}")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(token))
        return selected
    if getattr(args, "first_n", 0):
        return set(range(1, int(args.first_n) + 1))
    return None


def cmd_extract(args: argparse.Namespace) -> None:
    out = Path(args.out_dir).expanduser().resolve()
    gt = _index_gt()
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or PLATFORMS
    allow = _allowed_idxs(args)
    n_ok = n_fail = 0
    for (idx, plat), cell in sorted(gt.items()):
        if plat not in plats:
            continue
        if allow is not None and idx not in allow:
            continue
        dest = out / "gt_extract" / plat / f"g{idx:02d}_{cell['cue']}.json"
        if dest.is_file() and not args.force:
            n_ok += 1
            continue
        try:
            art = EXTRACTORS[plat](cell, hz=HZ)
            art.update(
                {
                    "idx": idx,
                    "cue": cell["cue"],
                    "description": cell.get("description") or "",
                }
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            n_ok += 1
            print(f"[extract] {plat} g{idx:02d} {cell['cue']} frames={len(art['joints_rad'])}", flush=True)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"[extract] FAIL {plat} g{idx:02d}: {exc}", flush=True)
    print(f"extract done ok={n_ok} fail={n_fail}", flush=True)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


def _load_extract(out: Path, plat: str, idx: int, cue: str) -> dict[str, Any]:
    p = out / "gt_extract" / plat / f"g{idx:02d}_{cue}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _shot_has_traj(art: dict[str, Any]) -> bool:
    """Reject hollow YAML rows (keys present, empty values) that break schema."""
    names = list(art.get("joint_names") or [])
    joints = list(art.get("joints_rad") or [])
    xyz = list(art.get("xyzrpy") or [])
    if not names or not joints:
        return False
    if any(not isinstance(row, (list, tuple)) or len(row) != len(names) for row in joints):
        return False
    if not xyz:
        return False
    width = len(xyz[0]) if isinstance(xyz[0], (list, tuple)) else 0
    if width < 3:
        return False
    if any(not isinstance(row, (list, tuple)) or len(row) != width for row in xyz):
        return False
    return True



def _fewshot_artifacts(
    out: Path,
    plat: str,
    gt: dict[tuple[int, str], dict[str, Any]],
    *,
    shots_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load few-shots (structured parts/links or legacy flat)."""
    if shots_path and shots_path.is_file():
        root = yaml.safe_load(shots_path.read_text(encoding="utf-8")) or {}
        rows = (root.get("robots") or root).get(plat) or {}
        # The per-part file currently specializes the full-body humanoid and
        # Tiago shots.  Keep the same frozen five-shot source for platforms not
        # materialized there (notably single_arm) instead of silently falling
        # back to target GT extracts from ``out/gt_extract``.
        if not rows and shots_path.name == "shots_fullbody_parts.yaml":
            # The structured file specializes humanoid / Tiago.  Single-arm
            # lives in the flat full-body file, while the already-correct
            # Stretch / Roomba shots live in the original five-robot file.
            for fallback_name in ("shots_fullbody.yaml", "shots.yaml"):
                fallback_path = shots_path.with_name(fallback_name)
                if not fallback_path.is_file():
                    continue
                fallback_root = yaml.safe_load(fallback_path.read_text(encoding="utf-8")) or {}
                rows = (fallback_root.get("robots") or fallback_root).get(plat) or {}
                if rows:
                    break
        arts = []
        for cue, block in rows.items():
            if not isinstance(block, dict):
                continue
            art = _shot_block_to_art(plat, cue, block)
            if art and _shot_has_traj(art):
                arts.append(art)
            elif block:
                print(f"[fewshot] skip hollow shot {plat}/{cue}", flush=True)
        if arts:
            return arts
    arts = []
    for idx in FEWSHOT_IDX:
        cell = gt.get((idx, plat))
        if not cell:
            continue
        path = out / "gt_extract" / plat / f"g{idx:02d}_{cell['cue']}.json"
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            art = _enrich_art_parts(plat, {
                "cue": cell["cue"],
                "hz": int(raw.get("hz", HZ)),
                "joint_names": list(raw.get("joint_names") or []),
                "joints_rad": list(raw.get("joints_rad") or []),
                "xyzrpy": list(raw.get("xyzrpy") or []),
                "link_names": list(raw.get("link_names") or []),
            })
            if _shot_has_traj(art):
                arts.append(art)
    return arts


def _shot_block_to_art(plat: str, cue: str, block: dict[str, Any]) -> dict[str, Any] | None:
    parts_block = block.get("parts")
    links_block = block.get("links")
    joint = block.get("joint_angles") or {}
    xyz = block.get("xyzrpy") or {}
    hz = int(block.get("hz") or joint.get("hz") or xyz.get("hz") or HZ)

    if parts_block and isinstance(parts_block, dict):
        names, values = flatten_joint_parts(parts_block, plat)
        art: dict[str, Any] = {
            "cue": cue, "hz": hz, "joint_names": names, "joints_rad": values, "parts": parts_block,
        }
    elif joint.get("joint_names") is not None:
        art = {
            "cue": cue,
            "hz": hz,
            "joint_names": list(joint.get("joint_names") or []),
            "joints_rad": list(joint.get("values") or []),
        }
        structured = structure_joint_shot(plat, art["joint_names"], art["joints_rad"], hz=hz)
        art["parts"] = structured["parts"]
        # Prefer gesture-only DOFs for humanoid/tiago prompts.
        if plat in PART_JOINTS:
            names, values = flatten_joint_parts(art["parts"], plat)
            art["joint_names"], art["joints_rad"] = names, values
    else:
        return None

    if links_block and isinstance(links_block, dict):
        link_names, xyz_vals = flatten_link_parts(links_block, plat)
        art["link_names"] = link_names
        art["xyzrpy"] = xyz_vals
        art["links"] = links_block
    else:
        art["link_names"] = list(xyz.get("link_names") or link_order(plat))
        art["xyzrpy"] = list(xyz.get("values") or [])
        structured = structure_xyz_shot(plat, art["link_names"], art["xyzrpy"], hz=hz)
        art["links"] = structured["links"]
        art["link_names"] = structured["link_names"]
    if plat == "stretch":
        art["stretch_cartesian"] = stretch_cartesian_from_art(art)
    art["motion_summary_joints"] = motion_summary_joints(art.get("parts") or {})
    art["motion_summary_links"] = motion_summary_links(art.get("links") or {})
    return art


def _enrich_art_parts(plat: str, art: dict[str, Any]) -> dict[str, Any]:
    if plat == "stretch":
        if "stretch_cartesian" not in art:
            art["stretch_cartesian"] = stretch_cartesian_from_art(art)
        return art
    if plat not in PART_JOINTS:
        return art
    if "parts" not in art:
        structured = structure_joint_shot(plat, art["joint_names"], art["joints_rad"], hz=int(art.get("hz", HZ)))
        art["parts"] = structured["parts"]
        names, values = flatten_joint_parts(art["parts"], plat)
        art["joint_names"], art["joints_rad"] = names, values
    if "links" not in art:
        structured = structure_xyz_shot(
            plat, art.get("link_names") or link_order(plat), art["xyzrpy"], hz=int(art.get("hz", HZ))
        )
        art["links"] = structured["links"]
        art["link_names"] = structured["link_names"]
    art["motion_summary_joints"] = motion_summary_joints(art.get("parts") or {})
    art["motion_summary_links"] = motion_summary_links(art.get("links") or {})
    return art


def _build_prompt(mode: Mode, *, plat: str, target: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    shots = [s for s in shots if s.get("cue") != target["cue"] and _shot_has_traj(s)]
    if not shots:
        raise RuntimeError(f"no few-shot examples for {plat}")
    shots = [_enrich_art_parts(plat, dict(s)) for s in shots]
    lines: list[str] = []

    if plat in PART_JOINTS and mode == "joint":
        parts = part_order(plat)
        lines.append(
            f"Few-shot **per-part joint-angle** trajectories for `{plat}` (radians).\n"
            f"Parts (fixed order): {parts}.\n"
            "Each example lists every part separately so you can see which body region moves.\n"
            "Still parts may appear as constant=true with a single joints_rad row; "
            "your output must expand every part to the same full n_frames of numeric rows."
        )
        for s in shots:
            payload = {
                "cue": s["cue"],
                "hz": s.get("hz", HZ),
                "n_frames": len(s["joints_rad"]),
                "motion_summary": s.get("motion_summary_joints"),
                "parts": compact_parts_for_prompt(s.get("parts") or {}),
            }
            lines.append(f"\n### Example: {s['cue']}")
            lines.append(json.dumps(payload, ensure_ascii=False))
        # Concrete numeric stub (never a string placeholder — models copy those literally).
        schema_parts = {
            p: {
                "joint_names": PART_JOINTS[plat][p],
                "joints_rad": [[0.0] * len(PART_JOINTS[plat][p])],
            }
            for p in parts
        }
        out_spec = (
            '{\n'
            f'  "cue": "{target["cue"]}",\n'
            f'  "hz": {HZ},\n'
            f'  "parts": {json.dumps(schema_parts, ensure_ascii=False)}\n'
            "}"
        )
        rest = REST_JOINTS.get(plat) or {}
        rest_rule = ""
        if rest:
            rest_rule = (
                "- Still-part rest poses are listed below; use them EXACTLY when a part does not move.\n"
                f"- Still rest rows: {json.dumps(rest, ensure_ascii=False)}\n"
                "- For tiago arms: right_arm and left_arm still rows are the SAME numbers "
                "(URDF axes mirror them); do NOT invent a different right-arm hang like [0,-0.3,…].\n"
            )
        rules = (
            f"- Return ALL parts: {parts}\n"
            "- Each part keeps its joint_names EXACTLY as in the examples\n"
            "- Every joints_rad value MUST be a JSON array of arrays of numbers "
            "(never a string like \"[[...]]\")\n"
            "- Still parts: repeat the same numeric row for all frames (span ≈ 0)\n"
            f"{rest_rule}"
            "- Moving parts: only move the joints the cue needs\n"
            f"- ~16–{MAX_FRAMES} frames; all parts share the same n_frames\n"
            "- Do NOT output a flat joints_rad vector; use the parts object"
        )
    elif plat == "stretch" and mode == "xyzrpy":
        lines.append(
            "Few-shot Cartesian trajectories for `stretch` with exactly 11 "
            "values per frame: base [x,y,yaw] + head [pan,tilt] + "
            "hand [x,y,z,roll,pitch,yaw].\n"
            "The hand pose is in the world frame; base and head values are "
            "separate controls and must not be inferred from hand coordinates."
        )
        for s in shots:
            cartesian = s["stretch_cartesian"]
            payload = {
                "cue": s["cue"],
                "hz": s.get("hz", HZ),
                "n_frames": len(cartesian["hand_xyzrpy"]),
                "base_xyyaw": cartesian["base_xyyaw"],
                "head_rp": cartesian["head_rp"],
                "hand_xyzrpy": cartesian["hand_xyzrpy"],
            }
            lines.append(f"\n### Example: {s['cue']}")
            lines.append(json.dumps(payload, ensure_ascii=False))
        out_spec = (
            '{\n'
            f'  "cue": "{target["cue"]}",\n'
            f'  "hz": {HZ},\n'
            '  "base_xyyaw": [[0.0, 0.0, 0.0]],\n'
            '  "head_rp": [[0.0, 0.0]],\n'
            '  "hand_xyzrpy": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]\n'
            "}"
        )
        rules = (
            "- Every frame contains base_xyyaw (3), head_rp (2), and hand_xyzrpy (6)\n"
            "- All three sequences have the same n_frames (about 16–"
            f"{MAX_FRAMES})\n"
            "- head_rp is [head_pan, head_tilt] in radians\n"
            "- CRITICAL: every hand_xyzrpy row is exactly [x,y,z,roll,pitch,yaw] "
            "with SIX floats; a 3D [x,y,z] row is invalid. If the wrist does "
            "not rotate, explicitly append [0.0, 0.0, 0.0] for roll/pitch/yaw\n"
            "- Keep base still unless the cue requires locomotion\n"
            "- Do NOT return a flat vector or omit a sequence"
        )
    elif plat in PART_JOINTS and mode == "xyzrpy":
        links = link_order(plat)
        lines.append(
            f"Few-shot **per-link xyzrpy** trajectories for `{plat}` (world frame).\n"
            f"Links (fixed order): {links}.\n"
            "Each link is a separate [x,y,z,roll,pitch,yaw] series so motion is localized.\n"
            "Still links may appear as constant=true with one pose; "
            "your output must expand every link to the same full n_frames of numeric rows."
        )
        for s in shots:
            payload = {
                "cue": s["cue"],
                "hz": s.get("hz", HZ),
                "n_frames": len(s["xyzrpy"]),
                "motion_summary": s.get("motion_summary_links"),
                "links": compact_links_for_prompt(s.get("links") or {}),
            }
            lines.append(f"\n### Example: {s['cue']}")
            lines.append(json.dumps(payload, ensure_ascii=False))
        schema_links = {lnk: {"xyzrpy": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]} for lnk in links}
        out_spec = (
            '{\n'
            f'  "cue": "{target["cue"]}",\n'
            f'  "hz": {HZ},\n'
            f'  "links": {json.dumps(schema_links, ensure_ascii=False)}\n'
            "}"
        )
        rules = (
            f"- Return ALL links: {links}\n"
            "- Each links.<name>.xyzrpy row is exactly 6 floats [x,y,z,roll,pitch,yaw]\n"
            "- xyzrpy MUST be a JSON array of numeric rows (never a string placeholder)\n"
            "- Still links: keep constant pose (xyz_span≈0 and rpy_span≈0)\n"
            "- Head nods/shakes: change head roll/pitch/yaw, NOT head height (z) and NOT torso height\n"
            "- Arm gestures: move the hand link(s); leave other links still unless needed\n"
            "- Still hands: copy the rest-pose hand xyzrpy from the examples "
            "(left/right rests differ; do not invent a new hang pose)\n"
            f"- ~16–{MAX_FRAMES} frames; all links share the same n_frames\n"
            "- Do NOT output a flat concatenated xyzrpy vector; use the links object"
        )
    else:
        joint_names = shots[0]["joint_names"]
        link_names = shots[0].get("link_names") or ["end_effector"]
        if mode == "joint":
            lines.append(f"Few-shot joint-angle trajectories for embodiment `{plat}` (radians):")
            for s in shots:
                payload = {
                    "cue": s["cue"], "hz": s.get("hz", HZ), "joint_names": s["joint_names"],
                    "n_frames": len(s["joints_rad"]), "joints_rad": s["joints_rad"],
                }
                lines.append(f"\n### Example: {s['cue']}")
                lines.append(json.dumps(payload, ensure_ascii=False))
            out_spec = (
                '{\n'
                f'  "cue": "{target["cue"]}",\n'
                f'  "hz": {HZ},\n'
                f'  "joint_names": {json.dumps(joint_names)},\n'
                '  "joints_rad": [[...], ...]\n'
                "}"
            )
            rules = (
                f"- joint_names MUST be exactly {json.dumps(joint_names)}\n"
                f"- each joints_rad row has exactly {len(joint_names)} floats\n"
                f"- ~16–{MAX_FRAMES} frames; smooth; stay near example numeric ranges"
            )
        else:
            lines.append(f"Few-shot named full-body xyzrpy trajectories for embodiment `{plat}`:")
            for s in shots:
                payload = {
                    "cue": s["cue"], "hz": s.get("hz", HZ), "n_frames": len(s["xyzrpy"]),
                    "link_names": s.get("link_names") or link_names, "xyzrpy": s["xyzrpy"],
                }
                lines.append(f"\n### Example: {s['cue']}")
                lines.append(json.dumps(payload, ensure_ascii=False))
            out_spec = (
                '{\n'
                f'  "cue": "{target["cue"]}",\n'
                f'  "hz": {HZ},\n'
                f'  "link_names": {json.dumps(link_names)},\n'
                '  "xyzrpy": [[...], ...]\n'
                "}"
            )
            rules = (
                f"- link_names MUST be exactly {json.dumps(link_names)}\n"
                f"- each xyzrpy row has exactly {6 * len(link_names)} floats (6 per named link)\n"
                f"- ~16–{MAX_FRAMES} frames; smooth; stay near example numeric ranges"
            )

    return f"""You are an expert robot motion designer for the `{plat}` embodiment.

Task: Given few-shot trajectories for iconic gesture cues, output a NEW trajectory for a target cue that does NOT appear in the examples.

Output format: return ONE JSON object only (no markdown fences, no prose):
{out_spec}

Rules:
{rules}
- Produce a readable iconic motion for the cue on this embodiment.
- Combine diverse motions to create a recognizable gesture that transcends the embodiment's limitations.
- Do NOT copy an example trajectory.

{chr(10).join(lines)}
---
Target cue (NOT in the examples above):
cue: {target['cue']}
description: {target.get('description') or ''}
embodiment: {plat}
---
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object")
    raw = m.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([\]}])", r"\1", raw)
        return json.loads(fixed)


def _validate(
    mode: Mode,
    obj: dict[str, Any],
    *,
    cue: str,
    joint_names: list[str],
    link_names: list[str] | None = None,
    plat: str = "",
) -> dict[str, Any]:
    hz = int(obj.get("hz", HZ))
    if mode == "joint" and isinstance(obj.get("parts"), dict) and plat in PART_JOINTS:
        names, rows = flatten_joint_parts(obj["parts"], plat)
        if not rows:
            raise ValueError("empty parts.joints_rad")
        clean = _round_rows(_subsample([[float(x) for x in row] for row in rows]))
        return {"cue": cue, "hz": hz, "joint_names": names, "joints_rad": clean, "parts": obj["parts"]}
    if mode == "xyzrpy" and plat == "stretch":
        fields = {
            "base_xyyaw": 3,
            "head_rp": 2,
            "hand_xyzrpy": 6,
        }
        clean_fields: dict[str, list[list[float]]] = {}
        n_frames: int | None = None
        for name, width in fields.items():
            rows = obj.get(name)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"empty {name}")
            parsed: list[list[float]] = []
            for index, row in enumerate(rows):
                if not isinstance(row, (list, tuple)) or len(row) != width:
                    raise ValueError(f"{name}[{index}] must contain {width} floats")
                parsed.append([float(value) for value in row])
            if n_frames is None:
                n_frames = len(parsed)
            elif len(parsed) != n_frames:
                raise ValueError("Stretch Cartesian sequences must share n_frames")
            clean_fields[name] = parsed
        shared = {
            name: _round_rows(_subsample(rows))
            for name, rows in clean_fields.items()
        }
        return {
            "cue": cue,
            "hz": hz,
            "link_names": ["base", "head", "hand"],
            "stretch_cartesian": shared,
        }
    if mode == "xyzrpy" and isinstance(obj.get("links"), dict) and plat in PART_JOINTS:
        names, rows = flatten_link_parts(obj["links"], plat)
        if not rows:
            raise ValueError("empty links.xyzrpy")
        clean = _round_rows(_subsample(rows))
        return {"cue": cue, "hz": hz, "link_names": names, "xyzrpy": clean, "links": obj["links"]}
    if mode == "joint":
        rows = obj.get("joints_rad")
        if not isinstance(rows, list) or not rows:
            raise ValueError("empty joints_rad")
        n = len(joint_names)
        clean = []
        for i, row in enumerate(rows):
            if not isinstance(row, (list, tuple)) or len(row) != n:
                raise ValueError(f"joints_rad[{i}] len != {n}")
            clean.append([float(x) for x in row])
        clean = _round_rows(_subsample(clean))
        return {"cue": cue, "hz": hz, "joint_names": joint_names, "joints_rad": clean}
    rows = obj.get("xyzrpy") or obj.get("xyz")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty xyzrpy")
    link_names = list(link_names or ["end_effector"])
    clean = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"xyzrpy[{i}] bad")
        if len(link_names) == 1 and len(row) == 3:
            row = list(row) + [0.0, 0.0, 0.0]
        if len(row) != 6 * len(link_names):
            raise ValueError(f"xyzrpy[{i}] must be {6 * len(link_names)} floats")
        clean.append([float(x) for x in row])
    clean = _round_rows(_subsample(clean))
    return {"cue": cue, "hz": hz, "link_names": link_names, "xyzrpy": clean}



def cmd_generate(args: argparse.Namespace) -> None:
    from google.genai import types
    from tqdm.auto import tqdm

    out = Path(args.out_dir).expanduser().resolve()
    gt = _index_gt()
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or PLATFORMS
    modes: list[Mode] = ["joint", "xyzrpy"] if args.mode == "both" else [args.mode]  # type: ignore
    client = _client()

    jobs = []
    for mode in modes:
        pred_path = out / f"predictions_{mode}.jsonl"
        done = set()
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("parse_ok"):
                        done.add(r["job_id"])
        for (idx, plat), cell in sorted(gt.items()):
            if plat not in plats:
                continue
            allow = _allowed_idxs(args)
            if allow is not None and idx not in allow:
                continue
            job_id = f"{mode}.{plat}.g{idx:02d}"
            if job_id in done and not args.force:
                continue
            jobs.append((mode, idx, plat, cell, pred_path))

    effective_thinking = 128 if args.model == "gemini-2.5-pro" and args.thinking_budget == 0 else args.thinking_budget
    print(
        f"generate pending={len(jobs)} model={args.model} "
        f"thinking_budget={effective_thinking} workers={args.workers}",
        flush=True,
    )

    def work(item: tuple) -> dict[str, Any]:
        mode, idx, plat, cell, pred_path = item
        job_id = f"{mode}.{plat}.g{idx:02d}"
        shots = _fewshot_artifacts(out, plat, gt, shots_path=args.fewshot_shots)
        # Target output schema is fixed by the five shots. Do not execute target
        # Motion Code here: this is inference and must not leak target kinematics.
        meta_path = out / "gt_extract" / plat / f"g{idx:02d}_{cell['cue']}.json"
        if not meta_path.is_file():
            art = {"platform": plat, "hz": HZ, "joint_names": shots[0]["joint_names"],
                   "link_names": shots[0].get("link_names") or ["end_effector"],
                   "idx": idx, "cue": cell["cue"], "description": cell.get("description") or ""}
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(art, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        joint_names = meta["joint_names"]
        prompt = _build_prompt(
            mode,
            plat=plat,
            target={"cue": cell["cue"], "description": cell.get("description") or ""},
            shots=shots or [meta],
        )
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                cfg_kwargs: dict[str, Any] = {"temperature": 0.3, "max_output_tokens": 16384}
                try:
                    thinking_budget = int(args.thinking_budget)
                    if args.model == "gemini-2.5-pro" and thinking_budget == 0:
                        thinking_budget = 128  # 2.5 Pro cannot disable thinking.
                    cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
                except Exception:
                    pass
                resp = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg_kwargs),
                )
                raw = (getattr(resp, "text", None) or "").strip()
                if not raw:
                    for cand in getattr(resp, "candidates", None) or []:
                        content = getattr(cand, "content", None)
                        for part in getattr(content, "parts", None) or []:
                            if getattr(part, "text", None):
                                raw += part.text
                    raw = raw.strip()
                # Prefer schema from structured shots when available.
                schema_joints = shots[0]["joint_names"] if shots else joint_names
                schema_links = (shots[0].get("link_names") if shots else None) or ["end_effector"]
                if plat in PART_JOINTS:
                    schema_joints = flat_joint_names(plat)
                    schema_links = link_order(plat)
                payload = _validate(
                    mode,
                    _extract_json(raw),
                    cue=cell["cue"],
                    joint_names=schema_joints,
                    link_names=schema_links,
                    plat=plat,
                )
                row = {
                    "job_id": job_id,
                    "mode": mode,
                    "platform": plat,
                    "idx": idx,
                    "cue": cell["cue"],
                    "description": cell.get("description") or "",
                    "payload": payload,
                    "raw_output": raw,
                    "model": args.model,
                    "parse_ok": True,
                }
                tqdm.write(f"[infer] {job_id} attempt={attempt + 1} parse=PASS")
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return row
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                last_raw = locals().get("raw") or ""
                tqdm.write(f"[infer] {job_id} attempt={attempt + 1} parse=FAIL error={last_err}")
                time.sleep(min(2**attempt, 15))
        return {
            "job_id": job_id,
            "mode": mode,
            "platform": plat,
            "idx": idx,
            "cue": cell["cue"],
            "error": last_err or "unknown",
            "raw_output": (last_raw[:8000] if "last_raw" in locals() else ""),
            "parse_ok": False,
        }

    def render_one(row: dict[str, Any]) -> None:
        mode: Mode = row["mode"]
        plat = row["platform"]
        idx = int(row["idx"])
        cue = row["cue"]
        gif = _trajectory_gif(out, mode, plat, idx, cue, row.get("model") or args.model)
        meta_path = out / "gt_extract" / plat / f"g{idx:02d}_{cue}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        try:
            RENDERERS[plat](row["payload"], mode, gif, meta)
            tqdm.write(f"[render] {row['job_id']} PASS -> {gif}")
            cmd_gallery(args)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[render] {row['job_id']} FAIL error={exc}")

    render_pool = (
        ThreadPoolExecutor(max_workers=max(1, args.render_workers))
        if args.render_as_you_go else None
    )
    with tqdm(total=len(jobs), desc="inference", unit="job") as bar, \
            ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            mode = row["mode"]
            pred_path = out / f"predictions_{mode}.jsonl"
            with pred_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            status = "PASS" if row.get("parse_ok") else "FAIL"
            tqdm.write(f"[saved] {row['job_id']} parse={status} -> {pred_path}")
            bar.update(1)
            if render_pool is not None and row.get("parse_ok"):
                render_pool.submit(render_one, row)
    if render_pool is not None:
        render_pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _model_slug(model: str) -> str:
    """Return a readable, filesystem-safe model identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(model).strip()).strip("-_") or "unknown-model"


def _trajectory_gif(out: Path, mode: Mode, plat: str, idx: int, cue: str, model: str) -> Path:
    return out / f"render_{mode}" / plat / f"emb_g{idx:02d}_{cue}_{_model_slug(model)}.gif"


def _write_gif(frames: list, path: Path, *, hz: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    pil = []
    for fr in frames:
        if isinstance(fr, Image.Image):
            pil.append(fr.convert("RGB"))
        else:
            arr = np.asarray(fr)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            pil.append(Image.fromarray(arr).convert("RGB"))
    if not pil:
        raise ValueError("no frames")
    dur = int(round(1000 / max(1, hz)))
    pil[0].save(path, save_all=True, append_images=pil[1:], duration=dur, loop=0, optimize=False)


def render_roomba_traj(payload: dict[str, Any], mode: Mode, out_gif: Path) -> None:
    import mujoco
    from embodiment_renderers import _ROOMBA_XML, _progress_overlay, _save_media

    if mode == "joint":
        states = [(r[0], r[1], r[2]) for r in payload["joints_rad"]]
    else:
        states = [(r[0], r[1], r[5]) for r in payload["xyzrpy"]]
    hz = int(payload.get("hz", HZ))
    model = mujoco.MjModel.from_xml_string(_ROOMBA_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=420, width=520)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.0)
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -50
    xs = [s[0] for s in states]
    ys = [s[1] for s in states]
    camera.lookat[:] = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 0.0)
    span = max(abs(max(xs) - min(xs)), abs(max(ys) - min(ys)), 0.4)
    camera.distance = max(1.6, 1.2 + 2.2 * span)
    frames = []
    payload["_landmark_names"] = ["base", "heading"]
    payload["_landmarks_xyz"] = []
    payload["_landmarks_rot"] = []
    for i, (x, y, yaw) in enumerate(states):
        data.qpos[:] = 0
        data.qpos[0:3] = (x, y, 0.09)
        data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        payload["_landmarks_xyz"].append([[x, y, 0.09], [x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw), 0.09]])
        rot = [[math.cos(yaw), -math.sin(yaw), 0.0],
               [math.sin(yaw), math.cos(yaw), 0.0],
               [0.0, 0.0, 1.0]]
        payload["_landmarks_rot"].append([rot, rot])
        renderer.update_scene(data, camera)
        frames.append(_progress_overlay(renderer.render(), i, len(states)))
    renderer.close()
    _save_media(frames, out_gif, fps=hz)


def render_stretch_traj(payload: dict[str, Any], mode: Mode, out_gif: Path, meta: dict[str, Any]) -> None:
    import mujoco
    from stretch_renderer import _STRETCH_SCENE, _apply, _ik
    from embodiment_renderers import _progress_overlay, _save_media

    hz = int(payload.get("hz", HZ))
    jkeys = meta.get("stretch_joint_keys") or [
        k for k in payload.get("joint_names", []) if not k.startswith("base_")
    ]
    if not jkeys:
        jkeys = ["lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll", "head_pan", "head_tilt"]
    model = mujoco.MjModel.from_xml_path(str(_STRETCH_SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=560)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145
    camera.elevation = -18
    frames = []
    trace_names = ["base", "head", "elbow", "wrist", "hand"]
    trace_bodies = ["base_link", "link_head_tilt", "link_arm_l4", "link_wrist_yaw", "link_grasp_center"]
    trace_ids = [int(model.body(n).id) for n in trace_bodies]
    payload["_landmark_names"], payload["_landmarks_xyz"] = trace_names, []
    payload["_landmarks_rot"] = []
    def capture_trace():
        payload["_landmarks_xyz"].append([np.asarray(data.xpos[b], float).tolist() for b in trace_ids])
        payload["_landmarks_rot"].append([
            np.asarray(data.xmat[b], float).reshape(3, 3).tolist()
            for b in trace_ids
        ])
    if mode == "joint":
        names = payload["joint_names"]
        # names: base_x, base_y, base_yaw, *jkeys
        for i, row in enumerate(payload["joints_rad"]):
            d = dict(zip(names, row))
            joints = {k: float(d[k]) for k in jkeys if k in d}
            _apply(model, data, x=d["base_x"], y=d["base_y"], yaw=d["base_yaw"], joints=joints)
            capture_trace()
            camera.lookat[:] = (d["base_x"] + 0.12, d["base_y"], 0.62)
            camera.distance = 2.5
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), i, len(payload["joints_rad"])))
    elif payload.get("stretch_cartesian"):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        cartesian = payload["stretch_cartesian"]
        bases = cartesian["base_xyyaw"]
        heads = cartesian["head_rp"]
        hands = cartesian["hand_xyzrpy"]
        if not (len(bases) == len(heads) == len(hands)):
            raise ValueError("Stretch Cartesian base, head, and hand frame counts differ")
        hand_site_id = None
        for name in ("ee", "link_grasp_center", "gripper", "wrist"):
            try:
                hand_site_id = model.site(name).id
                break
            except Exception:
                continue
        if hand_site_id is None and model.nsite:
            hand_site_id = 0
        if hand_site_id is None:
            raise ValueError("Stretch hand site is unavailable")
        solve_keys = ("lift", "arm", "wrist_yaw", "wrist_pitch", "wrist_roll")
        lo = np.asarray([0.0, 0.0, -math.pi / 2, -1.57, -3.14])
        hi = np.asarray([1.1, 0.52, 4.42, 0.56, 3.14])
        solved = {key: 0.0 for key in jkeys}
        solved.setdefault("grip", 0.0)

        for i, (base, head, hand) in enumerate(zip(bases, heads, hands)):
            x, y, yaw = [float(value) for value in base]
            head_pan, head_tilt = [float(value) for value in head]
            target_pos = np.asarray(hand[:3], dtype=float)
            target_rot = Rotation.from_euler("xyz", hand[3:6]).as_matrix()
            # Solve position first.  A direct all-DOF pose solve can sit at the
            # folded-arm local minimum when the requested world orientation is
            # infeasible, making an authored hand path look motionless.
            position_seed_joints = _ik(
                model,
                data,
                target=target_pos,
                x=x,
                y=y,
                yaw=yaw,
                joints={**solved, "head_pan": head_pan, "head_tilt": head_tilt},
                iters=20,
                max_step={"lift": 0.15, "arm": 0.10, "wrist_yaw": 0.45},
            )
            seed = np.asarray([position_seed_joints[key] for key in solve_keys])

            def residual(values: np.ndarray) -> np.ndarray:
                candidate = dict(solved)
                candidate.update(zip(solve_keys, values))
                candidate["head_pan"] = head_pan
                candidate["head_tilt"] = head_tilt
                _apply(model, data, x=x, y=y, yaw=yaw, joints=candidate)
                position_error = np.asarray(data.site_xpos[hand_site_id], dtype=float) - target_pos
                current_rot = np.asarray(data.site_xmat[hand_site_id], dtype=float).reshape(3, 3)
                rotation_error = Rotation.from_matrix(current_rot.T @ target_rot).as_rotvec()
                # Position controls the visible arm gesture.  Orientation is a
                # soft wrist preference because Stretch cannot realize every
                # six-axis world-frame hand pose independently of its arm.
                return np.concatenate((15.0 * position_error, 0.15 * rotation_error, 0.01 * (values - seed)))

            result = least_squares(
                residual,
                np.clip(seed, lo, hi),
                bounds=(lo, hi),
                max_nfev=80,
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
            )
            solved.update(zip(solve_keys, result.x))
            solved["head_pan"] = head_pan
            solved["head_tilt"] = head_tilt
            _apply(model, data, x=x, y=y, yaw=yaw, joints=solved)
            capture_trace()
            camera.lookat[:] = (x + 0.12, y, 0.62)
            camera.distance = 2.5
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), i, len(hands)))
    else:
        # Backward-compatible replay for historical 6D Stretch Cartesian artifacts.
        for i, row in enumerate(payload["xyzrpy"]):
            x, y, z, r, p, yaw = row
            joints = {k: 0.0 for k in jkeys}
            # map z loosely to lift if present
            for k in jkeys:
                if "lift" in k or k.endswith("z"):
                    joints[k] = float(np.clip(z, 0.0, 1.1))
            _apply(model, data, x=x, y=y, yaw=yaw, joints=joints)
            capture_trace()
            camera.lookat[:] = (x + 0.12, y, 0.62)
            camera.distance = 2.5
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), i, len(payload["xyzrpy"])))
    renderer.close()
    _save_media(frames, out_gif, fps=hz)


def render_humanoid_traj(payload: dict[str, Any], mode: Mode, out_gif: Path) -> None:
    import mujoco
    from embodiment_renderers import _make_g1_scene_xml, _G1_XML, _progress_overlay, _save_media, _g1_home, _joint_qpos_map

    if not _G1_XML.is_file():
        raise FileNotFoundError(_G1_XML)
    scene = _make_g1_scene_xml()
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=480)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.85)
        camera.distance = 2.7
        camera.azimuth = 145
        camera.elevation = -8
        jmap = _joint_qpos_map(model)
        home = _g1_home(model, jmap)
        hz = int(payload.get("hz", HZ))
        frames = []
        trace_names = ["root", "torso", "head", "right_elbow", "right_hand", "left_elbow", "left_hand"]
        trace_bodies = ["pelvis", "torso_link", "neck_pitch_link", "right_elbow_link", "right_wrist_yaw_link", "left_elbow_link", "left_wrist_yaw_link"]
        trace_ids = [int(model.body(n).id) for n in trace_bodies]
        payload["_landmark_names"], payload["_landmarks_xyz"] = trace_names, []
        payload["_landmarks_rot"] = []
        def capture_trace():
            payload["_landmarks_xyz"].append([np.asarray(data.xpos[b], float).tolist() for b in trace_ids])
            payload["_landmarks_rot"].append([
                np.asarray(data.xmat[b], float).reshape(3, 3).tolist()
                for b in trace_ids
            ])
        if mode == "joint":
            names = payload["joint_names"]
            for i, row in enumerate(payload["joints_rad"]):
                q = np.array(home, dtype=float)
                for n, v in zip(names, row):
                    if n in jmap:
                        q[jmap[n]] = float(v)
                data.qpos[:] = q
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                capture_trace()
                renderer.update_scene(data, camera)
                frames.append(_progress_overlay(renderer.render(), i, len(payload["joints_rad"])))
        else:
            from scipy.optimize import least_squares
            q = np.array(home, dtype=float)
            names_in_model = {model.body(i).name: i for i in range(model.nbody)}
            candidates = {
                "root": ("pelvis", "torso_link", "pelvis_contour_link"),
                "torso": ("torso_link", "waist_yaw_link", "pelvis"),
                "head": ("neck_pitch_link", "neck_yaw_link", "torso_link"),
                "right_elbow": ("right_elbow_link",),
                "right_hand": ("right_rubber_hand", "right_palm", "right_wrist_yaw_link"),
                "left_elbow": ("left_elbow_link",),
                "left_hand": ("left_rubber_hand", "left_palm", "left_wrist_yaw_link"),
            }
            bids = [next(names_in_model[n] for n in candidates[label] if n in names_in_model)
                    for label in payload["link_names"]]
            # Cartesian targets never include feet; if hip/knee/ankle stay free the
            # underconstrained IK uses legs as slack → dancing feet. Freeze legs.
            _leg_tok = ("hip", "knee", "ankle")
            jnames = [
                n for n in jmap
                if model.jnt_range[model.joint(n).id, 1] - model.jnt_range[model.joint(n).id, 0] > 1e-8
                and not any(tok in n for tok in _leg_tok)
            ]
            if payload.get("position_only"):
                # A vector program only authorizes its named body handles.
                # Freeze unrelated chains so underconstrained IK cannot move an
                # idle arm while solving a target for the other hand.
                active = set(payload["link_names"])

                def controls_active_chain(name: str) -> bool:
                    return (
                        ("right_hand" in active and name.startswith("right_") and any(tok in name for tok in ("shoulder", "elbow", "wrist")))
                        or ("left_hand" in active and name.startswith("left_") and any(tok in name for tok in ("shoulder", "elbow", "wrist")))
                        # Head-only vectors must stay in the real head pan/tilt
                        # chain. Waist is added only for an explicit full bow.
                        or (
                            "head" in active
                            and (
                                name in ("head_pan_joint", "head_tilt_joint")
                                or "neck" in name
                                or (payload.get("cue") == "bow_greeting" and "waist" in name)
                            )
                        )
                        or ("torso" in active and "waist" in name)
                    )

                jnames = [n for n in jnames if controls_active_chain(n)]
            qidx = np.asarray([jmap[n] for n in jnames], dtype=int)
            lo = np.asarray([model.jnt_range[model.joint(n).id, 0] for n in jnames])
            hi = np.asarray([model.jnt_range[model.joint(n).id, 1] for n in jnames])
            head_target_index = (
                payload["link_names"].index("head")
                if payload.get("position_only") and "head" in payload["link_names"]
                else None
            )
            head_neutral_target = None
            if head_target_index is not None:
                initial_targets = np.asarray(payload["xyzrpy"][0], dtype=float).reshape(-1, 6)
                head_neutral_target = initial_targets[head_target_index, :3].copy()
            head_pan_qpos = jmap.get("head_pan_joint")
            head_tilt_qpos = jmap.get("head_tilt_joint")
            if head_target_index is not None:
                direct_head_qpos = {x for x in (head_pan_qpos, head_tilt_qpos) if x is not None}
                keep = np.asarray([int(qi) not in direct_head_qpos for qi in qidx], dtype=bool)
                qidx, lo, hi = qidx[keep], lo[keep], hi[keep]
            for i, row in enumerate(payload["xyzrpy"]):
                targets = np.asarray(row, dtype=float).reshape(-1, 6)
                if head_target_index is not None and head_neutral_target is not None:
                    # The synthetic pan/tilt axes pass through the head body
                    # origin, so rotating them cannot match a translated head
                    # xyz through positional IK. Treat the authored head point
                    # as a virtual gaze marker instead: lateral displacement is
                    # pan, while forward/down displacement is a downward nod.
                    head_delta = targets[head_target_index, :3] - head_neutral_target
                    if head_pan_qpos is not None:
                        pan_joint = model.joint("head_pan_joint")
                        pan_lo, pan_hi = model.jnt_range[pan_joint.id]
                        q[head_pan_qpos] = float(np.clip(4.0 * head_delta[1], pan_lo, pan_hi))
                    if head_tilt_qpos is not None:
                        tilt_joint = model.joint("head_tilt_joint")
                        tilt_lo, tilt_hi = model.jnt_range[tilt_joint.id]
                        tilt = 6.0 * head_delta[0] - 2.0 * head_delta[2]
                        q[head_tilt_qpos] = float(np.clip(tilt, tilt_lo, tilt_hi))
                wrist_qpos = {
                    jmap[name] for name in jnames if "wrist" in name
                }
                wrist_reference = q[qidx].copy()
                wrist_weight = float(payload.get("wrist_continuity_weight", 0.0))
                wrist_weight *= (1.0 - i / max(1, len(payload["xyzrpy"]) - 1)) ** 2
                def residual(x, targets=targets):
                    q[qidx] = x
                    data.qpos[:] = q
                    mujoco.mj_forward(model, data)
                    chunks: list[np.ndarray] = []
                    direction_tracks = payload.get("fingertip_directions") or {}
                    local_axes = payload.get("fingertip_local_axes") or {}
                    palm_tracks = payload.get("palm_directions") or {}
                    palm_axes = payload.get("palm_local_axes") or {}
                    pos_weight = float(payload.get("hand_position_weight", 1.0))
                    elbow_weight = float(payload.get("elbow_position_weight", 0.8))
                    dir_weight = float(payload.get("fingertip_direction_weight", 0.15))
                    palm_weight = float(payload.get("palm_direction_weight", 0.15))
                    for label, b, t in zip(payload["link_names"], bids, targets):
                        position_error = np.asarray(data.xpos[b], dtype=float) - t[:3]
                        if label in direction_tracks:
                            weight = pos_weight
                        elif label.endswith("_elbow"):
                            weight = elbow_weight
                        else:
                            weight = 1.0
                        chunks.append(weight * position_error)
                        if label in direction_tracks and label in local_axes:
                            mat = np.asarray(data.xmat[b], dtype=float).reshape(3, 3)
                            current_direction = mat @ np.asarray(local_axes[label], dtype=float)
                            desired_direction = np.asarray(direction_tracks[label][i], dtype=float)
                            chunks.append(dir_weight * (current_direction - desired_direction))
                        if label in palm_tracks and label in palm_axes:
                            mat = np.asarray(data.xmat[b], dtype=float).reshape(3, 3)
                            current_palm = mat @ np.asarray(palm_axes[label], dtype=float)
                            desired_palm = np.asarray(palm_tracks[label][i], dtype=float)
                            chunks.append(palm_weight * (current_palm - desired_palm))
                        if not payload.get("position_only"):
                            # Full xyzrpy baseline matches orientation too. Vector
                            # Drag-and-Draw intentionally supplies xyz only.
                            quat = np.zeros(4)
                            mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[b], dtype=float).ravel())
                            rpy = np.asarray(_quat_wxyz_to_rpy(quat), dtype=float)
                            d = rpy - t[3:6]
                            d = (d + math.pi) % (2.0 * math.pi) - math.pi
                            chunks.append(0.35 * d)
                    wrist_mask = np.asarray([int(qi) in wrist_qpos for qi in qidx], dtype=bool)
                    if wrist_weight > 0.0 and np.any(wrist_mask):
                        chunks.append(wrist_weight * (x[wrist_mask] - wrist_reference[wrist_mask]))
                    return np.concatenate(chunks)

                if qidx.size:
                    q[qidx] = least_squares(
                        residual, np.clip(q[qidx], lo, hi), bounds=(lo, hi), max_nfev=100
                    ).x
                data.qpos[:] = q
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                capture_trace()
                renderer.update_scene(data, camera)
                frames.append(_progress_overlay(renderer.render(), i, len(payload["xyzrpy"])))
        renderer.close()
        _save_media(
            frames,
            out_gif,
            fps=hz,
            write_mp4=not bool(payload.get("skip_mp4", False)),
        )
    finally:
        scene.unlink(missing_ok=True)


def render_single_arm_traj(payload: dict[str, Any], mode: Mode, out_gif: Path) -> None:
    import mujoco
    from motion_generation_core import MotionGenerator
    from path_ee_ik import get_eef_quat_wxyz, ik_move_eef_to
    from PIL import Image

    def _arm_qpos_slices(gen):
        idxs = [int(i) for i in gen.robot._ref_joint_pos_indexes]
        names = [str(x) for x in list(gen.robot.robot_model.arm_joints)]
        return idxs, names

    def _to_pil_frame(fr):
        if isinstance(fr, Image.Image):
            return fr.convert("RGB")
        arr = np.asarray(fr)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")

    def capture_with_targets(spheres):
        """Render LLM-authored world-space handle targets into the MuJoCo scene."""
        context = gen.env.sim._render_context_offscreen
        camera_id = gen.env.sim.model.camera_name2id("frontview")
        context.render(512, 512, camera_id=camera_id)
        start = context.scn.ngeom
        for offset, (position, color) in enumerate(spheres):
            mujoco.mjv_initGeom(
                context.scn.geoms[start + offset],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.asarray([0.024, 0.0, 0.0], dtype=float),
                np.asarray(position, dtype=float),
                np.eye(3).ravel(),
                np.asarray(color, dtype=np.float32),
            )
        context.scn.ngeom = start + len(spheres)
        try:
            viewport = mujoco.MjrRect(0, 0, 512, 512)
            mujoco.mjr_render(viewport, context.scn, context.con)
            return _to_pil_frame(context.read_pixels(512, 512)[::-1])
        finally:
            context.scn.ngeom = start

    hz = int(payload.get("hz", HZ))
    gen = MotionGenerator(
        robot_name="IIWA",
        capture_image_width=512,
        capture_image_height=512,
        has_renderer=False,
        has_offscreen_renderer=True,
        hz=hz,
    )
    try:
        frames = []
        trace_names = ["base", "upper_arm", "elbow", "wrist", "hand"]
        trace_candidates = [("robot0_base", "base"), ("robot0_link_3", "link_3"), ("robot0_link_4", "link_4"), ("robot0_link_6", "link_6")]
        mm, md = gen.env.sim.model._model, gen.env.sim.data._data
        body_lookup = {mm.body(i).name: i for i in range(mm.nbody)}
        trace_ids = [next(body_lookup[n] for n in cand if n in body_lookup) for cand in trace_candidates]
        site_id_trace = gen.jacobian_calculator.ik_solver.site_ids[0]
        payload["_landmark_names"], payload["_landmarks_xyz"] = trace_names, []
        payload["_landmarks_rot"] = []
        def capture_trace():
            pts = [np.asarray(md.xpos[b], float).tolist() for b in trace_ids]
            pts.append(np.asarray(md.site_xpos[site_id_trace], float).tolist())
            payload["_landmarks_xyz"].append(pts)
            rotations = [
                np.asarray(md.xmat[b], float).reshape(3, 3).tolist()
                for b in trace_ids
            ]
            rotations.append(
                np.asarray(md.site_xmat[site_id_trace], float).reshape(3, 3).tolist()
            )
            payload["_landmarks_rot"].append(rotations)
        if mode == "joint":
            qpos_idxs, _ = _arm_qpos_slices(gen)
            gen._set_joint_positions(gen.initial_joint_pos)
            for qframe in payload["joints_rad"]:
                base = np.asarray(gen.env.sim.data.qpos, dtype=np.float64)
                base[qpos_idxs] = np.asarray(qframe, dtype=float)
                gen.env.sim.data.qpos[:] = base
                gen.env.sim.forward()
                capture_trace()
                if not payload.get("skip_capture", False):
                    frames.append(_to_pil_frame(gen._capture_image()))
        else:
            gen._set_joint_positions(gen.initial_joint_pos)
            mujoco_model = gen.env.sim.model._model
            mujoco_data = gen.env.sim.data._data
            ik = gen.jacobian_calculator.ik_solver
            site_id = ik.site_ids[0]
            elbow_targets = payload.get("elbow_targets")
            if elbow_targets:
                import mujoco
                from scipy.optimize import least_squares

                qpos_idxs, _ = _arm_qpos_slices(gen)
                qidx = np.asarray(qpos_idxs, dtype=int)
                names_in_model = {mujoco_model.body(i).name: i for i in range(mujoco_model.nbody)}
                elbow_id = next(names_in_model[n] for n in ("robot0_link_4", "link_4") if n in names_in_model)
                wrist_id = next(names_in_model[n] for n in ("robot0_link_6", "link_6") if n in names_in_model)
                upper_arm_id = next(names_in_model[n] for n in ("robot0_link_3", "link_3") if n in names_in_model)
                forearm_id = next(names_in_model[n] for n in ("robot0_link_5", "link_5") if n in names_in_model)
                upper_arm_targets = payload.get("upper_arm_targets")
                forearm_targets = payload.get("forearm_targets")
                wrist_targets = payload.get("wrist_targets")
                qadr_to_joint = {int(mujoco_model.jnt_qposadr[j]): j for j in range(mujoco_model.njnt)}
                lo, hi = [], []
                for qi in qidx:
                    jid = qadr_to_joint[int(qi)]
                    if mujoco_model.jnt_limited[jid]:
                        lo.append(float(mujoco_model.jnt_range[jid, 0]))
                        hi.append(float(mujoco_model.jnt_range[jid, 1]))
                    else:
                        lo.append(-2.0 * math.pi)
                        hi.append(2.0 * math.pi)
                lo, hi = np.asarray(lo), np.asarray(hi)
                q = np.asarray(gen.env.sim.data.qpos, dtype=float).copy()
                pos_weight = float(payload.get("ee_position_weight", 8.0))
                elbow_weight = float(payload.get("elbow_position_weight", 8.0))
                finger_rows = (payload.get("fingertip_directions") or {}).get("end_effector")
                fingertip_targets = (payload.get("fingertip_targets") or {}).get("end_effector")
                finger_axis = np.asarray(
                    (payload.get("fingertip_local_axes") or {}).get("end_effector", [0.0, 0.0, 1.0]),
                    dtype=float,
                )
                finger_weight = float(payload.get("fingertip_direction_weight", 2.0))
                palm_rows = (payload.get("palm_directions") or {}).get("end_effector")
                palm_axis = np.asarray(
                    (payload.get("palm_local_axes") or {}).get("end_effector", [1.0, 0.0, 0.0]),
                    dtype=float,
                )
                palm_weight = float(payload.get("palm_direction_weight", 2.0))
                continuity_weight = float(payload.get("joint_continuity_weight", 0.08))
                continuity_score_weight = float(payload.get("joint_continuity_score_weight", 0.04))
                max_joint_step = float(payload.get("max_joint_step_rad", 0.22))
                render_start_frame = int(payload.get("render_start_frame", 0))
                if not 0 <= render_start_frame < len(payload["xyzrpy"]):
                    raise ValueError(
                        f"render_start_frame={render_start_frame} outside "
                        f"0..{len(payload['xyzrpy']) - 1}"
                    )
                initial_arm_q = q[qidx].copy()
                rng = np.random.default_rng(731)

                def geometry(q_arm):
                    q[qidx] = q_arm
                    mujoco_data.qpos[:] = q
                    mujoco.mj_forward(mujoco_model, mujoco_data)
                    ee = np.asarray(mujoco_data.site_xpos[site_id], dtype=float)
                    elbow = np.asarray(mujoco_data.xpos[elbow_id], dtype=float)
                    wrist = np.asarray(mujoco_data.xpos[wrist_id], dtype=float)
                    upper_arm = np.asarray(mujoco_data.xpos[upper_arm_id], dtype=float)
                    forearm = np.asarray(mujoco_data.xpos[forearm_id], dtype=float)
                    rotation = np.asarray(mujoco_data.site_xmat[site_id], dtype=float).reshape(3, 3)
                    finger = rotation @ finger_axis
                    finger /= max(np.linalg.norm(finger), 1e-9)
                    palm = rotation @ palm_axis
                    palm -= finger * float(np.dot(palm, finger))
                    palm /= max(np.linalg.norm(palm), 1e-9)
                    return ee, upper_arm, elbow, forearm, wrist, finger, palm

                def solve_from(seed, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm):
                    seed = np.clip(np.asarray(seed, dtype=float), lo, hi)

                    def residual(x):
                        ee, upper_arm, elbow, forearm, wrist, finger, palm = geometry(x)
                        parts = [
                            pos_weight * (ee - target_ee),
                            elbow_weight * (elbow - target_elbow),
                        ]
                        if target_upper_arm is not None:
                            parts.append(elbow_weight * (upper_arm - target_upper_arm))
                        if target_forearm is not None:
                            parts.append(elbow_weight * (forearm - target_forearm))
                        if target_wrist is not None:
                            parts.append(elbow_weight * (wrist - target_wrist))
                        if target_finger is not None:
                            parts.append(finger_weight * (finger - target_finger))
                        if target_palm is not None:
                            parts.append(palm_weight * (palm - target_palm))
                        # Preserve the previous/substep joint branch. Without a
                        # meaningful continuity term, redundant IIWA solutions
                        # can alternate between two equally good Cartesian fits.
                        parts.append(continuity_weight * (x - seed))
                        return np.concatenate(parts)

                    return least_squares(
                        residual, seed, bounds=(lo, hi), max_nfev=240,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10,
                    ).x

                def errors(q_arm, reference_q, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm):
                    ee, upper_arm, elbow, forearm, wrist, finger, palm = geometry(q_arm)
                    ee_err = float(np.linalg.norm(ee - target_ee))
                    upper_arm_err = 0.0 if target_upper_arm is None else float(np.linalg.norm(upper_arm - target_upper_arm))
                    elbow_err = float(np.linalg.norm(elbow - target_elbow))
                    forearm_err = 0.0 if target_forearm is None else float(np.linalg.norm(forearm - target_forearm))
                    wrist_err = 0.0 if target_wrist is None else float(np.linalg.norm(wrist - target_wrist))
                    finger_err = 0.0 if target_finger is None else float(np.linalg.norm(finger - target_finger))
                    palm_err = 0.0 if target_palm is None else float(np.linalg.norm(palm - target_palm))
                    joint_delta = float(np.linalg.norm(np.asarray(q_arm) - np.asarray(reference_q)))
                    # EE dominates. Auxiliary points refine posture/orientation,
                    # while joint distance prevents a remote redundant branch
                    # from winning for a tiny Cartesian improvement.
                    score = (
                        ee_err
                        + 0.25 * (upper_arm_err + elbow_err + forearm_err + wrist_err)
                        + 0.05 * (finger_err + palm_err)
                        + continuity_score_weight * joint_delta
                    )
                    return (score, ee_err, upper_arm_err, elbow_err, forearm_err, wrist_err, finger_err, palm_err, joint_delta)

                previous_ee, previous_upper_arm, previous_elbow, previous_forearm, previous_wrist, previous_finger, previous_palm = geometry(initial_arm_q)
                previous_q = initial_arm_q.copy()
                for fi, row in enumerate(payload["xyzrpy"]):
                    target_ee = np.asarray(row[:3], dtype=float)
                    target_upper_arm = None if not upper_arm_targets else np.asarray(upper_arm_targets[fi], dtype=float)
                    target_elbow = np.asarray(elbow_targets[fi], dtype=float)
                    target_forearm = None if not forearm_targets else np.asarray(forearm_targets[fi], dtype=float)
                    target_wrist = None if not wrist_targets else np.asarray(wrist_targets[fi], dtype=float)
                    target_finger = None
                    if finger_rows:
                        target_finger = np.asarray(finger_rows[fi], dtype=float)
                        target_finger /= max(np.linalg.norm(target_finger), 1e-9)
                    target_palm = None
                    if palm_rows:
                        target_palm = np.asarray(palm_rows[fi], dtype=float)
                        if target_finger is not None:
                            target_palm -= target_finger * float(np.dot(target_palm, target_finger))
                        target_palm /= max(np.linalg.norm(target_palm), 1e-9)

                    # Continue through large Cartesian jumps in small steps so a
                    # one-frame neutral->pose transition cannot select a remote IK branch.
                    deltas = [
                        float(np.linalg.norm(target_ee - previous_ee)),
                        float(np.linalg.norm(target_elbow - previous_elbow)),
                    ]
                    if target_upper_arm is not None:
                        deltas.append(float(np.linalg.norm(target_upper_arm - previous_upper_arm)))
                    if target_forearm is not None:
                        deltas.append(float(np.linalg.norm(target_forearm - previous_forearm)))
                    if target_wrist is not None:
                        deltas.append(float(np.linalg.norm(target_wrist - previous_wrist)))
                    max_delta = max(deltas)
                    substeps = max(1, min(20, int(math.ceil(max_delta / 0.06))))
                    start_ee, start_upper_arm = previous_ee.copy(), previous_upper_arm.copy()
                    start_elbow, start_forearm = previous_elbow.copy(), previous_forearm.copy()
                    start_wrist = previous_wrist.copy()
                    start_finger = previous_finger.copy()
                    start_palm = previous_palm.copy()
                    solved = q[qidx].copy()
                    for si in range(1, substeps + 1):
                        alpha = si / substeps
                        waypoint_ee = start_ee + alpha * (target_ee - start_ee)
                        waypoint_upper_arm = None if target_upper_arm is None else start_upper_arm + alpha * (target_upper_arm - start_upper_arm)
                        waypoint_elbow = start_elbow + alpha * (target_elbow - start_elbow)
                        waypoint_forearm = None if target_forearm is None else start_forearm + alpha * (target_forearm - start_forearm)
                        waypoint_wrist = None if target_wrist is None else start_wrist + alpha * (target_wrist - start_wrist)
                        waypoint_finger = None
                        waypoint_palm = None
                        if target_finger is not None:
                            waypoint_finger = start_finger + alpha * (target_finger - start_finger)
                            waypoint_finger /= max(np.linalg.norm(waypoint_finger), 1e-9)
                        if target_palm is not None:
                            waypoint_palm = start_palm + alpha * (target_palm - start_palm)
                            if waypoint_finger is not None:
                                waypoint_palm -= waypoint_finger * float(np.dot(waypoint_palm, waypoint_finger))
                            waypoint_palm /= max(np.linalg.norm(waypoint_palm), 1e-9)
                        solved = solve_from(solved, waypoint_ee, waypoint_upper_arm, waypoint_elbow, waypoint_forearm, waypoint_wrist, waypoint_finger, waypoint_palm)

                    best = (errors(solved, previous_q, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm), solved)
                    # Retry remote seeds only when the authoritative EE target is
                    # genuinely missed. An inconsistent elbow/direction hint must
                    # not trigger a switch to another redundant IK branch.
                    if best[0][1] > 0.015:
                        seeds = [previous_q, initial_arm_q, 0.5 * (lo + hi)]
                        seeds.extend(rng.uniform(lo, hi) for _ in range(14))
                        for seed in seeds:
                            candidate = solve_from(seed, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm)
                            candidate_result = errors(candidate, previous_q, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm)
                            if candidate_result < best[0]:
                                best = (candidate_result, candidate)
                            if best[0][1] < 0.003 and best[0][8] < 0.35:
                                break
                    # Physical continuity guard: at 20 fps, a branch flip is a
                    # multi-radian one-frame jump. Clamp it and let subsequent
                    # frames converge instead of teleporting the joints. The
                    # hidden Drag-and-Draw warm-start is the exception: it must
                    # fully reach the source clip's first pose before frame 0 is
                    # captured.
                    joint_step = np.asarray(best[1]) - previous_q
                    limited = previous_q + np.clip(joint_step, -max_joint_step, max_joint_step)
                    is_hidden_warm_start = render_start_frame > 0 and fi == render_start_frame
                    if not is_hidden_warm_start and np.any(np.abs(joint_step) > max_joint_step):
                        best = (
                            errors(limited, previous_q, target_ee, target_upper_arm, target_elbow, target_forearm, target_wrist, target_finger, target_palm),
                            limited,
                        )
                    if payload.get("debug_ik_errors"):
                        print(
                            f"[single_arm_ik] frame={fi} ee={best[0][1]:.5f} "
                            f"upper={best[0][2]:.5f} elbow={best[0][3]:.5f} "
                            f"forearm={best[0][4]:.5f} wrist={best[0][5]:.5f} "
                            f"fingertip_dir={best[0][6]:.5f} palm_dir={best[0][7]:.5f} "
                            f"joint_delta={best[0][8]:.5f} q={np.round(best[1], 4).tolist()}"
                        )
                    q[qidx] = best[1]
                    mujoco_data.qpos[:] = q
                    mujoco.mj_forward(mujoco_model, mujoco_data)
                    previous_ee = np.asarray(mujoco_data.site_xpos[site_id], dtype=float).copy()
                    previous_upper_arm = np.asarray(mujoco_data.xpos[upper_arm_id], dtype=float).copy()
                    previous_elbow = np.asarray(mujoco_data.xpos[elbow_id], dtype=float).copy()
                    previous_forearm = np.asarray(mujoco_data.xpos[forearm_id], dtype=float).copy()
                    previous_wrist = np.asarray(mujoco_data.xpos[wrist_id], dtype=float).copy()
                    rotation = np.asarray(mujoco_data.site_xmat[site_id], dtype=float).reshape(3, 3)
                    previous_finger = rotation @ finger_axis
                    previous_finger /= max(np.linalg.norm(previous_finger), 1e-9)
                    previous_palm = rotation @ palm_axis
                    previous_palm -= previous_finger * float(np.dot(previous_palm, previous_finger))
                    previous_palm /= max(np.linalg.norm(previous_palm), 1e-9)
                    previous_q = np.asarray(best[1], dtype=float).copy()
                    # A Drag-and-Draw single-arm program has a neutral state at
                    # frame 0 for its declarative contract. It is warm-started
                    # into the first authored pose but deliberately not shown:
                    # source Motion-Code GIFs begin after their initial pose has
                    # already been applied.
                    if fi >= render_start_frame:
                        capture_trace()
                        if not payload.get("skip_capture", False):
                            if payload.get("show_llm_target_spheres") and fingertip_targets:
                                frames.append(capture_with_targets([
                                    (target_elbow, (0.96, 0.55, 0.12, 1.0)),
                                    (target_ee, (0.12, 0.47, 0.95, 1.0)),
                                    (fingertip_targets[fi], (0.16, 0.72, 0.32, 1.0)),
                                ]))
                            else:
                                frames.append(_to_pil_frame(gen._capture_image()))
            else:
                for row in payload["xyzrpy"]:
                    x, y, z, r, p, yaw = row
                    quat = get_eef_quat_wxyz(mujoco_model, mujoco_data, site_id) if payload.get("position_only") else _rpy_to_quat_wxyz(r, p, yaw)
                    ik_move_eef_to(
                        ik_solver=ik,
                        mujoco_model=mujoco_model,
                        mujoco_data=mujoco_data,
                        site_id=site_id,
                        target_pos=np.array([x, y, z], dtype=float),
                        quat_wxyz=quat,
                    )
                    capture_trace()
                    if not payload.get("skip_capture", False):
                        frames.append(_to_pil_frame(gen._capture_image()))
        if not payload.get("skip_capture", False):
            _write_gif(frames, out_gif, hz=hz)
    finally:
        try:
            gen.env.close()
        except Exception:
            pass


# Flat controllable names → MuJoCo qpos indices.
# qpos: 0-2 base, 3 torso, 4-5 head, 6-11 R arm, 12-17 R grip, 18-23 L arm, 24-29 L grip.
_TIAGO_JOINT_QPOS: dict[str, int] = {
    "base_forward": 0,
    "base_side": 1,
    "base_yaw": 2,
    "torso_lift": 3,
    "head_pan": 4,
    "head_tilt": 5,
    **{f"right_arm_{i}": 5 + i for i in range(1, 7)},
    **{f"left_arm_{i}": 17 + i for i in range(1, 7)},
}
_TIAGO_LINK_QPOS: dict[str, list[int]] = {
    "base": [0, 1, 2],
    "torso": [3],
    "head": [4, 5],
    "right_elbow": [6, 7, 8, 9, 10, 11],
    "right_hand": [6, 7, 8, 9, 10, 11],
    "left_elbow": [18, 19, 20, 21, 22, 23],
    "left_hand": [18, 19, 20, 21, 22, 23],
}


def _tiago_apply_joint_row(q: np.ndarray, joint_names: list[str], row: list[float]) -> None:
    """Write only named controllable DOFs; never clobber gripper slots."""
    for name, val in zip(joint_names, row):
        qi = _TIAGO_JOINT_QPOS.get(name)
        if qi is not None:
            q[qi] = float(val)


def _tiago_free_dofs_from_xyz(link_names: list[str], rows: list[list[float]], *, eps: float = 1e-3) -> list[int]:
    """Free only DOFs whose target link actually moves (freeze still torso/arms/base)."""
    if not rows:
        return [4, 5]
    arr = np.asarray(rows, dtype=float)
    free: list[int] = []
    for i, name in enumerate(link_names):
        block = arr[:, i * 6 : (i + 1) * 6]
        span = float(np.max(block.max(0) - block.min(0))) if block.size else 0.0
        if span > eps:
            free.extend(_TIAGO_LINK_QPOS.get(name, []))
    # Head-only nods still need head DOFs even if numeric span is tiny after rounding.
    if not free:
        free = list(_TIAGO_LINK_QPOS.get("head", [4, 5]))
    # unique preserve order
    seen: set[int] = set()
    out: list[int] = []
    for qi in free:
        if qi not in seen:
            seen.add(qi)
            out.append(qi)
    return out


def _tiago_ik_bounds(qidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(qidx), -3.0)
    hi = np.full(len(qidx), 3.0)
    for i, qi in enumerate(qidx):
        if int(qi) == 3:
            lo[i], hi[i] = 0.0, 0.35
        elif int(qi) == 2:
            lo[i], hi[i] = -math.pi, math.pi
        elif int(qi) in (0, 1):
            lo[i], hi[i] = -1.0, 1.0
    return lo, hi


def _tiago_ik_residual(
    env,
    q: np.ndarray,
    qidx: np.ndarray,
    bids: list[int],
    targets: np.ndarray,
    *,
    position_only: bool = False,
    link_names: list[str] | None = None,
    direction_targets: dict[str, list[float]] | None = None,
    local_axes: dict[str, list[float]] | None = None,
    palm_targets: dict[str, list[float]] | None = None,
    palm_axes: dict[str, list[float]] | None = None,
    position_weight: float = 1.0,
    elbow_position_weight: float = 0.8,
    direction_weight: float = 0.15,
    palm_weight: float = 0.15,
    reference_q: np.ndarray | None = None,
    wrist_qpos_ids: tuple[int, ...] = (),
    wrist_continuity_weight: float = 0.0,
):
    link_names = link_names or [str(i) for i in range(len(bids))]
    direction_targets = direction_targets or {}
    local_axes = local_axes or {}
    palm_targets = palm_targets or {}
    palm_axes = palm_axes or {}

    def residual(x: np.ndarray) -> np.ndarray:
        q[qidx] = x
        env.sim.data.qpos[:] = q
        env.sim.forward()
        chunks: list[np.ndarray] = []
        for label, b, t in zip(link_names, bids, targets):
            position_error = np.asarray(env.sim.data.body_xpos[b], dtype=float) - t[:3]
            if label in direction_targets:
                weight = position_weight
            elif label.endswith("_elbow"):
                weight = elbow_position_weight
            else:
                weight = 1.0
            chunks.append(weight * position_error)
            if label in direction_targets and label in local_axes:
                mat = np.asarray(env.sim.data.body_xmat[b], dtype=float).reshape(3, 3)
                current_direction = mat @ np.asarray(local_axes[label], dtype=float)
                desired_direction = np.asarray(direction_targets[label], dtype=float)
                chunks.append(direction_weight * (current_direction - desired_direction))
            if label in palm_targets and label in palm_axes:
                mat = np.asarray(env.sim.data.body_xmat[b], dtype=float).reshape(3, 3)
                current_palm = mat @ np.asarray(palm_axes[label], dtype=float)
                desired_palm = np.asarray(palm_targets[label], dtype=float)
                chunks.append(palm_weight * (current_palm - desired_palm))
            if not position_only:
                quat = np.zeros(4)
                mat = np.asarray(env.sim.data.body_xmat[b], dtype=float).reshape(3, 3)
                import mujoco
                mujoco.mju_mat2Quat(quat, mat.ravel())
                rpy = np.asarray(_quat_wxyz_to_rpy(quat), dtype=float)
                d = rpy - t[3:6]
                d = (d + math.pi) % (2.0 * math.pi) - math.pi
                chunks.append(0.25 * d)
        if reference_q is not None and wrist_continuity_weight > 0.0:
            wrist_mask = np.asarray(
                [int(qi) in wrist_qpos_ids for qi in qidx], dtype=bool
            )
            if np.any(wrist_mask):
                chunks.append(
                    wrist_continuity_weight * (x[wrist_mask] - reference_q[wrist_mask])
                )
        return np.concatenate(chunks)

    return residual


def _tiago_link_spans(link_names: list[str], rows: list[list[float]]) -> dict[str, float]:
    arr = np.asarray(rows, dtype=float)
    out: dict[str, float] = {}
    for i, name in enumerate(link_names):
        block = arr[:, i * 6 : (i + 1) * 6]
        out[name] = float(np.max(block.max(0) - block.min(0))) if block.size else 0.0
    return out


def _capture_tiago_target_spheres(env, mod, q: np.ndarray, spheres: list[tuple[np.ndarray, float, tuple[float, float, float, float]]]):
    """Render a Tiago frame with authored Cartesian target spheres in-scene."""
    import mujoco
    from PIL import Image

    env.sim.data.qpos[:] = q
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()
    context = env.sim._render_context_offscreen
    width, height = int(mod.W), int(mod.H)
    if width > context.con.offWidth or height > context.con.offHeight:
        context.update_offscreen_size(max(width, context.model.vis.global_.offwidth), max(height, context.model.vis.global_.offheight))
    context.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    context.cam.fixedcamid = int(env.sim.model.camera_name2id(mod.CAM))
    mujoco.mjv_updateScene(
        context.model._model, context.data._data, context.vopt, context.pert,
        context.cam, mujoco.mjtCatBit.mjCAT_ALL, context.scn,
    )
    for position, radius, rgba in spheres:
        index = int(context.scn.ngeom)
        if index >= int(context.scn.maxgeom):
            break
        mujoco.mjv_initGeom(
            context.scn.geoms[index],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray([radius, 0.0, 0.0], dtype=float),
            np.asarray(position, dtype=float),
            np.eye(3).ravel(),
            np.asarray(rgba, dtype=np.float32),
        )
        context.scn.ngeom = index + 1
    viewport = mujoco.MjrRect(0, 0, width, height)
    mujoco.mjr_render(viewport, context.scn, context.con)
    frame = context.read_pixels(width, height, depth=False)
    return Image.fromarray(np.flipud(frame).copy())


def render_tiago_traj(payload: dict[str, Any], mode: Mode, out_gif: Path) -> None:
    """Replay trajectories on the real Tiago model (never the old cylinder proxy)."""
    from correction_lib import _import_tiago_render
    from scipy.optimize import least_squares
    hz = int(payload.get("hz", HZ))
    make_env, _render_config, mod = _import_tiago_render()
    env = make_env()
    home = np.asarray(mod._default_qpos(env), dtype=float)
    frames = []
    trace_names = ["base", "torso", "head", "right_elbow", "right_hand", "left_elbow", "left_hand"]
    trace_bodies = ["robot0_base_link", "robot0_torso_lift_link", "robot0_head_2_link", "robot0_arm_right_4_link", "robot0_right_hand", "robot0_arm_left_4_link", "robot0_left_hand"]
    trace_ids = [int(env.sim.model.body_name2id(n)) for n in trace_bodies]
    payload["_landmark_names"], payload["_landmarks_xyz"] = trace_names, []
    payload["_landmarks_rot"] = []
    def capture(q, spheres=None):
        env.sim.data.qpos[:] = q
        env.sim.forward()
        payload["_landmarks_xyz"].append([np.asarray(env.sim.data.body_xpos[b], float).tolist() for b in trace_ids])
        payload["_landmarks_rot"].append([
            np.asarray(env.sim.data.body_xmat[b], float).reshape(3, 3).tolist()
            for b in trace_ids
        ])
        if spheres:
            return _capture_tiago_target_spheres(env, mod, q, spheres)
        return mod._capture(env, q, mod.CAM)
    if mode == "joint":
        # Faithful replay: controllable DOFs come only from the LLM row.
        # Grippers are not in the schema — keep whatever the env default uses.
        names = list(payload.get("joint_names") or [])
        for row in payload["joints_rad"]:
            q = home.copy()
            _tiago_apply_joint_row(q, names, row)
            frames.append(capture(q))
    else:
        body_map = {
            "base": "robot0_base_link",
            "torso": "robot0_torso_lift_link",
            "head": "robot0_head_2_link",
            "right_elbow": "robot0_arm_right_4_link",
            "right_hand": "robot0_right_hand",
            "left_elbow": "robot0_arm_left_4_link",
            "left_hand": "robot0_left_hand",
        }
        link_names = list(payload["link_names"])
        bids = [int(env.sim.model.body_name2id(body_map[n])) for n in link_names]
        rows = payload["xyzrpy"]
        spans = _tiago_link_spans(link_names, rows)
        eps = 1e-3
        # Moving-link DOFs only on subsequent frames (no torso/arm slack for nods).
        free_move = _tiago_free_dofs_from_xyz(link_names, rows, eps=eps)
        for name in (payload.get("fingertip_directions") or {}):
            free_move.extend(_TIAGO_LINK_QPOS.get(name, []))
        free_move = list(dict.fromkeys(free_move))
        # Seed frame: also solve still arm/hand (and still base) so frozen pose
        # matches the LLM xyzrpy targets — NOT the asymmetric render home.
        seed_links = [n for n in link_names if n in _TIAGO_LINK_QPOS]
        free_seed: list[int] = []
        for n in seed_links:
            if n == "torso" and spans.get("torso", 0.0) <= eps:
                continue  # never borrow torso_lift for head-only cues
            if n == "base" and spans.get("base", 0.0) <= eps:
                continue
            free_seed.extend(_TIAGO_LINK_QPOS[n])
        # unique
        seen: set[int] = set()
        free_seed = [qi for qi in free_seed if not (qi in seen or seen.add(qi))]
        if not free_seed:
            free_seed = list(free_move)

        q = home.copy()
        head_target_index = (
            link_names.index("head")
            if payload.get("position_only") and "head" in link_names
            else None
        )
        head_neutral_target = None
        if head_target_index is not None:
            initial_targets = np.asarray(rows[0], dtype=float).reshape(-1, 6)
            head_neutral_target = initial_targets[head_target_index, :3].copy()
        # If torso target is still, pin torso_lift to home (schema mid) — do not IK it.
        for fi, row in enumerate(rows):
            targets = np.asarray(row, dtype=float).reshape(-1, 6)
            use = list(free_seed if fi == 0 else free_move)
            if head_target_index is not None and head_neutral_target is not None:
                # As with G1, the authored xyz is a virtual gaze marker rather
                # than a literal body-origin translation. Tiago's joint signs
                # are opposite to the G1 synthetic neck: left/down are negative.
                head_delta = targets[head_target_index, :3] - head_neutral_target
                q[4] = float(np.clip(-4.0 * head_delta[1], -1.2, 1.2))
                tilt = -(6.0 * head_delta[0] - 2.0 * head_delta[2])
                q[5] = float(np.clip(tilt, -0.9, 0.7))
                # Keep these direct values fixed while any other active links
                # are solved. For head-only motion this leaves no positional IK.
                use = [qi for qi in use if qi not in (4, 5)]
            qidx = np.asarray(use, dtype=int)
            lo, hi = _tiago_ik_bounds(qidx)
            direction_tracks = payload.get("fingertip_directions") or {}
            direction_targets = {
                name: direction_tracks[name][fi]
                for name in link_names
                if name in direction_tracks
            }
            palm_tracks = payload.get("palm_directions") or {}
            palm_targets = {
                name: palm_tracks[name][fi]
                for name in link_names
                if name in palm_tracks
            }
            def authored_spheres():
                if not payload.get("show_authored_target_spheres"):
                    return None
                target_by_link = {
                    name: np.asarray(targets[index, :3], dtype=float)
                    for index, name in enumerate(link_names)
                }
                tip_tracks = payload.get("fingertip_targets") or {}
                spheres = []
                for side in ("right", "left"):
                    elbow, hand = f"{side}_elbow", f"{side}_hand"
                    if elbow in target_by_link:
                        spheres.append((target_by_link[elbow], 0.034, (0.95, 0.15, 0.70, 0.92)))
                    if hand in target_by_link:
                        spheres.append((target_by_link[hand], 0.036, (0.05, 0.85, 1.00, 0.92)))
                    if hand in tip_tracks:
                        spheres.append((np.asarray(tip_tracks[hand][fi], dtype=float), 0.026, (1.00, 0.78, 0.05, 0.92)))
                return spheres
            if qidx.size:
                hand_indices = {name: i for i, name in enumerate(link_names)}
                symmetric_hands = (
                    {"right_hand", "left_hand"}.issubset(hand_indices)
                    and "right_hand" in direction_targets
                    and "left_hand" in direction_targets
                )
                if symmetric_hands:
                    left_hand = targets[hand_indices["left_hand"], :3].copy()
                    left_hand[1] *= -1.0
                    left_direction = np.asarray(direction_targets["left_hand"], dtype=float).copy()
                    left_direction[1] *= -1.0
                    symmetric_hands = (
                        np.linalg.norm(targets[hand_indices["right_hand"], :3] - left_hand) <= 0.005
                        and np.linalg.norm(
                            np.asarray(direction_targets["right_hand"], dtype=float) - left_direction
                        ) <= 0.08
                    )
                elbow_indices = [
                    (i, bid) for i, (name, bid) in enumerate(zip(link_names, bids))
                    if name.endswith("_elbow")
                ]
                if (
                    elbow_indices
                    and bool(payload.get("position_only"))
                    and payload.get("ik_priority") != "hand_elbow"
                ):
                    # The vector DSL guarantees an authored neutral frame 0.
                    # Preserve the renderer's neutral configuration exactly,
                    # rather than letting a redundant IK solve pick a different
                    # but visually equivalent arm branch before the trajectory.
                    if fi == 0:
                        q = home.copy()
                        frames.append(capture(q, authored_spheres()))
                        continue
                    # Solve the hand / fingertip objective first from several
                    # IK basins. Elbow is deliberately absent from this residual:
                    # it chooses among visually equivalent hand solutions rather
                    # than pulling the hand away from its authored trajectory.
                    env.sim.data.qpos[:] = q
                    env.sim.forward()
                    previous_elbows = [
                        np.asarray(env.sim.data.body_xpos[bid], dtype=float).copy()
                        for _index, bid in elbow_indices
                    ]
                    seeds = [
                        np.clip(q[qidx], lo, hi),
                        np.clip(home[qidx], lo, hi),
                        0.5 * (lo + hi),
                    ]
                    rng = np.random.default_rng(fi)
                    for _ in range(4):
                        seeds.append(np.clip(q[qidx] + rng.normal(0.0, 0.65, len(qidx)), lo, hi))
                    # Branch-comparison renders deliberately explore remote
                    # basins instead of repeatedly rediscovering the local IK
                    # solution around the prior frame.
                    for _ in range(int(payload.get("branch_exploration_seeds", 0))):
                        seeds.append(rng.uniform(lo, hi))
                    candidates = []
                    for seed in seeds:
                        trial = q.copy()
                        residual = _tiago_ik_residual(
                            env, trial, qidx, bids, targets,
                            position_only=True, link_names=link_names,
                            direction_targets=direction_targets,
                            local_axes=payload.get("fingertip_local_axes") or {},
                            palm_targets=palm_targets,
                            palm_axes=payload.get("palm_local_axes") or {},
                            position_weight=float(payload.get("hand_position_weight", 1.0)),
                            elbow_position_weight=0.0,
                            direction_weight=float(payload.get("fingertip_direction_weight", 0.15)),
                            palm_weight=float(payload.get("palm_direction_weight", 0.15)),
                        )
                        solved = least_squares(
                            residual, seed, bounds=(lo, hi), max_nfev=100
                        ).x
                        trial[qidx] = solved
                        hand_error = float(np.linalg.norm(residual(solved)))
                        hand_position_error = 0.0
                        fingertip_direction_error = 0.0
                        for index, (name, bid) in enumerate(zip(link_names, bids)):
                            if name not in direction_targets:
                                continue
                            hand_position_error = max(
                                hand_position_error,
                                float(np.linalg.norm(np.asarray(env.sim.data.body_xpos[bid]) - targets[index, :3])),
                            )
                            local_axis = np.asarray(
                                (payload.get("fingertip_local_axes") or {})[name],
                                dtype=float,
                            )
                            current_direction = np.asarray(
                                env.sim.data.body_xmat[bid], dtype=float,
                            ).reshape(3, 3) @ local_axis
                            fingertip_direction_error = max(
                                fingertip_direction_error,
                                float(np.linalg.norm(current_direction - direction_targets[name])),
                            )
                        elbow_target_error = 0.0
                        elbow_step = 0.0
                        elbow_positions: dict[str, np.ndarray] = {}
                        for (index, bid), previous_elbow in zip(elbow_indices, previous_elbows):
                            current_elbow = np.asarray(env.sim.data.body_xpos[bid], dtype=float)
                            elbow_positions[link_names[index]] = current_elbow
                            elbow_target_error += float(np.linalg.norm(current_elbow - targets[index, :3]))
                            elbow_step = max(elbow_step, float(np.linalg.norm(current_elbow - previous_elbow)))
                        elbow_mirror_error = 0.0
                        if {"right_elbow", "left_elbow"}.issubset(elbow_positions):
                            mirrored_left = elbow_positions["left_elbow"].copy()
                            mirrored_left[1] *= -1.0
                            elbow_mirror_error = float(
                                np.linalg.norm(elbow_positions["right_elbow"] - mirrored_left)
                            )
                        elbow_height = float(np.mean([
                            position[2] for position in elbow_positions.values()
                        ]))
                        candidates.append({
                            "hand_error": hand_error,
                            "hand_position_error": hand_position_error,
                            "fingertip_direction_error": fingertip_direction_error,
                            "elbow_target_error": elbow_target_error,
                            "elbow_step": elbow_step,
                            "elbow_mirror_error": elbow_mirror_error,
                            "elbow_height": elbow_height,
                            "q": trial.copy(),
                        })

                    # The authored hand XYZ + fingertip direction are hard
                    # feasibility tests. The elbow threshold only selects among
                    # solutions that pass both tests; it never slows the hand.
                    position_tolerance = float(payload.get("hand_position_tolerance_m", 0.005))
                    direction_tolerance = float(payload.get("fingertip_direction_tolerance", 0.08))
                    primary = [
                        candidate for candidate in candidates
                        if candidate["hand_position_error"] <= position_tolerance
                        and candidate["fingertip_direction_error"] <= direction_tolerance
                    ]
                    if not primary:
                        primary = [min(candidates, key=lambda candidate: candidate["hand_error"])]
                        payload.setdefault("_ik_infeasible_frames", []).append({
                            "frame": fi,
                            "reason": "hand_or_fingertip",
                        })
                    if payload.get("elbow_selection_mode") == "previous":
                        # Ignore the authored elbow coordinates. With the hand
                        # and fingertip fixed, retain the most temporally
                        # continuous elbow among the feasible IK equilibria.
                        initial_variant = payload.get("initial_equilibrium_variant")
                        if fi == 1 and initial_variant == "llm_elbow":
                            selection_key = lambda candidate: (
                                candidate["elbow_target_error"],
                                candidate["hand_error"],
                            )
                        elif initial_variant == "elbow_up":
                            selection_key = lambda candidate: (
                                -candidate["elbow_height"],
                                candidate["elbow_step"],
                                candidate["hand_error"],
                            )
                        elif initial_variant == "elbow_down":
                            selection_key = lambda candidate: (
                                candidate["elbow_height"],
                                candidate["elbow_step"],
                                candidate["hand_error"],
                            )
                        else:
                            selection_key = (
                                (lambda candidate: (
                                    candidate["elbow_mirror_error"],
                                    candidate["elbow_step"],
                                    candidate["hand_error"],
                                ))
                                if symmetric_hands
                                else (lambda candidate: (
                                    candidate["elbow_step"],
                                    candidate["hand_error"],
                                ))
                            )
                        q = min(
                            primary,
                            key=selection_key,
                        )["q"]
                        frames.append(capture(q, authored_spheres()))
                        continue
                    max_elbow_step = float(payload.get("elbow_selection_tolerance_m", 0.08))
                    continuous = [
                        candidate for candidate in primary
                        if candidate["elbow_step"] <= max_elbow_step
                    ]
                    if not continuous:
                        payload.setdefault("_ik_infeasible_frames", []).append({
                            "frame": fi,
                            "reason": "elbow_continuity",
                        })
                        # Preserve the exact hand solution in the diagnostic
                        # render; the gallery labels this threshold infeasible.
                        continuous = primary
                    q = min(
                        continuous,
                        key=lambda candidate: (
                            candidate["elbow_target_error"],
                            candidate["hand_error"],
                        ),
                    )["q"]
                else:
                    progress = fi / max(1, len(rows) - 1)
                    # Wrist continuity fades to zero at the terminal target:
                    # intermediate poses stay calm while the final pose can fit.
                    wrist_weight = float(payload.get("wrist_continuity_weight", 0.0))
                    wrist_weight *= (1.0 - progress) ** 2
                    residual = _tiago_ik_residual(
                        env, q, qidx, bids, targets,
                        position_only=bool(payload.get("position_only")),
                        link_names=link_names, direction_targets=direction_targets,
                        local_axes=payload.get("fingertip_local_axes") or {},
                        palm_targets=palm_targets,
                        palm_axes=payload.get("palm_local_axes") or {},
                        position_weight=float(payload.get("hand_position_weight", 1.0)),
                        elbow_position_weight=float(payload.get("elbow_position_weight", 0.8)),
                        direction_weight=float(payload.get("fingertip_direction_weight", 0.15)),
                        palm_weight=float(payload.get("palm_direction_weight", 0.15)),
                        reference_q=q[qidx].copy(),
                        wrist_qpos_ids=(9, 10, 11, 21, 22, 23),
                        wrist_continuity_weight=wrist_weight,
                    )
                    q[qidx] = least_squares(
                        residual, np.clip(q[qidx], lo, hi), bounds=(lo, hi), max_nfev=100
                    ).x
            frames.append(capture(q, authored_spheres()))
    try:
        _write_gif(frames, out_gif, hz=hz)
    finally:
        try:
            env.close()
        except Exception:
            pass


RENDERERS = {
    "roomba": lambda payload, mode, gif, meta: render_roomba_traj(payload, mode, gif),
    "stretch": lambda payload, mode, gif, meta: render_stretch_traj(payload, mode, gif, meta),
    "humanoid": lambda payload, mode, gif, meta: render_humanoid_traj(payload, mode, gif),
    "single_arm": lambda payload, mode, gif, meta: render_single_arm_traj(payload, mode, gif),
    "tiago": lambda payload, mode, gif, meta: render_tiago_traj(payload, mode, gif),
}


def cmd_render(args: argparse.Namespace) -> None:
    out = Path(args.out_dir).expanduser().resolve()
    modes: list[Mode] = ["joint", "xyzrpy"] if args.mode == "both" else [args.mode]  # type: ignore
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or PLATFORMS
    for mode in modes:
        pred_path = out / f"predictions_{mode}.jsonl"
        if not pred_path.is_file():
            print(f"[render] missing {pred_path}", flush=True)
            continue
        rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        # dedupe keep last
        by = {}
        for r in rows:
            by[r["job_id"]] = r
        rows = list(by.values())
        allow = _allowed_idxs(args)
        ok = fail = skip = 0
        for r in rows:
            if r.get("error") or not r.get("parse_ok"):
                fail += 1
                continue
            plat = r["platform"]
            if plat not in plats:
                continue
            idx = int(r["idx"])
            if allow is not None and idx not in allow:
                continue
            cue = r["cue"]
            gif = _trajectory_gif(out, mode, plat, idx, cue, r.get("model") or args.model)
            if gif.is_file() and not args.force:
                skip += 1
                continue
            meta_path = out / "gt_extract" / plat / f"g{idx:02d}_{cue}.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            try:
                RENDERERS[plat](r["payload"], mode, gif, meta)
                ok += 1
                print(f"[render/{mode}] {plat} g{idx:02d} {cue}", flush=True)
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"[render/{mode}] FAIL {plat} g{idx:02d}: {exc}", flush=True)
        print(f"[render/{mode}] ok={ok} skip={skip} fail={fail}", flush=True)


def _discover_render_idxs(out: Path, plat: str) -> set[int]:
    """Cue indices that already have a joint or xyzrpy GIF under ``out``."""
    found: set[int] = set()
    for mode in ("joint", "xyzrpy"):
        folder = out / f"render_{mode}" / plat
        if not folder.is_dir():
            continue
        for path in folder.glob("emb_g*_*.gif"):
            m = re.match(r"emb_g(\d+)_", path.name)
            if m:
                found.add(int(m.group(1)))
    return found


def cmd_gallery(args: argparse.Namespace) -> None:
    out = Path(args.out_dir).expanduser().resolve()
    gt_render = _REPO / "data/results/render/embodiment_motion_code_gt"
    ours_render = (
        _REPO
        / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
    )
    eval_root = _REPO / "data/results/evaluation/embodiment_cue_id"

    def cue_labels(path: Path, section: str = "yesno") -> dict[tuple[str, int], str]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out_labels: dict[tuple[str, int], str] = {}
        for platform, block in (payload.get(section, {}).get("by_platform") or {}).items():
            wrong = {int(row["idx"]) for row in block.get("wrong_details") or []}
            for cue_idx in range(1, 31):
                out_labels[(platform, cue_idx)] = "no" if cue_idx in wrong else "yes"
        return out_labels

    gt_labels = cue_labels(eval_root / "cueid_human_yesno_acc_by_platform_gt_made.json")
    ours_labels = cue_labels(eval_root / "cueid_human_binary_acc_by_platform_gemini_made.json")
    baseline_eval = json.loads(
        (eval_root / "cueid_human_cue_identification_with_traj_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_labels = baseline_eval.get("cell_labels") or {}

    def baseline_label(platform: str, cue_idx: int, mode: str) -> str:
        raw = baseline_labels.get(platform, {}).get(f"g{cue_idx:02d}", {}).get(mode, "?")
        return {"o": "yes", "x": "no"}.get(raw, "na")
    sections = []
    gt = _index_gt()
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or PLATFORMS
    allow = _allowed_idxs(args)
    for plat in plats:
        if getattr(args, "exact_gallery", False) and allow is not None:
            idxs = set(allow)
        else:
            # Preserve the broad gallery for explicit standalone gallery runs.
            idxs = _discover_render_idxs(out, plat)
            if allow is not None:
                idxs |= set(allow)
        if not idxs:
            max_idx = args.first_n or 30
            idxs = set(range(1, max_idx + 1))
        for idx in sorted(idxs):
            cell = gt.get((idx, plat))
            if not cell:
                continue
            cue = cell["cue"]
            if plat == "single_arm":
                gtg = gt_render / "manipulator" / "gif" / "IIWA" / f"emb_g{idx:02d}_{cue}.gif"
                oursg = ours_render / "manipulator" / "gif" / "IIWA" / f"emb_g{idx:02d}_{cue}.gif"
            else:
                gtg = gt_render / plat / "gif" / f"emb_g{idx:02d}_{cue}.gif"
                oursg = ours_render / plat / "gif" / f"emb_g{idx:02d}_{cue}.gif"

            def media(path: Path, label: str, human_label: str) -> str:
                badge_text = {"yes": "YES", "no": "NO", "na": "?"}[human_label]
                if path.is_file():
                    rel = os.path.relpath(path, out)
                    bust = int(path.stat().st_mtime)
                    return (
                        f'<div class="col"><h3>{html_mod.escape(label)}</h3>'
                        f'<div class="media-box"><img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
                        f'<span class="human-badge {human_label}">{badge_text}</span></div></div>'
                    )
                return (
                    f'<div class="col"><h3>{html_mod.escape(label)}</h3>'
                    f'<div class="media-box missing-box"><p class="miss">missing</p>'
                    f'<span class="human-badge {human_label}">{badge_text}</span></div></div>'
                )

            def baseline_media(mode: Mode) -> str:
                folder = out / f"render_{mode}" / plat
                prefix = f"emb_g{idx:02d}_{cue}"
                cards = []
                named_models: set[str] = set()
                for path in sorted(folder.glob(f"{prefix}_*.gif")):
                    model = path.stem[len(prefix) + 1:]
                    named_models.add(model)
                    cards.append(media(path, f"Few-shot {mode} · {model}", baseline_label(plat, idx, mode)))
                legacy = folder / f"{prefix}.gif"
                # The original 275-success batch predates model-suffixed GIFs
                # and used gemini-2.5-pro. A newer named Pro result supersedes
                # that same (mode, robot, cue, model) cell rather than adding a
                # duplicate column.
                if legacy.is_file() and "gemini-2.5-pro" not in named_models:
                    cards.append(media(legacy, f"Few-shot {mode} · gemini-2.5-pro", baseline_label(plat, idx, mode)))
                if not cards:
                    cards.append(
                        media(
                            folder / f"{prefix}_gemini-2.5-pro.gif",
                            f"Few-shot {mode} · gemini-2.5-pro",
                            baseline_label(plat, idx, mode),
                        )
                    )
                return "".join(cards)

            sections.append(
                f'<section class="cue"><h2>{plat} · g{idx:02d} {html_mod.escape(cue)}</h2>'
                f'<div class="grid">{media(gtg, "GT Motion Code", gt_labels[(plat, idx)])}'
                f'{media(oursg, "Gemini-made Ours · gemini-2.5-pro", ours_labels[(plat, idx)])}'
                f'{baseline_media("joint")}{baseline_media("xyzrpy")}</div>'
                f"</section>"
            )
    html_path = out / "gallery_baseline_traj.html"
    html_path.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Baseline traj · joint vs xyzrpy</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}
.col h3{{font-size:.85rem;margin:0 0 6px}}
.media-box{{position:relative}}
.missing-box{{aspect-ratio:1;background:#090a0d;border-radius:8px;display:grid;place-items:center}}
.col img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}}
.human-badge{{position:absolute;right:8px;top:8px;padding:4px 8px;border-radius:999px;
font-size:.78rem;font-weight:800;line-height:1;color:#fff;box-shadow:0 1px 5px #000a}}
.human-badge.yes{{background:#188038}}
.human-badge.no{{background:#c5221f}}
.human-badge.na{{background:#6b7280}}
.miss{{color:#f28b82}}
@media (max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Baseline trajectories · few-shot joint vs xyzrpy</h1>
<p class="meta">n={len(sections)} cues · GT · Gemini-made Ours · model-specific few-shot joint / xyzrpy comparison</p>
{''.join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    print(f"gallery → {html_path} ({len(sections)} sections)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "cmd", nargs="?", default="live",
        choices=("live", "extract", "generate", "render", "gallery", "all"),
        help="live (default) generates, validates, saves, and renders each result.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "ours5")
    ap.add_argument(
        "--fewshot-shots", type=Path, default=DEFAULT_FEWSHOT_SHOTS,
        help="robot > cue > motion_code/xyzrpy/joint_angles YAML for both baselines",
    )
    ap.add_argument("--platforms", "--robots", dest="platforms", default=",".join(PLATFORMS))
    ap.add_argument(
        "--mode", "--baseline", dest="mode",
        choices=("joint", "xyzrpy", "both"), default="both",
    )
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--render-workers", type=int, default=1)
    ap.add_argument("--render-as-you-go", action="store_true")
    ap.add_argument("--exact-gallery", action="store_true", help="Show only --only cue indices in HTML.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--first-n", type=int, default=0, help="Only cue indices 1..N.")
    ap.add_argument(
        "--only-idx", "--only", dest="only_idx",
        default="",
        help="Cue indices/ranges (e.g. 1-10 or 1,4,8-10). Overrides --first-n.",
    )
    args = ap.parse_args()
    aliases = {
        "s": "single_arm", "single": "single_arm", "single-arm": "single_arm",
        "h": "humanoid", "t": "tiago", "r": "roomba", "st": "stretch",
    }
    requested = [p.strip() for p in args.platforms.split(",") if p.strip()]
    args.platforms = ",".join(aliases.get(p, p) for p in requested)
    unknown = sorted(set(args.platforms.split(",")) - set(PLATFORMS))
    if unknown:
        ap.error(f"unknown robot alias/name: {', '.join(unknown)}")
    if args.cmd == "live":
        args.force = True
        args.render_as_you_go = True
        args.exact_gallery = True
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "live":
        cmd_generate(args)
        cmd_gallery(args)
    if args.cmd in ("extract", "all"):
        cmd_extract(args)
    if args.cmd in ("generate", "all"):
        cmd_generate(args)
    if args.cmd in ("render", "all"):
        cmd_render(args)
    if args.cmd in ("gallery", "all"):
        cmd_gallery(args)


if __name__ == "__main__":
    main()
