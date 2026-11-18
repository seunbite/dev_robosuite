#!/usr/bin/env python3
"""Humanoid GT vs LLM: continuous location/direction + binary movement."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
GT_PATH = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
PRED_PATH = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
)
OUT_JSON = REPO / "data/results/evaluation/humanoid_gt_vs_gemini_loc_dir_move.json"
OUT_MD = REPO / "papers/icra2027_motion_abstraction/humanoid_gt_vs_gemini_loc_dir_move.md"

# Right-arm reach direction in a torso-relative frame (x forward, y left, z up).
REACH_VEC = {
    "front": np.array([1.0, 0.0, 0.0]),
    "back": np.array([-1.0, 0.0, 0.0]),
    "in": np.array([0.0, 1.0, 0.0]),
    "out": np.array([0.0, -1.0, 0.0]),
    "up": np.array([0.0, 0.0, 1.0]),
    "down": np.array([0.0, 0.0, -1.0]),
}

# Palm-pad facing (world), matching embodiment_renderers._G1_SEMANTIC_TO_VISUAL_VEC.
PALM_VEC = {
    "palmfront": np.array([1.0, 0.0, 0.0]),
    "palmback": np.array([-1.0, 0.0, 0.0]),
    "palmin": np.array([0.0, 1.0, 0.0]),
    "palmout": np.array([0.0, -1.0, 0.0]),
    "palmup": np.array([0.0, 0.0, 1.0]),
    "palmdown": np.array([0.0, 0.0, -1.0]),
}

# Canonical palm when only horiz/vert is given (dir|orient defaults).
DEFAULT_PALM = {
    ("front", "vertical"): "palmin",
    ("front", "horizontal"): "palmdown",
    ("back", "vertical"): "palmout",
    ("back", "horizontal"): "palmup",
    ("out", "vertical"): "palmfront",
    ("out", "horizontal"): "palmdown",
    ("in", "vertical"): "palmback",
    ("in", "horizontal"): "palmup",
    ("up", "vertical"): "palmin",
    ("up", "horizontal"): "palmfront",
    ("down", "vertical"): "palmout",
    ("down", "horizontal"): "palmback",
}


def _unwrap_pose(step: dict[str, Any]) -> dict[str, Any]:
    p = step.get("parameters") or {}
    if isinstance(p.get("pose"), dict):
        return p["pose"]
    return p


def _first_pose(row: dict[str, Any]) -> dict[str, Any] | None:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return _unwrap_pose(step)
    return None


def _parse_grip(raw: Any) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None
    parts = [t.strip().lower() for t in str(raw).replace(" ", "").split("|") if t.strip()]
    jaw = None
    palm = None
    for t in parts:
        if t in ("vertical", "vert"):
            jaw = "vertical"
        elif t in ("horizontal", "horiz"):
            jaw = "horizontal"
        elif t in PALM_VEC:
            palm = t
    return jaw, palm


def _xyz(pose: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k in ("x", "y", "z"):
        v = pose.get(k)
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


_G1_BANK_SELECT = None  # lazy: embodiment_renderers._g1_select_feasible_pose
_TIAGO_ENV = None  # lazy robosuite Tiago env for pose-bank resolve
_TIAGO_ENV_FAILED = False


def _g1_bank_select():
    """Lazy import so scoring still works if MuJoCo deps are unavailable."""
    global _G1_BANK_SELECT
    if _G1_BANK_SELECT is False:
        return None
    if _G1_BANK_SELECT is not None:
        return _G1_BANK_SELECT
    try:
        arm = REPO / "adhoc/generation/robotarm"
        if str(arm) not in sys.path:
            sys.path.insert(0, str(arm))
        from embodiment_renderers import _g1_select_feasible_pose  # type: ignore

        _G1_BANK_SELECT = _g1_select_feasible_pose
        return _G1_BANK_SELECT
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] G1 bank resolve unavailable: {exc}", flush=True)
        _G1_BANK_SELECT = False
        return None


def _tiago_env():
    """Lazy Tiago env for pose-bank candidate listing (matches GIF resolve)."""
    global _TIAGO_ENV, _TIAGO_ENV_FAILED
    if _TIAGO_ENV_FAILED:
        return None
    if _TIAGO_ENV is not None:
        return _TIAGO_ENV
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from adhoc.generation.tiago.legacy.render_mobile_manip_schema import (  # type: ignore
            _make_env,
        )

        _TIAGO_ENV = _make_env()
        return _TIAGO_ENV
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Tiago env unavailable for bank resolve: {exc}", flush=True)
        _TIAGO_ENV_FAILED = True
        return None


def _tiago_bank_select(pose: dict[str, Any]) -> dict[str, Any] | None:
    """Top-1 Tiago pose-bank entry for ``dir|grip`` (+ xyz rank), same as renderer."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    try:
        from adhoc.generation.tiago.legacy.render_mobile_config import (  # type: ignore
            list_pose_bank_candidates,
        )

        # The renderer ranks its curated plausible-pose metadata before touching
        # the kinematic fallback bank.  That first stage is entirely static, so
        # use it directly: this keeps scoring reproducible in headless shells
        # where constructing a MuJoCo off-screen context may fail.
        result = list_pose_bank_candidates(
            None,
            pose,
            top_k=1,
            plausible_only=True,
            ignore_xyz=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Tiago bank select failed: {exc}", flush=True)
        result = {}
    entries = result.get("entries") or []
    if entries:
        return entries[0]

    # Only uncommon poses absent from the curated plausible set need FK and a
    # MuJoCo environment.  Preserve the renderer's original fallback behavior.
    env = _tiago_env()
    if env is None:
        return None
    try:
        result = list_pose_bank_candidates(env, pose, top_k=1, ignore_xyz=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Tiago bank select failed: {exc}", flush=True)
        return None
    entries = result.get("entries") or []
    return entries[0] if entries else None


def _effective_xyz(pose: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Body-% xyz as the G1 renderer would realize them.

    Explicit Motion Code axes win. Unspecified axes are filled from the
    feasible-bank pose the renderer selects for ``dir|grip|elbow`` (when xyz
    is omitted the EE stays at that bank pose — see ``_g1_apply_ee_xyz_world``).
    ``bank=F`` / no bank hit → only explicit axes (no half-range penalty).
    """
    explicit = _xyz(pose)
    meta: dict[str, Any] = {
        "explicit": dict(explicit),
        "filled_axes": [],
        "source": "explicit",
        "pose_id": None,
    }
    if pose.get("use_feasible_bank") is False:
        meta["source"] = "bank_off"
        return explicit, meta
    select = _g1_bank_select()
    if select is None:
        meta["source"] = "bank_unavailable"
        return explicit, meta
    try:
        selected = select(pose)
    except Exception as exc:  # noqa: BLE001
        meta["source"] = f"bank_error:{exc}"
        return explicit, meta
    if not selected:
        meta["source"] = "no_bank_hit"
        return explicit, meta
    out = dict(explicit)
    for ax in ("x", "y", "z"):
        if ax not in out and isinstance(selected.get(ax), (int, float)):
            out[ax] = float(selected[ax])
            meta["filled_axes"].append(ax)
    meta["source"] = "bank_fill" if meta["filled_axes"] else "explicit"
    meta["pose_id"] = selected.get("pose_id")
    return out, meta


def _tiago_effective_xyz(pose: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """Body-% xyz as the Tiago renderer realizes them.

    Explicit Motion Code axes win (trim mode snaps EE to those %). Unspecified
    axes keep the selected pose-bank EE % — the coordinates that actually appear
    in the GIF — rather than an artificial miss penalty.
    """
    explicit = _xyz(pose)
    meta: dict[str, Any] = {
        "explicit": dict(explicit),
        "filled_axes": [],
        "source": "explicit",
        "tile_idx": None,
        "bank_source": None,
    }
    selected = _tiago_bank_select(pose)
    if not selected:
        meta["source"] = "no_bank_hit"
        return explicit, meta
    out = dict(explicit)
    for ax in ("x", "y", "z"):
        if ax not in out and isinstance(selected.get(ax), (int, float)):
            out[ax] = float(selected[ax])
            meta["filled_axes"].append(ax)
    meta["source"] = "bank_fill" if meta["filled_axes"] else "explicit"
    meta["tile_idx"] = selected.get("tile_idx")
    meta["bank_source"] = selected.get("source")
    return out, meta


def _reach(pose: dict[str, Any]) -> np.ndarray | None:
    d = str(pose.get("arm_position") or "still").lower()
    if d in ("", "still"):
        return None
    return REACH_VEC.get(d)


def _palm_normal(pose: dict[str, Any]) -> np.ndarray | None:
    d = str(pose.get("arm_position") or "still").lower()
    jaw, palm = _parse_grip(pose.get("gripper_orientation"))
    if palm is None and d in REACH_VEC and jaw in ("vertical", "horizontal"):
        palm = DEFAULT_PALM.get((d, jaw))
    if palm is None:
        return None
    return PALM_VEC[palm].copy()


def _rot_from_reach_palm(reach: np.ndarray, palm: np.ndarray) -> np.ndarray:
    """Columns: x=reach×palm, y=palm, z=reach (right-handed-ish)."""
    z = reach / (np.linalg.norm(reach) + 1e-12)
    y = palm - z * float(np.dot(palm, z))
    yn = np.linalg.norm(y)
    if yn < 1e-8:
        # Palm parallel to reach: pick an arbitrary perpendicular.
        tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        y = tmp - z * float(np.dot(tmp, z))
        yn = np.linalg.norm(y)
    y = y / (yn + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    return np.column_stack([x, y, z])


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX yaw-pitch-roll in degrees."""
    sy = -R[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    pitch = math.asin(sy)
    if abs(sy) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[0, 1], R[1, 1])
        yaw = 0.0
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    cos_th = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cos_th))


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _sign_from_endpoints(a: float, b: float, *, eps: float = 1e-9) -> str | None:
    """Map a degree/distance pair to ``+`` / ``-`` / ``+-`` (None if both ~0)."""
    pos = a > eps or b > eps
    neg = a < -eps or b < -eps
    if pos and neg:
        return "+-"
    if pos:
        return "+"
    if neg:
        return "-"
    return None


def _sign_from_degrees(deg: Any) -> str | None:
    if isinstance(deg, (list, tuple)) and len(deg) >= 2:
        return _sign_from_endpoints(float(deg[0]), float(deg[-1]))
    if isinstance(deg, (int, float)):
        # one-way scalar: treat as 0 → value
        return _sign_from_endpoints(0.0, float(deg))
    return None


def _sign_from_distance(dist: Any) -> str | None:
    if isinstance(dist, (list, tuple)) and len(dist) >= 2:
        return _sign_from_endpoints(float(dist[0]), float(dist[-1]))
    if isinstance(dist, (int, float)):
        return _sign_from_endpoints(0.0, float(dist))
    return None


def _norm_axis(raw: Any) -> str | None:
    if raw is None:
        return None
    a = str(raw).strip().lower()
    return a or None


def _move_keys(row: dict[str, Any]) -> set[tuple]:
    """Movement identity for scoring.

    - Joint ``movement`` and ``path(line, …)`` → ``("ax", axis, sign)`` with
      ``sign ∈ {+, -, +-}`` from degrees/distance endpoints. Line paths are
      interchangeable with joint moves that share the same axis+sign.
    - ``path(arc, …)`` stays path-like: ``("arc", plane)``.
    - Torso height levels (no degrees) → ``("height", level)``.
    """
    keys: set[tuple] = set()
    for step in row.get("movements") or []:
        t = step.get("type")
        p = step.get("parameters") or {}
        if t == "movement":
            mov = p.get("movement") if isinstance(p.get("movement"), dict) else p
            for j in mov.get("joints") or []:
                axis = _norm_axis(j.get("axis"))
                if axis == "height" and not isinstance(j.get("degrees"), (list, tuple, int, float)):
                    level = j.get("height")
                    if level is not None:
                        keys.add(("height", str(level).lower()))
                    continue
                if axis is None:
                    continue
                sign = _sign_from_degrees(j.get("degrees"))
                if sign is None:
                    continue
                keys.add(("ax", axis, sign))
        elif t == "path":
            path = p.get("path") if isinstance(p.get("path"), dict) else p
            shape = str(path.get("shape") or "").strip().lower()
            if shape == "line":
                axis = _norm_axis(path.get("axis"))
                sign = _sign_from_distance(path.get("distance"))
                if axis is not None and sign is not None:
                    keys.add(("ax", axis, sign))
            elif shape == "arc":
                plane = _norm_axis(path.get("plane") or path.get("axis"))
                if plane is not None:
                    keys.add(("arc", plane))
            else:
                # Unknown path shapes: keep coarse identity.
                tag = _norm_axis(path.get("axis") or path.get("plane") or path.get("mode"))
                if shape:
                    keys.add(("p", shape, tag))
    return _collapse_bidirectional(keys)


def _collapse_bidirectional(keys: set[tuple]) -> set[tuple]:
    """Merge same-axis ``+`` and ``-`` into ``+-`` (oscillation / scan).

    So GT ``(pan, +-)`` matches LLM ``(pan, +)`` + ``(pan, -)``, and GT that
    lists both one-way strokes collapses to one bidirectional key.
    """
    out = set(keys)
    signs_by_axis: dict[str, set[str]] = {}
    for k in keys:
        if len(k) == 3 and k[0] == "ax":
            signs_by_axis.setdefault(str(k[1]), set()).add(str(k[2]))
    for axis, signs in signs_by_axis.items():
        if ("+" in signs and "-" in signs) or "+-" in signs:
            out.discard(("ax", axis, "+"))
            out.discard(("ax", axis, "-"))
            out.add(("ax", axis, "+-"))
    return out


def _move_covers(gt_keys: set[tuple], llm_keys: set[tuple]) -> bool:
    """GT keys covered by LLM. One-way GT may be satisfied by LLM ``+-``."""
    gt = _collapse_bidirectional(gt_keys)
    llm = _collapse_bidirectional(llm_keys)
    for k in gt:
        if k in llm:
            continue
        if (
            len(k) == 3
            and k[0] == "ax"
            and k[2] in ("+", "-")
            and ("ax", k[1], "+-") in llm
        ):
            continue
        return False
    return True


def _plat(allj: list, cue: str, platform: str = "humanoid") -> dict[str, Any]:
    for e in allj:
        if e.get("cue") == cue:
            return (e.get("platforms") or {}).get(platform) or {}
    return {}


def score_pair(
    gt_row: dict[str, Any],
    llm_row: dict[str, Any],
    idx: int,
    cue: str,
    *,
    platform: str = "humanoid",
) -> dict[str, Any]:
    gp = _first_pose(gt_row) or {}
    lp = _first_pose(llm_row) or {}
    gxyz_raw, lxyz_raw = _xyz(gp), _xyz(lp)
    if platform == "humanoid":
        gxyz, gmeta = _effective_xyz(gp)
        lxyz, lmeta = _effective_xyz(lp)
    elif platform == "tiago":
        # Fill omitted axes from the Tiago pose-bank EE the renderer keeps.
        gxyz, gmeta = _tiago_effective_xyz(gp)
        lxyz, lmeta = _tiago_effective_xyz(lp)
    else:
        gxyz, gmeta = gxyz_raw, {"source": "explicit", "explicit": dict(gxyz_raw)}
        lxyz, lmeta = lxyz_raw, {"source": "explicit", "explicit": dict(lxyz_raw)}

    # --- Location (continuous, body-relative % units) ---
    # Compare render-effective xyz (explicit Motion Code + bank-filled axes).
    loc: dict[str, Any] = {
        "gt": gxyz_raw,
        "llm": lxyz_raw,
        "gt_effective": gxyz,
        "llm_effective": lxyz,
        "gt_resolve": gmeta,
        "llm_resolve": lmeta,
        "applicable": bool(gxyz),
    }
    if gxyz:
        deltas: dict[str, float] = {}
        missing: list[str] = []
        for ax, gv in gxyz.items():
            if ax in lxyz:
                deltas[ax] = abs(float(lxyz[ax]) - float(gv))
            else:
                missing.append(ax)
        loc["delta"] = {k: deltas.get(k) for k in ("x", "y", "z") if k in gxyz}
        loc["delta_with_penalty"] = dict(deltas)
        loc["missing_axes"] = missing
        vals = [v for v in deltas.values() if isinstance(v, (int, float))]
        if vals:
            loc["l1"] = float(sum(vals))
            loc["l2"] = float(math.sqrt(sum(v * v for v in vals)))
            loc["mean_abs"] = float(sum(vals) / len(vals))
            loc["max_abs"] = float(max(vals))
        else:
            loc["l1"] = loc["l2"] = loc["mean_abs"] = loc["max_abs"] = None
        loc["used_missing_penalty"] = False
        loc["miss_penalty"] = None
        loc["n_axes_scored"] = len(vals)

    # --- Direction (continuous, reach + palm → RPY / geodesic) ---
    gr, lr = _reach(gp), _reach(lp)
    gpalm, lpalm = _palm_normal(gp), _palm_normal(lp)
    direc: dict[str, Any] = {
        "applicable": gr is not None or gpalm is not None,
        "gt_dir": gp.get("arm_position"),
        "llm_dir": lp.get("arm_position"),
        "gt_grip": gp.get("gripper_orientation"),
        "llm_grip": lp.get("gripper_orientation"),
    }
    if gr is not None and lr is not None:
        direc["reach_angle_deg"] = _angle_between(gr, lr)
        direc["reach_ok"] = bool(direc["reach_angle_deg"] <= 1.0)
    elif gr is not None:
        direc["reach_angle_deg"] = None
        direc["reach_ok"] = False

    if gpalm is not None and lpalm is not None:
        ang = _angle_between(gpalm, lpalm)
        direc["palm_angle_deg"] = ang
        # Tiago parallel grippers: palm-normal flip (0° vs 180°) is visually
        # identical, so both count as a match.
        if platform == "tiago":
            direc["palm_ok"] = bool(ang <= 1.0 or abs(ang - 180.0) <= 1.0)
        else:
            direc["palm_ok"] = bool(ang <= 1.0)
    elif gpalm is not None:
        direc["palm_angle_deg"] = None
        direc["palm_ok"] = False

    if gr is not None and gpalm is not None and lr is not None and lpalm is not None:
        Ra = _rot_from_reach_palm(gr, gpalm)
        Rb = _rot_from_reach_palm(lr, lpalm)
        rpy_a, rpy_b = _rpy(Ra), _rpy(Rb)
        drpy = tuple(abs(_wrap180(b - a)) for a, b in zip(rpy_a, rpy_b))
        direc["gt_rpy_deg"] = rpy_a
        direc["llm_rpy_deg"] = rpy_b
        direc["delta_rpy_deg"] = {"roll": drpy[0], "pitch": drpy[1], "yaw": drpy[2]}
        direc["geodesic_deg"] = _geodesic_deg(Ra, Rb)
    else:
        direc["geodesic_deg"] = None
        direc["delta_rpy_deg"] = None

    # --- Movement (binary) ---
    gk, lk = _move_keys(gt_row), _move_keys(llm_row)
    move: dict[str, Any] = {"applicable": bool(gk)}
    if gk:
        move["ok"] = _move_covers(gk, lk)
        move["partial"] = bool(gk & lk) and not move["ok"]
        move["gt_keys"] = sorted(list(gk))
        move["llm_keys"] = sorted(list(lk))

    return {
        "idx": idx,
        "cue": cue,
        "location": loc,
        "direction": direc,
        "movement": move,
        "gt_code": (gt_row.get("source_code") or "").replace("\n", " "),
        "llm_code": (llm_row.get("source_code") or "").replace("\n", " "),
    }


def _mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred",
        type=Path,
        default=PRED_PATH,
        help="LLM master JSON (embodiment_motion_code_all.json)",
    )
    ap.add_argument("--gt", type=Path, default=GT_PATH)
    ap.add_argument(
        "--platform",
        type=str,
        default="humanoid",
        help="Platform key inside platforms{} (humanoid|tiago|…)",
    )
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    pred_path = args.pred
    platform = str(args.platform).strip().lower()
    out_json = args.out_json
    out_md = args.out_md
    if out_json is None:
        tag = pred_path.parent.name.replace("embodiment_motion_code_", "")
        if pred_path == PRED_PATH or "gemini" in tag:
            out_json = OUT_JSON
            out_md = out_md or OUT_MD
        else:
            out_json = REPO / "data/results/evaluation" / f"{platform}_gt_vs_{tag}_loc_dir_move.json"
            out_md = out_md or (
                REPO
                / "papers/icra2027_motion_abstraction"
                / f"{platform}_gt_vs_{tag}_loc_dir_move.md"
            )
    if out_md is None:
        out_md = OUT_MD

    gt_all = json.loads(args.gt.read_text())
    pred_all = json.loads(pred_path.read_text())
    order = [
        (int(e["idx"]), str(e["cue"]))
        for e in gt_all
        if platform in (e.get("platforms") or {})
    ]
    rows = []
    for idx, cue in order:
        rows.append(
            score_pair(
                _plat(gt_all, cue, platform),
                _plat(pred_all, cue, platform),
                idx,
                cue,
                platform=platform,
            )
        )

    loc_l2 = [r["location"]["l2"] for r in rows if r["location"].get("l2") is not None]
    loc_mean = [r["location"]["mean_abs"] for r in rows if r["location"].get("mean_abs") is not None]
    reach_ang = [
        r["direction"]["reach_angle_deg"]
        for r in rows
        if isinstance(r["direction"].get("reach_angle_deg"), (int, float))
    ]
    palm_ang = [
        r["direction"]["palm_angle_deg"]
        for r in rows
        if isinstance(r["direction"].get("palm_angle_deg"), (int, float))
    ]
    geo = [
        r["direction"]["geodesic_deg"]
        for r in rows
        if isinstance(r["direction"].get("geodesic_deg"), (int, float))
    ]
    move_app = [r for r in rows if r["movement"].get("applicable")]
    move_ok = sum(1 for r in move_app if r["movement"].get("ok"))

    ORI_OK = 60.0
    DIR_OK = 1.0  # discrete reach/palm dirs → 0° / 90° / 180°
    LOC_OK = 5.0  # legacy binary threshold (kept in JSON; table uses MAE)

    def _axis_mae(axis: str) -> dict[str, Any]:
        """Mean |GT−pred| body-% on render-effective coordinates."""
        vals: list[float] = []
        for r in rows:
            loc = r["location"]
            gt_eff = loc.get("gt_effective") or loc.get("gt") or {}
            if axis not in gt_eff:
                continue
            delt = loc.get("delta") or {}
            if axis in delt and isinstance(delt[axis], (int, float)):
                vals.append(float(delt[axis]))
        return {
            "n": len(vals),
            "mean_abs": _mean(vals),
            "rate": (
                (sum(1 for v in vals if v <= LOC_OK) / len(vals) * 100.0) if vals else None
            ),
            "loc_ok": LOC_OK,
        }

    def _dir_rate(kind: str) -> float | None:
        # Only cues where both sides yield an angle (same denominator as before).
        oks: list[bool] = []
        for r in rows:
            d = r["direction"]
            if kind == "reach":
                if not isinstance(d.get("reach_angle_deg"), (int, float)):
                    continue
                oks.append(bool(d.get("reach_ok", d["reach_angle_deg"] <= DIR_OK)))
            elif kind == "palm":
                if not isinstance(d.get("palm_angle_deg"), (int, float)):
                    continue
                if "palm_ok" in d:
                    oks.append(bool(d["palm_ok"]))
                else:
                    a = float(d["palm_angle_deg"])
                    if platform == "tiago":
                        oks.append(a <= DIR_OK or abs(a - 180.0) <= DIR_OK)
                    else:
                        oks.append(a <= DIR_OK)
        if not oks:
            return None
        return sum(oks) / len(oks) * 100.0

    ori_ok = [g <= ORI_OK for g in geo]
    axis_x, axis_y, axis_z = _axis_mae("x"), _axis_mae("y"), _axis_mae("z")
    reach_rate = _dir_rate("reach")
    palm_rate = _dir_rate("palm")
    table2 = {
        "reach": reach_rate,
        "palm": palm_rate,
        "orientation": (sum(ori_ok) / len(ori_ok) * 100.0) if ori_ok else None,
        # Location columns: continuous mean |Δ| (lower better), not binary rates.
        "locx": axis_x["mean_abs"],
        "locy": axis_y["mean_abs"],
        "locz": axis_z["mean_abs"],
        "locx_rate": axis_x["rate"],
        "locy_rate": axis_y["rate"],
        "locz_rate": axis_z["rate"],
        "movement": (move_ok / len(move_app) * 100.0) if move_app else None,
    }

    summary = {
        "n": len(rows),
        "pred": str(pred_path),
        "location": {
            "n_with_xyz": len(loc_l2),
            "mean_l2": _mean(loc_l2),
            "mean_axis_abs": _mean(loc_mean),
            "median_l2": float(np.median(loc_l2)) if loc_l2 else None,
            "locx": axis_x,
            "locy": axis_y,
            "locz": axis_z,
        },
        "direction": {
            "n_reach": len(reach_ang),
            "mean_reach_angle_deg": _mean(reach_ang),
            "rate_reach": reach_rate,
            "n_palm": len(palm_ang),
            "mean_palm_angle_deg": _mean(palm_ang),
            "rate_palm": palm_rate,
            "dir_ok_deg": DIR_OK,
            "n_rpy": len(geo),
            "mean_geodesic_deg": _mean(geo),
            "median_geodesic_deg": float(np.median(geo)) if geo else None,
            "rate_le_60": table2["orientation"],
        },
        "movement": {
            "n": len(move_app),
            "ok": move_ok,
            "acc": (move_ok / len(move_app)) if move_app else None,
            "rate": table2["movement"],
        },
        "table2_rates_pct": table2,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    def _row(metric: str, value: str) -> str:
        return f"| {metric} | {value} |"

    loc_s = summary["location"]
    dir_s = summary["direction"]
    mov_s = summary["movement"]
    lines = [
        f"# {platform} GT vs LLM — continuous location/direction, binary movement",
        "",
        f"Pred: `{pred_path}`",
        f"Platform: `{platform}` · N = {summary['n']} cues.",
        "",
        "## Table metrics (loc = mean |Δ| body-% ↓; rch/plm/mov = % ↑)",
        "",
        "| Reach↑ | Palm↑ | Locx↓ | Locy↓ | Locz↓ | Movement↑ |",
        "|---:|---:|---:|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{table2[k]:.1f}" if table2[k] is not None else "n/a"
            for k in ("reach", "palm", "locx", "locy", "locz", "movement")
        )
        + " |",
        "",
        "_Tiago Palm: 0° and 180° both count as match._",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        _row("Location cues with GT xyz", str(loc_s["n_with_xyz"])),
        _row(
            "Location mean L2 (body %)",
            f"{loc_s['mean_l2']:.1f}" if loc_s["mean_l2"] is not None else "n/a",
        ),
        _row(
            "Location median L2",
            f"{loc_s['median_l2']:.1f}" if loc_s["median_l2"] is not None else "n/a",
        ),
        _row(
            "Location mean per-axis |Δ|",
            f"{loc_s['mean_axis_abs']:.1f}" if loc_s["mean_axis_abs"] is not None else "n/a",
        ),
        _row(
            "Direction mean reach ∠ (°)",
            (
                f"{dir_s['mean_reach_angle_deg']:.1f} (n={dir_s['n_reach']})"
                if dir_s["mean_reach_angle_deg"] is not None
                else "n/a"
            ),
        ),
        _row(
            "Direction mean palm ∠ (°)",
            (
                f"{dir_s['mean_palm_angle_deg']:.1f} (n={dir_s['n_palm']})"
                if dir_s["mean_palm_angle_deg"] is not None
                else "n/a"
            ),
        ),
        _row(
            "Direction mean RPY geodesic (°)",
            (
                f"{dir_s['mean_geodesic_deg']:.1f} (n={dir_s['n_rpy']})"
                if dir_s["mean_geodesic_deg"] is not None
                else "n/a"
            ),
        ),
        _row(
            "Movement binary acc",
            (
                f"{move_ok}/{len(move_app)} ({100 * mov_s['acc']:.0f}%)"
                if mov_s["acc"] is not None
                else "n/a"
            ),
        ),
        "",
        "## Per cue",
        "",
        "| # | Cue | Δx | Δy | Δz | L2 | reach° | palm° | geo° | Δr,Δp,Δy | move |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        loc = r["location"]
        d = r["direction"]
        m = r["movement"]
        delt = loc.get("delta") or {}

        def fmt(v, nd=1):
            if v is None:
                return "—"
            if isinstance(v, float) and (math.isnan(v)):
                return "—"
            return f"{v:.{nd}f}"

        drpy = d.get("delta_rpy_deg")
        rpy_s = (
            f"{drpy['roll']:.0f},{drpy['pitch']:.0f},{drpy['yaw']:.0f}"
            if isinstance(drpy, dict)
            else "—"
        )
        if not m.get("applicable"):
            mov_cell = "n/a"
        else:
            mov_cell = "✓" if m.get("ok") else "✗"
        lines.append(
            f"| {r['idx']} | {r['cue']} | "
            f"{fmt(delt.get('x'))} | {fmt(delt.get('y'))} | {fmt(delt.get('z'))} | "
            f"{fmt(loc.get('l2'))} | {fmt(d.get('reach_angle_deg'), 0)} | "
            f"{fmt(d.get('palm_angle_deg'), 0)} | {fmt(d.get('geodesic_deg'), 0)} | "
            f"{rpy_s} | {mov_cell} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Location units are body-relative Motion Code percentages (0–100).",
            "- Unspecified LLM xyz axes are filled from the G1 feasible-bank pose "
            "the renderer selects for dir|grip|elbow (render-effective EE %; no 50 penalty).",
            f"- Location match: |Δaxis| ≤ {LOC_OK}.",
            "- Direction: `dir` → reach unit vector; `grip` palm token → palm normal; "
            "RPY from the reach–palm frame (geodesic = rotation angle between GT and LLM frames). "
            "Reach/palm angles are acos ∈ [0, 180°].",
            "- Movement binary: GT (axis, sign) set ⊆ LLM set; "
            "line paths share the same (axis, sign) keys as joint moves; "
            "arcs stay as (arc, plane).",
            "",
        ]
    )
    out_md.write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))
    print("wrote", out_json)
    print("wrote", out_md)


if __name__ == "__main__":
    main()
