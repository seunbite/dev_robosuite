from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .commands import FootTargetMap, JointMap, VelocityCommand
from .config import LEG_ORDER
from .controllers.high_level import HighLevelController
from .kinematics import QuadrupedKinematics, lerp_vec

_ARC_BASE_WZ = 1.0     # rad/s at speed=1.0
_ARC_FORWARD_VX = 0.3   # m/s forward per unit speed during arc
_LINE_BASE_VX = 0.5     # m/s at speed=1.0


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    """Normalize a config step so that ``duration`` is always present.

    * ``arc``  path with ``degrees`` + ``speed`` → computes ``duration`` and
      ``velocity``  (``wz`` from degrees, ``vx`` proportional to speed).
    * ``line`` path with ``distance`` + ``speed`` → computes ``duration`` and
      ``velocity``  (``vx`` from distance).
    * ``pose_to_pose`` and steps that already carry ``duration`` are returned
      unchanged (shallow-copied).
    """
    step = dict(step)
    if step.get("type") != "path":
        return step
    params = step.get("parameters", {})
    path_p = params.get("path", {})
    shape = path_p.get("shape", "")

    if shape == "arc" and "degrees" in path_p and "duration" not in step:
        deg = float(path_p["degrees"])
        spd = float(path_p.get("speed", 1.0))
        rad = deg * math.pi / 180.0
        wz = math.copysign(spd * _ARC_BASE_WZ, rad)
        # Use 3x theoretical duration as upper bound; the renderer will
        # detect heading completion and break early.
        dur = 3.0 * abs(rad) / max(abs(wz), 1e-6)
        vx = spd * _ARC_FORWARD_VX
        step["duration"] = round(dur, 3)
        path_p = dict(path_p)
        path_p["velocity"] = {"vx": round(vx, 4), "vy": 0.0, "wz": round(wz, 4)}
        step["parameters"] = {**params, "path": path_p}

    elif shape == "line" and "distance" in path_p and "duration" not in step:
        dist = float(path_p["distance"])
        spd = float(path_p.get("speed", 1.0))
        vx = math.copysign(spd * _LINE_BASE_VX, dist)
        # Use 3x theoretical duration; renderer detects distance completion.
        dur = 3.0 * abs(dist) / max(abs(vx), 1e-6)
        step["duration"] = round(dur, 3)
        path_p = dict(path_p)
        path_p["velocity"] = {"vx": round(vx, 4), "vy": 0.0, "wz": 0.0}
        step["parameters"] = {**params, "path": path_p}

    step.setdefault("duration", 1.5)
    return step


def path_step_target(step: dict[str, Any]) -> dict[str, float] | None:
    """Return the completion target for a path step, or None.

    For ``arc``  → ``{"heading_rad": <target>}``  (accumulated ``|wz * t|``)
    For ``line`` → ``{"distance_m": <target>}``    (accumulated ``|vx * t|``)
    """
    if step.get("type") != "path":
        return None
    path_p = step.get("parameters", {}).get("path", {})
    shape = path_p.get("shape", "")
    if shape == "arc" and "degrees" in path_p:
        return {"heading_rad": abs(float(path_p["degrees"])) * math.pi / 180.0}
    if shape == "line" and "distance" in path_p:
        return {"distance_m": abs(float(path_p["distance"]))}
    return None


def path_target_reached(target: dict[str, float], velocity: dict, local_t: float) -> bool:
    """Check if the accumulated motion has reached the target."""
    if "heading_rad" in target:
        wz = abs(float(velocity.get("wz", 0)))
        return wz * local_t >= target["heading_rad"]
    if "distance_m" in target:
        vx = abs(float(velocity.get("vx", 0)))
        return vx * local_t >= target["distance_m"]
    return False


@dataclass(frozen=True)
class PoseRecord:
    pose_id: str
    lifted_leg: str
    front_height: str
    rear_height: str
    direction: str
    description: str


@dataclass(frozen=True)
class MovementRecord:
    movement_id: str
    leg: str
    direction: str
    amplitude: str
    description: str


def _data_dir() -> Path:
    return Path(__file__).resolve().parent / "retrieval_data"


def default_pose_db_path() -> Path:
    return _data_dir() / "locomotion_pose_db.jsonl"


def default_movement_db_path() -> Path:
    return _data_dir() / "locomotion_movement_db.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


class LocomotionRetrievalDB:
    def __init__(
        self,
        pose_db_path: Path | None = None,
        movement_db_path: Path | None = None,
        kinematics: QuadrupedKinematics | None = None,
    ) -> None:
        self.pose_db_path = pose_db_path or default_pose_db_path()
        self.movement_db_path = movement_db_path or default_movement_db_path()
        self.kinematics = kinematics or QuadrupedKinematics()
        self._pose_rows = [PoseRecord(**row) for row in _load_jsonl(self.pose_db_path)]
        self._movement_rows = [MovementRecord(**row) for row in _load_jsonl(self.movement_db_path)]

    @property
    def pose_rows(self) -> tuple[PoseRecord, ...]:
        return tuple(self._pose_rows)

    @property
    def movement_rows(self) -> tuple[MovementRecord, ...]:
        return tuple(self._movement_rows)

    def retrieve_pose(self, query: dict[str, Any]) -> PoseRecord:
        def score(row: PoseRecord) -> tuple[int, str]:
            exact = 0
            if query.get("lifted_leg", row.lifted_leg) == row.lifted_leg:
                exact += 1
            if query.get("front_height", row.front_height) == row.front_height:
                exact += 1
            if query.get("rear_height", row.rear_height) == row.rear_height:
                exact += 1
            if query.get("direction", row.direction) == row.direction:
                exact += 1
            return (exact, row.pose_id)

        return max(self._pose_rows, key=score)

    def retrieve_movement(self, query: dict[str, Any]) -> MovementRecord:
        candidates = [
            row
            for row in self._movement_rows
            if row.leg == query["leg"] and row.direction == query["direction"]
        ]
        if not candidates:
            raise ValueError(f"No movement matches query={query}")
        amplitude = query.get("amplitude")
        if amplitude is not None:
            same_amp = [row for row in candidates if row.amplitude == amplitude]
            if same_amp:
                return same_amp[0]
        return candidates[0]

    def pose_to_feet(self, pose: PoseRecord, pose_params: dict[str, Any] | None = None) -> FootTargetMap:
        nominal = self.kinematics.nominal_stance()
        feet = {leg: nominal[leg] for leg in LEG_ORDER}

        front_shift = _height_to_z_delta(pose.front_height)
        rear_shift = _height_to_z_delta(pose.rear_height)
        support_shift = _support_shift_map(pose, front_shift, rear_shift)
        for leg in ("FL", "FR", "RL", "RR"):
            x, y, z = feet[leg]
            feet[leg] = (x, y, z + support_shift[leg])

        feet = _apply_direction_bias(feet, pose.direction)

        if pose.lifted_leg != "none":
            three_leg_pose = {}
            if isinstance(pose_params, dict):
                three_leg_pose = pose_params.get("three_leg_pose", {}) or {}
            feet = _apply_three_leg_pose(feet, pose.lifted_leg, three_leg_pose)
        return feet


def _pose_query_from_pose_params(params: dict[str, Any]) -> dict[str, str]:
    return {
        "lifted_leg": _lifted_leg_from_pose(str(params.get("dir", "neutral")), bool(params.get("three_legs", False))),
        "front_height": str(params.get("front_height", "mid")),
        "rear_height": str(params.get("back_height", params.get("rear_height", "mid"))),
        "direction": str(params.get("dir", "neutral")),
    }


def _pose_query_from_step(step: dict[str, Any]) -> dict[str, str]:
    pose_params = step.get("parameters", {}).get("pose")
    if pose_params is not None:
        return _pose_query_from_pose_params(pose_params)
    return dict(step["query"])


def _lifted_leg_from_pose(direction: str, three_legs: bool) -> str:
    if not three_legs:
        return "none"
    mapping = {
        "left": "FL",
        "right": "FR",
        "front": "FR",
        "back": "FR",
        "neutral": "none",
    }
    if direction not in mapping:
        raise ValueError(f"Unsupported pose direction for lifted leg selection: {direction}")
    return mapping[direction]


def _detect_lifted_leg(current_pose: FootTargetMap, nominal: FootTargetMap) -> str:
    best_leg = "none"
    best_delta = 0.06
    for leg in LEG_ORDER:
        delta = current_pose[leg][2] - nominal[leg][2]
        if delta > best_delta:
            best_leg = leg
            best_delta = delta
    return best_leg


def _movement_axis_from_name(axis_name: str) -> tuple[float, float, float]:
    axis = axis_name.lower()
    if axis == "x":
        return (1.0, 0.0, 0.0)
    if axis == "y":
        return (0.0, 1.0, 0.0)
    if axis == "z":
        return (0.0, 0.0, 1.0)
    raise ValueError(f"Unsupported movement axis: {axis_name}")


def _deg_range_from_value(val: Any) -> tuple[float, float]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return float(val[0]), float(val[1])
    if isinstance(val, (int, float)):
        span = abs(float(val))
        return (0.0, span)
    raise ValueError(f"Invalid degrees range: {val!r}")


def _primary_axis_from_degrees(degrees: dict[str, Any]) -> str:
    keys = [str(k).lower() for k in degrees if str(k).lower() in {"x", "y", "z"}]
    if not keys:
        raise ValueError("degrees must include at least one x/y/z key")
    order = {"x": 0, "y": 1, "z": 2}
    return sorted(keys, key=lambda k: order[k])[0]


# Maps sin(deg) to world-space foot target offset (meters) for x/z paw IK (y uses hip joint directly).
_HAND_SIN_METERS = {"x": 0.09, "z": 0.075}

_LEGACY_AMOUNT_TO_IK_DEG = {"small": 12.0, "medium": 22.0, "large": 35.0}
_LEGACY_AMOUNT_TO_HIP_DEG = {"small": 15.0, "medium": 28.0, "large": 40.0}


def _foot_ik_offset_from_degrees(degrees: dict[str, Any], progress: float) -> tuple[float, float, float]:
    dx = 0.0
    dy = 0.0
    dz = 0.0
    for axis in ("x", "z"):
        raw_key = next((k for k in degrees if str(k).lower() == axis), None)
        if raw_key is None:
            continue
        lo, hi = _deg_range_from_value(degrees[raw_key])
        deg = lo + (hi - lo) * progress
        s = math.sin(math.radians(deg))
        scale = _HAND_SIN_METERS[axis]
        if axis == "x":
            dx += s * scale
        else:
            dz += s * scale
    return (dx, dy, dz)


def _hip_y_delta_rad(joint_cfg: dict[str, Any], movement_params: dict[str, Any], phase: float) -> float:
    rep, spd = _movement_rep_speed(movement_params, joint_cfg)
    progress = _movement_line_progress(phase, rep, spd)
    degrees = joint_cfg.get("degrees")
    if isinstance(degrees, dict):
        ykey = next((k for k in degrees if str(k).lower() == "y"), None)
        if ykey is None:
            return 0.0
        lo, hi = _deg_range_from_value(degrees[ykey])
        deg = lo + (hi - lo) * progress
        # Match rest pose at the smaller configured angle (e.g. [-30,30] or [30,-30] -> -30).
        # Anchoring at `lo` only breaks [30,-30]: valleys sit at lo=+30 so the paw never returns to up at -30.
        rest_deg = min(lo, hi)
        return math.radians(deg) - math.radians(rest_deg)
    direction = str(joint_cfg.get("direction", "")).lower()
    if direction == "y":
        amount = str(joint_cfg.get("amount", "medium"))
        span = _LEGACY_AMOUNT_TO_HIP_DEG.get(amount, 28.0)
        lo, hi = 0.0, span
        deg = lo + (hi - lo) * progress
        return math.radians(deg) - math.radians(lo)
    return 0.0


def _joint_axis_delta_rad(
    joint_cfg: dict[str, Any],
    movement_params: dict[str, Any],
    phase: float,
    *,
    axis: str,
) -> float:
    rep, spd = _movement_rep_speed(movement_params, joint_cfg)
    progress = _movement_line_progress(phase, rep, spd)
    degrees = joint_cfg.get("degrees")
    if isinstance(degrees, dict):
        akey = next((k for k in degrees if str(k).lower() == axis), None)
        if akey is None:
            return 0.0
        lo, hi = _deg_range_from_value(degrees[akey])
        deg = lo + (hi - lo) * progress
        # For explicit joint-space controls, preserve range ordering semantics:
        # [0, +a] => positive delta, [0, -a] => negative delta.
        rest_deg = lo
        return math.radians(deg) - math.radians(rest_deg)
    return 0.0


def _movement_rep_speed(movement_params: dict[str, Any], joint_cfg: dict[str, Any]) -> tuple[int, float]:
    rep = int(joint_cfg.get("repetition", movement_params.get("repetition", 1)))
    spd = float(joint_cfg.get("speed", movement_params.get("speed", 1.0)))
    return max(1, rep), max(0.1, spd)


def _movement_query_from_step(
    step: dict[str, Any],
    current_pose: FootTargetMap,
    nominal: FootTargetMap,
) -> dict[str, str]:
    movement_params = step.get("parameters", {}).get("movement")
    if movement_params is None:
        return dict(step["query"])

    joints = movement_params.get("joints", [])
    if not joints:
        leg = _detect_lifted_leg(current_pose, nominal)
        if leg == "none":
            leg = "FR"
        return {"leg": leg, "direction": "front", "amplitude": "medium"}

    leg = str(movement_params.get("leg", _detect_lifted_leg(current_pose, nominal)))
    if leg == "none":
        leg = "FR"
    first_joint = joints[0]
    deg_map = first_joint.get("degrees")
    if isinstance(deg_map, dict):
        axis = _primary_axis_from_degrees(deg_map)
        return {"leg": leg, "direction": axis, "amplitude": "medium"}
    return {
        "leg": leg,
        "direction": str(first_joint.get("direction", "x")),
        "amplitude": str(first_joint.get("amount", "medium")),
    }


def _height_to_z_delta(height: str) -> float:
    table = {"low": -0.03, "mid": 0.015, "high": 0.065}
    if height not in table:
        raise ValueError(f"Unsupported height level: {height}")
    return table[height]


def _support_shift_map(pose: PoseRecord, front_shift: float, rear_shift: float) -> dict[str, float]:
    shifts = {"FL": front_shift, "FR": front_shift, "RL": rear_shift, "RR": rear_shift}
    leg = pose.lifted_leg
    if leg == "none":
        return shifts
    if leg in ("FL", "FR"):
        support_front = "FR" if leg == "FL" else "FL"
        shifts[support_front] = min(0.0, 0.25 * front_shift)
        shifts["RL"] = min(shifts["RL"], rear_shift - 0.01)
        shifts["RR"] = min(shifts["RR"], rear_shift - 0.01)
    return shifts


def _apply_direction_bias(feet: FootTargetMap, direction: str) -> FootTargetMap:
    updated = dict(feet)
    if direction == "front":
        for leg in ("FL", "FR"):
            x, y, z = updated[leg]
            updated[leg] = (x + 0.03, y, z)
        for leg in ("RL", "RR"):
            x, y, z = updated[leg]
            updated[leg] = (x - 0.02, y, z)
    elif direction == "back":
        for leg in ("FL", "FR"):
            x, y, z = updated[leg]
            updated[leg] = (x - 0.02, y, z)
        for leg in ("RL", "RR"):
            x, y, z = updated[leg]
            updated[leg] = (x + 0.03, y, z)
    elif direction == "left":
        for leg in ("FL", "RL"):
            x, y, z = updated[leg]
            updated[leg] = (x, y + 0.02, z)
        for leg in ("FR", "RR"):
            x, y, z = updated[leg]
            updated[leg] = (x, y - 0.02, z)
    elif direction == "right":
        for leg in ("FL", "RL"):
            x, y, z = updated[leg]
            updated[leg] = (x, y - 0.02, z)
        for leg in ("FR", "RR"):
            x, y, z = updated[leg]
            updated[leg] = (x, y + 0.02, z)
    elif direction != "neutral":
        raise ValueError(f"Unsupported pose direction: {direction}")
    return updated


def _pingpong01(cycle_pos: float) -> float:
    frac = cycle_pos - math.floor(cycle_pos)
    return frac * 2.0 if frac < 0.5 else (1.0 - frac) * 2.0


def _movement_line_progress(phase: float, repetition: int, speed: float) -> float:
    clamped = min(max(phase, 0.0), 1.0)
    cycles = max(1, repetition) * max(0.1, speed)
    return _pingpong01(clamped * cycles)


def _normalize_hand_position(hand_position: str) -> str:
    token = hand_position.strip().lower()
    alias = {
        "위": "up",
        "앞": "front",
        "오른쪽": "right",
        "뒤": "back",
        "왼쪽": "left",
    }
    return alias.get(token, token)


def _three_leg_base_offsets(hand_position: str) -> tuple[float, float, float]:
    hp = _normalize_hand_position(hand_position)
    # Keep a lifted-foot proxy in current_pose so subsequent movement step can detect the same lifted leg.
    mapping = {
        "up": (0.0, 0.0, 0.14),
        "front": (0.10, 0.0, 0.06),
        "right": (0.0, -0.10, 0.10),
        "back": (-0.04, 0.0, 0.12),
        "left": (0.0, 0.10, 0.10),
    }
    return mapping.get(hp, mapping["back"])


def _three_leg_pose_joint_overrides(
    leg: str,
    hand_position: str,
) -> JointMap:
    """Absolute joint targets for one lifted paw pose (5-way hand_position mapping)."""
    hip_j = f"{leg}_hip_joint"
    thigh_j = f"{leg}_thigh_joint"
    calf_j = f"{leg}_calf_joint"
    hp = _normalize_hand_position(hand_position)
    targets = {
        "up": (0.000, -1.571, -1.780),
        "front": (0.000, -1.571, -1.110),
        "right": (-1.047, 0.960, -0.838),
        "back": (0.000, 0.960, -2.723),
        "left": (1.047, 0.960, -0.838),
    }
    hip, thigh, calf = targets.get(hp, targets["back"])
    return {hip_j: hip, thigh_j: thigh, calf_j: calf}


def _apply_three_leg_pose(
    feet: FootTargetMap,
    lifted_leg: str,
    three_leg_pose: dict[str, Any],
) -> FootTargetMap:
    updated = dict(feet)
    hand_position = str(three_leg_pose.get("hand_position", "back"))
    x, y, z = updated[lifted_leg]
    dx, dy, dz = _three_leg_base_offsets(hand_position)
    updated[lifted_leg] = (x + dx, y + dy, z + dz)
    return updated


def _movement_step_offsets(step: dict[str, Any], phase: float) -> tuple[float, float, float]:
    movement_params = step.get("parameters", {}).get("movement", {})
    joints = movement_params.get("joints", [])
    dx = 0.0
    dy = 0.0
    dz = 0.0
    for joint_cfg in joints:
        jtok = str(joint_cfg.get("joint", "")).lower()
        degrees = joint_cfg.get("degrees")
        rep, spd = _movement_rep_speed(movement_params, joint_cfg)
        progress = _movement_line_progress(phase, rep, spd)
        if isinstance(degrees, dict):
            if jtok in {"hip", "thigh", "calf", "knee"}:
                part_dx, part_dy, part_dz = (0.0, 0.0, 0.0)
            else:
                sub_deg = {k: v for k, v in degrees.items() if str(k).lower() in ("x", "z")}
                part_dx, part_dy, part_dz = _foot_ik_offset_from_degrees(sub_deg, progress)
        else:
            direction = str(joint_cfg.get("direction", "y")).lower()
            amount = str(joint_cfg.get("amount", "medium"))
            if direction in ("x", "z"):
                span = _LEGACY_AMOUNT_TO_IK_DEG.get(amount, 22.0)
                part_dx, part_dy, part_dz = _foot_ik_offset_from_degrees({direction: (0.0, span)}, progress)
            else:
                part_dx, part_dy, part_dz = (0.0, 0.0, 0.0)
        dx += part_dx
        dy += part_dy
        dz += part_dz
    return (dx, dy, dz)


def _movement_axes_used(step: dict[str, Any]) -> set[str]:
    movement_params = step.get("parameters", {}).get("movement", {})
    joints = movement_params.get("joints", [])
    axes: set[str] = set()
    for joint_cfg in joints:
        jtok = str(joint_cfg.get("joint", "")).lower()
        degrees = joint_cfg.get("degrees")
        if isinstance(degrees, dict):
            if jtok in {"hip", "thigh", "calf", "knee"}:
                continue
            for k in degrees:
                kk = str(k).lower()
                if kk in {"x", "z"}:
                    axes.add(kk)
        else:
            d = str(joint_cfg.get("direction", "y")).lower()
            if d in {"x", "z"}:
                axes.add(d)
    return axes


def _clamp_lateral_with_fixed_z(
    leg: str,
    foot: tuple[float, float, float],
    anchor: tuple[float, float, float],
    kin: QuadrupedKinematics,
) -> tuple[float, float, float]:
    hip = kin.config.hip_origin(leg)
    x, y, z = foot
    z_down = -(z - hip[2])
    min_radius = kin.config.hip_offset + 0.01
    min_lateral_sq = max(min_radius * min_radius - z_down * z_down, 0.0)
    min_lateral = math.sqrt(min_lateral_sq)
    local_y = y - hip[1]
    if abs(local_y) >= min_lateral:
        return foot
    anchor_local_y = anchor[1] - hip[1]
    sign = 1.0 if anchor_local_y >= 0.0 else -1.0
    return (x, hip[1] + sign * min_lateral, z)


def _infer_three_leg_base_joints_from_current_pose(
    leg: str,
    current_pose: FootTargetMap,
    kin: QuadrupedKinematics,
) -> JointMap:
    """Infer closest 3-leg canonical joint set (up/front/right/back/left) for lifted leg."""
    target = current_pose[leg]
    hip_j = f"{leg}_hip_joint"
    thigh_j = f"{leg}_thigh_joint"
    calf_j = f"{leg}_calf_joint"
    best: JointMap | None = None
    best_dist = float("inf")
    for hp in ("up", "front", "right", "back", "left"):
        cand = _three_leg_pose_joint_overrides(leg, hp)
        foot = kin.foot_position(leg, (cand[hip_j], cand[thigh_j], cand[calf_j]))
        dx = foot[0] - target[0]
        dy = foot[1] - target[1]
        dz = foot[2] - target[2]
        dist = dx * dx + dy * dy + dz * dz
        if dist < best_dist:
            best_dist = dist
            best = cand
    assert best is not None
    return best


def _movement_joint_overrides(
    step: dict[str, Any],
    leg: str,
    phase: float,
    base_joint_map: JointMap,
    current_pose: FootTargetMap | None = None,
    kin: QuadrupedKinematics | None = None,
) -> JointMap:
    movement_params = step.get("parameters", {}).get("movement", {})
    joints = movement_params.get("joints", [])
    overrides: JointMap = {}
    hip_j = f"{leg}_hip_joint"
    thigh_j = f"{leg}_thigh_joint"
    calf_j = f"{leg}_calf_joint"
    hip_delta = 0.0
    inferred_base: JointMap | None = None
    explicit_joint_axes: set[str] = set()

    for joint_cfg in joints:
        jtok = str(joint_cfg.get("joint", "")).lower()
        if jtok == "knee":
            jtok = "calf"
        if jtok in {"hip", "thigh", "calf"}:
            explicit_joint_axes.add(jtok)
        if jtok == "thigh":
            delta = _joint_axis_delta_rad(joint_cfg, movement_params, phase, axis="x")
            has_base = "base_rad" in joint_cfg
            if has_base:
                base = float(joint_cfg["base_rad"])
            else:
                if inferred_base is None and current_pose is not None and kin is not None:
                    inferred_base = _infer_three_leg_base_joints_from_current_pose(leg, current_pose, kin)
                base = inferred_base[thigh_j] if inferred_base is not None else base_joint_map[thigh_j]
            overrides[thigh_j] = base + delta
            continue
        if jtok == "calf":
            delta = _joint_axis_delta_rad(joint_cfg, movement_params, phase, axis="x")
            has_base = "base_rad" in joint_cfg
            if has_base:
                base = float(joint_cfg["base_rad"])
            else:
                if inferred_base is None and current_pose is not None and kin is not None:
                    inferred_base = _infer_three_leg_base_joints_from_current_pose(leg, current_pose, kin)
                base = inferred_base[calf_j] if inferred_base is not None else base_joint_map[calf_j]
            overrides[calf_j] = base + delta
            continue
        if jtok == "hip":
            delta = _joint_axis_delta_rad(joint_cfg, movement_params, phase, axis="y")
            has_base = "base_rad" in joint_cfg
            if has_base:
                base = float(joint_cfg["base_rad"])
            else:
                if inferred_base is None and current_pose is not None and kin is not None:
                    inferred_base = _infer_three_leg_base_joints_from_current_pose(leg, current_pose, kin)
                base = inferred_base[hip_j] if inferred_base is not None else base_joint_map[hip_j]
            overrides[hip_j] = base + delta
            continue
        # Backward-compatible route: y movement on unspecified/hand entries drives hip abduction.
        hip_delta += _hip_y_delta_rad(joint_cfg, movement_params, phase)

    if hip_j not in overrides and abs(hip_delta) > 1e-15:
        overrides[hip_j] = base_joint_map[hip_j] + hip_delta

    # When any explicit leg joint is commanded, treat unspecified leg joints as zero-delta
    # from the inferred pose baseline to preserve the same leg shape.
    if explicit_joint_axes and current_pose is not None and kin is not None:
        if inferred_base is None:
            inferred_base = _infer_three_leg_base_joints_from_current_pose(leg, current_pose, kin)
        if "hip" not in explicit_joint_axes and hip_j not in overrides:
            overrides[hip_j] = inferred_base[hip_j]
        if "thigh" not in explicit_joint_axes and thigh_j not in overrides:
            overrides[thigh_j] = inferred_base[thigh_j]
        if "calf" not in explicit_joint_axes and calf_j not in overrides:
            overrides[calf_j] = inferred_base[calf_j]
    return overrides


class ConfigMotionComposer:
    def __init__(self, controller: HighLevelController | None = None, db: LocomotionRetrievalDB | None = None) -> None:
        self.controller = controller or HighLevelController()
        self.db = db or LocomotionRetrievalDB(kinematics=self.controller.low_level.kinematics)
        self.kinematics = self.controller.low_level.kinematics
        self.nominal = self.kinematics.nominal_stance()

    def feet_for_step(
        self,
        step: dict[str, Any],
        local_t: float,
        duration: float,
        current_pose: FootTargetMap,
    ) -> tuple[FootTargetMap, FootTargetMap, JointMap]:
        kind = step["type"]
        if kind == "pose":
            pose_params = step.get("parameters", {}).get("pose", {})
            pose = self.db.retrieve_pose(_pose_query_from_step(step))
            feet = self._sanitize_feet(self.db.pose_to_feet(pose, pose_params))
            overrides: JointMap = {}
            if pose_params.get("three_legs"):
                tpl = pose_params.get("three_leg_pose") or {}
                hand_p = str(tpl.get("hand_position", "back"))
                overrides = _three_leg_pose_joint_overrides(pose.lifted_leg, hand_p)
                posed_joint_map = self.kinematics.feet_to_joint_map(feet)
                posed_joint_map.update(overrides)
                feet = self.kinematics.joint_map_to_feet(posed_joint_map)
            return feet, feet, overrides

        if kind == "movement":
            movement_query = _movement_query_from_step(step, current_pose, self.nominal)
            leg = movement_query["leg"]
            feet = dict(current_pose)
            phase = min(max(local_t / max(duration, 1e-6), 0.0), 1.0)
            anchor = current_pose[leg]
            dx, dy, dz = _movement_step_offsets(step, phase)
            used_axes = _movement_axes_used(step)
            x = anchor[0] + (dx if "x" in used_axes else 0.0)
            y = anchor[1] + (dy if "y" in used_axes else 0.0)
            z = anchor[2] + (dz if "z" in used_axes else 0.0)
            if used_axes == {"y"}:
                feet[leg] = _clamp_lateral_with_fixed_z(leg, (x, y, z), anchor, self.kinematics)
            else:
                feet[leg] = (x, y, z)
            feet = self._sanitize_feet(feet)
            # Base IK must use the same foot targets as `feet` (post-sanitize). Using `current_pose`
            # here desyncs thigh/calf from hip when delta=0, so "rest" frames drift (A-B-C-B-C).
            base_joint_map = self.kinematics.feet_to_joint_map(feet)
            joint_overrides = _movement_joint_overrides(
                step,
                leg,
                phase,
                base_joint_map,
                current_pose=dict(current_pose),
                kin=self.kinematics,
            )
            return feet, dict(current_pose), joint_overrides

        if kind == "path":
            path_params = step.get("parameters", {}).get("path")
            shape = step["shape"] if path_params is None else str(path_params["shape"])
            velocity = VelocityCommand(**(step.get("velocity", {}) if path_params is None else path_params.get("velocity", {})))
            if shape in {"line", "arc"}:
                return self.controller.gait.foot_targets(velocity, local_t), dict(current_pose), {}
            if shape == "pose_to_pose":
                if path_params is None:
                    start_params = step["start_pose"]
                    end_params = step["end_pose"]
                else:
                    start_params = path_params["start_pose"]
                    end_params = path_params["end_pose"]
                start_query = _pose_query_from_pose_params(start_params)
                end_query = _pose_query_from_pose_params(end_params)
                start_pose = self.db.pose_to_feet(self.db.retrieve_pose(start_query), start_params)
                end_pose = self.db.pose_to_feet(self.db.retrieve_pose(end_query), end_params)
                alpha = min(max(local_t / max(duration, 1e-6), 0.0), 1.0)
                blended_pose = {leg: lerp_vec(start_pose[leg], end_pose[leg], alpha) for leg in LEG_ORDER}
                walk_feet = self.controller.gait.foot_targets(velocity, local_t)
                feet = {}
                for leg in LEG_ORDER:
                    nx, ny, nz = self.nominal[leg]
                    wx, wy, wz = walk_feet[leg]
                    px, py, pz = blended_pose[leg]
                    feet[leg] = (
                        px + 0.7 * (wx - nx),
                        py + 0.7 * (wy - ny),
                        pz + 0.35 * (wz - nz),
                    )
                return self._sanitize_feet(feet), blended_pose, {}
            raise ValueError(f"Unsupported path shape: {shape}")

        raise ValueError(f"Unsupported step type: {kind}")

    def _sanitize_feet(self, feet: FootTargetMap) -> FootTargetMap:
        safe: FootTargetMap = {}
        min_radius = self.kinematics.config.hip_offset + 0.01
        for leg, foot in feet.items():
            hip = self.kinematics.config.hip_origin(leg)
            lx = foot[0] - hip[0]
            ly = foot[1] - hip[1]
            lz = foot[2] - hip[2]
            z_down = -lz
            radial = math.sqrt(max(ly * ly + z_down * z_down, 1e-8))
            if radial < min_radius:
                scale = min_radius / radial
                ly *= scale
                z_down *= scale
                lz = -z_down
            safe[leg] = (hip[0] + lx, hip[1] + ly, hip[2] + lz)
        return safe
