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

_ARC_BASE_WZ = 1.0     # rad/s at speed=1.0 (in-place yaw)
_ARC_FORWARD_VX = 0.3   # m/s forward per unit speed during arc (r>0)
_LINE_BASE_VX = 0.5     # m/s at speed=1.0


def _arc_degrees_value(degrees: Any) -> float:
    """Scalar sweep for a single arc segment (lists must be expanded upstream)."""
    if isinstance(degrees, (list, tuple)):
        nums = [float(x) for x in degrees if isinstance(x, (int, float))]
        if not nums:
            return 0.0
        # Prefer the last non-zero signed value if a ± pair slipped through.
        for v in reversed(nums):
            if abs(v) > 1e-9:
                return float(v)
        return float(nums[-1])
    return float(degrees)


def normalize_step(step: dict[str, Any]) -> dict[str, Any]:
    """Normalize a config step so that ``duration`` is always present.

    * ``arc``  path with ``degrees`` + ``speed`` → computes ``duration`` and
      ``velocity``  (``wz`` from degrees, ``vx`` forward for r>0 / 0 for spin).
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
        deg = _arc_degrees_value(path_p["degrees"])
        spd = max(0.2, float(path_p.get("speed", 1.0)))
        rad = deg * math.pi / 180.0
        radius = abs(float(path_p.get("radius", 0.0) or 0.0))
        # Always walk forward on circling arcs; in-place yaw when r≈0.
        # Positive deg → CCW (+wz), negative → CW (−wz).
        if radius < 1e-4:
            vx = 0.0
            wz = math.copysign(spd * _ARC_BASE_WZ, rad if rad != 0.0 else 1.0)
        else:
            vx = abs(spd) * _ARC_FORWARD_VX
            wz = math.copysign(vx / radius, rad if rad != 0.0 else 1.0)
        # Use 3x theoretical duration as upper bound; the renderer will
        # detect heading completion and break early.
        dur = 3.0 * abs(rad) / max(abs(wz), 1e-6)
        step["duration"] = round(dur, 3)
        path_p = dict(path_p)
        path_p["degrees"] = deg
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
        return {"heading_rad": abs(_arc_degrees_value(path_p["degrees"])) * math.pi / 180.0}
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

        params = pose_params if isinstance(pose_params, dict) else {}
        front_level = str(params.get("front_height", pose.front_height))
        rear_level = str(params.get("back_height", params.get("rear_height", pose.rear_height)))
        direction = str(params.get("dir", pose.direction))
        lifted_leg = _lifted_leg_from_pose(direction, bool(params.get("three_legs", False)))

        front_shift = _height_to_z_delta(front_level)
        rear_shift = _height_to_z_delta(rear_level)
        # Optional continuous offsets (m) for soft planted pair-leg nods — stacked on
        # categorical height so stance crouch/bow stay strong while move(forelegs,z)
        # can bob gently without slamming into play_bow.
        try:
            front_shift += float(params.get("front_z_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            rear_shift += float(params.get("back_z_delta", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        support_shift = _support_shift_map(lifted_leg, front_shift, rear_shift)
        for leg in ("FL", "FR", "RL", "RR"):
            x, y, z = feet[leg]
            feet[leg] = (x, y, z + support_shift[leg])

        feet = _apply_direction_bias(feet, direction)

        if lifted_leg != "none":
            three_leg_pose = {}
            if isinstance(pose_params, dict):
                three_leg_pose = pose_params.get("three_leg_pose", {}) or {}
            feet = _apply_three_leg_pose(feet, lifted_leg, three_leg_pose)
            feet = _apply_three_leg_weight_shift(feet, lifted_leg)
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
        "neutral": "FR",  # schema has no dir; default right-front paw
    }
    if direction not in mapping:
        return "FR"
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
    # Semantic contract:
    # - low  => crouched / joints more flexed (play_bow front, crouch, sit rear)
    # - high => tall / joints more extended (sit front, play_bow rear)
    # In this body frame, increasing foot z lifts the foot toward the body (more flex),
    # while decreasing z pushes it farther down (more extension).
    # mid is 0 so nominal stand actually reaches the ground under hold-base.
    # Magnitudes are large enough to read under gravity-plant PD (was ±3cm — too subtle).
    table = {"low": 0.085, "mid": 0.0, "high": -0.055}
    if height not in table:
        raise ValueError(f"Unsupported height level: {height}")
    return table[height]


# Push feet slightly past the kinematic tip so MJCF foot spheres (r≈0.022) plant.
_FOOT_PLANT_Z = -0.025


def _apply_foot_plant(feet: FootTargetMap) -> FootTargetMap:
    return {leg: (p[0], p[1], p[2] + _FOOT_PLANT_Z) for leg, p in feet.items()}


def _support_shift_map(lifted_leg: str, front_shift: float, rear_shift: float) -> dict[str, float]:
    shifts = {"FL": front_shift, "FR": front_shift, "RL": rear_shift, "RR": rear_shift}
    leg = lifted_leg
    if leg == "none":
        return shifts
    if leg in ("FL", "FR"):
        support_front = "FR" if leg == "FL" else "FL"
        # Keep the remaining front foot more extended (planted) and crouch the
        # rear pair a bit more so CoM stays inside the three-leg support triangle.
        shifts[support_front] = min(0.0, 0.15 * front_shift)
        shifts["RL"] = min(shifts["RL"], rear_shift - 0.02)
        shifts["RR"] = min(shifts["RR"], rear_shift - 0.02)
    return shifts


def _apply_three_leg_weight_shift(feet: FootTargetMap, lifted_leg: str) -> FootTargetMap:
    """Nudge support feet under the body so CoM stays over the triangle."""
    updated = dict(feet)
    if lifted_leg == "FR":
        # Aggressive left/back shift onto FL+RL+RR before the paw leaves the ground.
        for leg, dx, dy in (("FL", -0.02, 0.06), ("RL", 0.025, 0.05), ("RR", 0.02, -0.025)):
            x, y, z = updated[leg]
            updated[leg] = (x + dx, y + dy, z)
    elif lifted_leg == "FL":
        for leg, dx, dy in (("FR", -0.02, -0.06), ("RR", 0.025, -0.05), ("RL", 0.02, 0.025)):
            x, y, z = updated[leg]
            updated[leg] = (x + dx, y + dy, z)
    return updated


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


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def _movement_line_progress(phase: float, repetition: int, speed: float) -> float:
    """Map phase∈[0,1] through ``repetition`` ping-pong cycles.

    ``speed`` only shortens step duration (set by the config converter); it must
    not multiply cycle count — otherwise ``rep=3, spd=3`` becomes 9 frantic pumps.
    """
    del speed  # duration already encodes speed; keep signature for callers
    clamped = min(max(phase, 0.0), 1.0)
    cycles = max(1, int(repetition))
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
        # Keep lifts small enough that PD + weight-shift can brace under gravity.
        "up": (0.0, 0.0, 0.08),
        "front": (0.06, 0.0, 0.04),
        "right": (0.0, -0.06, 0.05),
        "back": (-0.02, 0.0, 0.06),
        "left": (0.0, 0.06, 0.05),
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
    # Kinematic three-leg path can hold a clearer raise without tipping.
    targets = {
        "up": (0.000, -1.20, -1.50),
        "front": (0.000, -0.90, -1.10),
        "right": (-0.70, 0.70, -0.90),
        "back": (0.000, 0.70, -1.80),
        "left": (0.70, 0.70, -0.90),
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
        self._pose_blend_start: FootTargetMap | None = None
        # Persist lifted-paw joint locks so empty ``hold`` keeps the silhouette.
        self._hold_joint_overrides: JointMap = {}
        self._hold_feet: FootTargetMap | None = None

    def feet_for_step(
        self,
        step: dict[str, Any],
        local_t: float,
        duration: float,
        current_pose: FootTargetMap,
    ) -> tuple[FootTargetMap, FootTargetMap, JointMap]:
        kind = step["type"]
        if kind == "pose":
            pose_params = step.get("parameters", {}).get("pose", {}) or {}
            pose = self.db.retrieve_pose(_pose_query_from_step(step))
            # Four-foot planted silhouette (no lift) — base for bracing under gravity.
            planted_params = dict(pose_params)
            planted_params["three_legs"] = False
            # Bow + lifted paw is statically unstable under free-base gravity; brace
            # from a near-stand support polygon, then raise the paw.
            if pose_params.get("three_legs"):
                fh = str(planted_params.get("front_height", "mid")).lower()
                bh = str(planted_params.get("back_height", planted_params.get("rear_height", "mid"))).lower()
                if fh == "low" and bh == "high":
                    planted_params["front_height"] = "mid"
                    planted_params["back_height"] = "mid"
            planted = self._sanitize_feet(self.db.pose_to_feet(pose, planted_params), plant=True)

            overrides_full: JointMap = {}
            target = dict(planted)
            if pose_params.get("three_legs"):
                lifted = pose.lifted_leg if pose.lifted_leg not in (None, "none") else "FR"
                # Shift support feet first so CoM can settle inside the triangle.
                planted = _apply_three_leg_weight_shift(planted, lifted)
                tpl = pose_params.get("three_leg_pose") or {}
                hand_p = str(tpl.get("hand_position", "back"))
                overrides_full = _three_leg_pose_joint_overrides(lifted, hand_p)
                lifted_feet = _apply_three_leg_pose(dict(planted), lifted, tpl)
                posed_joint_map = self.kinematics.feet_to_joint_map(lifted_feet)
                posed_joint_map.update(overrides_full)
                target = self.kinematics.joint_map_to_feet(posed_joint_map)
            else:
                target = planted
                self._hold_joint_overrides = {}
                self._hold_feet = None

            # Ease into the silhouette so PD can brace under gravity (no root pin).
            if local_t <= 1e-9 or self._pose_blend_start is None:
                self._pose_blend_start = dict(current_pose)
            alpha = _smoothstep01(min(max(local_t / max(duration, 1e-6), 0.0), 1.0))
            start = self._pose_blend_start

            overrides: JointMap = {}
            if pose_params.get("three_legs"):
                # Phase 1: weight-shift only (4 feet). Phase 2: raise the paw.
                if alpha < 0.65:
                    a = _smoothstep01(alpha / 0.65)
                    feet = {
                        leg: (
                            start[leg][0] + (planted[leg][0] - start[leg][0]) * a,
                            start[leg][1] + (planted[leg][1] - start[leg][1]) * a,
                            start[leg][2] + (planted[leg][2] - start[leg][2]) * a,
                        )
                        for leg in planted
                    }
                else:
                    a = _smoothstep01((alpha - 0.65) / 0.35)
                    feet = {
                        leg: (
                            planted[leg][0] + (target[leg][0] - planted[leg][0]) * a,
                            planted[leg][1] + (target[leg][1] - planted[leg][1]) * a,
                            planted[leg][2] + (target[leg][2] - planted[leg][2]) * a,
                        )
                        for leg in target
                    }
                    start_q = self.kinematics.feet_to_joint_map(planted)
                    overrides = {
                        j: float(start_q.get(j, v)) + (float(v) - float(start_q.get(j, v))) * a
                        for j, v in overrides_full.items()
                    }
                    # Persist as soon as the paw lifts so a following empty hold
                    # keeps the extended silhouette (stop_palm_out / highfive).
                    self._hold_joint_overrides = dict(overrides_full)
                    self._hold_feet = dict(target) if a >= 0.95 else dict(feet)
            else:
                feet = {
                    leg: (
                        start[leg][0] + (target[leg][0] - start[leg][0]) * alpha,
                        start[leg][1] + (target[leg][1] - start[leg][1]) * alpha,
                        start[leg][2] + (target[leg][2] - start[leg][2]) * alpha,
                    )
                    for leg in target
                }

            if alpha >= 1.0 - 1e-6:
                self._pose_blend_start = None
            return feet, dict(target) if alpha >= 1.0 - 1e-6 else dict(feet), overrides

        self._pose_blend_start = None

        if kind == "movement":
            movement_params = step.get("parameters", {}).get("movement") or {}
            joints = movement_params.get("joints") or []
            # Empty-joint hold: freeze last silhouette (keep three-leg joint locks).
            if not joints:
                hold_feet = dict(self._hold_feet) if self._hold_feet is not None else dict(current_pose)
                return hold_feet, hold_feet, dict(self._hold_joint_overrides)
            # Active leg motion clears the static three-leg hold lock.
            self._hold_joint_overrides = {}
            self._hold_feet = None
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
            # Radial clamp only — current_pose is already planted from the prior pose step.
            feet = self._sanitize_feet(feet, plant=False)
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
            self._hold_joint_overrides = {}
            self._hold_feet = None
            path_params = step.get("parameters", {}).get("path")
            shape = step["shape"] if path_params is None else str(path_params["shape"])
            velocity = VelocityCommand(**(step.get("velocity", {}) if path_params is None else path_params.get("velocity", {})))
            if shape in {"line", "arc"}:
                return self.controller.gait.foot_targets(velocity, local_t), dict(current_pose), {}
            if shape == "pose_to_pose":
                if path_params is None:
                    start_params = step["start_pose"]
                    end_params = step["end_pose"]
                    repetition = int(step.get("repetition", 1))
                else:
                    start_params = path_params["start_pose"]
                    end_params = path_params["end_pose"]
                    repetition = int(path_params.get("repetition", step.get("repetition", 1)))
                repetition = max(1, repetition)
                start_query = _pose_query_from_pose_params(start_params)
                end_query = _pose_query_from_pose_params(end_params)
                start_pose = self.db.pose_to_feet(self.db.retrieve_pose(start_query), start_params)
                end_pose = self.db.pose_to_feet(self.db.retrieve_pose(end_query), end_params)
                alpha_total = min(max(local_t / max(duration, 1e-6), 0.0), 1.0)
                alpha = _pingpong01(alpha_total * float(repetition))
                # Ease in/out so gravity plant can track without thrashing.
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                blended_pose = {leg: lerp_vec(start_pose[leg], end_pose[leg], alpha) for leg in LEG_ORDER}
                walk_feet = self.controller.gait.foot_targets(velocity, local_t)
                gb_xy = float(path_params.get("gait_blend_xy", 0.92)) if path_params is not None else 0.92
                gb_z = float(path_params.get("gait_blend_z", 0.55)) if path_params is not None else 0.55
                gb_xy = max(0.0, min(1.0, gb_xy))
                gb_z = max(0.0, min(1.0, gb_z))
                feet = {}
                for leg in LEG_ORDER:
                    nx, ny, nz = self.nominal[leg]
                    wx, wy, wz = walk_feet[leg]
                    px, py, pz = blended_pose[leg]
                    feet[leg] = (
                        px + gb_xy * (wx - nx),
                        py + gb_xy * (wy - ny),
                        pz + gb_z * (wz - nz),
                    )
                # pose_to_feet is unplanted; plant once after the blend.
                return self._sanitize_feet(feet, plant=True), blended_pose, {}
            raise ValueError(f"Unsupported path shape: {shape}")

        raise ValueError(f"Unsupported step type: {kind}")

    def _sanitize_feet(self, feet: FootTargetMap, *, plant: bool = False) -> FootTargetMap:
        """Radial reachability clamp; optionally apply one-shot foot plant delta.

        Plant must not be re-applied on movement/hold steps that reuse an already
        planted ``current_pose`` — that double-dips ``_FOOT_PLANT_Z`` and looks
        like the robot sinking into the floor for the last few frames.
        """
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
        return _apply_foot_plant(safe) if plant else safe
