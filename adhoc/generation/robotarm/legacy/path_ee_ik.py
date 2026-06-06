"""EE IK path execution and config validation (line / arc, no joint field)."""
from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T
from PIL import Image

VALID_LINE_AXES = frozenset({"x", "y", "z"})
VALID_PLANES = frozenset({"xy", "yz", "xz", "zx"})
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# EE path speed scale: speed=1.0 -> 0.12 m/s along path
PATH_EE_M_PER_S_AT_UNIT_SPEED = 0.12
PATH_IK_MAX_ITERS = 80
PATH_IK_POS_TOL_M = 0.004


def normalize_plane(plane: str) -> str:
    p = str(plane).lower().strip()
    return "xz" if p == "zx" else p


def _plane_axis_indices(plane: str) -> tuple[int, int]:
    p = normalize_plane(plane)
    if p == "xy":
        return 0, 1
    if p == "yz":
        return 1, 2
    return 0, 2  # xz


def normalize_path_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Drop legacy few-shot ``joint`` when EE IK fields are present."""
    out = dict(params)
    if out.get("joint") is None:
        return out
    shape = str(out.get("shape", "line")).lower()
    if shape == "line" and out.get("axis") in VALID_LINE_AXES and isinstance(
        out.get("distance"), (int, float)
    ):
        out.pop("joint", None)
    elif shape in ("arc", "circle") and out.get("plane") is not None and isinstance(
        out.get("radius"), (int, float)
    ):
        out.pop("joint", None)
    return out


def validate_path_parameters(params: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    params = normalize_path_parameters(params)
    if params.get("joint") is not None:
        errors.append("path must not include 'joint' (EE IK draws line/arc in Cartesian space)")

    shape = params.get("shape")
    speed = params.get("speed", 1.0)
    if isinstance(speed, (int, float)) and not (0.5 <= float(speed) <= 4.0):
        errors.append(f"path speed {speed} outside [0.5, 4.0]")

    hold = params.get("hold_time", 0.0)
    if isinstance(hold, (int, float)) and hold < 0:
        errors.append(f"path hold_time {hold} must be non-negative")

    if shape == "line":
        axis = params.get("axis")
        if axis not in VALID_LINE_AXES:
            errors.append(f"line path axis must be one of {sorted(VALID_LINE_AXES)}, got {axis!r}")
        dist = params.get("distance")
        if not isinstance(dist, (int, float)):
            errors.append("line path requires numeric distance (meters)")
        else:
            dm = _line_distance_meters(float(dist))
            if not (0.02 <= abs(dm) <= 0.5):
                errors.append(f"line distance {dm:.3f}m outside recommended [0.02, 0.5] m")
    elif shape in ("arc", "circle"):
        plane = params.get("plane")
        if normalize_plane(str(plane)) not in {"xy", "yz", "xz"}:
            errors.append(f"arc plane must be xy, yz, or xz/zx, got {plane!r}")
        radius = params.get("radius")
        sweep = params.get("sweep")
        if not isinstance(radius, (int, float)):
            errors.append("arc path requires numeric radius (meters)")
        else:
            rm = _arc_radius_meters(float(radius))
            if not (0.02 <= float(rm) <= 0.35):
                errors.append(f"arc radius {rm:.3f}m outside recommended [0.02, 0.35] m")
        if not isinstance(sweep, (int, float)):
            errors.append("arc path requires numeric sweep (degrees)")
        elif not (15 <= abs(float(sweep)) <= 1080):
            errors.append(f"arc sweep {sweep} outside recommended [15, 1080] degrees")
        direction = params.get("direction", "ccw")
        if direction not in ("cw", "ccw"):
            errors.append(f"arc direction must be cw or ccw, got {direction!r}")
    elif shape is not None:
        errors.append(f"unknown path shape {shape!r}")
    else:
        errors.append("path requires shape line or arc")
    return errors


def _line_distance_meters(distance: float) -> float:
    """Configs may use meters (|d|<=1.5) or legacy large degree-like values."""
    if abs(distance) <= 1.5:
        return float(distance)
    return float(np.sign(distance) * min(0.45, abs(distance) * 0.008))


def _arc_radius_meters(radius: float) -> float:
    if radius <= 1.5:
        return float(radius)
    return float(min(0.35, radius * 0.01))


def path_length_meters(params: dict[str, Any]) -> float:
    shape = params.get("shape")
    if shape == "line":
        return abs(_line_distance_meters(float(params["distance"])))
    sweep = float(params.get("sweep", 360))
    radius = _arc_radius_meters(float(params["radius"]))
    sweep_rad = math.radians(abs(sweep))
    if sweep >= 360:
        return 2.0 * math.pi * radius
    return radius * sweep_rad


def path_duration_seconds(params: dict[str, Any]) -> float:
    speed = max(0.5, float(params.get("speed", 1.0)))
    length_m = max(0.01, path_length_meters(params))
    return length_m / (speed * PATH_EE_M_PER_S_AT_UNIT_SPEED)


def _axis_unit(axis: str) -> np.ndarray:
    v = np.zeros(3, dtype=float)
    v[AXIS_INDEX[axis]] = 1.0
    return v


def _plane_basis(plane: str) -> tuple[np.ndarray, np.ndarray]:
    i, j = _plane_axis_indices(plane)
    e1 = np.zeros(3, dtype=float)
    e2 = np.zeros(3, dtype=float)
    e1[i] = 1.0
    e2[j] = 1.0
    return e1, e2


def line_waypoints(params: dict[str, Any], start_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = str(params["axis"])
    dist_m = _line_distance_meters(float(params["distance"]))
    end_pos = start_pos + _axis_unit(axis) * dist_m
    return start_pos, end_pos


def arc_waypoints(
    params: dict[str, Any],
    start_pos: np.ndarray,
    num_samples: int,
) -> list[np.ndarray]:
    plane = normalize_plane(str(params["plane"]))
    radius_m = _arc_radius_meters(float(params["radius"]))
    sweep_deg = float(params.get("sweep", 360))
    direction = str(params.get("direction", "ccw"))
    sign = -1.0 if direction == "cw" else 1.0

    e1, e2 = _plane_basis(plane)
    center = start_pos - radius_m * e1

    if sweep_deg >= 360:
        total_rad = 2.0 * math.pi
    else:
        total_rad = math.radians(abs(sweep_deg))

    points: list[np.ndarray] = []
    for k in range(num_samples):
        t = k / max(1, num_samples - 1) if num_samples > 1 else 1.0
        theta = sign * t * total_rad
        pos = center + radius_m * (math.cos(theta) * e1 + math.sin(theta) * e2)
        points.append(pos.astype(float))
    return points


def get_eef_quat_wxyz(mujoco_model, mujoco_data, site_id: int) -> np.ndarray:
    """Site orientation as wxyz quaternion (MuJoCo layout)."""
    raw = np.array(mujoco_data.site(site_id).xmat, dtype=float)
    mat = raw.reshape(3, 3) if raw.size == 9 else raw
    quat = np.zeros(4, dtype=float)
    # mju_mat2Quat expects a 1d length-9 rotation matrix (row-major), not 3x3 object
    mujoco.mju_mat2Quat(quat, mat.ravel())
    if float(np.linalg.norm(quat)) < 1e-8:
        quat[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat


def ik_move_eef_to(
    *,
    ik_solver,
    mujoco_model,
    mujoco_data,
    site_id: int,
    target_pos: np.ndarray,
    quat_wxyz: np.ndarray,
    max_iters: int = PATH_IK_MAX_ITERS,
    tol_m: float = PATH_IK_POS_TOL_M,
) -> None:
    axis_angle = T.quat2axisangle(np.roll(quat_wxyz, shift=-1))
    action = np.concatenate([np.asarray(target_pos, dtype=float), axis_angle.reshape(3)])
    dof_ids = ik_solver.dof_ids

    for _ in range(max_iters):
        q_des = ik_solver.solve(action.reshape(1, -1))
        mujoco_data.qpos[dof_ids] = q_des
        mujoco.mj_forward(mujoco_model, mujoco_data)
        err = float(np.linalg.norm(target_pos - mujoco_data.site(site_id).xpos))
        if err < tol_m:
            break


def execute_path_ee_ik(
    generator: Any,
    parameters: dict[str, Any],
    *,
    hz: int,
    frames: list,
    capture_image_fn,
    hesitation_strength: float = 0.0,
) -> None:
    """Append GIF frames for one path step using EE IK (orientation held)."""
    shape = parameters.get("shape")
    hold_time = float(parameters.get("hold_time", 0.0))
    path_speed = float(parameters.get("speed", 1.0))

    mujoco_model = generator.env.sim.model._model
    mujoco_data = generator.env.sim.data._data
    ik = generator.jacobian_calculator.ik_solver
    site_id = ik.site_ids[0]

    mujoco.mj_forward(mujoco_model, mujoco_data)
    start_pos = generator._get_eef_position()
    hold_quat = get_eef_quat_wxyz(mujoco_model, mujoco_data, site_id)

    effective_speed = max(0.5, path_speed * (1.0 - 0.45 * hesitation_strength))
    params_eff = {**parameters, "speed": effective_speed}
    duration = path_duration_seconds(params_eff)
    # Short arcs/lines need enough samples for visible motion and reliable EE IK
    num_frames = max(12, int(duration * hz))

    pre_pause = int(hz * 0.12 * hesitation_strength)
    for _ in range(pre_pause):
        frames.append(Image.fromarray(capture_image_fn()))

    if shape == "line":
        _, end_pos = line_waypoints(parameters, start_pos)
        print(
            f"\n--- Path (line EE IK) ---\n"
            f"axis={parameters.get('axis')}, distance={_line_distance_meters(float(parameters['distance'])):.3f}m, "
            f"speed={effective_speed}, frames={num_frames}"
        )
        for k in range(num_frames):
            t = (k + 1) / num_frames
            target = start_pos * (1.0 - t) + end_pos * t
            ik_move_eef_to(
                ik_solver=ik,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                site_id=site_id,
                target_pos=target,
                quat_wxyz=hold_quat,
            )
            generator._set_joint_positions(generator._get_joint_positions())
            frames.append(Image.fromarray(capture_image_fn()))
    elif shape in ("arc", "circle"):
        plane = normalize_plane(str(parameters.get("plane", "xy")))
        sweep = float(parameters.get("sweep", 360))
        radius_m = _arc_radius_meters(float(parameters["radius"]))
        mode = "full circle (EE on circumference)" if sweep >= 360 else "arc end anchored at current EE"
        print(
            f"\n--- Path (arc EE IK) ---\n"
            f"plane={plane}, radius={radius_m:.3f}m, sweep={sweep}°, {mode}, "
            f"speed={effective_speed}, frames={num_frames}"
        )
        targets = arc_waypoints(parameters, start_pos, num_frames)
        for target in targets:
            ik_move_eef_to(
                ik_solver=ik,
                mujoco_model=mujoco_model,
                mujoco_data=mujoco_data,
                site_id=site_id,
                target_pos=target,
                quat_wxyz=hold_quat,
            )
            generator._set_joint_positions(generator._get_joint_positions())
            frames.append(Image.fromarray(capture_image_fn()))
    else:
        raise ValueError(f"Unsupported EE path shape: {shape}")

    if hold_time > 0:
        n_hold = int(hold_time * hz)
        for _ in range(n_hold):
            frames.append(Image.fromarray(capture_image_fn()))
        print(f"  Captured {n_hold} hold frames (hold_time: {hold_time}s)")
