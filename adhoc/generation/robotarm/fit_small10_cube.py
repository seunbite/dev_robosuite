#!/usr/bin/env python3
"""Fit small10 dump EE trajectories into the ±15 cm workspace cube and re-render GIFs.

    python adhoc/generation/robotarm/fit_small10_cube.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from export_droid_small10 import SMALL10, OUT, _traj_stats, rebuild_gallery  # noqa: E402
from render_fr3_neutral import _env, _grab  # noqa: E402
from replay_cube import _HALF_M, _clip_q, cube_center  # noqa: E402

_MARGIN = 0.98
# FR3 site_xmat from Panda pose-bank dir=front (approach +X / camera).
# Bank horiz/vert labels are swapped vs motion-code grip= on Franka.
_R_BANK_FRONT_HORIZ = np.array(
    [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
)
_R_BANK_FRONT_VERT = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
)
_Q_FRONT_SEED = np.deg2rad([41.0, 39.3, 143.1, -123.7, 67.7, 12.3, 45.8])


def _site_id(env) -> int:
    robot = env.robots[0]
    arm = "right" if "right" in robot.eef_site_id else next(iter(robot.eef_site_id))
    return int(robot.eef_site_id[arm])


def _dof_ids(env) -> np.ndarray:
    names = list(env.robots[0].robot_model.joints)[:7]
    model = env.sim.model
    return np.array([int(model.jnt_dofadr[model.joint_name2id(n)]) for n in names])


def _mj(env):
    sim = env.sim
    model = sim.model._model if hasattr(sim.model, "_model") else sim.model
    data = sim.data._data if hasattr(sim.data, "_data") else sim.data
    return model, data


def _set_q(env, q: np.ndarray) -> None:
    env.robots[0].set_robot_joint_positions(_clip_q(q))
    env.sim.forward()


def fk_ee_R(env, q: np.ndarray, sid: int) -> tuple[np.ndarray, np.ndarray]:
    _set_q(env, q)
    p = np.asarray(env.sim.data.site_xpos[sid], dtype=np.float64).copy()
    R = np.asarray(env.sim.data.site_xmat[sid], dtype=np.float64).reshape(3, 3).copy()
    return p, R


def _rot_err(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    R = R_des @ R_cur.T
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def ik_to(
    env,
    target: np.ndarray,
    R_des: np.ndarray,
    dof_ids,
    sid: int,
    q_bias: np.ndarray,
    iters: int = 160,
) -> tuple[np.ndarray, float]:
    """Position + locked front horiz/vert orientation."""
    model, data = _mj(env)
    target = np.asarray(target, dtype=np.float64)
    q = _clip_q(q_bias)
    _set_q(env, q)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    w_ori = 0.9
    err = 1.0
    for _ in range(iters):
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        Jp = jacp[:, dof_ids]
        Jr = jacr[:, dof_ids]
        xpos = np.asarray(data.site_xpos[sid], dtype=np.float64)
        R_cur = np.asarray(data.site_xmat[sid], dtype=np.float64).reshape(3, 3)
        e_p = target - xpos
        e_r = _rot_err(R_des, R_cur)
        err = float(np.linalg.norm(e_p))
        if err < 0.005 and float(np.linalg.norm(e_r)) < 0.08:
            break
        J = np.vstack([Jp, w_ori * Jr])
        e = np.concatenate([e_p, w_ori * e_r])
        jj = J @ J.T + 2e-3 * np.eye(6)
        dq = J.T @ np.linalg.solve(jj, e)
        q = _clip_q(q + np.clip(dq, -0.18, 0.18) + 0.02 * (q_bias - q))
        _set_q(env, q)
    return q, err


def ik_translate(
    env,
    p_from: np.ndarray,
    p_to: np.ndarray,
    R_des: np.ndarray,
    dof_ids,
    sid: int,
    q_bias: np.ndarray,
    step_m: float = 0.03,
) -> tuple[np.ndarray, float]:
    """Translate EE from p_from to p_to in small steps, holding orientation."""
    p_from = np.asarray(p_from, dtype=np.float64)
    p_to = np.asarray(p_to, dtype=np.float64)
    travel = float(np.linalg.norm(p_to - p_from))
    n = max(1, int(np.ceil(travel / step_m)))
    q = _clip_q(q_bias)
    err = 0.0
    for i in range(1, n + 1):
        p = p_from + (i / n) * (p_to - p_from)
        q, err = ik_to(env, p, R_des, dof_ids, sid, q, iters=80)
    q, err = ik_to(env, p_to, R_des, dof_ids, sid, q, iters=160)
    return q, err


def _closest_in_cube(p: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    lo, hi = center - half, center + half
    return np.clip(np.asarray(p, dtype=np.float64), lo, hi)


def _parse_only() -> set[int] | None:
    if "--only" not in sys.argv:
        return None
    i = sys.argv.index("--only")
    return {int(x) for x in sys.argv[i + 1].split(",")}


def _grip_token(code: str) -> str:
    m = re.search(r"grip\s*=\s*([a-z]+)", str(code or ""), re.I)
    raw = (m.group(1).lower() if m else "vert")
    if raw.startswith("horiz"):
        return "horiz"
    return "vert"


def _first_dir(code: str) -> str:
    m = re.search(r"dir\s*=\s*([a-z]+)", str(code or ""), re.I)
    return (m.group(1).lower() if m else "front")


def _R_front_grip(code: str) -> np.ndarray:
    # Motion-code grip=horiz/vert after FR3.apply_bank_gripper_clock (−45° q7).
    if _grip_token(code) == "horiz":
        return _R_BANK_FRONT_VERT
    return _R_BANK_FRONT_HORIZ


def _path_x_dist(code: str) -> float | None:
    m = re.search(
        r"path\s*\(\s*ee\s*,\s*line\s*,\s*axis\s*=\s*x\s*,\s*dist\s*=\s*([+-]?[0-9]*\.?[0-9]+)",
        str(code or ""),
        re.I,
    )
    return float(m.group(1)) if m else None


def _ik_along(
    env,
    targets: np.ndarray,
    R_des: np.ndarray,
    dof_ids,
    sid: int,
    q0: np.ndarray,
) -> tuple[np.ndarray, list[float]]:
    q_new = np.zeros((len(targets), 7), dtype=np.float64)
    errs: list[float] = []
    q_prev = _clip_q(q0)
    for i in range(len(targets)):
        q_try, err = ik_to(env, targets[i], R_des, dof_ids, sid, q_prev)
        q_new[i] = q_try
        q_prev = q_try
        errs.append(err)
    return q_new, errs


def _dump_paths(idx: int, cue: str, src: str) -> tuple[Path, Path, Path]:
    stem = f"{idx:02d}_{cue}"
    dest = OUT / stem / src
    return dest / f"{stem}_{src}.npz", dest / f"{stem}_{src}.json", dest / "preview.gif"


def _write_gif(env, qs: np.ndarray, path: Path, hz: float) -> None:
    frames: list[Image.Image] = []
    for q in qs:
        _set_q(env, q)
        frames.append(_grab(env))
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / max(hz, 1.0))),
        loop=0,
    )


def _fit_one(
    env,
    dof_ids,
    sid,
    npz_path: Path,
    json_path: Path,
    gif_path: Path,
    center: np.ndarray,
    code: str = "",
) -> dict:
    z = np.load(npz_path)
    q = np.asarray(z["joints_rad"], dtype=np.float64)
    hz = float(np.asarray(z["hz"]).reshape(-1)[0]) if "hz" in z.files else 15.0
    g = (
        np.asarray(z["gripper"], dtype=np.float64)
        if "gripper" in z.files
        else np.zeros(len(q), dtype=np.float64)
    )
    p0, R0 = fk_ee_R(env, q[0], sid)
    p_cube = _closest_in_cube(p0, center, _HALF_M)
    shift = p_cube - p0
    use_front = _first_dir(code) == "front"
    R_des = _R_front_grip(code) if use_front else R0
    ori_note = f"front+{_grip_token(code)}" if use_front else f"keep {_first_dir(code)}"
    print(
        f"  start EE {np.round(p0, 3).tolist()} → cube {np.round(p_cube, 3).tolist()} "
        f"(shift {np.round(100 * shift, 1).tolist()} cm); {ori_note}; replay original Δq"
    )
    if use_front:
        p_seed, _ = fk_ee_R(env, _Q_FRONT_SEED, sid)
        q0, err0 = ik_translate(env, p_seed, p_cube, R_des, dof_ids, sid, _clip_q(_Q_FRONT_SEED))
        q0b, errb = ik_to(env, p_cube, R_des, dof_ids, sid, _clip_q(_Q_FRONT_SEED), iters=200)
        if errb < err0:
            q0, err0 = q0b, errb
    else:
        q0, err0 = ik_translate(env, p0, p_cube, R_des, dof_ids, sid, _clip_q(q[0]))
        q0b, errb = ik_to(env, p_cube, R_des, dof_ids, sid, _clip_q(q[0]), iters=200)
        if errb < err0:
            q0, err0 = q0b, errb
    _, R_chk = fk_ee_R(env, q0, sid)
    p_start, _ = fk_ee_R(env, q0, sid)
    print(f"  start approach z {np.round(R_chk[:, 2], 3).tolist()}  (front is +X)")
    ee_orig = np.stack([fk_ee_R(env, q[i], sid)[0] for i in range(len(q))], axis=0)
    dee = ee_orig - ee_orig[0]
    dist_x = _path_x_dist(code)
    q_joint = np.stack([_clip_q(q0 + (q[i] - q[0])) for i in range(len(q))], axis=0)
    orig_span = float(np.max(np.rad2deg(q.max(0) - q.min(0))))
    joint_span = float(np.max(np.rad2deg(q_joint.max(0) - q_joint.min(0))))
    if dist_x is not None and dist_x > 0:
        x = dee[:, 0].copy()
        x = x - x[0]
        if float(np.mean(x)) < 0:
            x = -x
        span = float(np.max(x) - np.min(x))
        if span < 0.02:
            x = np.linspace(0.0, dist_x, len(q))
        else:
            x = x - np.min(x)
            x = dist_x * x / max(float(np.max(x)), 1e-9)
        dee = np.zeros_like(dee)
        dee[:, 0] = x
        targets = p_start + dee
        q_new, cart_errs = _ik_along(env, targets, R_des, dof_ids, sid, q0)
        errs = [err0] + cart_errs
        print(f"  path +X {100 * dist_x:.0f} cm from start (was going backward in generate)")
    elif orig_span > 3.0 and joint_span < 0.4 * orig_span:
        targets = p_start + dee
        q_new, cart_errs = _ik_along(env, targets, R_des, dof_ids, sid, q0)
        errs = [err0] + cart_errs
        print(f"  joint replay clipped ({joint_span:.1f}° vs orig {orig_span:.1f}°); Cartesian ΔEE")
    else:
        q_new = q_joint
        errs = [err0]
    ee_out = np.stack([fk_ee_R(env, q_new[i], sid)[0] for i in range(len(q))], axis=0)
    scale = float(np.linalg.norm(shift))
    pose_frame = int(np.asarray(z["pose_frame"]).reshape(-1)[0]) if "pose_frame" in z.files else 0
    np.savez(
        npz_path,
        hz=np.array(hz),
        joints_rad=q_new,
        gripper=g,
        ee_xyz=ee_out,
        robot=np.array(str(np.asarray(z["robot"]).reshape(-1)[0]) if "robot" in z.files else "FR3"),
        pose_frame=np.array(pose_frame, dtype=np.int32),
        cube_scale=np.array(scale),
        cube_center=center,
    )
    if json_path.is_file():
        data = json.loads(json_path.read_text())
        data["joints_rad"] = q_new.tolist()
        data["ee_xyz"] = ee_out.tolist()
        data["n_frames"] = int(len(q_new))
        data["duration_s"] = float(len(q_new) / max(hz, 1.0))
        data["cube_fit"] = {
            "center_m": center.tolist(),
            "half_m": _HALF_M,
            "start_shift_m": shift.tolist(),
            "ik_err_cm_max": float(100.0 * np.max(errs)),
        }
        json_path.write_text(json.dumps(data, indent=2))
    _write_gif(env, q_new, gif_path, hz)
    p0_out, R_out = fk_ee_R(env, q_new[0], sid)
    start_inside = np.all((p0_out >= center - _HALF_M - 0.01) & (p0_out <= center + _HALF_M + 0.01))
    print(
        f"  start inside={bool(start_inside)}  IK {100 * err0:.1f} cm  "
        f"Δq kept  EE span cm {np.round((ee_out.max(0) - ee_out.min(0)) * 100, 1).tolist()}  "
        f"{gif_path.name}"
    )
    return {
        "scale": scale,
        "ik_err_max_cm": float(100.0 * np.max(errs)),
        "ee_span_cm": ((ee_out.max(0) - ee_out.min(0)) * 100).tolist(),
    }


def main() -> None:
    only = _parse_only()
    index = {}
    idx_path = OUT / "index.json"
    if idx_path.is_file():
        for p in json.loads(idx_path.read_text()):
            for src in ("human", "llm"):
                index[(p["idx"], src)] = (p.get(src) or {}).get("code") or ""

    center = cube_center()
    print("cube center", np.round(center, 3).tolist(), "half", _HALF_M)
    env = _env()
    sid = _site_id(env)
    dof_ids = _dof_ids(env)
    pairs: list[dict] = []
    for idx, cue in SMALL10:
        if only is not None and idx not in only:
            continue
        pair: dict = {"idx": idx, "cue": cue}
        for src in ("human", "llm"):
            npz, js, gif = _dump_paths(idx, cue, src)
            print(f"\n==== {idx:02d} {cue} {src} ====")
            rec: dict = {
                "idx": idx,
                "cue": cue,
                "src": src,
                "code": index.get((idx, src), ""),
                "gif": str(gif),
                "npz": str(npz),
                "error": "",
            }
            if js.is_file() and not rec["code"]:
                data = json.loads(js.read_text())
                rec["code"] = data.get("code") or data.get("source_code") or ""
            if not npz.is_file():
                rec["error"] = f"missing {npz}"
                pair[src] = rec
                continue
            try:
                rec["cube"] = _fit_one(env, dof_ids, sid, npz, js, gif, center, rec["code"])
                rec["stats"] = _traj_stats(npz)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)
                print("  FAIL", exc)
            pair[src] = rec
        pairs.append(pair)
    html_path = rebuild_gallery({(p["idx"], src): (p.get(src) or {}).get("code") or "" for p in pairs for src in ("human", "llm")})
    print("gallery", html_path)


if __name__ == "__main__":
    main()
