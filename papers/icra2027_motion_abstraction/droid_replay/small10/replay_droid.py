#!/usr/bin/env python3
"""Select and replay a bundled small10 cue on a DROID FR3.

    python replay_droid.py --list
    python replay_droid.py --write-gif-plan
    python replay_droid.py --idx 2 --author LLM --dry-run --slow 1
    python replay_droid.py --idx 2 --author LLM --play --slow 0.1 --return-slow 0.3

Single-trajectory NPZ/JSON files remain supported:

    python replay_droid.py --traj 02_shake_head_no_llm.npz --play --slow 0.1

``--play`` ends with a return to ``neutral_joints.json`` (``--return-slow``,
default 0.3) so the next ``RobotEnv()`` reset is already at that pose.
Approach and dump playback stay on ``--slow`` (default 0.1).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from droid_traj import HOME as _HOME
from droid_traj import GIF_HOLD_S, GIF_SLOW
from droid_traj import approach_qs, cue_start_index, plan_from_start, strip_to_cue

_DEFAULT_BUNDLE = Path(__file__).resolve().with_name("small10_all.npz")
_DEFAULT_NEUTRAL = Path(__file__).resolve().with_name("neutral_joints.json")
_DEFAULT_GIF_PLAN = Path(__file__).resolve().with_name("gif_from_neutral.npz")
_FR3_Q_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_FR3_Q_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
_QDOT_AT_SLOW_1 = np.deg2rad(30.0)
_AT_POSE_DEG = 4.0
_NUDGE_DEG = 12.0


def _clip_q(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float64).reshape(-1)[:7], _FR3_Q_LO, _FR3_Q_HI)


def _far(a: np.ndarray, b: np.ndarray, deg: float = 6.0) -> bool:
    return float(np.max(np.abs(_clip_q(a) - _clip_q(b)))) > np.deg2rad(deg)


def _max_abs_dq(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(_clip_q(a) - _clip_q(b))))


def _smoothstep01(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def labeled_pose_path(q0: np.ndarray, q_goal: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Current joints → cue pose. Same single segment the GIF approach used."""
    q0 = _clip_q(q0)
    goal = _clip_q(q_goal)
    if not _far(q0, goal, _AT_POSE_DEG):
        return []
    return [("cue pose", goal)]


def plan_pose_path(q0: np.ndarray, q_goal: np.ndarray) -> list[np.ndarray]:
    return [q for _name, q in labeled_pose_path(q0, q_goal)]


def _bundle_text(z, key: str, n: int, default: str = "") -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=str)
    return np.asarray(z[key]).astype(str).reshape(-1)


def _is_bundle(z) -> bool:
    if not {"idx", "author", "offsets", "joints_rad"}.issubset(z.files):
        return False
    if "format_name" not in z.files:
        return True
    return str(np.asarray(z["format_name"]).reshape(-1)[0]) == "droid_small10_bundle"


def _print_bundle(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"trajectory bundle not found: {path}")
    z = np.load(path, allow_pickle=False)
    if not _is_bundle(z):
        raise SystemExit(f"--list requires a small10 bundle, got: {path}")
    idxs = np.asarray(z["idx"], dtype=np.int32).reshape(-1)
    authors = _bundle_text(z, "author", len(idxs))
    cues = _bundle_text(z, "cue", len(idxs))
    offsets = np.asarray(z["offsets"], dtype=np.int64).reshape(-1)
    hz = np.asarray(z["hz"], dtype=np.float64).reshape(-1)
    print(f"bundle={path} trajectories={len(idxs)}")
    for i, (idx, author, cue) in enumerate(zip(idxs, authors, cues)):
        frames = int(offsets[i + 1] - offsets[i])
        rate = float(hz[i] if len(hz) > 1 else hz[0])
        print(f"  idx={idx:>2} author={author.upper():<5} frames={frames:>3} hz={rate:g} cue={cue}")


def _load_bundle(z, path: Path, idx: int | None, author: str | None) -> dict:
    if idx is None or author is None:
        raise SystemExit(
            f"{path.name} contains multiple trajectories; pass both --idx and --author "
            "(or use --list)"
        )
    idxs = np.asarray(z["idx"], dtype=np.int32).reshape(-1)
    authors = np.char.lower(_bundle_text(z, "author", len(idxs)))
    matches = np.flatnonzero((idxs == int(idx)) & (authors == author.lower()))
    if len(matches) != 1:
        available = sorted({int(v) for v in idxs})
        raise SystemExit(
            f"no unique trajectory for idx={idx} author={author}; available idx={available}"
        )
    k = int(matches[0])
    offsets = np.asarray(z["offsets"], dtype=np.int64).reshape(-1)
    start, stop = int(offsets[k]), int(offsets[k + 1])
    q = np.asarray(z["joints_rad"][start:stop], dtype=np.float64)
    if "gripper" in z.files:
        g = np.asarray(z["gripper"][start:stop], dtype=np.float64)
    else:
        g = np.zeros(len(q), dtype=np.float64)
    hz_all = np.asarray(z["hz"], dtype=np.float64).reshape(-1)
    hz = float(hz_all[k] if len(hz_all) > 1 else hz_all[0])
    cues = _bundle_text(z, "cue", len(idxs))
    robots = _bundle_text(z, "robot", len(idxs), "FR3")
    pose_all = (
        np.asarray(z["pose_frame"], dtype=np.int32).reshape(-1)
        if "pose_frame" in z.files
        else np.zeros(len(idxs), dtype=np.int32)
    )
    return {
        "hz": hz,
        "joints_rad": q,
        "gripper": g,
        "cue": str(cues[k]),
        "robot": str(robots[k]),
        "hardware": {},
        "pose_frame": int(pose_all[k]),
        "idx": int(idxs[k]),
        "author": str(authors[k]),
    }


def _load(path: Path, *, idx: int | None = None, author: str | None = None) -> dict:
    if not path.is_file():
        raise SystemExit(f"trajectory file not found: {path}")
    if path.suffix == ".json":
        if idx is not None or author is not None:
            raise SystemExit("--idx/--author select from a bundle, not a single JSON trajectory")
        data = json.loads(path.read_text())
        stored = data.get("pose_frame")
        q, g, _ee, pose_frame = strip_to_cue(
            np.asarray(data["joints_rad"], dtype=np.float64),
            np.asarray(data["gripper"], dtype=np.float64),
            None,
            None if stored is None else int(stored),
        )
        return {
            "hz": float(data["hz"]),
            "joints_rad": q,
            "gripper": g,
            "cue": data.get("cue", path.stem),
            "robot": data.get("robot", "FR3"),
            "hardware": data.get("hardware") or {},
            "pose_frame": pose_frame,
        }
    z = np.load(path, allow_pickle=False)
    if _is_bundle(z):
        return _load_bundle(z, path, idx, author)
    if idx is not None or author is not None:
        raise SystemExit("--idx/--author require a small10 bundle")
    hz = float(np.asarray(z["hz"]).reshape(-1)[0])
    robot = "FR3"
    if "robot" in z.files:
        robot = str(np.asarray(z["robot"]).reshape(-1)[0])
    stored = None
    if "pose_frame" in z.files:
        stored = int(np.asarray(z["pose_frame"]).reshape(-1)[0])
    q, g, _ee, pose_frame = strip_to_cue(
        np.asarray(z["joints_rad"], dtype=np.float64),
        np.asarray(z["gripper"], dtype=np.float64) if "gripper" in z.files else None,
        None,
        stored,
    )
    return {
        "hz": hz,
        "joints_rad": q,
        "gripper": g,
        "cue": path.stem,
        "robot": robot,
        "hardware": {},
        "pose_frame": pose_frame,
    }


def _joints_from_obs(obs: dict) -> np.ndarray:
    if obs is None:
        raise RuntimeError("empty observation")
    for key in ("joint_positions", "joint_position"):
        if key in obs:
            return np.asarray(obs[key], dtype=np.float64).reshape(-1)[:7]
    rs = obs.get("robot_state") or obs.get("observation") or {}
    for key in ("joint_positions", "joint_position"):
        if key in rs:
            return np.asarray(rs[key], dtype=np.float64).reshape(-1)[:7]
    raise RuntimeError(f"could not find 7-DoF joints in obs keys={list(obs)}")


def _build_env(control_hz: float):
    from droid.robot_env import RobotEnv  # noqa: WPS433

    # DROID defaults to do_reset=True, which blocking-moves to its own
    # reset_joints (not desk-neutral). That looks identical to check_droid_env
    # connecting and never reaches the cue. Stay at the current pose.
    print("RobotEnv(joint_position, do_reset=False) — will not yank to DROID home", flush=True)
    kwargs = dict(
        action_space="joint_position",
        gripper_action_space="position",
        do_reset=False,
    )
    try:
        env = RobotEnv(control_hz=int(round(control_hz)), **kwargs)
    except TypeError:
        kwargs.pop("do_reset", None)
        try:
            env = RobotEnv(control_hz=int(round(control_hz)), **kwargs)
        except TypeError:
            env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    print("  connected", type(env), flush=True)
    return env


def _current_q(env) -> np.ndarray:
    if hasattr(env, "get_observation"):
        obs = env.get_observation()
    elif hasattr(env, "get_state"):
        obs = env.get_state()
    else:
        raise RuntimeError("no get_observation/get_state; refusing env.reset() (it can yank the arm)")
    if isinstance(obs, tuple):
        obs = obs[0]
    return _clip_q(_joints_from_obs(obs))


def _control_iface(env):
    """NUC path is ServerInterface; local path is FrankaRobot wrapping polymetis."""
    robot = getattr(env, "_robot", None)
    if robot is None:
        return None
    if hasattr(robot, "start_joint_impedance"):
        return robot
    inner = getattr(robot, "_robot", None)
    return inner if inner is not None else robot


def _start_joint_policy(env) -> None:
    """DROID joint_position still starts *cartesian* impedance.

    That controller holds the Cartesian pose and hunts in Z (gravity) —
    the arm bobs in place while ``update_desired_joint_positions`` RPCs
    fail and are swallowed. Joint impedance is what tracks a joint cue.
    On the laptop ``env._robot`` is ServerInterface, not polymetis; the
    NUC RPC must expose ``start_joint_impedance``.
    """
    iface = _control_iface(env)
    if iface is None or not hasattr(iface, "start_joint_impedance"):
        print("  no start_joint_impedance RPC; using env.update_robot blocking", flush=True)
        return
    try:
        if hasattr(iface, "is_running_policy") and iface.is_running_policy():
            iface.terminate_current_policy()
    except Exception as exc:
        print(f"  terminate_current_policy: {exc}", flush=True)
    iface.start_joint_impedance()
    t0 = time.time()
    while hasattr(iface, "is_running_policy") and not iface.is_running_policy():
        if time.time() - t0 > 5.0:
            break
        time.sleep(0.02)
    running = bool(iface.is_running_policy()) if hasattr(iface, "is_running_policy") else False
    print(f"  joint impedance running={running}", flush=True)


def _send(env, q: np.ndarray, gripper: float, *, blocking: bool = False) -> None:
    action = np.concatenate([_clip_q(q), [float(gripper)]])
    if hasattr(env, "update_robot"):
        env.update_robot(
            action,
            action_space="joint_position",
            gripper_action_space="position",
            blocking=blocking,
        )
        return
    env.step(action)


def _lerp_to(
    env,
    q_goal: np.ndarray,
    gripper: float,
    *,
    slow: float,
    hz: float,
    q0: np.ndarray | None = None,
    min_s: float = 0.4,
) -> np.ndarray:
    q0 = _clip_q(q0 if q0 is not None else _current_q(env))
    q_goal = _clip_q(q_goal)
    qs, meta = approach_qs(q0, q_goal, hz=hz, slow=slow, min_s=min_s)
    dt = 1.0 / hz
    print(
        f"  smoothstep max_|Δq|={meta['max_dq_deg']:.1f}°  "
        f"{meta['qdot_deg_s']:.1f}°/s  {meta['duration_s']:.1f}s  ({meta['n']} steps)"
    )
    q = q0
    for q in qs:
        t0 = time.time()
        _send(env, q, gripper, blocking=not _streaming(env))
        leftover = dt - (time.time() - t0)
        if leftover > 0:
            time.sleep(leftover)
    return q_goal


def _streaming(env) -> bool:
    iface = _control_iface(env)
    try:
        return bool(iface is not None and iface.is_running_policy())
    except Exception:
        return False


def _goto_pose(env, q_goal: np.ndarray, gripper: float, *, slow: float, hz: float) -> np.ndarray:
    q0 = _current_q(env)
    goal = _clip_q(q_goal)
    max_dq = _max_abs_dq(q0, goal)
    max_deg = float(np.degrees(max_dq))
    print(f"  start_deg {np.round(np.degrees(q0), 1).tolist()}")
    print(f"  goal_deg  {np.round(np.degrees(goal), 1).tolist()}")
    if max_deg <= _AT_POSE_DEG:
        print(f"already at cue pose (max |Δq|={max_deg:.1f}°); skip approach")
        return goal
    if max_deg <= _NUDGE_DEG:
        print(f"pose path (nudge → cue)  max |Δq|={max_deg:.1f}°")
        _lerp_to(env, goal, gripper, slow=slow, hz=hz, q0=q0, min_s=0.15)
        return goal
    print(f"pose path (current → cue, GIF lerp)  max |Δq|={max_deg:.1f}°")
    _lerp_to(env, goal, gripper, slow=slow, hz=hz, q0=q0)
    return goal


def _load_start(path: Path | None) -> tuple[np.ndarray | None, float, str]:
    if path is not None and path.is_file():
        data = json.loads(path.read_text())
        g = float(data.get("gripper", 0.0))
        return _clip_q(np.asarray(data["q_rad"], dtype=np.float64)), g, str(path)
    return None, 0.0, "no saved_neutral.json"


def _print_plan(plan: dict) -> None:
    print(f"  start_deg {np.round(np.degrees(plan['q_start']), 2).tolist()}")
    print(f"  goal_deg  {np.round(np.degrees(plan['q_pose']), 2).tolist()}")
    print(
        f"  approach smoothstep max_|Δq|={plan['max_dq_deg']:.1f}°  "
        f"{plan['qdot_deg_s']:.1f}°/s  {plan['duration_s']:.1f}s  ({plan['n']} steps)"
    )
    print(f"  hold={len(plan['hold'])} approach={len(plan['approach'])} play={len(plan['play'])} total={len(plan['joints_rad'])}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = int(round(frac * max(len(plan["approach"]) - 1, 0)))
        q = plan["approach"][i] if len(plan["approach"]) else plan["q_pose"]
        print(f"  approach u={frac:.2f}  {np.round(np.degrees(q), 2).tolist()}")


def _write_plan(path: Path, plan: dict, traj: dict) -> None:
    path = Path(path)
    payload = {
        "cue": traj.get("cue"),
        "idx": traj.get("idx"),
        "author": traj.get("author"),
        "slow": plan["slow"],
        "hz": plan["hz"],
        "hold_s": plan["hold_s"],
        "max_dq_deg": plan["max_dq_deg"],
        "q_start_deg": np.degrees(plan["q_start"]).tolist(),
        "q_pose_deg": np.degrees(plan["q_pose"]).tolist(),
        "joints_rad": plan["joints_rad"].tolist(),
        "n_hold": int(len(plan["hold"])),
        "n_approach": int(len(plan["approach"])),
        "n_play": int(len(plan["play"])),
    }
    if path.suffix == ".npz":
        np.savez(
            path,
            joints_rad=plan["joints_rad"],
            q_start=plan["q_start"],
            q_pose=plan["q_pose"],
            n_hold=np.array(len(plan["hold"])),
            n_approach=np.array(len(plan["approach"])),
            n_play=np.array(len(plan["play"])),
            hz=np.array(plan["hz"]),
            slow=np.array(plan["slow"]),
        )
    else:
        path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote plan", path)


def _compare_to_gif(plan: dict, gif_path: Path, *, idx: int | None, author: str | None) -> None:
    if not gif_path.is_file():
        print(f"compare: no GIF joint file {gif_path}")
        return
    z = np.load(gif_path, allow_pickle=False)
    if "idx" in z.files and idx is not None and author is not None:
        idxs = np.asarray(z["idx"]).reshape(-1)
        authors = np.char.lower(np.asarray(z["author"]).astype(str).reshape(-1))
        hits = np.flatnonzero((idxs == int(idx)) & (authors == author.lower()))
        if len(hits) != 1:
            print(f"compare: no gif plan for idx={idx} author={author}")
            return
        k = int(hits[0])
        off = np.asarray(z["offsets"], dtype=np.int64).reshape(-1)
        gif_q = np.asarray(z["joints_rad"][int(off[k]) : int(off[k + 1])], dtype=np.float64)
        gif_start = np.asarray(z["q_start"][k], dtype=np.float64)
        n_hold = int(np.asarray(z["n_hold"]).reshape(-1)[k])
        n_app = int(np.asarray(z["n_approach"]).reshape(-1)[k])
        n_play = int(np.asarray(z["n_play"]).reshape(-1)[k])
    else:
        gif_q = np.asarray(z["joints_rad"], dtype=np.float64)
        gif_start = np.asarray(z["q_start"], dtype=np.float64).reshape(-1)[:7]
        n_hold = int(np.asarray(z["n_hold"]).reshape(-1)[0])
        n_app = int(np.asarray(z["n_approach"]).reshape(-1)[0])
        n_play = int(np.asarray(z["n_play"]).reshape(-1)[0])
    gif_pose = gif_q[n_hold + n_app - 1] if n_app else gif_q[n_hold]
    start_err = float(np.degrees(_max_abs_dq(plan["q_start"], gif_start)))
    pose_err = float(np.degrees(_max_abs_dq(plan["q_pose"], gif_pose)))
    play_err = float(np.degrees(np.max(np.abs(plan["play"] - gif_q[n_hold + n_app : n_hold + n_app + n_play]))))
    samples = np.linspace(0.0, 1.0, 11)
    geo = []
    for u in samples:
        s = _smoothstep01(u)
        a = (1.0 - s) * plan["q_start"] + s * plan["q_pose"]
        b = (1.0 - s) * gif_start + s * gif_pose
        geo.append(float(np.degrees(np.max(np.abs(a - b)))))
    geo_err = max(geo) if geo else 0.0
    ok = start_err < 0.05 and pose_err < 0.05 and play_err < 0.05 and geo_err < 0.05
    print(
        f"compare vs {gif_path.name}: start {start_err:.3f}°  "
        f"pose {pose_err:.3f}°  play {play_err:.3f}°  "
        f"approach_geometry {geo_err:.3f}°  {'MATCH' if ok else 'DIFF'}"
    )
    if abs(plan["slow"] - GIF_SLOW) < 1e-9 and abs(plan["hold_s"] - GIF_HOLD_S) < 1e-9:
        if gif_q.shape == plan["joints_rad"].shape:
            frame_err = float(np.degrees(np.max(np.abs(plan["joints_rad"] - gif_q))))
            print(f"  frame-wise vs GIF (slow=1, hold={GIF_HOLD_S}s): {frame_err:.3f}°")
        else:
            print(
                f"  frame counts differ plan={plan['joints_rad'].shape[0]} gif={len(gif_q)} "
                "(timing/hold mismatch; geometry check above is the path test)"
            )


def _write_gif_plans(bundle: Path, start_path: Path, out: Path) -> None:
    q_start, _, _ = _load_start(start_path)
    if q_start is None:
        raise SystemExit(f"need start joints JSON: {start_path}")
    z = np.load(bundle, allow_pickle=False)
    idxs = np.asarray(z["idx"], dtype=np.int32).reshape(-1)
    authors = np.asarray(z["author"]).astype(str).reshape(-1)
    plans = []
    for idx, author in zip(idxs, authors):
        traj = _load(bundle, idx=int(idx), author=str(author).lower())
        plan = plan_from_start(
            q_start,
            traj["joints_rad"],
            hz=float(traj["hz"]),
            slow=GIF_SLOW,
            hold_s=GIF_HOLD_S,
        )
        plan["idx"] = int(idx)
        plan["author"] = str(author).lower()
        plan["cue"] = str(traj.get("cue") or "")
        plans.append(plan)
    lengths = np.asarray([len(p["joints_rad"]) for p in plans], dtype=np.int64)
    offsets = np.concatenate([np.array([0], dtype=np.int64), np.cumsum(lengths)])
    np.savez_compressed(
        out,
        format_name=np.array(["droid_gif_from_neutral"]),
        idx=np.asarray([p["idx"] for p in plans], dtype=np.int32),
        author=np.asarray([p["author"] for p in plans]),
        cue=np.asarray([p["cue"] for p in plans]),
        offsets=offsets,
        joints_rad=np.concatenate([p["joints_rad"] for p in plans], axis=0),
        q_start=np.stack([p["q_start"] for p in plans], axis=0),
        n_hold=np.asarray([len(p["hold"]) for p in plans], dtype=np.int32),
        n_approach=np.asarray([len(p["approach"]) for p in plans], dtype=np.int32),
        n_play=np.asarray([len(p["play"]) for p in plans], dtype=np.int32),
        hz=np.asarray([p["hz"] for p in plans], dtype=np.float64),
    )
    print(f"wrote {out} ({len(plans)} GIF joint plans from {start_path.name})")


def replay(
    traj: dict,
    *,
    slow: float,
    return_slow: float,
    do_pose: bool,
    do_play: bool,
    dry_run: bool,
    start_q: np.ndarray | None = None,
    start_g: float = 0.0,
    start_label: str = "sim-home",
    write_plan: Path | None = None,
    compare: Path | None = None,
    hold_s: float = 0.0,
) -> None:
    q = traj["joints_rad"]
    g = traj["gripper"]
    if q.ndim != 2 or q.shape[1] < 7:
        raise SystemExit(f"expected (N, >=7) joints, got {q.shape}")
    hz_src = float(traj["hz"])
    pose_i = int(traj.get("pose_frame") or 0)
    pose_i = cue_start_index(q, pose_i)
    q_pose = q[pose_i, :7]
    g0 = float(g[pose_i]) if pose_i < len(g) else (float(g[0]) if len(g) else 0.0)
    play_n = q.shape[0] - pose_i
    play_s = play_n / max(hz_src * float(slow), 1e-6)
    print(
        f"cue={traj['cue']} robot={traj.get('robot')} frames={q.shape[0]} "
        f"pose_frame={pose_i}/{q.shape[0]} dump_hz={hz_src} "
        f"slow={slow} return_slow={return_slow}"
    )
    print(f"  q_pose_deg={np.round(np.degrees(q_pose), 1).tolist()}")
    print(f"  play remainder {play_n} frames → {play_s:.1f}s at this --slow")
    if dry_run:
        q0 = _clip_q(start_q if start_q is not None else _HOME)
        plan = plan_from_start(q0, q[pose_i:], hz=hz_src, slow=slow, hold_s=hold_s)
        ret = approach_qs(q[pose_i:][-1], q0, hz=hz_src, slow=return_slow)
        print(f"dry-run from {start_label} (same planner as --play / GIF):")
        print("  sequence: saved_neutral → cue → play → saved_neutral")
        _print_plan(plan)
        print(
            f"  return to_neutral (--return-slow {return_slow:g}) "
            f"max_|Δq|={ret[1]['max_dq_deg']:.1f}°  "
            f"{ret[1]['qdot_deg_s']:.1f}°/s  {ret[1]['duration_s']:.1f}s  ({ret[1]['n']} steps)"
        )
        if compare is not None:
            _compare_to_gif(plan, compare, idx=traj.get("idx"), author=traj.get("author"))
        if write_plan is not None:
            _write_plan(write_plan, plan, traj)
        print("dry-run: not contacting the robot")
        return
    if not do_pose and not do_play:
        raise SystemExit("pass --pose and/or --play")

    env = _build_env(hz_src)
    _start_joint_policy(env)
    q_neutral = _clip_q(start_q) if start_q is not None else None
    try:
        q_now = _current_q(env)
        print(f"  now_deg   {np.round(np.degrees(q_now), 1).tolist()}", flush=True)
        if do_play and q_neutral is not None:
            if _far(q_now, q_pose, _AT_POSE_DEG):
                if _far(q_now, q_neutral, _AT_POSE_DEG):
                    print(
                        f"return to origin first ({start_label}, --return-slow {return_slow:g})"
                    )
                    _goto_pose(env, q_neutral, start_g, slow=return_slow, hz=hz_src)
                print("pose path origin → cue (GIF lerp)", flush=True)
                _goto_pose(env, q_pose, g0, slow=slow, hz=hz_src)
                got = _current_q(env)
                err = float(np.degrees(_max_abs_dq(got, q_pose)))
                print(f"  arrived max |Δq| vs cue = {err:.1f}°  now {np.round(np.degrees(got), 1).tolist()}", flush=True)
            else:
                print("already at cue pose; skip approach", flush=True)
        elif do_pose or do_play:
            _goto_pose(env, q_pose, g0, slow=slow, hz=hz_src)
        print("at cue pose (npz start)")
        got = _current_q(env)
        print(
            f"  now vs cue max |Δq|={float(np.degrees(_max_abs_dq(got, q_pose))):.1f}°  "
            f"{np.round(np.degrees(got), 1).tolist()}",
            flush=True,
        )
        if not do_play:
            print("pose only (--play to continue; staying at cue)")
            return
        dt = 1.0 / max(hz_src * float(slow), 1e-6)
        print(f"playing cue ({play_n} frames, dt={dt:.3f}s)", flush=True)
        block = not _streaming(env)
        for i in range(pose_i, q.shape[0]):
            t0 = time.time()
            gi = float(g[i]) if i < len(g) else g0
            _send(env, q[i, :7], gi, blocking=block)
            leftover = dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)
    finally:
        if do_play and q_neutral is not None:
            print(
                f"return to origin ({start_label}, --return-slow {return_slow:g}) "
                "before the next RobotEnv"
            )
            _goto_pose(env, q_neutral, start_g, slow=return_slow, hz=hz_src)
            print("done at origin")
        elif do_play:
            print("done (no saved_neutral.json; staying at finish pose)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--traj",
        type=Path,
        default=_DEFAULT_BUNDLE,
        help=f"single trajectory or bundle (default: {_DEFAULT_BUNDLE.name})",
    )
    p.add_argument("--idx", type=int, help="cue index in a trajectory bundle")
    p.add_argument(
        "--author",
        type=str.lower,
        choices=("human", "llm"),
        help="trajectory author in a bundle (case-insensitive)",
    )
    p.add_argument("--list", action="store_true", help="list bundled idx/author choices and exit")
    p.add_argument("--pose", action="store_true", help="smoothstep lerp current joints to npz[0]")
    p.add_argument("--play", action="store_true", help="from saved_neutral to cue, play, return to_neutral")
    p.add_argument(
        "--slow",
        type=float,
        default=0.1,
        help="cue approach and dump playback; 0.1 is 10x slower than GIF timing",
    )
    p.add_argument(
        "--return-slow",
        type=float,
        default=0.3,
        help="return to saved_neutral only; 0.3 is 3x slower than GIF timing",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--start",
        type=Path,
        default=_DEFAULT_NEUTRAL,
        help="start joints JSON from read_joints.py --write (dry-run / GIF compare)",
    )
    p.add_argument("--write-plan", type=Path, help="write planned joints JSON/NPZ")
    p.add_argument(
        "--compare",
        type=Path,
        default=_DEFAULT_GIF_PLAN,
        help=f"GIF joint file for dry-run compare (default: {_DEFAULT_GIF_PLAN.name})",
    )
    p.add_argument(
        "--write-gif-plan",
        action="store_true",
        help="rebuild gif_from_neutral.npz from --start + bundle, then exit",
    )
    p.add_argument("--hold", type=float, default=0.0, help="dry-run/GIF hold at start, seconds")
    args = p.parse_args()
    if args.slow <= 0:
        raise SystemExit("--slow must be > 0")
    if args.return_slow <= 0:
        raise SystemExit("--return-slow must be > 0")
    if args.write_gif_plan:
        _write_gif_plans(args.traj, args.start, _DEFAULT_GIF_PLAN)
        return
    if (args.idx is None) != (args.author is None):
        raise SystemExit("pass --idx and --author together")
    if args.list:
        _print_bundle(args.traj)
        return
    if not args.pose and not args.play and not args.dry_run:
        raise SystemExit("need --pose and/or --play (or --dry-run)")
    traj = _load(args.traj, idx=args.idx, author=args.author)
    start_q, start_g, start_label = _load_start(args.start)
    do_pose = bool(args.pose or args.play)
    compare = args.compare if args.dry_run else None
    hold_s = float(args.hold)
    if args.dry_run and hold_s == 0.0:
        hold_s = GIF_HOLD_S
    replay(
        traj,
        slow=args.slow,
        return_slow=args.return_slow,
        do_pose=do_pose,
        do_play=bool(args.play),
        dry_run=args.dry_run,
        start_q=start_q,
        start_g=start_g,
        start_label=start_label,
        write_plan=args.write_plan,
        compare=compare,
        hold_s=hold_s,
    )


if __name__ == "__main__":
    main()
