#!/usr/bin/env python3
"""Sim preview of replay_droid.py --pose: lerp current (reset) → first cue pose."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_fr3_neutral import _CAM_ZOOM, _OUT, _SIZE, _env, _grab, _label  # noqa: E402
from replay_droid import (  # noqa: E402
    _QDOT_AT_SLOW_1,
    _clip_q,
    _load,
    labeled_pose_path,
)

_TRAJ = _OUT / "02_shake_head_no/llm/02_shake_head_no_llm.npz"


def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _pose_steps(q0: np.ndarray, q1: np.ndarray, *, slow: float, hz: float):
    q0 = _clip_q(q0)
    q1 = _clip_q(q1)
    max_dq = float(np.max(np.abs(q1 - q0)))
    qdot = _QDOT_AT_SLOW_1 * max(float(slow), 1e-4)
    duration = max(max_dq / max(qdot, 1e-6), 0.5)
    n = max(2, int(round(duration * hz)))
    qs = [(_clip_q((1.0 - i / n) * q0 + (i / n) * q1), i / n) for i in range(0, n + 1)]
    return qs, max_dq, qdot, duration, n


def main() -> None:
    slow = 1.0
    traj = _load(_TRAJ)
    q_all = traj["joints_rad"]
    hz = float(traj["hz"])
    pose_i = int(traj.get("pose_frame") or 0)
    q_goal = q_all[pose_i, :7]

    env = _env()
    q0 = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
    env.robots[0].set_robot_joint_positions(q0)
    env.sim.forward()

    path = labeled_pose_path(q0, q_goal)
    print(f"pose_frame={pose_i}  waypoints={len(path)}")
    print(f"  q0_deg   {np.round(np.degrees(q0), 1).tolist()}")
    for i, (name, wp) in enumerate(path):
        env.robots[0].set_robot_joint_positions(_clip_q(wp))
        env.sim.forward()
        arm = "right"
        sid = env.robots[0].eef_site_id[arm]
        ee = np.asarray(env.sim.data.site_xpos[sid])
        print(f"  wp{i}_{name}_deg {np.round(np.degrees(wp), 1).tolist()}  EE {np.round(ee, 3).tolist()}")

    hold = max(1, int(round(0.35 * hz)))
    frames: list[Image.Image] = []
    keys: list[tuple[str, Image.Image]] = []
    tags = [name for name, _ in path]
    wps = [wp for _name, wp in path]

    def capture(q, tag: str) -> Image.Image:
        env.robots[0].set_robot_joint_positions(_clip_q(q))
        env.sim.forward()
        return _label(
            _grab(env),
            [
                f"replay_droid --pose  {tag}",
                "neutral EE (0.46, 0, 1.01) m  ·  |ΔY| ≤ 15 cm",
                f"wp q° {np.round(np.degrees(_clip_q(q)), 1).tolist()}",
            ],
        )

    for _ in range(hold):
        frames.append(capture(q0, "hold start"))
    keys.append(("start (neutral)", frames[-1].copy()))

    cur = q0
    for wi, wp in enumerate(wps):
        steps, max_dq, qdot, duration, n = _pose_steps(cur, wp, slow=slow, hz=hz)
        print(f"  seg{wi} {np.degrees(max_dq):.1f}°  {duration:.1f}s")
        for q, _u in steps[1:]:
            frames.append(capture(q, tags[min(wi, len(tags) - 1)]))
        keys.append((tags[min(wi, len(tags) - 1)], frames[-1].copy()))
        cur = wp
    for _ in range(hold):
        frames.append(capture(cur, "hold cue pose"))

    gif_path = _OUT / "pose_goto.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / hz)),
        loop=0,
    )
    print("wrote", gif_path, "frames", len(frames))

    ncol = len(keys)
    strip = Image.new("RGB", (_SIZE * ncol + 8 * (ncol - 1), _SIZE + 28), (20, 20, 24))
    draw = ImageDraw.Draw(strip)
    font = _font(14)
    for i, (title, im) in enumerate(keys):
        x = i * (_SIZE + 8)
        draw.text((x + 8, 6), title, fill=(230, 230, 230), font=font)
        strip.paste(im, (x, 28))
    strip_path = _OUT / "pose_goto_strip.png"
    strip.save(strip_path)
    print("wrote", strip_path)


if __name__ == "__main__":
    main()
