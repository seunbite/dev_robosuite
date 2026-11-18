#!/usr/bin/env python3
"""Trace ±15 cm cube about FR3 reset EE: 4 front + 1 depth + 4 back (9 edges)."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from render_fr3_neutral import _OUT, _SIZE, _env, _grab, _label  # noqa: E402
from replay_cube import cube_center  # noqa: E402
from replay_droid import _Y_HALF_M, _clip_q  # noqa: E402

_HALF = float(_Y_HALF_M)
_HZ = 15
_SAMPLES_PER_EDGE = 18
_HOLD = 4


def _font(size: int):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _mj(env):
    sim = env.sim
    model = sim.model._model if hasattr(sim.model, "_model") else sim.model
    data = sim.data._data if hasattr(sim.data, "_data") else sim.data
    return model, data


def _dof_ids(env) -> np.ndarray:
    names = list(env.robots[0].robot_model.joints)[:7]
    model = env.sim.model
    out = []
    for name in names:
        jid = model.joint_name2id(name)
        out.append(int(model.jnt_dofadr[jid]))
    return np.asarray(out, dtype=int)


def _site_id(env) -> int:
    arm = "right" if "right" in env.robots[0].eef_site_id else next(iter(env.robots[0].eef_site_id))
    return int(env.robots[0].eef_site_id[arm])


def _q_from_data(env, dof_ids: np.ndarray) -> np.ndarray:
    _, data = _mj(env)
    return np.asarray(data.qpos[dof_ids], dtype=np.float64).copy()


def _set_q(env, q: np.ndarray) -> None:
    env.robots[0].set_robot_joint_positions(_clip_q(q))
    env.sim.forward()


def ik_to(env, target: np.ndarray, dof_ids: np.ndarray, sid: int, q_bias: np.ndarray, iters: int = 60) -> float:
    model, data = _mj(env)
    target = np.asarray(target, dtype=np.float64)
    q = _q_from_data(env, dof_ids)
    err = 1.0
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iters):
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        j = jacp[:, dof_ids]
        xpos = np.asarray(data.site_xpos[sid], dtype=np.float64)
        err_v = target - xpos
        err = float(np.linalg.norm(err_v))
        if err < 0.006:
            break
        jj = j @ j.T + 2e-3 * np.eye(3)
        dq = j.T @ np.linalg.solve(jj, err_v)
        q = _clip_q(q + np.clip(dq, -0.12, 0.12) + 0.04 * (q_bias - q))
        _set_q(env, q)
    return err


def _corners(c: np.ndarray, h: float) -> dict[str, list[np.ndarray]]:
    xf, xb = c[0] + h, c[0] - h
    ym, yp = c[1] - h, c[1] + h
    zm, zp = c[2] - h, c[2] + h
    front = [
        np.array([xf, ym, zm]),
        np.array([xf, yp, zm]),
        np.array([xf, yp, zp]),
        np.array([xf, ym, zp]),
    ]
    back = [
        np.array([xb, ym, zm]),
        np.array([xb, yp, zm]),
        np.array([xb, yp, zp]),
        np.array([xb, ym, zp]),
    ]
    return {"front": front, "back": back}


def _edge_points(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    return [(1.0 - k / n) * a + (k / n) * b for k in range(1, n + 1)]


def _connected_path(cube: dict[str, list[np.ndarray]], n: int, start: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """9 edges: front square (4) → −X 30 cm (1) → back square (4).

    Front starts and ends at the same corner; that corner is also the start
    of the depth edge onto the back face.
    """
    front, back = cube["front"], cube["back"]
    path: list[tuple[str, np.ndarray]] = [("to front", p) for p in _edge_points(start, front[0], n)]
    for i in range(4):
        path.extend(("front", p) for p in _edge_points(front[i], front[(i + 1) % 4], n))
    path.extend(("depth −X 30cm", p) for p in _edge_points(front[0], back[0], n))
    for i in range(4):
        path.extend(("back", p) for p in _edge_points(back[i], back[(i + 1) % 4], n))
    return path


def main() -> None:
    c = cube_center()
    h = _HALF
    cube = _corners(c, h)
    env = _env()
    sid = _site_id(env)
    dof_ids = _dof_ids(env)
    q0 = np.asarray(env.robots[0].robot_model.init_qpos, dtype=np.float64).reshape(-1)[:7]
    _set_q(env, q0)
    q_bias = q0.copy()
    ee0 = np.asarray(env.sim.data.site_xpos[sid], dtype=np.float64).copy()
    path = _connected_path(cube, _SAMPLES_PER_EDGE, ee0)

    print("robot EE", np.round(ee0, 3).tolist())
    print("cube center", np.round(c, 3).tolist(), "(+40 cm Z)")
    print("front x", round(c[0] + h, 3), "back x", round(c[0] - h, 3), "z", [round(c[2] - h, 3), round(c[2] + h, 3)])

    frames: list[Image.Image] = []
    errs: list[float] = []

    def capture(tag: str) -> Image.Image:
        ee = np.asarray(env.sim.data.site_xpos[sid], dtype=np.float64)
        return _label(
            _grab(env),
            [
                f"±15 cm cube  {tag}",
                f"center {np.round(c, 3).tolist()} m",
                f"EE {np.round(ee, 3).tolist()} m",
            ],
        )

    for _ in range(_HOLD):
        frames.append(capture("hold start (neutral)"))

    cur = ""
    for tag, tgt in path:
        if tag != cur:
            cur = tag
            print(f"-- {tag} --")
        err = ik_to(env, tgt, dof_ids, sid, q_bias)
        errs.append(err)
        frames.append(capture(f"{tag}  IK err {err*100:.1f} cm"))

    for _ in range(_HOLD):
        frames.append(capture("hold end"))

    print(f"IK err median {np.median(errs)*100:.1f} cm  max {np.max(errs)*100:.1f} cm")
    gif = _OUT / "workspace_cube.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(round(1000.0 / _HZ)), loop=0)
    print("wrote", gif, "frames", len(frames))

    n = _SAMPLES_PER_EDGE
    keys = [
        ("neutral", frames[_HOLD]),
        ("front 4", frames[min(len(frames) - 1, _HOLD + n + n * 2)]),
        ("back 4", frames[min(len(frames) - 1, _HOLD + n + n * 6)]),
    ]
    strip = Image.new("RGB", (_SIZE * 3 + 16, _SIZE + 28), (20, 20, 24))
    draw = ImageDraw.Draw(strip)
    font = _font(14)
    for i, (title, im) in enumerate(keys):
        x = i * (_SIZE + 8)
        draw.text((x + 8, 6), title, fill=(230, 230, 230), font=font)
        strip.paste(im, (x, 28))
    strip_path = _OUT / "workspace_cube_strip.png"
    strip.save(strip_path)
    print("wrote", strip_path)


if __name__ == "__main__":
    main()
