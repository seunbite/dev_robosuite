#!/usr/bin/env python3
"""Render a cue GIF: real-robot neutral → smoothstep to dump[0] → play dump.

    python render_from_neutral.py --idx 1
    python render_from_neutral.py --idx 2 --idx 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_ADHOC = Path("/Users/sb/Downloads/workspace/dev_robosuite/adhoc/generation/robotarm")
sys.path.insert(0, str(_ADHOC))
sys.path.insert(0, str(_HERE))

from render_fr3_neutral import _env, _grab, _open_gripper  # noqa: E402
from replay_cube import _clip_q  # noqa: E402
from droid_traj import GIF_HOLD_S, GIF_SLOW, plan_from_start  # noqa: E402

_NEUTRAL = _HERE / "neutral_joints.json"


def _set_q(env, q: np.ndarray) -> None:
    env.robots[0].set_robot_joint_positions(_clip_q(q))
    _open_gripper(env)


def _write_gif(env, qs: np.ndarray, path: Path, hz: float) -> None:
    frames: list[Image.Image] = []
    for q in qs:
        _set_q(env, q)
        frames.append(_grab(env))
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = [f.convert("RGB") for f in frames]
    dur = int(round(1000.0 / max(hz, 1.0)))
    # One global palette. Per-frame ADAPTIVE palettes composite into flicker.
    palette = rgb[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in rgb]
    pal[0].save(
        path,
        save_all=True,
        append_images=pal[1:],
        duration=dur,
        loop=0,
        disposal=2,
        optimize=False,
    )


def _dump_path(idx: int, cue: str, author: str) -> Path:
    stem = f"{idx:02d}_{cue}"
    return _HERE / stem / author / f"{stem}_{author}.npz"


def _index_cues() -> dict[int, str]:
    return {int(e["idx"]): str(e["cue"]) for e in json.loads((_HERE / "index.json").read_text())}


def _render_one(env, q_real: np.ndarray, idx: int, cue: str, author: str) -> None:
    npz = _dump_path(idx, cue, author)
    z = np.load(npz)
    q_play = np.asarray(z["joints_rad"], dtype=np.float64)[:, :7]
    hz = float(np.asarray(z["hz"]).reshape(-1)[0])
    plan = plan_from_start(q_real, q_play, hz=hz, slow=GIF_SLOW, hold_s=GIF_HOLD_S)
    out = npz.with_name("preview.gif")
    _write_gif(env, plan["joints_rad"], out, hz)
    print(
        f"{idx:02d}_{cue} {author}: frames={len(plan['joints_rad'])} hold={GIF_HOLD_S:.2f}s "
        f"approach_max_|Δq|={plan['max_dq_deg']:.1f}° play={len(q_play)} → {out}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, action="append", dest="idxs")
    p.add_argument("--cue", type=str, default=None)
    p.add_argument("--author", type=str.lower, choices=("human", "llm", "both"), default="both")
    args = p.parse_args()
    catalog = _index_cues()
    idxs = args.idxs or [1]
    jobs: list[tuple[int, str]] = []
    for idx in idxs:
        cue = args.cue if args.cue and len(idxs) == 1 else catalog.get(idx)
        if cue is None:
            raise SystemExit(f"no small10 dump for idx={idx}; have {sorted(catalog)}")
        jobs.append((idx, cue))
    q_real = np.asarray(json.loads(_NEUTRAL.read_text())["q_rad"], dtype=np.float64)
    authors = ("human", "llm") if args.author == "both" else (args.author,)
    for idx, cue in jobs:
        for author in authors:
            env = _env()
            try:
                _render_one(env, q_real, idx, cue, author)
            finally:
                env.close()


if __name__ == "__main__":
    main()
