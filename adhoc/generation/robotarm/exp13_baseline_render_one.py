#!/usr/bin/env python3
"""
Exp13 baseline demo (one cue):
1) Take exp1 generated motion config for one cue.
2) Simulate + extract:
   - EE xyz trajectory per frame
   - arm joint qpos slice per frame (radians)
3) Re-render from extracted trajectories:
   - fewshot_xyz: use EE IK to drive through xyz points (fixed orientation)
   - fewshot_joint: set qpos slice per frame directly
4) Write gifs + xyz/joints json + a small HTML preview.

This is a *renderer validation* step (proves the xyz/joint re-render plumbing works).
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc" / "generation" / "robotarm" / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from legacy.path_ee_ik import get_eef_quat_wxyz, ik_move_eef_to  # noqa: E402
from legacy.motion_generation_core import MotionGenerator  # noqa: E402
from pilot90_paths import SHOTS as SHOTS_MANIP, GT_PATH  # noqa: E402


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _load_exp1_row(*, exp1_json: Path, cue: str | None, idx: int | None) -> dict[str, Any]:
    rows = json.loads(exp1_json.read_text(encoding="utf-8"))
    if cue is not None:
        for r in rows:
            if r.get("cue") == cue:
                return r
        raise KeyError(f"cue not found in {exp1_json}: {cue!r}")
    if idx is not None:
        for r in rows:
            if int(r.get("idx")) == int(idx):
                return r
        raise KeyError(f"idx not found in {exp1_json}: {idx!r}")
    raise ValueError("Provide cue or idx")


def _arm_qpos_slices(gen: MotionGenerator) -> tuple[list[int], list[str]]:
    robot = gen.robot
    idxs = getattr(robot, "_ref_joint_pos_indexes", None)
    if isinstance(idxs, (list, tuple)) and idxs and all(isinstance(i, (int, np.integer)) for i in idxs):
        qpos_idxs = [int(i) for i in idxs]
        names: list[str] = []
        rm = getattr(robot, "robot_model", None)
        if rm is not None and hasattr(rm, "arm_joints"):
            try:
                names = [str(x) for x in list(rm.arm_joints)]
            except Exception:
                names = []
        if len(names) != len(qpos_idxs) or not names:
            names = [f"DOF_{i}" for i in range(len(qpos_idxs))]
        return qpos_idxs, names

    # Fallback: use initial_joint_pos length.
    init = getattr(gen, "initial_joint_pos", None)
    if init is None:
        raise RuntimeError("Could not determine arm qpos slice indices.")
    init_arr = np.asarray(init, dtype=float)
    qpos_idxs = list(range(len(init_arr)))
    names = [f"DOF_{i}" for i in range(len(qpos_idxs))]
    return qpos_idxs, names


def _to_pil_frame(fr: Any) -> Image.Image:
    if isinstance(fr, Image.Image):
        return fr.convert("RGB")
    arr = np.asarray(fr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _write_gif(frames: list[Image.Image], out_path: Path, *, hz: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dur_ms = int(round(1000 / max(1, int(hz))))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=dur_ms,
        loop=0,
        optimize=False,
    )


def extract_from_exp1_row(*, gen: MotionGenerator, row: dict[str, Any], cfg_path: Path, hz: int) -> tuple[list[Image.Image], list[list[float]], list[list[float]], list[str]]:
    cue = str(row["cue"])
    cue_idx = int(row["idx"])
    qpos_idxs, joint_names = _arm_qpos_slices(gen)

    xyz: list[list[float]] = []
    joints: list[list[float]] = []

    orig_capture = gen._capture_image

    def _capture_with_tracks():
        tip = gen._get_eef_position()
        xyz.append([float(tip[0]), float(tip[1]), float(tip[2])])
        q = np.asarray(gen.env.sim.data.qpos[qpos_idxs], dtype=np.float64).copy()
        joints.append([float(x) for x in q.tolist()])
        return orig_capture()

    gen._capture_image = _capture_with_tracks
    try:
        # generate *frames* and throw away gif writing.
        frames, _pose_id = gen.execute_cue(
            cue=cue,
            cue_idx=cue_idx,
            pose_index=None,
            config_path=str(cfg_path),
            hz=int(hz),
            save_gif=False,
            overlay_progress_bar=False,
        )
    finally:
        gen._capture_image = orig_capture

    pil_frames = [_to_pil_frame(fr) for fr in frames]
    return pil_frames, xyz, joints, joint_names


def render_from_xyz(*, gen: MotionGenerator, xyz: list[list[float]], out_gif: Path, hz: int) -> list[Image.Image]:
    if not xyz:
        raise ValueError("xyz empty")

    # Reset to initial joint pos so we have a valid IK starting pose.
    gen._set_joint_positions(gen.initial_joint_pos)

    mujoco_model = gen.env.sim.model._model
    mujoco_data = gen.env.sim.data._data
    ik = gen.jacobian_calculator.ik_solver
    site_id = ik.site_ids[0]
    quat_wxyz = get_eef_quat_wxyz(mujoco_model, mujoco_data, site_id)

    frames: list[Image.Image] = []
    for p in xyz:
        target = np.asarray(p, dtype=float)
        ik_move_eef_to(
            ik_solver=ik,
            mujoco_model=mujoco_model,
            mujoco_data=mujoco_data,
            site_id=site_id,
            target_pos=target,
            quat_wxyz=quat_wxyz,
        )
        fr = _to_pil_frame(gen._capture_image())
        frames.append(fr)

    _write_gif(frames, out_gif, hz=hz)
    return frames


def render_from_joints(*, gen: MotionGenerator, joints: list[list[float]], out_gif: Path, hz: int) -> list[Image.Image]:
    if not joints:
        raise ValueError("joints empty")

    qpos_idxs, _ = _arm_qpos_slices(gen)
    # Reset to initial state.
    gen._set_joint_positions(gen.initial_joint_pos)

    frames: list[Image.Image] = []
    for qframe in joints:
        qframe = np.asarray(qframe, dtype=float)
        base_qpos = np.asarray(gen.env.sim.data.qpos, dtype=np.float64)
        if len(qpos_idxs) != len(qframe):
            raise ValueError(f"joint length mismatch: qpos_idxs={len(qpos_idxs)} qframe={len(qframe)}")
        base_qpos[qpos_idxs] = qframe
        gen.env.sim.data.qpos[:] = base_qpos
        gen.env.sim.forward()
        fr = _to_pil_frame(gen._capture_image())
        frames.append(fr)

    _write_gif(frames, out_gif, hz=hz)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp13 baseline xyz/joint renderer demo for one cue.")
    ap.add_argument("--exp1-json", type=Path, default=Path("data/results/motion_configs/manipulator/exp/result_exp1_gemini-2.5-pro.json"))
    ap.add_argument("--cue", type=str, default="raise_fists_victory")
    ap.add_argument("--hz", type=int, default=8)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/results/baseline_fewshot/exp13/demo_one"),
    )
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    row = _load_exp1_row(exp1_json=args.exp1_json, cue=args.cue, idx=None)

    # Create a single-row config json for execute_cue().
    cfg_path = out_dir / f"target_cfg_{row['idx']}.json"
    cfg_path.write_text(json.dumps([row], indent=2, ensure_ascii=False), encoding="utf-8")

    # Use offscreen rendering + track capture.
    gen = MotionGenerator(
        robot_name="IIWA",
        capture_image_width=512,
        capture_image_height=512,
        has_renderer=False,
        has_offscreen_renderer=True,
        hz=int(args.hz),
    )

    frames, xyz, joints, joint_names = extract_from_exp1_row(gen=gen, row=row, cfg_path=cfg_path, hz=args.hz)

    # Save extracted trajectories.
    xyz_path = out_dir / "xyz_pred.json"
    xyz_path.write_text(json.dumps({"cue": args.cue, "idx": row["idx"], "hz": args.hz, "xyz": xyz}, indent=2, ensure_ascii=False), encoding="utf-8")
    joints_path = out_dir / "joints_pred.json"
    joints_path.write_text(
        json.dumps(
            {"cue": args.cue, "idx": row["idx"], "hz": args.hz, "joint_names": joint_names, "joints_rad": joints},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Original frames gif.
    gif_orig = out_dir / "gif_original_from_exp1.gif"
    _write_gif(frames, gif_orig, hz=args.hz)

    # Re-render validations.
    gif_xyz = out_dir / "gif_render_from_xyz.gif"
    gif_joint = out_dir / "gif_render_from_joint.gif"
    render_from_xyz(gen=gen, xyz=xyz, out_gif=gif_xyz, hz=args.hz)
    render_from_joints(gen=gen, joints=joints, out_gif=gif_joint, hz=args.hz)

    # HTML preview.
    out_html = out_dir / "preview.html"
    out_html.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<title>Exp13 demo one cue</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px 48px;background:#fafafa;color:#111}}
h1{{font-size:22px}}
.row{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{flex:1;min-width:320px;border:1px solid #ddd;border-radius:12px;background:#fff;padding:12px}}
video,img{{max-width:100%}}
code{{background:#f0f0f2;padding:2px 6px;border-radius:6px}}
.meta{{color:#555;margin:8px 0 14px}}
</style></head><body>
<h1>Exp13 demo — one cue: {_esc(args.cue)}</h1>
<div class="meta">
idx={_esc(row['idx'])} · hz={_esc(args.hz)}<br/>
extracted: <code>xyz_pred.json</code> and <code>joints_pred.json</code>
</div>
<div class="row">
  <div class="card"><h3>Original (exp1 config)</h3><img src="gif_original_from_exp1.gif"/></div>
  <div class="card"><h3>Re-render from xyz</h3><img src="gif_render_from_xyz.gif"/></div>
  <div class="card"><h3>Re-render from joint</h3><img src="gif_render_from_joint.gif"/></div>
</div>
<p>Tip: both re-rendered GIFs should visually match the original trajectory.</p>
</body></html>""",
        encoding="utf-8",
    )

    print(f"Wrote preview: {out_html}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out_html)], check=False)


if __name__ == "__main__":
    main()

