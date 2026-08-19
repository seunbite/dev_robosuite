#!/usr/bin/env python3
"""
Exp13 (baseline few-shot) — build few-shot artifacts from the *same* exp1 shot configs.

For each exp1 few-shot example (manipulator / IIWA):
  1) simulate + render a GIF
  2) extract EE xyz trajectory per frame
  3) extract joint-position list per frame (qpos over arm joints)

These artifacts are then used as the few-shot "shots" for two baselines:
  - fewshot_xyz  (shot includes EE xyz trajectory)
  - fewshot_joint (shot includes joint-angle trajectory)

This script only builds the few-shot artifacts and an HTML preview.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import re

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

# Keep robotarm on path (but after repo root).
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pilot90_paths import SHOTS as SHOTS_MANIP  # noqa: E402


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


@dataclass(frozen=True)
class FewshotArtifact:
    idx: int
    cue: str
    description: str
    gif_path: Path
    xyz_json: Path
    joints_json: Path
    n_frames: int
    joint_names: list[str]


def _load_exp1_fewshot_rows(*, shots_json: Path = SHOTS_MANIP, max_examples: int = 6) -> list[dict[str, Any]]:
    """Match exp1's few-shot selection: first N rows with state=='handmade'."""
    rows = json.loads(shots_json.read_text(encoding="utf-8"))
    handmade = [r for r in rows if r.get("state") == "handmade"]
    return handmade[:max_examples]


def _safe_slug(text: str, *, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.-]+", "_", (text or "").strip()).strip("_")
    return s[:max_len] if s else "cue"


def _arm_joint_qpos_slices(gen: Any) -> tuple[list[int], list[str]]:
    """
    Return qpos indices and joint names for the arm joints.

    We prefer robot-provided indices when available, to avoid depending on MuJoCo model details.
    """
    robot = gen.robot
    idxs = getattr(robot, "_ref_joint_pos_indexes", None)
    if isinstance(idxs, (list, tuple)) and idxs and all(isinstance(i, (int, np.integer)) for i in idxs):
        qpos_idxs = [int(i) for i in idxs]
        # Best-effort names: align with joint order in robot model if available.
        names = []
        rm = getattr(robot, "robot_model", None)
        if rm is not None and hasattr(rm, "arm_joints"):
            try:
                names = [str(x) for x in list(rm.arm_joints)]
            except Exception:
                names = []
        if not names:
            names = [f"DOF_{i}" for i in range(len(qpos_idxs))]
        return qpos_idxs, names

    # Fallback: treat initial_joint_pos as the arm vector; map it to qpos[0:len()]
    init = getattr(gen, "initial_joint_pos", None)
    if init is None:
        raise RuntimeError("Could not determine arm qpos indices (missing robot._ref_joint_pos_indexes and initial_joint_pos).")
    n = int(np.asarray(init).shape[0])
    return list(range(n)), [f"DOF_{i}" for i in range(n)]


def _simulate_with_xyz_and_joints(
    *,
    gen: Any,
    cue: str,
    cue_idx: int,
    config_path: Path,
    hz: int,
    pose_index: int | None = None,
) -> tuple[list[Image.Image], list[list[float]], list[list[float]], list[str]]:
    """
    Run one MuJoCo simulation pass and capture:
      - frames (PIL)
      - xyz trajectory: [[x,y,z], ...]
      - joint trajectory: [[q1,...,qN], ...] in radians
    """
    # Import from legacy since it already implements consistent cue execution.
    from legacy.vlm_pose_benchmark import _get_gripper_tip_world  # noqa: WPS433

    qpos_idxs, joint_names = _arm_joint_qpos_slices(gen)
    xyz: list[list[float]] = []
    joints: list[list[float]] = []

    orig_capture = gen._capture_image

    def _capture_with_tracks():
        tip = _get_gripper_tip_world(gen.env, gen.robot)
        xyz.append([float(tip[0]), float(tip[1]), float(tip[2])])
        q = np.asarray(gen.env.sim.data.qpos[qpos_idxs], dtype=np.float64).copy()
        joints.append([float(x) for x in q.tolist()])
        return orig_capture()

    gen._capture_image = _capture_with_tracks
    try:
        gen._set_joint_positions(gen.initial_joint_pos)
        frames, _pose_id = gen.execute_cue(
            cue=cue,
            cue_idx=int(cue_idx),
            pose_index=pose_index,
            config_path=str(config_path),
            hz=int(hz),
            save_gif=False,
            overlay_progress_bar=False,
        )
    finally:
        gen._capture_image = orig_capture

    # Ensure PIL frames
    pil_frames: list[Image.Image] = []
    for fr in frames:
        if isinstance(fr, Image.Image):
            pil_frames.append(fr.convert("RGB"))
        else:
            pil_frames.append(Image.fromarray(np.asarray(fr, dtype=np.uint8)).convert("RGB"))

    return pil_frames, xyz, joints, joint_names


def _write_gif(frames: list[Image.Image], out_gif: Path, *, hz: int) -> None:
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write.")
    dur_ms = int(round(1000 / max(1, int(hz))))
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=dur_ms,
        loop=0,
        optimize=False,
    )


def build_fewshot_artifacts(
    *,
    out_dir: Path,
    shots_json: Path = SHOTS_MANIP,
    max_examples: int = 6,
    hz: int = 8,
) -> list[FewshotArtifact]:
    rows = _load_exp1_fewshot_rows(shots_json=shots_json, max_examples=max_examples)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Local import here so the script can be imported without MuJoCo deps.
    from motion_generation import MotionGenerator  # noqa: WPS433

    gen = MotionGenerator(
        robot_name="IIWA",
        capture_image_width=512,
        capture_image_height=512,
        has_renderer=False,
        has_offscreen_renderer=True,
        hz=int(hz),
    )
    arts: list[FewshotArtifact] = []

    for r in rows:
        idx = int(r.get("idx", 0))
        cue = str(r.get("cue", f"cue_{idx}"))
        slug = _safe_slug(cue)
        desc = str(r.get("description", "")).strip()

        # Write a single-row config list for execute_cue().
        cfg_path = out_dir / f"shot_c{idx:03d}_{slug}.json"
        cfg_path.write_text(json.dumps([r], indent=2, ensure_ascii=False), encoding="utf-8")

        frames, xyz, joints, joint_names = _simulate_with_xyz_and_joints(
            gen=gen,
            cue=cue,
            cue_idx=idx,
            config_path=cfg_path,
            hz=hz,
        )

        gif_path = out_dir / "gifs" / f"shot_c{idx:03d}_{slug}.gif"
        _write_gif(frames, gif_path, hz=hz)

        xyz_json = out_dir / "xyz" / f"shot_c{idx:03d}_{slug}_xyz.json"
        xyz_json.parent.mkdir(parents=True, exist_ok=True)
        xyz_json.write_text(
            json.dumps({"idx": idx, "cue": cue, "hz": hz, "xyz": xyz}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        joints_json = out_dir / "joints" / f"shot_c{idx:03d}_{slug}_joints.json"
        joints_json.parent.mkdir(parents=True, exist_ok=True)
        joints_json.write_text(
            json.dumps(
                {"idx": idx, "cue": cue, "hz": hz, "joint_names": joint_names, "joints_rad": joints},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        arts.append(
            FewshotArtifact(
                idx=idx,
                cue=cue,
                description=desc,
                gif_path=gif_path,
                xyz_json=xyz_json,
                joints_json=joints_json,
                n_frames=len(frames),
                joint_names=joint_names,
            )
        )

    return arts


def write_preview_html(arts: list[FewshotArtifact], out_html: Path) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for a in arts:
        gif_rel = a.gif_path.relative_to(out_html.parent).as_posix()
        xyz_rel = a.xyz_json.relative_to(out_html.parent).as_posix()
        joints_rel = a.joints_json.relative_to(out_html.parent).as_posix()
        cards.append(
            f"""
<section class="card">
  <h2>c{a.idx:03d} · {_esc(a.cue)}</h2>
  <p class="meta">frames={a.n_frames} · joints={len(a.joint_names)} · xyz=<a href="{_esc(xyz_rel)}">{_esc(a.xyz_json.name)}</a> · joints=<a href="{_esc(joints_rel)}">{_esc(a.joints_json.name)}</a></p>
  <p class="desc">{_esc(a.description)}</p>
  <img src="{_esc(gif_rel)}" loading="lazy"/>
</section>
"""
        )
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Exp13 few-shot artifacts (exp1 shots)</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px 48px;background:#fafafa;color:#111}}
h1{{font-size:22px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:12px;padding:14px;margin:14px 0}}
.meta{{color:#555;font-size:12px}}
.desc{{color:#222;font-size:13px}}
img{{width:100%;max-width:720px;border-radius:8px;border:1px solid #ddd;background:#000}}
</style></head><body>
<h1>Exp13 baseline few-shot — rendered shots + extracted xyz/joints</h1>
<p class="meta">These are the exp1 few-shot examples rendered in MuJoCo, with per-frame EE xyz and arm joint qpos captured. The same artifacts feed both fewshot_xyz and fewshot_joint baselines.</p>
{''.join(cards)}
</body></html>
"""
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def main() -> None:
    p = argparse.ArgumentParser(description="Build exp13 baseline few-shot artifacts (gif + xyz + joints) from exp1 shots.")
    p.add_argument("--shots-json", type=Path, default=SHOTS_MANIP)
    p.add_argument("--max-examples", type=int, default=6)
    p.add_argument("--hz", type=int, default=8)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "data/results/baseline_fewshot/exp13",
    )
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    arts = build_fewshot_artifacts(
        out_dir=args.out_dir,
        shots_json=args.shots_json,
        max_examples=int(args.max_examples),
        hz=int(args.hz),
    )
    out_html = write_preview_html(arts, args.out_dir / "fewshot_preview.html")
    print(f"Wrote HTML → {out_html}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(out_html)], check=False)


if __name__ == "__main__":
    main()

