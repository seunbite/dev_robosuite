"""Resolve pilot-40 motion GIF/MP4 paths for VLM verify."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROBOT = "IIWA"
PILOT40_MOTION_CFG = (
    "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
PILOT40_GIF_OUT = "data/results/visualize/gt_fixed_pose_pilot20_hz10"
PILOT40_MANIFEST = "data/results/render/manipulator/motion_vlm_verify_pilot40/manifest_pilot40.json"


def gif_matches_cue(name: str, cue: str, *, robot: str = ROBOT) -> bool:
    """Match ``*_IIWA_{cue}_p123.gif`` and ``*_IIWA_{cue}.gif``."""
    token = f"_{robot}_{cue}"
    if token not in name:
        return False
    pos = name.index(token) + len(token)
    rest = name[pos:]
    return not rest or rest[0] in "._"


def gif_dirs(repo: Path) -> list[Path]:
    return [
        repo / "data/results/visualize/gt_fixed_pose_pilot20_hz10/IIWA",
        repo / "data/results/visualize/gt_fixed_pose_pilot40_hz10/IIWA",
        repo / "data/results/visualize/gt_fixed_pose_pilot20_hz10",
        repo / "data/results/render/manipulator/motion_gt_compare/gt_positive/IIWA",
    ]


def mp4_dirs(repo: Path) -> list[Path]:
    return [
        repo / "data/results/render/manipulator/motion_vlm_verify_pilot40/mp4",
        repo / "data/results/render/manipulator/motion_gt_compare/media/generation/mp4",
    ]


def default_mp4_out(repo: Path, idx: int, cue: str) -> Path:
    return (
        repo
        / "data/results/render/manipulator/motion_vlm_verify_pilot40/mp4"
        / f"{idx:03d}_{cue}.mp4"
    )


def pick_latest_gif(repo: Path, cue: str, *, robot: str = ROBOT) -> Path | None:
    cands: list[Path] = []
    for d in gif_dirs(repo):
        if not d.is_dir():
            continue
        for p in d.glob("*.gif"):
            if gif_matches_cue(p.name, cue, robot=robot):
                cands.append(p)
    if not cands:
        viz = repo / "data/results/visualize"
        if viz.is_dir():
            for p in viz.rglob("*.gif"):
                if gif_matches_cue(p.name, cue, robot=robot):
                    cands.append(p)
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_existing_mp4(repo: Path, idx: int, cue: str) -> Path | None:
    names = (f"{idx:03d}_{cue}.mp4", f"{idx:03d}_{cue}_gen.mp4")
    for d in mp4_dirs(repo):
        for name in names:
            p = d / name
            if p.is_file():
                return p
    for root in (
        repo / "data/results/render/manipulator",
        repo / "data/results/visualize",
    ):
        if not root.is_dir():
            continue
        for pattern in (f"**/{idx:03d}_{cue}.mp4", f"**/{idx:03d}_{cue}_gen.mp4"):
            hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if hits:
                return hits[0]
    return None


def gif_to_mp4(gif: Path, mp4: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = [
        ff,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(gif),
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(mp4),
    ]
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "").strip() or "ffmpeg failed")


def resolve_mp4(
    repo: Path,
    item: dict[str, Any],
    idx: int,
    cue: str,
    *,
    build: bool = True,
) -> tuple[Path | None, str | None]:
    """Return (mp4 path, skip reason)."""
    raw = item.get("mp4")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        if path.is_file():
            return path, None

    found = find_existing_mp4(repo, idx, cue)
    if found:
        return found, None

    out = default_mp4_out(repo, idx, cue)
    if out.is_file():
        return out, None

    if not build:
        return None, "no mp4"

    gif = pick_latest_gif(repo, cue)
    if not gif:
        searched = ", ".join(str(d.relative_to(repo)) for d in gif_dirs(repo))
        return None, f"no mp4 (no gif; searched {searched} + visualize/**)"

    try:
        print(f"[mp4] {cue} from {gif.relative_to(repo)}", flush=True)
        gif_to_mp4(gif, out)
    except Exception as e:
        return None, f"mp4 build failed: {e}"

    return (out if out.is_file() else None), (None if out.is_file() else "mp4 build failed")


def _pose_index_from_row(row: dict[str, Any]) -> int | None:
    gfp = row.get("gt_fixed_first_pose")
    if isinstance(gfp, dict) and gfp.get("pose_id") is not None:
        return int(gfp["pose_id"])
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose")
        if isinstance(pose, dict) and pose.get("pose_id") is not None:
            return int(pose["pose_id"])
        break
    return None


def _render_pilot40_cues(
    repo: Path,
    robotarm_dir: Path,
    cfg: Path,
    gif_out: Path,
    cue_indices: list[int],
    *,
    hz: int = 10,
) -> None:
    """Render GIFs via motion_generation_core.generate (no render.py dependency)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))
    from legacy.motion_generation_core import generate  # noqa: WPS433

    rows = json.loads(cfg.read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): r for r in rows}
    jpath = str(repo / "data/seed/_remainder/closest_poses_results.jsonl")
    gif_out.mkdir(parents=True, exist_ok=True)

    for idx in cue_indices:
        row = by_idx.get(idx)
        if not row:
            continue
        cue = str(row["cue"])
        if pick_latest_gif(repo, cue):
            continue
        try:
            print(f"[render] c{idx} {cue} ...", flush=True)
            generate(
                robot=ROBOT,
                cue=cue,
                cue_idx=idx,
                pose_index=_pose_index_from_row(row),
                jsonl_path=jpath,
                config_path=str(cfg),
                output_dir=str(gif_out),
                hz=hz,
                top_k=1,
            )
        except Exception as e:
            print(f"[render fail] c{idx} {cue}: {e}", flush=True)


def prepare_pilot40_motion_mp4s(
    repo: Path,
    robotarm_dir: Path,
    items: list[tuple[int, str]],
    *,
    config_json: Path | None = None,
    hz: int = 10,
) -> int:
    """Render missing MuJoCo GIFs (if needed) and build MP4s. Returns ready count."""
    cfg = config_json or (repo / PILOT40_MOTION_CFG)
    gif_out = repo / PILOT40_GIF_OUT

    need_render: list[int] = []
    for idx, cue in items:
        mp4, _ = resolve_mp4(repo, {}, idx, cue, build=False)
        if mp4 or pick_latest_gif(repo, cue):
            continue
        need_render.append(idx)

    if need_render:
        print(
            f"[render] {len(need_render)} cues missing GIF → {gif_out.relative_to(repo)}/IIWA",
            flush=True,
        )
        _render_pilot40_cues(repo, robotarm_dir, cfg, gif_out, need_render, hz=hz)

    ready = 0
    for idx, cue in items:
        mp4, _ = resolve_mp4(repo, {}, idx, cue, build=True)
        if mp4:
            ready += 1
    return ready


def write_pilot40_manifest(repo: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = repo / PILOT40_MANIFEST
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["idx"])
        cue = str(row["cue"])
        mp4, _ = resolve_mp4(repo, {}, idx, cue, build=False)
        gif = pick_latest_gif(repo, cue)
        manifest_rows.append(
            {
                "cue_idx": idx,
                "cue": cue,
                "description": row.get("description", ""),
                "config_path": str(repo / PILOT40_MOTION_CFG),
                "gif": str(gif) if gif else None,
                "mp4": str(mp4) if mp4 else None,
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"rows": manifest_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path
