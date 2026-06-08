"""Resolve pilot-40 motion GIF/MP4 paths for VLM verify."""
from __future__ import annotations

import json
import os
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
PILOT90_MOTION_CFG = (
    "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
PILOT90_MP4_DIR = "data/results/render/manipulator/motion_vlm_verify_pilot90/mp4"
PILOT90_MANIFEST = "data/results/render/manipulator/motion_vlm_verify_pilot90/manifest_pilot90.json"
PILOT90_PAIRWISE_DIR = "data/results/verify/samples/motion_gt_neg_pairwise_pilot90"
DEFAULT_JSONL = "data/seed/_remainder/closest_poses_results.jsonl"


def pose_jsonl(repo: Path) -> Path:
    raw = os.getenv("MOTION_POSE_JSONL")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else repo / p
    canonical = repo / DEFAULT_JSONL
    if canonical.is_file():
        return canonical
    legacy = repo / "data/seed/closest_poses_results.jsonl"
    return legacy if legacy.is_file() else canonical


def _count_pose_jsonl(jpath: Path, *, robot: str = ROBOT) -> tuple[int, int]:
    """Return (total lines, poses for ``robot``)."""
    total = 0
    n_robot = 0
    with jpath.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("robot") == robot:
                n_robot += 1
    return total, n_robot


def check_pose_jsonl(
    repo: Path,
    *,
    robot: str = ROBOT,
    min_robot_poses: int = 100,
) -> Path:
    """Ensure pose JSONL exists and has enough entries for MuJoCo pose lookup."""
    jpath = pose_jsonl(repo)
    if not jpath.is_file():
        raise FileNotFoundError(
            f"Missing pose database: {jpath}\n"
            "Copy data/seed/_remainder/closest_poses_results.jsonl to the cluster "
            "(git pull, or rsync from laptop) or set MOTION_POSE_JSONL."
        )
    total, n_robot = _count_pose_jsonl(jpath, robot=robot)
    if total == 0 or n_robot < min_robot_poses:
        raise RuntimeError(
            f"Pose database unusable: {jpath}\n"
            f"  lines={total}, {robot} poses={n_robot} (need >={min_robot_poses})\n"
            "Likely an empty placeholder file on the cluster. Fix:\n"
            "  git pull   # file is tracked (~3MB, ~3400 lines, ~548 IIWA)\n"
            "  # or from laptop:\n"
            "  rsync -avz data/seed/_remainder/closest_poses_results.jsonl "
            "USER@babel:.../dev_robosuite/data/seed/_remainder/"
        )
    return jpath


def check_pilot40_render_prereqs(repo: Path) -> Path:
    jpath = check_pose_jsonl(repo)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH (needed for GIF→MP4)")
    return jpath


def gif_matches_cue(name: str, cue: str, *, robot: str = ROBOT) -> bool:
    """Match ``*_IIWA_{cue}_p123.gif``, ``*_IIWA_{cue}_c7_tiled.gif``, etc."""
    token = f"_{robot}_{cue}"
    if token not in name:
        return False
    pos = name.index(token) + len(token)
    rest = name[pos:]
    return not rest or rest[0] in "._"


def gif_dirs(repo: Path, *, pilot90: bool = False) -> list[Path]:
    root = repo / PILOT40_GIF_OUT
    dirs = [
        root / "IIWA",
        root,
        repo / "data/results/visualize/gt_fixed_pose_pilot40_hz10/IIWA",
        repo / "data/results/render/manipulator/motion_gt_compare/gt_positive/IIWA",
    ]
    if pilot90:
        dirs[:0] = [
            repo / "run/IIWA",
            repo / "data/results/render/manipulator/motion_vlm_verify_pilot90/gif/IIWA",
        ]
    return dirs


def mp4_dirs(repo: Path, *, pilot90: bool = False) -> list[Path]:
    extra = os.getenv("MOTION_MP4_DIR")
    dirs = [
        repo / "data/results/render/manipulator/motion_vlm_verify_pilot40/mp4",
        repo / "data/results/render/manipulator/motion_gt_compare/media/generation/mp4",
    ]
    if pilot90:
        dirs.insert(0, repo / PILOT90_MP4_DIR)
    if extra:
        p = Path(extra)
        dirs.insert(0, p if p.is_absolute() else repo / p)
    return dirs


def default_mp4_out(repo: Path, idx: int, cue: str, *, pilot90: bool = False) -> Path:
    base = repo / (PILOT90_MP4_DIR if pilot90 else "data/results/render/manipulator/motion_vlm_verify_pilot40/mp4")
    return base / f"{idx:03d}_{cue}.mp4"


def pick_latest_gif(repo: Path, cue: str, *, robot: str = ROBOT, pilot90: bool = False) -> Path | None:
    cands: list[Path] = []
    for d in gif_dirs(repo, pilot90=pilot90):
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


def find_existing_mp4(repo: Path, idx: int, cue: str, *, pilot90: bool = False) -> Path | None:
    names = (f"{idx:03d}_{cue}.mp4", f"{idx:03d}_{cue}_gen.mp4")
    for d in mp4_dirs(repo, pilot90=pilot90):
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
    pilot90: bool = False,
) -> tuple[Path | None, str | None]:
    """Return (mp4 path, skip reason)."""
    raw = item.get("mp4")
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        if path.is_file():
            return path, None

    found = find_existing_mp4(repo, idx, cue, pilot90=pilot90)
    if found:
        return found, None

    out = default_mp4_out(repo, idx, cue, pilot90=pilot90)
    if out.is_file():
        return out, None

    if not build:
        return None, "no mp4"

    gif = pick_latest_gif(repo, cue, pilot90=pilot90)
    if not gif:
        return None, "no mp4 (no gif found after render/search)"

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


def _render_motion_cues(
    repo: Path,
    robotarm_dir: Path,
    cfg: Path,
    gif_out: Path,
    cue_indices: list[int],
    *,
    jpath: Path,
    hz: int = 10,
    pilot90: bool = False,
) -> list[str]:
    """Render GIFs via motion_generation_core.generate. Returns failure messages."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))
    from legacy.motion_generation_core import generate  # noqa: WPS433

    rows = json.loads(cfg.read_text(encoding="utf-8"))
    by_idx = {int(r["idx"]): r for r in rows}
    gif_out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for idx in cue_indices:
        row = by_idx.get(idx)
        if not row:
            continue
        cue = str(row["cue"])
        if pick_latest_gif(repo, cue, pilot90=pilot90):
            continue
        try:
            print(f"[render] c{idx} {cue} ...", flush=True)
            generate(
                robot=ROBOT,
                cue=cue,
                cue_idx=idx,
                pose_index=_pose_index_from_row(row),
                jsonl_path=str(jpath),
                config_path=str(cfg),
                output_dir=str(gif_out),
                hz=hz,
                top_k=1,
            )
            if not pick_latest_gif(repo, cue, pilot90=pilot90):
                rel = gif_out.relative_to(repo) if gif_out.is_relative_to(repo) else gif_out
                failures.append(f"c{idx} {cue}: generate finished but no GIF found under {rel}")
        except Exception as e:
            msg = f"c{idx} {cue}: {e}"
            failures.append(msg)
            print(f"[render fail] {msg}", flush=True)
    return failures


# Back-compat alias
_render_pilot40_cues = _render_motion_cues


def prepare_motion_mp4s(
    repo: Path,
    robotarm_dir: Path,
    items: list[tuple[int, str]],
    *,
    config_json: Path | None = None,
    hz: int = 10,
    pilot90: bool = False,
    render_missing: bool | None = None,
) -> tuple[int, list[str]]:
    """Render missing MuJoCo GIFs (if needed) and build MP4s. Returns (ready count, failures)."""
    cfg = config_json or (repo / (PILOT90_MOTION_CFG if pilot90 else PILOT40_MOTION_CFG))
    gif_out = (
        repo / "data/results/render/manipulator/motion_vlm_verify_pilot90/gif"
        if pilot90
        else repo / PILOT40_GIF_OUT
    )
    failures: list[str] = []

    need_render: list[int] = []
    for idx, cue in items:
        mp4, _ = resolve_mp4(repo, {}, idx, cue, build=False, pilot90=pilot90)
        if mp4 or pick_latest_gif(repo, cue, pilot90=pilot90):
            continue
        need_render.append(idx)

    do_render = render_missing if render_missing is not None else os.getenv("MOTION_RENDER_MISSING", "0") == "1"

    if need_render and do_render:
        try:
            jpath = check_pilot40_render_prereqs(repo)
        except (FileNotFoundError, RuntimeError) as e:
            failures.append(str(e))
            print(f"[render abort] {e}", flush=True)
            need_render = []
        else:
            print(
                f"[render] {len(need_render)} cues missing GIF → {gif_out.relative_to(repo)}/IIWA",
                flush=True,
            )
            failures.extend(
                _render_motion_cues(
                    repo,
                    robotarm_dir,
                    cfg,
                    gif_out,
                    need_render,
                    jpath=jpath,
                    hz=hz,
                    pilot90=pilot90,
                )
            )
    elif need_render:
        hint = (
            "bash scripts/prepare_pilot90_motion_mp4.sh --render-missing"
            if pilot90
            else "bash scripts/prepare_pilot40_motion_mp4.sh"
        )
        print(
            f"[render skip] {len(need_render)} cues missing GIF "
            f"(sync run/IIWA GIFs or run: {hint})",
            flush=True,
        )

    ready = 0
    for idx, cue in items:
        mp4, reason = resolve_mp4(repo, {}, idx, cue, build=True, pilot90=pilot90)
        if mp4:
            ready += 1
        elif reason:
            failures.append(f"{cue}: {reason}")
    return ready, failures


def prepare_pilot40_motion_mp4s(
    repo: Path,
    robotarm_dir: Path,
    items: list[tuple[int, str]],
    *,
    config_json: Path | None = None,
    hz: int = 10,
) -> tuple[int, list[str]]:
    return prepare_motion_mp4s(
        repo, robotarm_dir, items, config_json=config_json, hz=hz, pilot90=False
    )


def prepare_pilot90_motion_mp4s(
    repo: Path,
    robotarm_dir: Path,
    items: list[tuple[int, str]],
    *,
    config_json: Path | None = None,
    hz: int = 10,
    render_missing: bool | None = None,
) -> tuple[int, list[str]]:
    return prepare_motion_mp4s(
        repo,
        robotarm_dir,
        items,
        config_json=config_json,
        hz=hz,
        pilot90=True,
        render_missing=render_missing,
    )


def write_motion_manifest(
    repo: Path,
    rows: list[dict[str, Any]],
    *,
    pilot90: bool = False,
    config_rel: str | None = None,
) -> Path:
    manifest_path = repo / (PILOT90_MANIFEST if pilot90 else PILOT40_MANIFEST)
    cfg_rel = config_rel or (PILOT90_MOTION_CFG if pilot90 else PILOT40_MOTION_CFG)
    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row["idx"])
        cue = str(row["cue"])
        mp4, _ = resolve_mp4(repo, {}, idx, cue, build=False, pilot90=pilot90)
        gif = pick_latest_gif(repo, cue, pilot90=pilot90)
        manifest_rows.append(
            {
                "cue_idx": idx,
                "cue": cue,
                "description": row.get("description", ""),
                "config_path": str(repo / cfg_rel),
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


def write_pilot40_manifest(repo: Path, rows: list[dict[str, Any]]) -> Path:
    return write_motion_manifest(repo, rows, pilot90=False)


def write_pilot90_manifest(repo: Path, rows: list[dict[str, Any]]) -> Path:
    return write_motion_manifest(repo, rows, pilot90=True)
