"""Resolve pilot-40 motion GIF/MP4 paths for VLM verify."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

ROBOT = "IIWA"


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
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def find_existing_mp4(repo: Path, idx: int, cue: str) -> Path | None:
    names = (f"{idx:03d}_{cue}.mp4", f"{idx:03d}_{cue}_gen.mp4")
    for d in mp4_dirs(repo):
        for name in names:
            p = d / name
            if p.is_file():
                return p
    root = repo / "data/results/render/manipulator"
    if root.is_dir():
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
        return None, f"no mp4 (no gif; searched {searched})"

    try:
        print(f"[mp4] {cue} from {gif.relative_to(repo)}", flush=True)
        gif_to_mp4(gif, out)
    except Exception as e:
        return None, f"mp4 build failed: {e}"

    return (out if out.is_file() else None), (None if out.is_file() else "mp4 build failed")
