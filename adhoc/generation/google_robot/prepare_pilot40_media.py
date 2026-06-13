#!/usr/bin/env python3
"""Export pose PNG + motion MP4 per cue for pilot-40 Google Robot verify."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEFAULT_CFG = _REPO / "data/seed/shots/google_robot/shot_configs_pilot40_mobile.json"
DEFAULT_GIF_DIR = _REPO / "data/results/render/google_robot"
DEFAULT_MEDIA = _REPO / "data/results/render/google_robot/pilot40_media"


def _safe_cue(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _gif_path(gif_dir: Path, row: dict) -> Path:
    idx = int(row["idx"])
    cue = _safe_cue(str(row["cue"]))
    exact = gif_dir / f"mm19_g{idx:02d}_{cue}.gif"
    if exact.is_file():
        return exact
    cands = sorted(gif_dir.glob(f"*g{idx:02d}*{cue}*.gif"))
    if cands:
        return cands[0]
    return exact


def _gif_to_mp4(gif_path: Path, mp4_path: Path) -> bool:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(gif_path),
        "-movflags",
        "faststart",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return mp4_path.is_file()
    except subprocess.CalledProcessError:
        return False


def _gif_first_frame_png(gif_path: Path, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(gif_path) as im:
        im.seek(0)
        im.convert("RGB").save(png_path)


def run(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    gif_dir = Path(args.gif_dir)
    pose_dir = Path(args.media_dir) / "pose"
    mp4_dir = Path(args.media_dir) / "mp4"
    pose_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    if args.render_missing:
        missing = [r for r in rows if not _gif_path(gif_dir, r).is_file()]
        if missing:
            from render import run as render_run

            render_run(
                config_json=str(args.config_json),
                output_dir=str(gif_dir),
                do_html=False,
                html_only=False,
            )

    n_pose = n_mp4 = 0
    for row in tqdm(rows, desc="pilot40_media"):
        gif_path = _gif_path(gif_dir, row)
        if not gif_path.is_file():
            tqdm.write(f"missing gif idx={row['idx']} cue={row['cue']}")
            continue
        idx = int(row["idx"])
        cue = _safe_cue(str(row["cue"]))
        stem = f"mm19_g{idx:02d}_{cue}"
        png_path = pose_dir / f"{stem}_pose.png"
        mp4_path = mp4_dir / f"{stem}.mp4"
        _gif_first_frame_png(gif_path, png_path)
        n_pose += 1
        if _gif_to_mp4(gif_path, mp4_path):
            n_mp4 += 1
        else:
            tqdm.write(f"mp4 fail {stem} (ffmpeg?)")

    manifest = {
        "config_json": str(Path(args.config_json).relative_to(_REPO)),
        "gif_dir": str(gif_dir.relative_to(_REPO)),
        "media_dir": str(Path(args.media_dir).relative_to(_REPO)),
        "n_cues": len(rows),
        "n_pose_png": n_pose,
        "n_mp4": n_mp4,
    }
    manifest_path = Path(args.media_dir) / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"pose PNG: {n_pose}/{len(rows)} → {pose_dir}")
    print(f"mp4: {n_mp4}/{len(rows)} → {mp4_dir}")
    print(f"manifest → {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", default=str(DEFAULT_CFG))
    p.add_argument("--gif-dir", default=str(DEFAULT_GIF_DIR))
    p.add_argument("--media-dir", default=str(DEFAULT_MEDIA))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--render-missing", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
