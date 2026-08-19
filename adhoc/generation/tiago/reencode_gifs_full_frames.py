#!/usr/bin/env python3
"""Re-encode existing GIFs with disposal=2 full-frame writer (no VLM / sim)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_GEN = _REPO / "adhoc/generation"
for p in (_REPO, _GEN, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gif_encode import save_gif_frames  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"


def _load_composited_frames(gif_path: Path) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with Image.open(gif_path) as im:
        try:
            while True:
                frames.append(im.convert("RGB").copy())
                im.seek(im.tell() + 1)
        except EOFError:
            pass
    return frames


def _partial_tile_count(gif_path: Path) -> int:
    with Image.open(gif_path) as im:
        w, h = im.size
        partial = 0
        try:
            while True:
                ext = im.tile[0].extents if im.tile else (0, 0, w, h)
                if ext != (0, 0, w, h):
                    partial += 1
                im.seek(im.tell() + 1)
        except EOFError:
            pass
        return partial


def reencode_gif(gif_path: Path, *, duration_ms: int = 50, regen_mp4: bool = True) -> bool:
    if not gif_path.is_file():
        return False
    if _partial_tile_count(gif_path) == 0:
        return False
    frames = _load_composited_frames(gif_path)
    if not frames:
        return False
    save_gif_frames(frames, gif_path, duration_ms=duration_ms)
    if regen_mp4:
        mp4 = gif_path.with_suffix(".mp4")
        from shutil import which
        import subprocess

        if which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(gif_path),
                    "-movflags",
                    "+faststart",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4),
                ],
                check=False,
            )
    return True


def run(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    include_originals: bool = True,
    include_corr1: bool = True,
    dry_run: bool = False,
) -> None:
    changed = 0
    scanned = 0
    targets: list[Path] = []
    if include_originals:
        targets.extend(sorted((_REPO / "data/results/render/tiago" / f"exp1_{model_tag}").glob("mm19_*.gif")))
    if include_corr1:
        root = _REPO / "data/results/render/tiago/correction/exp1" / model_tag
        for cue_dir in sorted(root.iterdir()):
            if not cue_dir.is_dir():
                continue
            targets.extend(
                g
                for g in sorted((cue_dir / "correction").glob("mm19_*.gif"))
                if "_pose_pool" not in g.name
            )

    for gif in targets:
        scanned += 1
        partial = _partial_tile_count(gif)
        if partial == 0:
            continue
        if dry_run:
            print(f"would reencode {gif} (partial_frames={partial})")
            changed += 1
            continue
        if reencode_gif(gif):
            changed += 1
            print(f"reencoded {gif.name} (was partial={partial})")

    print(f"done: scanned={scanned} reencoded={changed}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--no-originals", action="store_true")
    ap.add_argument("--no-corr1", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(
        model_tag=args.model_tag,
        include_originals=not args.no_originals,
        include_corr1=not args.no_corr1,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
