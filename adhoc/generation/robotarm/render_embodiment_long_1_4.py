#!/usr/bin/env python3
"""Render Core + Long1–4 configs by platform."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.robotarm.embodiment_renderers import (  # noqa: E402
    render_roomba,
    render_unitree_g1,
)
from adhoc.generation.robotarm.motion_code_to_config import (  # noqa: E402
    render_tiago,
    render_manipulator,
    render_quadruped,
)

CONFIG_ROOT = _REPO / "data/results/motion_configs/embodiment_long_1_4"
RENDER_ROOT = _REPO / "data/results/render/embodiment_long_1_4"
GT_RENDER_ROOT = _REPO / "data/results/render/embodiment_motion_code_gt"
LEVELS = ("long1", "long2", "long3", "long4")
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")


def _gif_dir(root: Path, platform: str) -> Path:
    if platform == "single_arm":
        return root / "manipulator/gif/IIWA"
    return root / platform / "gif"


def copy_core() -> None:
    for platform in PLATFORMS:
        source = _gif_dir(GT_RENDER_ROOT, platform)
        target = _gif_dir(RENDER_ROOT / "core", platform)
        target.mkdir(parents=True, exist_ok=True)
        for ext in ("*.gif", "*.mp4"):
            for path in source.glob(ext):
                shutil.copy2(path, target / path.name)


def _rows(level: str, platform: str) -> list[dict]:
    data = json.loads(
        (CONFIG_ROOT / level / "embodiment_motion_code_all.json").read_text(
            encoding="utf-8"
        )
    )
    return [entry["platforms"][platform] for entry in data]


def render(platform: str, levels: list[str], *, force: bool) -> None:
    for level in levels:
        rows = _rows(level, platform)
        out_root = RENDER_ROOT / level
        print(f"[long-render] {platform}/{level}: {len(rows)} rows", flush=True)
        if platform == "humanoid":
            render_unitree_g1(
                rows, _gif_dir(out_root, platform), force=force
            )
        elif platform == "roomba":
            render_roomba(rows, _gif_dir(out_root, platform), force=force)
        elif platform == "tiago":
            render_tiago(rows, out_root=out_root, force=force)
        elif platform == "single_arm":
            render_manipulator(rows, out_root=out_root, force=force)
            shutil.rmtree(out_root / "manipulator/_scratch", ignore_errors=True)
        else:
            render_quadruped(
                rows,
                config_path=CONFIG_ROOT / level / "result_quadruped.json",
                out_root=out_root,
                force=force,
            )
            gif_dir = out_root / "quadruped/gif"
            for duplicate in gif_dir.glob("lc19_*"):
                if duplicate.is_file():
                    duplicate.unlink()
            shutil.rmtree(gif_dir / "_per_cue_configs", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=PLATFORMS)
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-core", action="store_true")
    args = parser.parse_args()
    if args.copy_core:
        copy_core()
    if args.platform:
        render(args.platform, args.levels, force=args.force)


if __name__ == "__main__":
    main()
