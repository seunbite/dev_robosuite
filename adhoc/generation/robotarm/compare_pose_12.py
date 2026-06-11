#!/usr/bin/env python3
"""Exp 6: 12-tile pose comparison (multitile GT identification)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from compare_pose_multitile import DEFAULT_IMG_DIR, DEFAULT_OUT, run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Exp 6 — compare pose (12-tile grid)")
    p.add_argument("--consolidated-json", type=Path, default=None)
    p.add_argument("--tile-dir", type=Path, default=None)
    p.add_argument("--tile-pick-json", type=Path, default=None)
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMG_DIR.parent / "pose_multitile_gt_pilot90_grid12")
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT.parent / "pilot90_pose_multitile_grid12.json")
    p.add_argument("--model", default=None)
    p.add_argument(
        "--vlm-backend",
        default=os.getenv("VLM_BACKEND", "transformers"),
        choices=["transformers", "hf", "local", "vllm-local", "vllm", "openai", "qwen", "gemini"],
    )
    p.add_argument("--max-cues", type=int, default=None)
    p.add_argument("--cue-indices", type=str, default=None)
    p.add_argument("--cues", type=str, default=None)
    p.add_argument("--temporal-prompt", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.grid_sizes = "12"
    if args.consolidated_json is None:
        from pilot90_paths import GT_PATH  # noqa: WPS433

        args.consolidated_json = GT_PATH
    if args.tile_dir is None:
        from pilot90_paths import TILE_DIR  # noqa: WPS433

        args.tile_dir = TILE_DIR
    if args.tile_pick_json is None:
        from pilot90_paths import TILE_PICK  # noqa: WPS433

        args.tile_pick_json = TILE_PICK
    if args.max_cues is None:
        from pilot90_paths import N_CUES  # noqa: WPS433

        args.max_cues = N_CUES
    run(args)


if __name__ == "__main__":
    main()
