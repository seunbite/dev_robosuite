"""Backward-compat shim — use verify_pose_vlm.py."""
from verify_pose_vlm import *  # noqa: F403
from verify_pose_vlm import (  # noqa: F401
    APPROPRIATE_MEANS_LINE,
    _extract_json,
    _fewshot_block,
    _first_pose,
    _load_json,
    _load_tile_pick,
    _movement_summary,
    _prompt,
    _resolve_pose_image,
    run,
)
