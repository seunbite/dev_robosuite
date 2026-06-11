"""Backward-compat shim — use compare_pose_2.py."""
from compare_pose_2 import *  # noqa: F403
from compare_pose_2 import (  # noqa: F401 — import * skips leading _
    REPRESENTATIVE_MEANS_LINE,
    _configs_by_cue,
    _pair_prompt,
    _stitch_pair,
    run,
)
