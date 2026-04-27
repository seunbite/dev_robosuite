#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import dev_robosuite_root  # noqa: E402

_DEFAULT_JSONL = dev_robosuite_root() / "data" / "seed" / "_remainder" / "closest_poses_results.jsonl"
_DEFAULT_ROBOTS = ["IIWA", "Panda", "Sawyer", "Kinova3", "Jaco", "UR5e", "XArm7"]


def run(
    robots: list[str] | None = None,
    reset: bool = False,
    angle_step: float = 90.0,
    jsonl_path: str | None = None,
    run_motion_generation: bool = True,
    cue_group: str = "iconic",
) -> None:
    stack = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
    stack.parent.mkdir(parents=True, exist_ok=True)
    if reset and stack.exists():
        stack.unlink()
        print(f"Removed {stack}")

    robots = robots or list(_DEFAULT_ROBOTS)
    py = sys.executable
    find_script = Path(__file__).with_name("legacy") / "find_closest_poses.py"
    for robot in robots:
        cmd = [
            py,
            str(find_script),
            "--robot",
            robot,
            "--brute_force",
            "True",
            "--angle_step",
            str(angle_step),
            "--stack_jsonl_path",
            str(stack),
        ]
        r = subprocess.run(cmd, cwd=str(_REPO))
        if r.returncode != 0:
            print(f"Warning: find_closest_poses failed for {robot} (exit {r.returncode})")

    if run_motion_generation:
        from motion_generation import run as motion_run

        motion_run(cue_group=cue_group)


if __name__ == "__main__":
    fire.Fire(run)
