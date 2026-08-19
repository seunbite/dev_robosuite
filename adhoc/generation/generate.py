#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import announce_output  # noqa: E402


def run(robot: str = "robotarm", stage: str = "all", **kwargs) -> None:
    stage = stage.strip().lower()
    aliases = {"pre": "preprocess", "prep": "preprocess", "motion": "motion_generation", "gen": "motion_generation", "rendering": "render"}
    stage = aliases.get(stage, stage)
    if stage not in {"all", "preprocess", "motion_generation", "render"}:
        raise ValueError("stage must be all|preprocess|motion_generation|render")

    if stage == "all":
        announce_output(ROOT / robot, None, kind="pipeline output")
        from preprocess import run as run_pre

        run_pre(robot=robot, **kwargs)
        return

    announce_output(ROOT / robot, None, kind=f"{stage} output")
    script = ROOT / f"{stage}.py"
    cmd = f"cd '{ROOT.parent.parent}' && \"{sys.executable}\" \"{script}\" --robot='{robot}'"
    for k, v in kwargs.items():
        if v is None:
            continue
        cmd += f" --{k}='{v}'"
    print(cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    fire.Fire(run)
