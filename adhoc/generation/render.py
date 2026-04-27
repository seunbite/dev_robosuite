#!/usr/bin/env python3
"""Thin dispatcher: run ``generation/<robot>/render.py``."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parent


def run(robot: str = "robotarm", **kwargs) -> None:
    aliases = {"robotarm": "robotarm", "manipulator": "robotarm", "google_robot": "google_robot", "quadruped": "quadruped"}
    rk = robot.strip().lower().replace("-", "_")
    if rk not in aliases:
        raise ValueError("robot must be robotarm/manipulator, google_robot, or quadruped")
    script = ROOT / aliases[rk] / "render.py"
    cmd = f"cd '{ROOT.parent.parent}' && \"{sys.executable}\" \"{script}\""
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
