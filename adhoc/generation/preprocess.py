#!/usr/bin/env python3
"""Thin dispatcher: run ``generation/<robot>/preprocess.py``."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import announce_output  # noqa: E402


def run(robot: str = "robotarm", **kwargs) -> None:
    aliases = {"robotarm": "robotarm", "manipulator": "robotarm", "tiago": "tiago", "quadruped": "quadruped"}
    rk = robot.strip().lower().replace("-", "_")
    if rk not in aliases:
        raise ValueError("robot must be robotarm/manipulator, tiago, or quadruped")
    script = ROOT / aliases[rk] / "preprocess.py"
    announce_output(script.parent, None, kind="파일")
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
