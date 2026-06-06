#!/usr/bin/env python3
"""Prepare pilot-40 motion GIFs + MP4s for step 8 (no VLM / no GPU model load)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_media_paths import (  # noqa: E402
    PILOT40_MOTION_CFG,
    check_pilot40_render_prereqs,
    prepare_pilot40_motion_mp4s,
    write_pilot40_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Render pilot-40 motion GIFs and build MP4s for step 8")
    p.add_argument("--config-json", type=Path, default=_REPO / PILOT40_MOTION_CFG)
    p.add_argument("--hz", type=int, default=10)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    check_pilot40_render_prereqs(_REPO)
    rows = json.loads(args.config_json.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    todo = [(int(r["idx"]), str(r["cue"])) for r in rows]
    ready, failures = prepare_pilot40_motion_mp4s(
        _REPO, _HERE, todo, config_json=args.config_json, hz=args.hz
    )
    manifest = write_pilot40_manifest(_REPO, rows)
    print(f"[done] {ready}/{len(todo)} mp4 ready → {manifest}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} render failures (first 5):", flush=True)
        for line in failures[:5]:
            print(f"  {line}", flush=True)
    if ready == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
