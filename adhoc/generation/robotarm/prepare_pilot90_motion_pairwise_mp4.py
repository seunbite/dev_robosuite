#!/usr/bin/env python3
"""Render GT vs neg-axis pairwise MP4s for pilot-90 step 10."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_pilot90_motion_pairwise_specs import main as refresh_specs  # noqa: E402
from motion_pairwise_media import prepare_pilot90_pairwise_mp4s  # noqa: E402
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare pilot-90 motion pairwise MP4s (step 10)")
    p.add_argument("--out-dir", type=Path, default=MOTION_PAIRWISE_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-spec-refresh", action="store_true")
    args = p.parse_args()

    ready, failures = prepare_pilot90_pairwise_mp4s(
        out_dir=args.out_dir,
        force=args.force,
        limit=args.limit,
    )
    if not args.skip_spec_refresh:
        refresh_specs()
    print(f"[done] {ready} pairwise mp4 ready → {args.out_dir}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} issues (first 5):", flush=True)
        for line in failures[:5]:
            print(f"  {line}", flush=True)
    if ready == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
