#!/usr/bin/env python3
"""Build MP4 manifest for pilot-90 step 8 (GIF→MP4 from run/IIWA)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from motion_media_paths import PILOT90_MOTION_CFG, prepare_pilot90_motion_mp4s, write_pilot90_manifest  # noqa: E402
from pilot90_experiment_suite import manifest90_rows_from_cfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare pilot-90 motion MP4s for step 8")
    p.add_argument("--config-json", type=Path, default=_REPO / PILOT90_MOTION_CFG)
    p.add_argument(
        "--render-missing",
        action="store_true",
        help="MuJoCo-render missing GIFs (needs closest_poses_results.jsonl + ffmpeg). "
        "Default: only GIF→MP4 from run/IIWA or motion_vlm_verify_pilot90/gif.",
    )
    args = p.parse_args()

    rows = manifest90_rows_from_cfg(json.loads(args.config_json.read_text(encoding="utf-8")))
    todo = [(int(r["idx"]), str(r["cue"])) for r in rows]
    ready, failures = prepare_pilot90_motion_mp4s(
        _REPO,
        _HERE,
        todo,
        config_json=args.config_json,
        render_missing=args.render_missing,
    )
    manifest = write_pilot90_manifest(_REPO, rows)
    print(f"[done] {ready}/{len(todo)} mp4 ready → {manifest}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} issues (first 5):", flush=True)
        for line in failures[:5]:
            print(f"  {line}", flush=True)
    if ready == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
