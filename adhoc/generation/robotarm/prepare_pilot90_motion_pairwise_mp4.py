#!/usr/bin/env python3
"""Render GT vs neg-axis pairwise MP4s for pilot-90 step 10."""
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

from build_pilot90_motion_pairwise_specs import main as refresh_specs  # noqa: E402
from motion_media_paths import check_pose_jsonl  # noqa: E402
from motion_pairwise_media import (  # noqa: E402
    load_idx_subset,
    prepare_pilot90_pairwise_mp4s,
    write_gemini_correct_subset,
    write_motion_gt_correct_subset,
)
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare pilot-90 motion pairwise MP4s (step 10)")
    p.add_argument("--out-dir", type=Path, default=MOTION_PAIRWISE_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--subset-json", type=Path, default=None, help="Only render cues listed in idxs[]")
    p.add_argument(
        "--from-gemini-exp10",
        type=Path,
        default=None,
        help="Build subset-json from vlm_correct rows in exp10 results, then render",
    )
    p.add_argument(
        "--from-motion-gt-score",
        action="store_true",
        help="Score 88 configs vs movement component GT; render matching cues only",
    )
    p.add_argument("--skip-spec-refresh", action="store_true")
    args = p.parse_args()

    subset_path = args.subset_json
    if args.from_gemini_exp10:
        subset_path = write_gemini_correct_subset(args.from_gemini_exp10)
        print(f"[subset] gemini correct -> {subset_path}", flush=True)
    if args.from_motion_gt_score:
        subset_path = write_motion_gt_correct_subset()
        data = json.loads(subset_path.read_text(encoding="utf-8"))
        print(
            f"[subset] motion GT correct {data['n_correct']}/{data['n_pairwise']} -> {subset_path}",
            flush=True,
        )

    jpath = check_pose_jsonl(_REPO)
    print(f"[preflight] pose JSONL ok: {jpath}", flush=True)

    idx_subset = load_idx_subset(subset_path)
    if idx_subset:
        print(f"[subset] rendering {len(idx_subset)} cues", flush=True)

    ready, failures = prepare_pilot90_pairwise_mp4s(
        out_dir=args.out_dir,
        force=args.force,
        limit=args.limit,
        idx_subset=idx_subset,
    )
    if not args.skip_spec_refresh:
        if idx_subset:
            from motion_pairwise_media import write_pairwise_specs  # noqa: WPS433

            sidecars = sorted(args.out_dir.glob("*_pair_spec.json"))
            entries = []
            for sc in sidecars:
                entry = json.loads(sc.read_text(encoding="utf-8"))
                if int(entry.get("idx", -1)) in idx_subset:
                    entries.append(entry)
            if entries:
                stem = (subset_path.stem if subset_path else "subset").replace("_subset", "")
                specs_name = f"pairwise_specs_{stem}.json"
                path = write_pairwise_specs(entries, args.out_dir, specs_name=specs_name)
                print(f"[specs] subset {len(entries)} -> {path}", flush=True)
        else:
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
