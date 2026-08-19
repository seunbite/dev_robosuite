#!/usr/bin/env python3
"""Render GT vs neg-repetition pairwise MP4s for pilot-90 exp11."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from build_pilot90_motion_rep_pairwise_preview import (  # noqa: E402
    REP_PAIRWISE_DIR,
    build_rep_pairwise_specs,
    load_vlm_results,
    write_rep_pairwise_html,
)
from pilot90_paths import VERIFY_EXP_DIR  # noqa: E402
from motion_media_paths import check_pose_jsonl  # noqa: E402
from motion_pairwise_rep_media import (  # noqa: E402
    SPECS_NAME,
    prepare_pilot90_rep_pairwise_mp4s,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare pilot-90 exp11 rep-flip pairwise MP4s")
    p.add_argument("--out-dir", type=Path, default=REP_PAIRWISE_DIR)
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--refresh-specs", action="store_true", help="Rebuild specs JSON before render")
    p.add_argument("--open-html", action="store_true")
    args = p.parse_args()

    if args.refresh_specs:
        _, payload = build_rep_pairwise_specs()
        spec_path = args.out_dir / SPECS_NAME
        spec_path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[specs] refreshed -> {spec_path}", flush=True)

    jpath = check_pose_jsonl(_REPO)
    print(f"[preflight] pose JSONL ok: {jpath}", flush=True)

    ready, failures = prepare_pilot90_rep_pairwise_mp4s(
        out_dir=args.out_dir,
        force=args.force,
        limit=args.limit,
    )
    specs_payload = __import__("json").loads((args.out_dir / SPECS_NAME).read_text(encoding="utf-8"))
    vlm_by_idx, vlm_model = load_vlm_results(
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro_pilot30.json",
        VERIFY_EXP_DIR / "result_exp11_gemini-2.5-pro.json",
    )
    html_path = write_rep_pairwise_html(
        specs_payload,
        args.out_dir / "pairwise_rep_input_review.html",
        vlm_by_idx=vlm_by_idx,
        vlm_model=vlm_model,
    )
    print(f"[done] {ready} pairwise rep mp4 ready → {args.out_dir}", flush=True)
    print(f"[html] {html_path}", flush=True)
    if failures:
        print(f"[warn] {len(failures)} issues (first 5):", flush=True)
        for line in failures[:5]:
            print(f"  {line}", flush=True)
    if args.open_html and sys.platform == "darwin":
        import subprocess

        subprocess.run(["open", str(html_path)], check=False)
    if ready == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
