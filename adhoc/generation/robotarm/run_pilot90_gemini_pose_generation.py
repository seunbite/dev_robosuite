#!/usr/bin/env python3
"""Gemini pose generation (task 1) for pilot-90 pairwise-eligible cues (88).

Skips cues already listed in the checkpoint JSON (resume default).

  source APIKEY.sh
  python adhoc/generation/robotarm/run_pilot90_gemini_pose_generation.py
  python ... --dry-run
  python ... --force --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _HERE / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

POSE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"
)
CHECKPOINT = _REPO / "data/results/verify/pilot90_gemini/pose_generation_checkpoint.json"
MERGED = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_no_reasoning_merged_104.json"
)


def _pairwise88_cues() -> list[tuple[int, str]]:
    from motion_pairwise_media import _pairwise_specs

    return [(int(s["idx"]), str(s["cue"])) for s in _pairwise_specs()]


def _load_checkpoint(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(c) for c in (data.get("done_cues") or [])}


def _save_checkpoint(path: Path, done: set[str], *, model: str, errors: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "model": model,
                "n_done": len(done),
                "done_cues": sorted(done),
                "errors": errors,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Gemini pose generation for pilot-90 (88 pairwise cues)")
    p.add_argument("--out-json", type=Path, default=POSE_CFG)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    p.add_argument("--model", default="gemini-2.5-pro")
    p.add_argument("--delay", type=float, default=2.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Re-run even if cue is in checkpoint")
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoint")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    work = _pairwise88_cues()
    if args.limit:
        work = work[: args.limit]
    done = set() if args.no_resume else _load_checkpoint(args.checkpoint)
    pending = [(i, c) for i, c in work if args.force or c not in done]

    print(f"pairwise-eligible: {len(work)}  done: {len(done)}  pending: {len(pending)}", flush=True)
    if not pending:
        print("Nothing to generate.", flush=True)
        return

    if args.dry_run:
        print("Would generate:", [c for _, c in pending], flush=True)
        return

    from adhoc.generation.embodiment_sources import resolve_embodiment_paths
    from legacy.config_gen_single import generate_motion_config

    prompt_path, shots_path, _, _ = resolve_embodiment_paths("manipulator")
    out_path = args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.is_file():
        out_path.write_text("[]\n", encoding="utf-8")

    errors: list[str] = list(
        json.loads(args.checkpoint.read_text(encoding="utf-8")).get("errors", [])
        if args.checkpoint.is_file()
        else []
    )

    for idx, cue in pending:
        ok = False
        last_err = ""
        for attempt in range(args.max_retries + 1):
            try:
                generate_motion_config(
                    cue_name=cue,
                    cue_idx=idx,
                    model_name=args.model,
                    prompt_file=str(prompt_path),
                    shots_json=str(shots_path),
                    config_json=str(out_path),
                    require_reasoning=True,
                )
                ok = True
                break
            except Exception as e:
                last_err = str(e)
                if attempt < args.max_retries:
                    time.sleep(args.delay * (2**attempt))
        if ok:
            done.add(cue)
            print(f"[OK] {idx} {cue}", flush=True)
            _save_checkpoint(args.checkpoint, done, model=args.model, errors=errors)
        else:
            msg = f"{cue}: {last_err[:240]}"
            errors.append(msg)
            print(f"[FAIL] {msg}", flush=True)
            _save_checkpoint(args.checkpoint, done, model=args.model, errors=errors)
        time.sleep(args.delay)

    print(f"Finished. done={len(done)} errors={len(errors)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
