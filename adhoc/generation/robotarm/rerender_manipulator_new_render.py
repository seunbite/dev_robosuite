#!/usr/bin/env python3
"""Re-render manipulator exp1 configs with xyz-trim renderer → new_render/."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pilot90_paths import load_config_list, result_config_path  # noqa: E402
from render_paths import MANIPULATOR_RENDER_HZ, manipulator_cue_render_dir  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
RENDER_VARIANT = "new_render"
LOG_PATH = (
    _REPO
    / "data/results/verify/manipulator/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "orig_new_render_log.jsonl"
)


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    only: list[str] | None = None,
    dry_run: bool = False,
    skip_done: bool = True,
    build_html: bool = False,
    hz: int = MANIPULATOR_RENDER_HZ,
) -> None:
    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    from legacy.motion_generation_core import generate  # noqa: WPS433

    cfg_path = result_config_path(exp_id, model_tag)
    rows = sorted(load_config_list(cfg_path), key=lambda r: int(r.get("idx", 0)))
    want = set(only) if only else None
    todo = [r for r in rows if str(r.get("cue", "")) and (want is None or str(r["cue"]) in want)]

    print(
        f"new_render cues={len(todo)} hz={hz} "
        f"(dry_run={dry_run}, skip_done={skip_done})"
    )
    summary = {"ok": 0, "skipped": 0, "errors": 0}

    for n, row in enumerate(todo, start=1):
        idx = int(row.get("idx", -1))
        cue = str(row.get("cue", ""))
        out = manipulator_cue_render_dir(model_tag, exp_id, cue, RENDER_VARIANT)
        iwa = out / "IIWA"
        if skip_done and iwa.is_dir() and any(iwa.glob("*.gif")):
            summary["skipped"] += 1
            if dry_run:
                print(f"  [{n}/{len(todo)}] c{idx} {cue} (skip)")
            continue
        if dry_run:
            print(f"  [{n}/{len(todo)}] c{idx} {cue} → {iwa}/")
            continue

        t0 = time.time()
        try:
            print(f"[{n}/{len(todo)}] c{idx} {cue} ...", flush=True)
            iwa.mkdir(parents=True, exist_ok=True)
            generate(
                robot="IIWA",
                cue=cue,
                cue_idx=idx,
                pose_index=None,
                jsonl_path=str(_REPO / "data/seed/_remainder/closest_poses_results.jsonl"),
                config_path=str(cfg_path),
                output_dir=str(out),
                hz=hz,
                top_k=1,
                ignore_xyz=False,
            )
            cands = sorted(iwa.glob(f"*_IIWA_{cue}_*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
            gif = cands[0] if cands else None
            ok = gif is not None and gif.is_file()
            entry = {
                "variant": RENDER_VARIANT,
                "idx": idx,
                "cue": cue,
                "hz": hz,
                "ok": ok,
                "gif": str(gif) if gif else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
            if ok:
                summary["ok"] += 1
            else:
                summary["errors"] += 1
            print(f"  → {gif}", flush=True)
        except Exception as exc:
            entry = {
                "variant": RENDER_VARIANT,
                "idx": idx,
                "cue": cue,
                "hz": hz,
                "ok": False,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1),
            }
            summary["errors"] += 1
            print(f"  ERR: {exc}", flush=True)
        _append_log(entry)

    print(f"done: {summary}")
    if build_html and not dry_run:
        from build_manipulator_exp1_correction_html import build  # noqa: WPS433

        path = build(model_tag=model_tag, exp_id=exp_id)
        print(f"[html] {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Manipulator exp1 → new_render (xyz-trim)")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--only", default=None, help="Comma-separated cue names")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-render even if GIF exists")
    ap.add_argument("--hz", type=int, default=MANIPULATOR_RENDER_HZ)
    ap.add_argument("--build-html", action="store_true")
    args = ap.parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None
    run(
        model_tag=args.model_tag,
        exp_id=args.exp,
        only=only,
        dry_run=args.dry_run,
        skip_done=not args.force,
        build_html=args.build_html,
        hz=args.hz,
    )


if __name__ == "__main__":
    main()
