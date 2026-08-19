#!/usr/bin/env python3
"""Run corr1 (--from-base) for all exp1 tiago cues missing round-1 correction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    load_config_list,
    run_instance_correction,
)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
LOG_PATH = (
    _REPO
    / "data/results/verify/tiago/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "batch_corr1_log.jsonl"
)


def _has_corr1(cue: str, *, model_tag: str, exp_id: str) -> bool:
    cue_dir = correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
    return correction_record_path(cue_dir, 1).is_file()


def cues_to_run(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    force: bool = False,
    only: list[str] | None = None,
) -> list[tuple[int, str]]:
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg_path), key=lambda r: (int(r.get("idx", 0)), str(r.get("cue", ""))))
    out: list[tuple[int, str]] = []
    want = set(only) if only else None
    for row in rows:
        cue = str(row.get("cue", ""))
        if not cue:
            continue
        if want is not None and cue not in want:
            continue
        if not force and _has_corr1(cue, model_tag=model_tag, exp_id=exp_id):
            continue
        out.append((int(row.get("idx", -1)), cue))
    return out


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_batch(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    force: bool = False,
    only: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    todo = cues_to_run(model_tag=model_tag, exp_id=exp_id, force=force, only=only)
    results: list[dict] = []
    print(f"corr1 batch: {len(todo)} cues to run (force={force})")
    if dry_run:
        for idx, cue in todo:
            print(f"  g{idx:02d} {cue}")
        return results

    for n, (idx, cue) in enumerate(todo, start=1):
        t0 = time.time()
        print(f"[{n}/{len(todo)}] g{idx:02d} {cue} ...", flush=True)
        try:
            result = run_instance_correction(
                domain="tiago",
                exp_id=exp_id,  # type: ignore[arg-type]
                cue=cue,
                config_path=cfg_path,
                model_tag=model_tag,
                backend="gemini",
                chain_from_latest=False,
                inner_exp_id="0",
                ignore_xyz=False,
            )
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": True,
                "action": result.action,
                "is_correct": result.is_correct,
                "round": result.round,
                "elapsed_s": round(time.time() - t0, 1),
                "correction_json": str(result.correction_json) if result.correction_json else None,
            }
        except Exception as exc:
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": False,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1),
            }
            print(f"  ERROR: {exc}", flush=True)
        results.append(entry)
        _append_log(entry)
        print(f"  -> {entry.get('action', 'error')} ({entry['elapsed_s']}s)", flush=True)

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch corr1 for exp1 tiago cues")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--force", action="store_true", help="Re-run even if correction.json exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cues", default=None, help="Comma-separated cue subset")
    ap.add_argument("--build-html", action="store_true", help="Build overview HTML after batch")
    args = ap.parse_args()
    only = [c.strip() for c in args.cues.split(",") if c.strip()] if args.cues else None
    results = run_batch(
        model_tag=args.model_tag,
        exp_id=args.exp,
        force=args.force,
        only=only,
        dry_run=args.dry_run,
    )
    if args.build_html and not args.dry_run:
        from build_correction_overview_html import build_overview_html  # noqa: WPS433

        path = build_overview_html(model_tag=args.model_tag, exp_id=args.exp)
        print(f"Overview HTML: {path}")
    summary = {
        "n_run": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
