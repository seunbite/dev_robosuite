#!/usr/bin/env python3
"""Batch corr1 from new_render (orig config) for exp1 tiago cues."""
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
    default_config_path,
    load_config_list,
)
from run_corr1_from_new_render import run  # noqa: E402

from render_paths import render_gif_path  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
LOG_PATH = (
    _REPO
    / "data/results/verify/tiago/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "corr1_log.jsonl"
)


def _corr_gif(model_tag: str, exp_id: str, cue: str, idx: int) -> Path:
    return render_gif_path("tiago", model_tag, exp_id, idx, cue, "corr")


def _generation_gif(model_tag: str, exp_id: str, cue: str, idx: int) -> Path:
    return render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen")


def _done_cues_from_log() -> set[str]:
    if not LOG_PATH.is_file():
        return set()
    out: set[str] = set()
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.add(json.loads(line)["cue"])
    return out


def cues_to_run(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    only: list[str] | None = None,
    force: bool = False,
) -> list[tuple[int, str]]:
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg_path), key=lambda r: (int(r.get("idx", 0)), str(r.get("cue", ""))))
    done = _done_cues_from_log()
    want = set(only) if only else None
    out: list[tuple[int, str]] = []
    for row in rows:
        cue = str(row.get("cue", ""))
        if not cue:
            continue
        if want is not None and cue not in want:
            continue
        idx = int(row.get("idx", -1))
        if not force:
            if cue in done:
                continue
            if _corr_gif(model_tag, exp_id, cue, idx).is_file():
                continue
        if not _generation_gif(model_tag, exp_id, cue, idx).is_file():
            continue
        out.append((idx, cue))
    return out


def run_batch(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    only: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
    inner_exp_id: str = "0",
    build_html: bool = False,
) -> list[dict]:
    todo = cues_to_run(model_tag=model_tag, exp_id=exp_id, only=only, force=force)
    results: list[dict] = []
    print(f"corr1-from-new-render batch: {len(todo)} cues (force={force}, dry_run={dry_run})")
    if dry_run:
        for idx, cue in todo:
            print(f"  g{idx:02d} {cue}")
        return results

    for n, (idx, cue) in enumerate(todo, start=1):
        t0 = time.time()
        print(f"[{n}/{len(todo)}] g{idx:02d} {cue} ...", flush=True)
        try:
            summary = run(
                cue=cue,
                model_tag=model_tag,
                exp_id=exp_id,
                force=True,
                inner_exp_id=inner_exp_id,
            )
            entry = {"idx": idx, "cue": cue, "ok": True, **summary}
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
        print(f"  -> {entry.get('action', 'error')} ({entry.get('elapsed_s', round(time.time() - t0, 1))}s)", flush=True)
        if build_html and entry.get("ok"):
            from build_correction_overview_html import build_overview_html  # noqa: WPS433

            html_path = build_overview_html(model_tag=model_tag, exp_id=exp_id)
            print(f"  HTML updated: {html_path}", flush=True)

    summary = {
        "n_run": len(results),
        "n_ok": sum(1 for r in results if r.get("ok")),
        "n_fail": sum(1 for r in results if not r.get("ok")),
    }
    print(json.dumps(summary, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch corr1 from new_render for exp1 tiago")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--cues", default=None, help="Comma-separated cue subset")
    ap.add_argument("--force", action="store_true", help="Re-run even if already in log or new_render/correction exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inner-exp", default="0")
    ap.add_argument("--build-html", action="store_true")
    args = ap.parse_args()
    only = [c.strip() for c in args.cues.split(",") if c.strip()] if args.cues else None
    results = run_batch(
        model_tag=args.model_tag,
        exp_id=args.exp,
        only=only,
        force=args.force,
        dry_run=args.dry_run,
        inner_exp_id=args.inner_exp,
        build_html=args.build_html,
    )
    if args.build_html and not args.dry_run and results:
        from build_correction_overview_html import build_overview_html  # noqa: WPS433

        path = build_overview_html(model_tag=args.model_tag, exp_id=args.exp)
        print(f"Overview HTML: {path}")


if __name__ == "__main__":
    main()
