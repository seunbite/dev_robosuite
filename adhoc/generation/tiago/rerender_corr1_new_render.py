#!/usr/bin/env python3
"""Re-render exp original configs (xyz-trim) → flat exp render directory."""
from __future__ import annotations

import argparse
import json
import os
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
    default_render_dir,
    load_config_list,
    render_instance,
)
from render_paths import render_gif_path  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
LOG_PATH = (
    _REPO
    / "data/results/verify/tiago/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "orig_render_log.jsonl"
)


def _generation_gif(model_tag: str, exp_id: str, cue: str, idx: int) -> Path:
    return render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen")


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
) -> None:
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg_path), key=lambda r: int(r.get("idx", 0)))
    want = set(only) if only else None
    todo: list[dict] = []
    for row in rows:
        cue = str(row.get("cue", ""))
        if not cue or (want is not None and cue not in want):
            continue
        todo.append(row)

    print(f"orig config → generation render: {len(todo)} cues (dry_run={dry_run})")
    summary = {"ok": 0, "errors": 0}

    for n, row in enumerate(todo, start=1):
        idx = int(row.get("idx", -1))
        cue = str(row.get("cue", ""))
        out = default_render_dir("tiago", model_tag, cue, exp_id, 0)  # type: ignore[arg-type]
        gif = _generation_gif(model_tag, exp_id, cue, idx)
        if dry_run:
            print(f"  [{n}/{len(todo)}] g{idx:02d} {cue} → {gif}")
            continue

        t0 = time.time()
        try:
            print(f"[{n}/{len(todo)}] g{idx:02d} {cue} ...", flush=True)
            art = render_instance(
                "tiago",
                row,
                config_path=cfg_path,
                render_dir=out,
                model_tag=model_tag,
                exp_id=exp_id,  # type: ignore[arg-type]
                round_n=0,
            )
            ok = art.gif is not None and art.gif.is_file()
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": ok,
                "gif": str(art.gif) if art.gif else None,
                "elapsed_s": round(time.time() - t0, 1),
            }
            if ok:
                summary["ok"] += 1
            else:
                summary["errors"] += 1
            print(f"  → {art.gif}", flush=True)
        except Exception as exc:
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": False,
                "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1),
            }
            summary["errors"] += 1
            print(f"  ERR: {exc}", flush=True)
        _append_log(entry)

    print(f"done: {summary}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--only", default=None, help="Comma-separated cue names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None
    run(model_tag=args.model_tag, exp_id=args.exp, only=only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
