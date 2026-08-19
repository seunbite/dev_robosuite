#!/usr/bin/env python3
"""Re-render all exp1 corr1 tiago GIFs and report flicker (RGB spike) stats."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    config_snapshot_path,
    default_config_path,
    default_render_dir,
    load_config_list,
    render_instance,
)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
LOG_PATH = (
    _REPO
    / "data/results/verify/tiago/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "corr1_rerender_log.jsonl"
)


def _gif_path(cue: str, idx: int) -> Path:
    safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return (
        _REPO
        / "data/results/render/tiago/correction"
        / f"exp{DEFAULT_EXP_ID}"
        / DEFAULT_MODEL_TAG
        / cue
        / "correction"
        / f"mm19_g{idx:02d}_{safe}.gif"
    )


def spike_stats(path: Path, *, thresh: float = 15.0) -> dict:
    if not path.is_file():
        return {"exists": False, "frames": 0, "spikes": 0, "max_diff": 0.0}
    im = Image.open(path)
    frames: list[np.ndarray] = []
    try:
        while True:
            frames.append(np.array(im.convert("RGB")))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    spikes = 0
    max_diff = 0.0
    spike_frames: list[int] = []
    for i in range(1, len(frames)):
        d = float(np.mean(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))))
        max_diff = max(max_diff, d)
        if d > thresh:
            spikes += 1
            if len(spike_frames) < 8:
                spike_frames.append(i)
    return {
        "exists": True,
        "frames": len(frames),
        "spikes": spikes,
        "max_diff": round(max_diff, 2),
        "spike_frames": spike_frames,
    }


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
    skip_existing_ok: bool = False,
) -> None:
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg_path), key=lambda r: int(r.get("idx", 0)))
    want = set(only) if only else None
    todo: list[tuple[int, str]] = []
    for row in rows:
        cue = str(row.get("cue", ""))
        if not cue or (want is not None and cue not in want):
            continue
        snap = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
        if not snap.is_file():
            continue
        todo.append((int(row.get("idx", -1)), cue))

    print(f"corr1 rerender: {len(todo)} cues (dry_run={dry_run})")
    summary = {"ok": 0, "still_bad": 0, "errors": 0, "skipped": 0}

    for n, (idx, cue) in enumerate(todo, start=1):
        gif = _gif_path(cue, idx)
        before = spike_stats(gif)
        if skip_existing_ok and before.get("spikes", 0) == 0:
            summary["skipped"] += 1
            continue

        snap = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
        if dry_run:
            print(f"  [{n}/{len(todo)}] g{idx:02d} {cue} before_spikes={before.get('spikes')}")
            continue

        t0 = time.time()
        try:
            snap_rows = json.loads(snap.read_text(encoding="utf-8"))
            row = snap_rows[0] if isinstance(snap_rows, list) else snap_rows
            out = default_render_dir("tiago", model_tag, cue, exp_id, 1)  # type: ignore[arg-type]
            print(f"[{n}/{len(todo)}] g{idx:02d} {cue} ...", flush=True)
            render_instance(
                "tiago",
                row,
                config_path=snap,
                render_dir=out,
                model_tag=model_tag,
                exp_id=exp_id,  # type: ignore[arg-type]
                round_n=1,
                ignore_xyz=False,
            )
            after = spike_stats(gif)
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": True,
                "before": before,
                "after": after,
                "elapsed_s": round(time.time() - t0, 1),
            }
            if after.get("spikes", 0) == 0:
                summary["ok"] += 1
            else:
                summary["still_bad"] += 1
            print(
                f"  spikes {before.get('spikes')} → {after.get('spikes')} "
                f"max_diff {before.get('max_diff')} → {after.get('max_diff')}",
                flush=True,
            )
        except Exception as exc:
            entry = {
                "idx": idx,
                "cue": cue,
                "ok": False,
                "error": str(exc),
                "before": before,
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
    ap.add_argument("--skip-existing-ok", action="store_true")
    args = ap.parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None
    run(
        model_tag=args.model_tag,
        exp_id=args.exp,
        only=only,
        dry_run=args.dry_run,
        skip_existing_ok=args.skip_existing_ok,
    )


if __name__ == "__main__":
    main()
