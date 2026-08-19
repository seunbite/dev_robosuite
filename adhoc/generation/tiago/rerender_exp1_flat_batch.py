#!/usr/bin/env python3
"""Re-render tiago exp1 GIFs into flat exp1/ (+ corr1/, corr2/) from saved configs."""
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
    config_snapshot_path,
    correction_stem,
    default_config_path,
    default_render_dir,
    load_config_list,
    render_instance,
)
from render_paths import render_media_paths, render_role_for_round  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
CORR2_SNAP = _REPO / (
    "data/results/motion_configs/tiago/exp/correction"
    f"/{DEFAULT_MODEL_TAG}/exp{DEFAULT_EXP_ID}"
)
LOG_PATH = (
    _REPO
    / "data/results/verify/tiago/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "flat_rerender_log.jsonl"
)


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_snapshot(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def _corr2_snapshot(cue: str) -> Path | None:
    cue_dir = CORR2_SNAP / cue
    for name in (
        "correction_correction_new_render_config.json",
        "correction_correction_config.json",
    ):
        p = cue_dir / name
        if p.is_file():
            return p
    return None


def cues_with_corr2() -> list[str]:
    out: list[str] = []
    if not CORR2_SNAP.is_dir():
        return out
    for cue_dir in sorted(CORR2_SNAP.iterdir()):
        if not cue_dir.is_dir():
            continue
        if _corr2_snapshot(cue_dir.name) is not None:
            out.append(cue_dir.name)
    return out


def _render_round(
    *,
    row: dict,
    config_path: Path,
    model_tag: str,
    exp_id: str,
    cue: str,
    round_n: int,
) -> Path | None:
    role = render_role_for_round(round_n)
    idx = int(row.get("idx", -1))
    gif, _, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, role)
    out = default_render_dir("tiago", model_tag, cue, exp_id, round_n)  # type: ignore[arg-type]
    art = render_instance(
        "tiago",
        row,
        config_path=config_path,
        render_dir=out,
        model_tag=model_tag,
        exp_id=exp_id,  # type: ignore[arg-type]
        round_n=round_n,
        ignore_xyz=False,
    )
    return art.gif if art.gif and art.gif.is_file() else gif if gif.is_file() else None


def run(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    only: list[str] | None = None,
    corr2_first: bool = False,
    include_corr2_only: bool = False,
    remaining_only: bool = False,
    skip_gen: bool = False,
    skip_corr1: bool = False,
    skip_corr2: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg_path), key=lambda r: int(r.get("idx", 0)))
    by_cue = {str(r.get("cue", "")): r for r in rows if r.get("cue")}

    corr2_cues = set(cues_with_corr2())
    todo: list[str] = []
    if only:
        todo = [c for c in only if c in by_cue]
    elif include_corr2_only or corr2_first:
        todo = [c for c in sorted(corr2_cues) if c in by_cue]
        if not include_corr2_only and not remaining_only:
            todo += [c for c in by_cue if c not in corr2_cues]
    elif remaining_only:
        todo = [c for c in by_cue if c not in corr2_cues]
    else:
        todo = list(by_cue.keys())

    # air_quotes first within corr2 batch when requested
    if corr2_first and "air_quotes" in todo:
        todo.remove("air_quotes")
        todo.insert(0, "air_quotes")

    summary = {"gen": 0, "corr1": 0, "corr2": 0, "skip": 0, "err": 0}
    print(f"flat rerender: {len(todo)} cues (dry_run={dry_run})", flush=True)

    for n, cue in enumerate(todo, start=1):
        row = by_cue[cue]
        idx = int(row.get("idx", -1))
        has_corr2 = cue in corr2_cues
        print(f"[{n}/{len(todo)}] g{idx:02d} {cue} corr2={has_corr2}", flush=True)

        jobs: list[tuple[str, int, Path]] = []
        if not skip_gen:
            jobs.append(("gen", 0, cfg_path))
        snap1 = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
        if not skip_corr1 and snap1.is_file():
            jobs.append(("corr1", 1, snap1))
        snap2 = _corr2_snapshot(cue)
        if not skip_corr2 and has_corr2 and snap2 is not None:
            jobs.append(("corr2", 2, snap2))

        for label, round_n, snap in jobs:
            gif, _, _ = render_media_paths(
                "tiago",
                model_tag,
                exp_id,
                idx,
                cue,
                render_role_for_round(round_n),
            )
            if dry_run:
                print(f"  [dry] {label} -> {gif}", flush=True)
                summary[label] += 1
                continue
            t0 = time.time()
            try:
                cfg_row = _load_snapshot(snap) if round_n > 0 else row
                out_gif = _render_round(
                    row=cfg_row,
                    config_path=snap,
                    model_tag=model_tag,
                    exp_id=exp_id,
                    cue=cue,
                    round_n=round_n,
                )
                ok = out_gif is not None and out_gif.is_file()
                entry = {
                    "idx": idx,
                    "cue": cue,
                    "round": label,
                    "ok": ok,
                    "gif": str(out_gif) if out_gif else None,
                    "elapsed_s": round(time.time() - t0, 1),
                }
                print(f"  {label}: {'OK' if ok else 'FAIL'} {out_gif}", flush=True)
                if ok:
                    summary[label] += 1
                else:
                    summary["err"] += 1
            except Exception as exc:
                entry = {
                    "idx": idx,
                    "cue": cue,
                    "round": label,
                    "ok": False,
                    "error": str(exc),
                    "elapsed_s": round(time.time() - t0, 1),
                }
                summary["err"] += 1
                print(f"  {label}: ERR {exc}", flush=True)
            _append_log(entry)

    print(f"done: {summary}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-render exp1 flat GIFs from configs")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--only", default=None, help="Comma-separated cues")
    ap.add_argument(
        "--corr2-first",
        action="store_true",
        help="Process corr2 cues first (air_quotes at front), then the rest",
    )
    ap.add_argument(
        "--corr2-only",
        action="store_true",
        help="Only cues with correction_correction_config.json",
    )
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-corr1", action="store_true")
    ap.add_argument("--skip-corr2", action="store_true")
    ap.add_argument(
        "--remaining",
        action="store_true",
        help="Only cues without corr2 config (after corr2 batch)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None
    run(
        model_tag=args.model_tag,
        exp_id=args.exp,
        only=only,
        corr2_first=args.corr2_first,
        include_corr2_only=args.corr2_only,
        remaining_only=args.remaining,
        skip_gen=args.skip_gen,
        skip_corr1=args.skip_corr1,
        skip_corr2=args.skip_corr2,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
