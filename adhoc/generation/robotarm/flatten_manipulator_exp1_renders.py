#!/usr/bin/env python3
"""Copy manipulator exp1 nested renders → flat exp1 dir (generation / corr1 / corr2)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    load_config_list,
)
from pilot90_paths import row_generation_done  # noqa: E402
from render_paths import (  # noqa: E402
    manipulator_cue_render_dir,
    render_gif_path,
    render_media_paths,
)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
LOG_PATH = (
    _REPO
    / "data/results/verify/manipulator/correction"
    / f"exp{DEFAULT_EXP_ID}"
    / DEFAULT_MODEL_TAG
    / "flatten_exp1_renders_log.jsonl"
)


def _latest_gif(dir_path: Path) -> Path | None:
    iwa = dir_path / "IIWA"
    if not iwa.is_dir():
        return None
    cands = sorted(iwa.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _copy_sidecar(src_gif: Path, dst_gif: Path) -> None:
    dst_gif.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_gif, dst_gif)
    for ext in (".mp4",):
        side = src_gif.with_suffix(ext)
        if side.is_file():
            shutil.copy2(side, dst_gif.with_suffix(ext))


def _append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def flatten(
    *,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, int]:
    cfg_path = default_config_path("manipulator", model_tag, exp_id)  # type: ignore[arg-type]
    rows = [r for r in load_config_list(cfg_path) if row_generation_done(r)]
    summary = {"gen": 0, "corr1": 0, "corr2": 0, "skip": 0, "missing": 0}

    for row in sorted(rows, key=lambda r: int(r.get("idx", 0))):
        cue = str(row.get("cue", ""))
        idx = int(row.get("idx", -1))
        if not cue or idx < 0:
            continue

        jobs: list[tuple[str, Path | None, str]] = [
            (
                "gen",
                _latest_gif(manipulator_cue_render_dir(model_tag, exp_id, cue, "new_render")),
                "gen",
            ),
            (
                "corr1",
                _latest_gif(manipulator_cue_render_dir(model_tag, exp_id, cue, "correction")),
                "corr",
            ),
        ]

        cue_dir = correction_cue_dir("manipulator", exp_id, model_tag, cue)  # type: ignore[arg-type]
        rec2 = correction_record_path(cue_dir, 2)
        if rec2.is_file():
            jobs.append(
                (
                    "corr2",
                    _latest_gif(manipulator_cue_render_dir(model_tag, exp_id, cue, "correction")),
                    "corr_r2",
                ),
            )

        for label, src, role in jobs:
            dst, _, _ = render_media_paths("manipulator", model_tag, exp_id, idx, cue, role)
            if src is None or not src.is_file():
                if label != "corr2":
                    summary["missing"] += 1
                continue
            if dst.is_file() and not force:
                summary["skip"] += 1
                continue
            entry = {
                "idx": idx,
                "cue": cue,
                "role": label,
                "src": str(src),
                "dst": str(dst),
                "dry_run": dry_run,
            }
            if dry_run:
                print(f"[dry-run] {label} g{idx:02d} {cue}\n  {src}\n  -> {dst}")
            else:
                _copy_sidecar(src, dst)
                print(f"[{label}] g{idx:02d} {cue} -> {dst.name}")
            _append_log(entry)
            summary[label] += 1

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten manipulator exp1 GIFs into gemini-2.5-pro/exp1")
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="Overwrite existing flat GIFs")
    args = ap.parse_args()
    summary = flatten(
        model_tag=args.model_tag,
        exp_id=args.exp,
        dry_run=args.dry_run,
        force=args.force,
    )
    print("summary:", summary)


if __name__ == "__main__":
    main()
