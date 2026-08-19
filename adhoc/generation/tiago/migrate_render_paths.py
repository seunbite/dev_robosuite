#!/usr/bin/env python3
"""Migrate legacy render trees → render/{robot}/{llm}/exp{N}/<role>_g{idx}_{cue}.gif"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import _safe_cue, correction_stem, default_config_path, load_config_list  # noqa: E402
from render_paths import (  # noqa: E402
    gt_render_role,
    render_exp_dir,
    render_gif_path,
    render_media_paths,
    render_role_for_round,
)

DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_EXP = "1"


def _copy_pair(src_gif: Path, dst_gif: Path, *, dry_run: bool) -> bool:
    if not src_gif.is_file():
        return False
    dst_mp4 = dst_gif.with_suffix(".mp4")
    src_mp4 = src_gif.with_suffix(".mp4")
    dst_pose = dst_gif.parent / f"{dst_gif.stem}_pose.png"
    src_pose = src_gif.parent / f"{src_gif.stem}_pose.png"
    if dry_run:
        print(f"  {src_gif} → {dst_gif}")
        return True
    dst_gif.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_gif, dst_gif)
    if src_mp4.is_file():
        shutil.copy2(src_mp4, dst_mp4)
    if src_pose.is_file():
        shutil.copy2(src_pose, dst_pose)
    return True


def _legacy_stem(idx: int, cue: str) -> str:
    return f"mm19_g{idx:02d}_{_safe_cue(cue)}"


def migrate_cue(
    *,
    model_tag: str,
    exp_id: str,
    idx: int,
    cue: str,
    dry_run: bool,
) -> int:
    n = 0
    stem = _legacy_stem(idx, cue)
    old_root = _REPO / "data/results/render/tiago"
    cue_tree = old_root / "correction" / f"exp{exp_id}" / model_tag / cue

    mappings: list[tuple[Path, Path]] = []

    # new_render orig → generation
    mappings.append(
        (
            cue_tree / "new_render" / f"{stem}.gif",
            render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen"),
        )
    )
    # legacy exp{N}_{model}
    mappings.append(
        (
            old_root / f"exp{exp_id}_{model_tag}" / f"{stem}.gif",
            render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen"),
        )
    )
    # flat legacy at tiago root
    mappings.append(
        (
            old_root / f"{stem}.gif",
            render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen"),
        )
    )

    for rn in (1, 2, 3):
        role = render_role_for_round(rn)
        dst = render_gif_path("tiago", model_tag, exp_id, idx, cue, role)
        sub = correction_stem(rn)
        mappings.append((cue_tree / "new_render" / sub / f"{stem}.gif", dst))
        mappings.append((cue_tree / sub / f"{stem}.gif", dst))

    for variant in ("gt", "y"):
        role = gt_render_role(variant, 0)
        sub = "gt" if variant == "gt" else "gt_y"
        mappings.append(
            (
                cue_tree / sub / f"{stem}.gif",
                render_gif_path("tiago", model_tag, exp_id, idx, cue, role),
            )
        )
        for rn in (1, 2):
            corr_role = gt_render_role(variant, rn)
            corr_sub = "gt_corr" if rn == 1 else f"gt_corr_{correction_stem(rn).removeprefix('correction_')}"
            mappings.append(
                (
                    cue_tree / sub / corr_sub / f"{stem}.gif",
                    render_gif_path("tiago", model_tag, exp_id, idx, cue, corr_role),
                )
            )

    seen_dst: set[Path] = set()
    for src, dst in mappings:
        if dst in seen_dst:
            continue
        if _copy_pair(src, dst, dry_run=dry_run):
            seen_dst.add(dst)
            n += 1
    return n


def cleanup_legacy(*, model_tag: str, exp_id: str, dry_run: bool) -> None:
    old_root = _REPO / "data/results/render/tiago"
    targets = [
        old_root / "correction",
        old_root / f"exp{exp_id}_{model_tag}",
        old_root / f"exp{exp_id}_{model_tag}_media",
    ]
    for p in old_root.glob("mm19_g*.gif"):
        targets.append(p)
    for t in targets:
        if not t.exists():
            continue
        if dry_run:
            print(f"rm -rf {t}")
        elif t.is_file():
            t.unlink()
        else:
            shutil.rmtree(t, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default=DEFAULT_MODEL)
    ap.add_argument("--exp", default=DEFAULT_EXP)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-cleanup", action="store_true")
    args = ap.parse_args()

    cfg = default_config_path("tiago", args.model_tag, args.exp)  # type: ignore[arg-type]
    rows = sorted(load_config_list(cfg), key=lambda r: int(r.get("idx", 0)))
    total = 0
    for row in rows:
        cue = str(row.get("cue", ""))
        if not cue:
            continue
        idx = int(row.get("idx", -1))
        n = migrate_cue(
            model_tag=args.model_tag,
            exp_id=args.exp,
            idx=idx,
            cue=cue,
            dry_run=args.dry_run,
        )
        total += n
    print(f"migrated {total} gif(s) → {render_exp_dir('tiago', args.model_tag, args.exp)}")
    if not args.skip_cleanup:
        cleanup_legacy(model_tag=args.model_tag, exp_id=args.exp, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
