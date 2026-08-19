#!/usr/bin/env python3
"""Run flex_bicep correction triplet: (1) original→corr1, (2) GT→corr, (3) original+GT→corr."""
from __future__ import annotations

import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    CorrectionResult,
    RenderArtifacts,
    compose_tiled_motion_gif,
    default_config_path,
    load_config_row,
    render_instance,
    row_from_parsed_config,
    run_correction_vlm,
    write_config_snapshot,
    _ensure_mp4,
    _gif_first_frame_png,
    _safe_cue,
)
from correction_prompts import parse_correction_response  # noqa: E402

CUE = "flex_bicep"
MODEL_TAG = "gemini-2.5-pro"
EXP_ID = "1"
DOMAIN = "tiago"


def _gt_config_path() -> Path:
    return (
        _REPO
        / "data/results/motion_configs/tiago/exp/correction"
        / MODEL_TAG
        / f"exp{EXP_ID}"
        / CUE
        / "gt_config.json"
    )


def _gt_gif_path() -> Path:
    idx = 9
    return (
        _REPO
        / "data/results/render/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / "gt"
        / f"mm19_g{idx:02d}_{CUE}.gif"
    )


def _orig_gif_path() -> Path:
    return _REPO / "data/results/render/tiago" / f"exp{EXP_ID}_{MODEL_TAG}" / f"mm19_g09_{CUE}.gif"


def _orig_gt_corr_config_path() -> Path:
    return (
        _REPO
        / "data/results/motion_configs/tiago/exp/correction"
        / MODEL_TAG
        / f"exp{EXP_ID}"
        / CUE
        / "orig_gt_correction_config.json"
    )


def _orig_gt_corr_gif_path() -> Path:
    return (
        _REPO
        / "data/results/render/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / "orig_gt_corr"
        / f"mm19_g09_{CUE}.gif"
    )


def _clear_main_correction_rounds() -> None:
    cue_dir = _REPO / "data/results/verify/tiago/correction" / f"exp{EXP_ID}" / MODEL_TAG / CUE
    cfg_dir = (
        _REPO
        / "data/results/motion_configs/tiago/exp/correction"
        / MODEL_TAG
        / f"exp{EXP_ID}"
        / CUE
    )
    render_base = (
        _REPO / "data/results/render/tiago/correction" / f"exp{EXP_ID}" / MODEL_TAG / CUE
    )
    for stem in ("correction", "correction_correction", "correction_correction_correction"):
        for p in (
            cue_dir / f"{stem}.json",
            cfg_dir / f"{stem}_config.json",
            render_base / stem,
        ):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)


def _save_adhoc_record(
    *,
    tag: str,
    parent_round: int,
    input_row: dict,
    correction: dict,
    action: str,
    output_row: dict | None,
    media: RenderArtifacts,
    config_snapshot: Path | None,
    extra: dict | None = None,
) -> Path:
    verify_dir = _REPO / "data/results/verify/tiago/correction" / f"exp{EXP_ID}" / MODEL_TAG / CUE
    verify_dir.mkdir(parents=True, exist_ok=True)
    path = verify_dir / f"{tag}.json"
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "domain": DOMAIN,
        "exp_id": EXP_ID,
        "model_tag": MODEL_TAG,
        "cue": CUE,
        "parent_round": parent_round,
        "action": action,
        "is_correct": correction.get("is_correct"),
        "correction": correction,
        "input_row": input_row,
        "output_row": output_row,
        "config_snapshot_json": str(config_snapshot) if config_snapshot else None,
        "media": {
            "gif": str(media.gif) if media.gif else None,
            "mp4": str(media.mp4) if media.mp4 else None,
            "cumulative_tile_gif": str(media.cumulative_tile_gif) if media.cumulative_tile_gif else None,
        },
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _apply_correction(
    *,
    tag: str,
    input_row: dict,
    correction: dict,
    render_dir: Path,
    parent_round: int,
) -> tuple[str, dict | None, RenderArtifacts, Path | None]:
    cfg_path = default_config_path(DOMAIN, MODEL_TAG, EXP_ID)  # type: ignore[arg-type]
    is_correct = correction.get("is_correct")
    media = RenderArtifacts()
    config_snapshot: Path | None = None
    output_row: dict | None = None
    render_dir.mkdir(parents=True, exist_ok=True)

    if is_correct is True:
        action = "verified_ok"
        output_row = input_row
        config_snapshot = (
            _REPO
            / "data/results/motion_configs/tiago/exp/correction"
            / MODEL_TAG
            / f"exp{EXP_ID}"
            / CUE
            / f"{tag}_config.json"
        )
        write_config_snapshot(config_snapshot, output_row)
        src = None
        if tag == "gt_correction":
            src = _gt_gif_path()
        elif tag in ("orig_gt_correction", "orig_gt_triple_correction", "orig_gt_triple_correction_shuf"):
            src = _orig_gt_corr_gif_path()
        if src and src.is_file():
            out_gif = render_dir / src.name
            shutil.copy2(src, out_gif)
            media.gif = out_gif
            media.mp4 = _ensure_mp4(out_gif, render_dir / out_gif.with_suffix(".mp4").name)
        return action, output_row, media, config_snapshot

    config_parsed = correction.get("config_parsed")
    if not isinstance(config_parsed, dict) or not config_parsed.get("movements"):
        action = "parse_failed"
        return action, None, media, None

    output_row = row_from_parsed_config(
        DOMAIN, EXP_ID, input_row, config_parsed, model=MODEL_TAG  # type: ignore[arg-type]
    )
    output_row["correction_q1"] = correction.get("correction_q1")
    output_row["correction_q2"] = correction.get("correction_q2")
    if correction.get("inner_reasoning"):
        output_row["reasoning"] = correction["inner_reasoning"]
    config_snapshot = (
        _REPO
        / "data/results/motion_configs/tiago/exp/correction"
        / MODEL_TAG
        / f"exp{EXP_ID}"
        / CUE
        / f"{tag}_config.json"
    )
    write_config_snapshot(config_snapshot, output_row)
    media = render_instance(
        DOMAIN,
        output_row,
        config_path=config_snapshot,
        render_dir=render_dir,
        model_tag=MODEL_TAG,
        exp_id=EXP_ID,  # type: ignore[arg-type]
        round_n=1,
        ignore_xyz=True,
    )
    return "corrected", output_row, media, config_snapshot


def run_gt_correction() -> dict:
    gt_row = load_config_row(_gt_config_path(), cue=CUE)
    gif = _gt_gif_path()
    media = RenderArtifacts(
        gif=gif,
        mp4=_ensure_mp4(gif, gif.with_suffix(".mp4")),
        pose_png=_gif_first_frame_png(gif, gif.parent / f"{gif.stem}_pose.png"),
    )
    correction = run_correction_vlm(
        DOMAIN,
        EXP_ID,  # type: ignore[arg-type]
        gt_row,
        media,
        model=MODEL_TAG,
        inner_exp_id="0",
        prior_attempts=None,
    )
    render_dir = gif.parent / "gt_corr"
    action, output_row, out_media, snap = _apply_correction(
        tag="gt_correction",
        input_row=gt_row,
        correction=correction,
        render_dir=render_dir,
        parent_round=0,
    )
    rec = _save_adhoc_record(
        tag="gt_correction",
        parent_round=0,
        input_row=gt_row,
        correction=correction,
        action=action,
        output_row=output_row,
        media=out_media if out_media.gif else media,
        config_snapshot=snap,
    )
    return {
        "scenario": "2. GT → correction",
        "action": action,
        "q1": correction.get("correction_q1"),
        "q2": correction.get("correction_q2"),
        "record": str(rec),
        "gif": str((out_media.gif or media.gif) or ""),
    }


def run_orig_gt_correction() -> dict:
    base_cfg = default_config_path(DOMAIN, MODEL_TAG, EXP_ID)  # type: ignore[arg-type]
    orig_row = load_config_row(base_cfg, cue=CUE)
    gt_row = load_config_row(_gt_config_path(), cue=CUE)
    orig_gif = _orig_gif_path()
    gt_gif = _gt_gif_path()
    out_dir = (
        _REPO
        / "data/results/render/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / "orig_gt_corr"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tiled = out_dir / "cumulative_2tiles.gif"
    compose_tiled_motion_gif([orig_gif, gt_gif], tiled)
    media = RenderArtifacts(
        gif=tiled,
        cumulative_tile_gif=tiled,
        pose_png=_gif_first_frame_png(tiled, out_dir / "cumulative_tile_pose.png"),
    )
    prior = [("Original (exp1)", orig_row), ("GT", gt_row)]
    correction = run_correction_vlm(
        DOMAIN,
        EXP_ID,  # type: ignore[arg-type]
        gt_row,
        media,
        model=MODEL_TAG,
        inner_exp_id="0",
        prior_attempts=prior,
    )
    action, output_row, out_media, snap = _apply_correction(
        tag="orig_gt_correction",
        input_row=gt_row,
        correction=correction,
        render_dir=out_dir,
        parent_round=1,
    )
    rec = _save_adhoc_record(
        tag="orig_gt_correction",
        parent_round=1,
        input_row=gt_row,
        correction=correction,
        action=action,
        output_row=output_row,
        media=out_media if out_media.gif else media,
        config_snapshot=snap,
    )
    return {
        "scenario": "3. Original + GT → correction (corr2-style prompt)",
        "action": action,
        "q1": correction.get("correction_q1"),
        "q2": correction.get("correction_q2"),
        "record": str(rec),
        "gif": str((out_media.gif or media.gif) or ""),
    }


def run_orig_gt_triple_correction(*, shuffle_order: bool = False) -> dict:
    base_cfg = default_config_path(DOMAIN, MODEL_TAG, EXP_ID)  # type: ignore[arg-type]
    orig_row = load_config_row(base_cfg, cue=CUE)
    gt_row = load_config_row(_gt_config_path(), cue=CUE)
    orig_gt_row = load_config_row(_orig_gt_corr_config_path(), cue=CUE)

    items: list[tuple[str, dict, Path]] = [
        ("Original (exp1)", orig_row, _orig_gif_path()),
        ("GT", gt_row, _gt_gif_path()),
        ("Original + GT → correction", orig_gt_row, _orig_gt_corr_gif_path()),
    ]
    if shuffle_order:
        random.shuffle(items)
    tile_order = [label for label, _, _ in items]
    prior = [(label, row) for label, row, _ in items]
    gif_paths = [gif for _, _, gif in items]

    tag = "orig_gt_triple_correction_shuf" if shuffle_order else "orig_gt_triple_correction"
    render_subdir = "orig_gt_triple_corr_shuf" if shuffle_order else "orig_gt_triple_corr"
    out_dir = (
        _REPO
        / "data/results/render/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / render_subdir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tiled_name = f"cumulative_{len(gif_paths)}tiles"
    if shuffle_order:
        tiled_name += "_shuf"
    tiled = out_dir / f"{tiled_name}.gif"
    compose_tiled_motion_gif(gif_paths, tiled)
    media = RenderArtifacts(
        gif=tiled,
        cumulative_tile_gif=tiled,
        pose_png=_gif_first_frame_png(tiled, out_dir / "cumulative_tile_pose.png"),
    )

    correction = run_correction_vlm(
        DOMAIN,
        EXP_ID,  # type: ignore[arg-type]
        orig_gt_row,
        media,
        model=MODEL_TAG,
        inner_exp_id="0",
        prior_attempts=prior,
    )
    action, output_row, out_media, snap = _apply_correction(
        tag=tag,
        input_row=orig_gt_row,
        correction=correction,
        render_dir=out_dir,
        parent_round=2,
    )
    rec = _save_adhoc_record(
        tag=tag,
        parent_round=2,
        input_row=orig_gt_row,
        correction=correction,
        action=action,
        output_row=output_row,
        media=out_media if out_media.gif else media,
        config_snapshot=snap,
        extra={
            "shuffle_order": shuffle_order,
            "tile_order": tile_order,
        },
    )
    return {
        "scenario": "Original + GT + Orig+GT corr → correction (3-tile)",
        "shuffle_order": shuffle_order,
        "tile_order": tile_order,
        "action": action,
        "q1": correction.get("correction_q1"),
        "q2": correction.get("correction_q2"),
        "record": str(rec),
        "gif": str((out_media.gif or media.gif) or ""),
        "cumulative_gif": str(tiled),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="flex_bicep correction experiments")
    ap.add_argument("--triple-only", action="store_true", help="Run only 3-tile cumulative correction")
    ap.add_argument(
        "--shuffle-order",
        action="store_true",
        default=False,
        help="Shuffle tile/prior order (Original, GT, Orig+GT corr)",
    )
    ap.add_argument(
        "--no-shuffle-order",
        action="store_false",
        dest="shuffle_order",
        help="Keep canonical tile order (default)",
    )
    ap.add_argument("--full-triplet", action="store_true", help="Run full corr1 + GT + orig+GT triplet")
    args = ap.parse_args()

    if args.triple_only:
        print(f"=== 3-tile correction (shuffle_order={args.shuffle_order}) ===")
        result = run_orig_gt_triple_correction(shuffle_order=args.shuffle_order)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if not args.full_triplet:
        ap.print_help()
        return

    import subprocess

    _clear_main_correction_rounds()
    print("=== 1. Original → correction (corr1 re-run) ===")
    subprocess.run(
        [
            sys.executable,
            str(_HERE / "run_correction.py"),
            "--cue",
            CUE,
            "--exp",
            EXP_ID,
            "--inner-exp",
            "0",
            "--ignore-xyz",
            "--from-base",
        ],
        cwd=str(_HERE),
        check=True,
    )
    corr1_path = (
        _REPO
        / "data/results/verify/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / "correction.json"
    )
    corr1 = json.loads(corr1_path.read_text())
    r1 = {
        "scenario": "1. Original → correction (corr1)",
        "action": corr1.get("action"),
        "q1": (corr1.get("correction") or {}).get("correction_q1"),
        "q2": (corr1.get("correction") or {}).get("correction_q2"),
        "record": str(corr1_path),
        "gif": (corr1.get("media") or {}).get("gif"),
    }
    print(json.dumps(r1, indent=2, ensure_ascii=False))

    print("\n=== 2. GT → correction ===")
    r2 = run_gt_correction()
    print(json.dumps(r2, indent=2, ensure_ascii=False))

    print("\n=== 3. Original + GT → correction ===")
    r3 = run_orig_gt_correction()
    print(json.dumps(r3, indent=2, ensure_ascii=False))

    summary_path = (
        _REPO
        / "data/results/verify/tiago/correction"
        / f"exp{EXP_ID}"
        / MODEL_TAG
        / CUE
        / "triplet_summary.json"
    )
    summary_path.write_text(
        json.dumps([r1, r2, r3], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
