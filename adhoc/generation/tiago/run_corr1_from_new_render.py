#!/usr/bin/env python3
"""Run corr1 using generation render as VLM input; output to flat exp render dir."""
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

from correction_prompts import correction_indicates_verified_ok  # noqa: E402
from correction_lib import (  # noqa: E402
    CorrectionResult,
    RenderArtifacts,
    _ensure_mp4,
    _gif_first_frame_png,
    config_snapshot_path,
    correction_cue_dir,
    correction_record_path,
    default_config_path,
    default_render_dir,
    load_config_list,
    render_instance,
    row_from_parsed_config,
    run_correction_vlm,
    save_correction_record,
    write_config_snapshot,
)
from render_paths import render_media_paths  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"


def _media_from_gif(gif: Path) -> RenderArtifacts:
    mp4 = gif.with_suffix(".mp4")
    png = gif.parent / f"{gif.stem}_pose.png"
    if not png.is_file():
        png = _gif_first_frame_png(gif, png)
    if not mp4.is_file():
        mp4 = _ensure_mp4(gif, mp4)
    return RenderArtifacts(gif=gif, mp4=mp4 if mp4.is_file() else None, pose_png=png)


def run(
    *,
    cue: str,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    force: bool = False,
    inner_exp_id: str = "0",
) -> dict:
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = load_config_list(cfg_path)
    input_row = next((r for r in rows if str(r.get("cue", "")) == cue), None)
    if input_row is None:
        raise SystemExit(f"cue not found: {cue}")

    idx = int(input_row.get("idx", -1))
    vlm_gif, _, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, "gen")
    if not vlm_gif.is_file():
        raise SystemExit(f"generation GIF missing: {vlm_gif}")

    cue_dir = correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
    rec1 = correction_record_path(cue_dir, 1)
    snap1 = config_snapshot_path("tiago", exp_id, model_tag, cue, 1)  # type: ignore[arg-type]
    if rec1.is_file() and not force:
        raise SystemExit(f"corr1 exists ({rec1}); use --force")
    if force:
        for p in (rec1, snap1):
            if p.is_file():
                p.unlink()

    round_n = 1
    parent_round = 0
    out_render_dir = default_render_dir("tiago", model_tag, cue, exp_id, round_n)  # type: ignore[arg-type]
    media = _media_from_gif(vlm_gif)

    t0 = time.time()
    correction = run_correction_vlm(
        "tiago",
        exp_id,  # type: ignore[arg-type]
        input_row,
        media,
        backend="gemini",
        inner_exp_id=inner_exp_id,  # type: ignore[arg-type]
    )
    is_correct = correction.get("is_correct")
    if correction_indicates_verified_ok(correction):
        is_correct = True
        correction["is_correct"] = True

    action = "none"
    output_row = None
    config_snapshot_json = None

    if is_correct is True:
        action = "verified_ok"
        output_row = dict(input_row)
        config_snapshot_json = snap1
        write_config_snapshot(config_snapshot_json, output_row)
        media = render_instance(
            "tiago",
            output_row,
            config_path=config_snapshot_json,
            render_dir=out_render_dir,
            model_tag=model_tag,
            exp_id=exp_id,  # type: ignore[arg-type]
            round_n=round_n,
        )
    else:
        config_parsed = correction.get("config_parsed")
        if not isinstance(config_parsed, dict) or not config_parsed.get("movements"):
            raise SystemExit(
                f"Correction failed parse: {correction.get('parse_error')!r}\n"
                f"Q1: {correction.get('correction_q1')}"
            )
        output_row = row_from_parsed_config(
            "tiago",
            exp_id,  # type: ignore[arg-type]
            input_row,
            config_parsed,
            model=model_tag,
        )
        output_row["correction_q1"] = correction.get("correction_q1")
        output_row["correction_q2"] = correction.get("correction_q2")
        if correction.get("inner_reasoning"):
            output_row["reasoning"] = correction["inner_reasoning"]
        action = "corrected"
        config_snapshot_json = snap1
        write_config_snapshot(config_snapshot_json, output_row)
        media = render_instance(
            "tiago",
            output_row,
            config_path=config_snapshot_json,
            render_dir=out_render_dir,
            model_tag=model_tag,
            exp_id=exp_id,  # type: ignore[arg-type]
            round_n=round_n,
        )

    result = CorrectionResult(
        cue=cue,
        domain="tiago",
        exp_id=exp_id,  # type: ignore[arg-type]
        model_tag=model_tag,
        round=round_n,
        parent_round=parent_round,
        config_json=cfg_path,
        input_row=input_row,
        correction=correction,
        is_correct=is_correct if isinstance(is_correct, bool) else None,
        action=action,
        output_row=output_row,
        media=media,
        config_snapshot_json=config_snapshot_json,
    )
    save_correction_record(result)

    summary = {
        "cue": cue,
        "idx": idx,
        "action": action,
        "is_correct": is_correct,
        "elapsed_s": round(time.time() - t0, 1),
        "vlm_input_gif": str(vlm_gif),
        "corr1_gif": str(media.gif) if media.gif else None,
        "correction_json": str(result.correction_json) if result.correction_json else None,
        "correction_q1": correction.get("correction_q1"),
    }
    log = (
        _REPO
        / "data/results/verify/tiago/correction"
        / f"exp{exp_id}"
        / model_tag
        / "corr1_log.jsonl"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", required=True)
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inner-exp", default="0")
    args = ap.parse_args()
    run(cue=args.cue, model_tag=args.model_tag, exp_id=args.exp, force=args.force, inner_exp_id=args.inner_exp)


if __name__ == "__main__":
    main()
