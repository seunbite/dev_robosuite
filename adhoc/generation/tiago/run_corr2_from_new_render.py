#!/usr/bin/env python3
"""Run corrN (N>=2) with new-render cumulative input; render with new renderer."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
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
    compose_tiled_motion_gif,
    config_snapshot_path,
    correction_stem,
    correction_cue_dir,
    default_config_path,
    default_render_dir,
    load_config_list,
    motion_gif_path_for_round,
    render_instance,
    row_from_parsed_config,
    run_correction_vlm,
    write_config_snapshot,
)
from render_paths import cumulative_tile_path  # noqa: E402

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"


def _base_row(model_tag: str, exp_id: str, cue: str) -> dict:
    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = load_config_list(cfg_path)
    row = next((r for r in rows if str(r.get("cue", "")) == cue), None)
    if row is None:
        raise SystemExit(f"cue not found: {cue}")
    return row


def _load_snapshot(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"snapshot missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[0] if isinstance(data, list) else data


def _new_render_gif(model_tag: str, exp_id: str, cue: str, idx: int, attempt_round: int) -> Path:
    return motion_gif_path_for_round("tiago", model_tag, exp_id, cue, idx, attempt_round)  # type: ignore[arg-type]


def _new_render_out_dir(model_tag: str, exp_id: str, cue: str, round_n: int) -> Path:
    del cue, round_n
    return default_render_dir("tiago", model_tag, "", exp_id, 1)  # type: ignore[arg-type]


def _new_render_record_path(model_tag: str, exp_id: str, cue: str, round_n: int) -> Path:
    cue_dir = correction_cue_dir("tiago", exp_id, model_tag, cue)  # type: ignore[arg-type]
    return cue_dir / f"{correction_stem(round_n)}_new_render.json"


def _new_render_snapshot_path(model_tag: str, exp_id: str, cue: str, round_n: int) -> Path:
    return config_snapshot_path("tiago", exp_id, model_tag, cue, round_n).with_name(
        f"{correction_stem(round_n)}_new_render_config.json"
    )


def _load_round_row(model_tag: str, exp_id: str, cue: str, round_n: int) -> dict:
    if round_n == 0:
        return _base_row(model_tag, exp_id, cue)
    if round_n == 1:
        return _load_snapshot(config_snapshot_path("tiago", exp_id, model_tag, cue, 1))  # type: ignore[arg-type]
    return _load_snapshot(_new_render_snapshot_path(model_tag, exp_id, cue, round_n))


def _build_cumulative_media(
    *,
    model_tag: str,
    exp_id: str,
    cue: str,
    idx: int,
    parent_round: int,
) -> RenderArtifacts:
    inputs = [_new_render_gif(model_tag, exp_id, cue, idx, rn) for rn in range(parent_round + 1)]
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"cumulative input missing: {p}")
    tiled = cumulative_tile_path("tiago", model_tag, exp_id, idx, cue, len(inputs))
    compose_tiled_motion_gif(inputs, tiled)
    from correction_lib import _gif_first_frame_png  # noqa: WPS433

    return RenderArtifacts(
        gif=tiled,
        pose_png=_gif_first_frame_png(tiled, tiled.parent / f"{tiled.stem}_pose.png"),
        cumulative_tile_gif=tiled,
    )


def _save_custom_record(path: Path, result: CorrectionResult, *, stem: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "domain": result.domain,
        "exp_id": result.exp_id,
        "model_tag": result.model_tag,
        "round": result.round,
        "parent_round": result.parent_round,
        "correction_stem": stem,
        "cue": result.cue,
        "base_config_json": str(result.config_json),
        "config_snapshot_json": (
            str(result.config_snapshot_json) if result.config_snapshot_json else None
        ),
        "is_correct": result.is_correct,
        "action": result.action,
        "correction": result.correction,
        "input_row": result.input_row,
        "output_row": result.output_row,
        "media": {
            "gif": str(result.media.gif) if result.media.gif else None,
            "mp4": str(result.media.mp4) if result.media.mp4 else None,
            "pose_png": str(result.media.pose_png) if result.media.pose_png else None,
            "cumulative_tile_gif": (
                str(result.media.cumulative_tile_gif) if result.media.cumulative_tile_gif else None
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run(
    *,
    cue: str,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    round_n: int = 2,
    force: bool = False,
    inner_exp_id: str = "0",
) -> dict:
    if round_n < 2:
        raise SystemExit("--round must be >= 2")
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")
    base_row = _base_row(model_tag, exp_id, cue)
    idx = int(base_row.get("idx", -1))
    parent_round = round_n - 1

    parent_row = _load_round_row(model_tag, exp_id, cue, parent_round)
    prior_attempts = [("Original (exp1)", base_row)]
    for prev_round in range(1, parent_round + 1):
        prior_attempts.append((f"Correction {prev_round}", _load_round_row(model_tag, exp_id, cue, prev_round)))

    record_path = _new_render_record_path(model_tag, exp_id, cue, round_n)
    snapshot_path = _new_render_snapshot_path(model_tag, exp_id, cue, round_n)
    out_dir = _new_render_out_dir(model_tag, exp_id, cue, round_n)
    if record_path.is_file() and not force:
        raise SystemExit(f"new-render corr{round_n} exists ({record_path}); use --force")
    if force:
        for p in (record_path, snapshot_path):
            if p.is_file():
                p.unlink()
        if out_dir.is_dir():
            shutil.rmtree(out_dir)

    media = _build_cumulative_media(
        model_tag=model_tag,
        exp_id=exp_id,
        cue=cue,
        idx=idx,
        parent_round=parent_round,
    )

    t0 = time.time()
    correction = run_correction_vlm(
        "tiago",
        exp_id,  # type: ignore[arg-type]
        parent_row,
        media,
        backend="gemini",
        inner_exp_id=inner_exp_id,  # type: ignore[arg-type]
        prior_attempts=prior_attempts,
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
        output_row = dict(parent_row)
        config_snapshot_json = snapshot_path
        write_config_snapshot(config_snapshot_json, output_row)
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
            parent_row,
            config_parsed,
            model=model_tag,
        )
        output_row["correction_q1"] = correction.get("correction_q1")
        output_row["correction_q2"] = correction.get("correction_q2")
        if correction.get("inner_reasoning"):
            output_row["reasoning"] = correction["inner_reasoning"]
        action = "corrected"
        config_snapshot_json = snapshot_path
        write_config_snapshot(config_snapshot_json, output_row)

    new_art = render_instance(
        "tiago",
        output_row,
        config_path=config_snapshot_json,
        render_dir=out_dir,
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
        config_json=default_config_path("tiago", model_tag, exp_id),  # type: ignore[arg-type]
        input_row=parent_row,
        correction=correction,
        is_correct=is_correct if isinstance(is_correct, bool) else None,
        action=action,
        output_row=output_row,
        media=new_art,
        config_snapshot_json=config_snapshot_json,
    )
    result.correction_json = _save_custom_record(
        record_path, result, stem=f"{correction_stem(round_n)}_new_render"
    )

    summary = {
        "cue": cue,
        "idx": idx,
        "action": action,
        "is_correct": is_correct,
        "elapsed_s": round(time.time() - t0, 1),
        "vlm_input_gif": str(media.gif) if media.gif else None,
        f"corr{round_n}_gif": str(new_art.gif) if new_art.gif else None,
        "correction_json": str(result.correction_json) if result.correction_json else None,
        "correction_q1": correction.get("correction_q1"),
    }
    log = (
        _REPO
        / "data/results/verify/tiago/correction"
        / f"exp{exp_id}"
        / model_tag
        / f"corr{round_n}_from_new_render_log.jsonl"
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
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--inner-exp", default="0")
    args = ap.parse_args()
    run(
        cue=args.cue,
        model_tag=args.model_tag,
        exp_id=args.exp,
        round_n=args.round,
        force=args.force,
        inner_exp_id=args.inner_exp,
    )


if __name__ == "__main__":
    main()
