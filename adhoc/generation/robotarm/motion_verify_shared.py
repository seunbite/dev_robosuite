#!/usr/bin/env python3
"""Shared motion verify prompt + component parsing."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from verify_pose_tiles_gemini import APPROPRIATE_MEANS_LINE, _first_pose, _movement_summary


def motion_verify_prompt(row: dict[str, Any], fewshot_text: str, *, modality: str) -> str:
    from prompt_loader import fill_template  # noqa: WPS433

    p = _first_pose(row)
    fixed = row.get("gt_fixed_first_pose") or p
    tail_json = json.dumps(
        [s for s in row.get("movements", []) if s.get("type") != "pose"],
        indent=2,
        ensure_ascii=False,
    )
    appropriate = APPROPRIATE_MEANS_LINE.replace("this pose", "this fixed start pose").replace(
        "subsequent movements", "the shown tail movement"
    )
    if modality == "vlm":
        from prompt_loader import fill_template as _fill  # noqa: WPS433

        return _fill(
            "exp08_motion_verify_vlm.txt",
            {
                "APPROPRIATE_MEANS": appropriate,
                "FEWSHOT": fewshot_text,
                "CUE": str(row.get("cue", "")),
                "DESCRIPTION": str(row.get("description", "")),
                "FIXED_DIR": str(fixed.get("dir", "")),
                "FIXED_GRIPPER": str(fixed.get("gripper_orientation", "")),
                "TAIL_SUMMARY": _movement_summary(row),
            },
        )
    return fill_template(
        "exp09_motion_verify_text.txt",
        {
            "APPROPRIATE_MEANS": appropriate,
            "FEWSHOT": fewshot_text,
            "CUE": str(row.get("cue", "")),
            "DESCRIPTION": str(row.get("description", "")),
            "FIXED_DIR": str(fixed.get("dir", "")),
            "FIXED_GRIPPER": str(fixed.get("gripper_orientation", "")),
            "TAIL_SUMMARY": _movement_summary(row),
            "TAIL_JSON": tail_json,
        },
    )


def normalize_component(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    if not raw or not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in ("movement", "path_arc", "path_line"):
        return None
    out: dict[str, Any] = {"kind": kind}
    if kind == "path_arc":
        plane = raw.get("plane") or "xz"
        if plane == "null":
            return None
        out["plane"] = str(plane).lower()
        return out
    if kind == "path_line":
        axis = raw.get("axis")
        if not axis or axis == "null":
            return None
        out["axis"] = str(axis).lower()
        return out
    axes = raw.get("axes") or {}
    if isinstance(axes, dict):
        clean = {}
        for ax in "xyz":
            if ax in axes and axes[ax] in ("+", "-", "+-"):
                clean[ax] = axes[ax]
        if clean:
            out["axes"] = clean
    j = raw.get("joint")
    if j and j != "null":
        out["joint"] = str(j).lower()
    rep = raw.get("repetition")
    if rep and rep != "null":
        out["repetition"] = str(rep).lower()
    if raw.get("hold") is True:
        out["hold"] = True
    return out if len(out) > 1 else None


def resume_default() -> bool:
    return os.getenv("RESUME", "1") != "0"


def load_verify_done_indices(
    out_path: Path,
    *,
    rows_key: str | None = None,
    idx_key: str = "cue_idx",
) -> set[int]:
    """Cue indices with a completed verify record in a prior output JSON."""
    if not out_path.is_file():
        return set()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    if rows_key:
        rows = data.get(rows_key) or []
    else:
        rows = data.get("rows") or data.get("mp4") or []
    done: set[int] = set()
    for r in rows:
        idx_raw = r.get(idx_key, r.get("idx"))
        if idx_raw is None:
            continue
        if (
            r.get("movement_is_appropriate") is not None
            or r.get("verify_result")
            or r.get("vlm_correct") is not None
            or r.get("correct") is not None
        ):
            done.add(int(idx_raw))
    return done


def record_from_parsed(parsed: dict[str, Any]) -> dict[str, Any]:
    rec_comp = None
    if not parsed.get("movement_is_appropriate"):
        rec_comp = normalize_component(
            (parsed.get("if_not_appropriate") or {}).get("recommended_component")
        )
    return {
        "verify_result": parsed,
        "movement_is_appropriate": parsed.get("movement_is_appropriate"),
        "recommended_component": rec_comp,
    }
