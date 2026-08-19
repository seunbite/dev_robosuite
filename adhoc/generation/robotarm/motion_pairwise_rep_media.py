"""Build GT vs neg-repetition pairwise MP4s for pilot-90 exp11."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[3]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import testset_utils  # noqa: E402
from compare_pose_2 import _stitch_pair  # noqa: E402
from motion_media_paths import pose_jsonl  # noqa: E402
from motion_pairwise_media import (  # noqa: E402
    HZ,
    N_OUT_FRAMES,
    ROBOT,
    _frames_to_mp4,
    _panel_at_progress_plain,
    _sample_for_cfg,
    _write_single_cfg,
    compact_tail_label,
)
from pilot90_paths import (  # noqa: E402
    MOTION_REP_PAIRWISE_DIR,
    REP_PAIRWISE_SPECS_NAME,
    prompt_exp_path,
)

REP_PAIRWISE_DIR = MOTION_REP_PAIRWISE_DIR
SPECS_NAME = REP_PAIRWISE_SPECS_NAME
PROMPT_EXP11 = prompt_exp_path(11)


def build_pairwise_rep_prompt(*, row: dict[str, Any]) -> str:
    tpl = PROMPT_EXP11.read_text(encoding="utf-8")
    return tpl.replace("{{CUE}}", str(row.get("cue", "")))


def build_pair_rep_mp4(
    *,
    idx: int,
    cue: str,
    row: dict[str, Any],
    gt_cfg: dict[str, Any],
    neg_cfg: dict[str, Any],
    out_mp4: Path,
    scratch: Path,
    gt_side: str,
) -> dict[str, Any] | None:
    """Render one stitched left|right plain MP4 (no alpha trajectory)."""
    gt_side = gt_side.lower()
    if gt_side not in ("left", "right"):
        gt_side = "left"

    scratch.mkdir(parents=True, exist_ok=True)
    gt_path = scratch / f"{idx:03d}_{cue}_gt_rep.json"
    neg_path = scratch / f"{idx:03d}_{cue}_neg_rep.json"
    _write_single_cfg(gt_cfg, gt_path)
    _write_single_cfg(neg_cfg, neg_path)

    gt_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(gt_cfg, gt_path, idx, cue, "gt_rep"),
        ROBOT,
        HZ,
        force=True,
        disk_cache=False,
    )
    neg_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(neg_cfg, neg_path, idx, cue, "neg_rep"),
        ROBOT,
        HZ,
        force=True,
        disk_cache=False,
    )

    gt_label = compact_tail_label(gt_cfg)
    neg_label = compact_tail_label(neg_cfg)
    if gt_side == "left":
        left_lbl, right_lbl = gt_label, neg_label
        left_kind, right_kind = "gt", "neg_rep"
    else:
        left_lbl, right_lbl = neg_label, gt_label
        left_kind, right_kind = "neg_rep", "gt"

    out_frames: list[Image.Image] = []
    for k in range(N_OUT_FRAMES):
        progress = k / max(1, N_OUT_FRAMES - 1)
        if gt_side == "left":
            lp, rp = _panel_at_progress_plain(gt_bundle, progress), _panel_at_progress_plain(neg_bundle, progress)
        else:
            lp, rp = _panel_at_progress_plain(neg_bundle, progress), _panel_at_progress_plain(gt_bundle, progress)
        out_frames.append(_stitch_pair(lp, rp, cue=cue, left_label=left_lbl, right_label=right_lbl))

    _frames_to_mp4(out_frames, out_mp4, scratch=scratch)
    if not out_mp4.is_file():
        return None

    start_pose = gt_cfg.get("gt_fixed_first_pose") or {}
    return {
        "idx": idx,
        "cue": cue,
        "description": row.get("description", ""),
        "start_pose_source": gt_cfg.get("start_pose_source"),
        "start_pose_dir": start_pose.get("dir"),
        "start_pose_gripper": start_pose.get("gripper_orientation"),
        "start_pose_id": start_pose.get("pose_id"),
        "gt_side": gt_side,
        "left": left_kind,
        "right": right_kind,
        "left_tail_summary": left_lbl,
        "right_tail_summary": right_lbl,
        "pair_mp4": out_mp4.name,
        "layout": "left_right_plain_mp4_stitched",
        "prompt": build_pairwise_rep_prompt(row=row),
    }


def prepare_pilot90_rep_pairwise_mp4s(
    *,
    spec_path: Path | None = None,
    out_dir: Path | None = None,
    scratch: Path | None = None,
    force: bool = False,
    limit: int = 0,
) -> tuple[int, list[str]]:
    jpath = pose_jsonl(_REPO)
    os.environ["MOTION_POSE_JSONL"] = str(jpath)

    od = out_dir or REP_PAIRWISE_DIR
    sp = spec_path or (od / SPECS_NAME)
    if not sp.is_file():
        raise FileNotFoundError(f"Missing specs: {sp} — run build_pilot90_motion_rep_pairwise_preview.py first")

    payload = json.loads(sp.read_text(encoding="utf-8"))
    entries = list(payload.get("mp4") or [])
    if limit:
        entries = entries[:limit]

    sc = scratch or (od / "_build_scratch_rep")
    od.mkdir(parents=True, exist_ok=True)
    ready = 0
    failures: list[str] = []
    rendered: list[dict[str, Any]] = []

    for entry in entries:
        idx = int(entry["idx"])
        cue = str(entry["cue"])
        out_mp4 = od / f"{idx:03d}_{cue}_pair_rep.mp4"
        sidecar = od / f"{idx:03d}_{cue}_rep_spec.json"

        if out_mp4.is_file() and not force and sidecar.is_file():
            rendered_meta = json.loads(sidecar.read_text(encoding="utf-8"))
            rendered.append({**entry, **rendered_meta})
            ready += 1
            continue

        gt_cfg = entry.get("gt_cfg")
        neg_cfg = entry.get("neg_cfg")
        if not gt_cfg or not neg_cfg:
            failures.append(f"{cue}: missing gt_cfg/neg_cfg in spec")
            continue

        print(f"[rep-pairwise] c{idx} {cue} gt_side={entry.get('gt_side')} ...", flush=True)
        try:
            row = {"cue": cue, "description": entry.get("description", "")}
            built = build_pair_rep_mp4(
                idx=idx,
                cue=cue,
                row=row,
                gt_cfg=gt_cfg,
                neg_cfg=neg_cfg,
                out_mp4=out_mp4,
                scratch=sc,
                gt_side=str(entry.get("gt_side", "left")),
            )
            if built:
                built.update(
                    {
                        "gt_repetition_label": entry.get("gt_repetition_label"),
                        "gt_rep_override": entry.get("gt_rep_override"),
                        "neg_rep_override": entry.get("neg_rep_override"),
                        "gt_movement_repetition_param": entry.get("gt_movement_repetition_param"),
                        "neg_movement_repetition_param": entry.get("neg_movement_repetition_param"),
                        "component": entry.get("component"),
                    }
                )
                sidecar.write_text(json.dumps(built, indent=2, ensure_ascii=False), encoding="utf-8")
                rendered.append({**entry, **built})
                ready += 1
            else:
                failures.append(f"{cue}: build returned false")
        except Exception as e:
            failures.append(f"{cue}: {e}")
            print(f"[rep-pairwise fail] {cue}: {e}", flush=True)

    if rendered:
        by_idx = {int(e["idx"]): e for e in entries}
        for meta in rendered:
            by_idx[int(meta["idx"])] = {**by_idx[int(meta["idx"])], **meta}
        merged = sorted(by_idx.values(), key=lambda e: int(e["idx"]))
        out_payload = dict(payload)
        out_payload["layout"] = "left_right_plain_mp4_stitched"
        out_payload["n_with_mp4"] = sum(
            1
            for e in merged
            if (od / f"{int(e['idx']):03d}_{e['cue']}_pair_rep.mp4").is_file()
        )
        out_payload["mp4"] = merged
        (od / SPECS_NAME).write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return ready, failures
