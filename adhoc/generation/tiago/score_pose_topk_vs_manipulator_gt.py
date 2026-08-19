#!/usr/bin/env python3
"""Score Tiago pose picks vs manipulator pilot-40 human GT (pose + movement)."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation/robotarm"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from legacy.render_mobile_config import (  # noqa: E402
    _apply_bank_r_arm,
    _apply_pose_nominal,
    _default_qpos,
    _make_env,
    rank_pose_bank_topk,
)
from pilot40_experiment_suite import (  # noqa: E402
    CONSOLIDATED,
    pose_generation_correct,
)
from pose_bank_topk_tiles import first_pose_from_row  # noqa: E402
from score_pilot40_motion_gt_components import (  # noqa: E402
    _build_annotation_map,
    _tail_matches_component,
    _tail_steps,
)

DEFAULT_PICK = _REPO / "data/results/verify/tiago/pose_topk_gemini_pick.json"
DEFAULT_SHOTS = _REPO / "data/seed/shots/tiago/shot_configs_pilot40_mobile.json"
DEFAULT_OUT = _REPO / "data/results/verify/tiago/pilot40_manipulator_gt_score.json"

ARM_TO_DIR: dict[str, str] = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "left",
    "up": "up",
    "down": "down",
}

CUE_GT_ALIASES = {
    "scratch_head_confused": ["scratch_head_confused", "self_scratch_head_confused"],
}


def _load_gt_by_cue(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}


def _lookup_gt(gt_by_cue: dict[str, dict[str, Any]], cue: str) -> dict[str, Any] | None:
    if cue in gt_by_cue:
        return gt_by_cue[cue]
    for alt in CUE_GT_ALIASES.get(cue, []):
        if alt in gt_by_cue:
            return gt_by_cue[alt]
    return None


def _compute_dir_orient(env, pose: dict[str, Any], r_arm_rad: list[float] | None = None) -> dict[str, str]:
    from adhoc.generation.tiago.legacy import render_mobile_manip_schema as sch

    q = _default_qpos(env)
    if r_arm_rad is not None:
        _apply_bank_r_arm(q, pose, r_arm_rad)
    else:
        _apply_pose_nominal(q, pose)
    env.sim.data.qpos[:] = q
    env.sim.forward()
    d = sch._dir_six_way_from_anchor(env, q, cam_name=sch.CAM)
    o = sch._gripper_horizontal_vertical_by_reach_dir(env.sim, d)
    return {"dir": str(d), "gripper_orientation": str(o)}


def _normalize_mobile_tail_step(step: dict[str, Any]) -> dict[str, Any]:
    p = dict(step.get("parameters") or {})
    if step.get("type") == "path" and isinstance(p.get("path"), dict):
        inner = dict(p["path"])
        return {"type": "path", "parameters": inner}
    return step


def _mobile_tail(movements: list) -> list[dict[str, Any]]:
    return [_normalize_mobile_tail_step(s) for s in _tail_steps(movements)]


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    scored = [r for r in rows if r.get(key) is not None]
    ok = sum(1 for r in scored if r[key])
    n = len(scored)
    return {
        "n_correct": ok,
        "n": n,
        "accuracy": ok / n if n else None,
        "pct": f"{ok}/{n} ({100.0 * ok / n:.1f}%)" if n else "n/a",
    }


def _nominal_pose_label(pose: dict[str, Any]) -> dict[str, str]:
    arm = str(pose.get("arm_position", "front")).strip().lower()
    if arm == "still":
        return {"dir": "still", "gripper_orientation": str(pose.get("gripper_orientation", "vertical"))}
    return {
        "dir": ARM_TO_DIR.get(arm, arm),
        "gripper_orientation": str(pose.get("gripper_orientation", "horizontal")),
    }


def run(args: argparse.Namespace) -> None:
    pick_path = Path(args.pick_json)
    pick_by_idx: dict[int, dict[str, Any]] = {}
    if pick_path.is_file():
        pick = json.loads(pick_path.read_text(encoding="utf-8"))
        for rec in pick.get("results") or []:
            pick_by_idx[int(rec["idx"])] = rec

    shot_rows = json.loads(Path(args.shots_json).read_text(encoding="utf-8"))
    shots = {int(r["idx"]): r for r in shot_rows}
    gt_by_cue = _load_gt_by_cue(Path(args.consolidated_json))
    motion_ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}

    env = _make_env()
    rows_out: list[dict[str, Any]] = []
    try:
        for idx in sorted(shots):
            shot = shots[idx]
            cue = str(shot["cue"])
            rec = pick_by_idx.get(idx, {})
            pose = first_pose_from_row(shot)
            gt_row = _lookup_gt(gt_by_cue, cue)
            gt_str = (gt_row or {}).get("groundtruth", "")

            nominal = _nominal_pose_label(pose)
            sim_pose = _compute_dir_orient(env, pose, r_arm_rad=None)

            ranked = rank_pose_bank_topk(env, pose, top_k=1)
            entries = ranked.get("entries") or []
            xyz_pose = None
            if entries:
                xyz_pose = _compute_dir_orient(env, pose, r_arm_rad=entries[0]["r_arm_rad"])
            elif rec.get("xyz_rank1"):
                xyz_pose = _compute_dir_orient(env, pose, r_arm_rad=rec["xyz_rank1"]["r_arm_rad"])

            vlm_pose = None
            if rec.get("gemini_pick"):
                vlm_pose = _compute_dir_orient(env, pose, r_arm_rad=rec["gemini_pick"]["r_arm_rad"])

            gen_ok = pose_generation_correct(nominal, gt_str) if gt_row else None
            sim_ok = pose_generation_correct(sim_pose, gt_str) if gt_row else None
            xyz_ok = pose_generation_correct(xyz_pose, gt_str) if gt_row and xyz_pose else None
            vlm_ok = pose_generation_correct(vlm_pose, gt_str) if gt_row and vlm_pose else None

            motion_ok = None
            motion_comp = None
            ann = motion_ann.get(int((gt_row or {}).get("cue_idx", idx)))
            if ann and ann.get("component"):
                motion_comp = ann["component"]
                tail = _mobile_tail(shot.get("movements") or [])
                motion_ok, _ = _tail_matches_component(tail, motion_comp)

            bank = rec.get("bank_meta") or ranked
            rows_out.append(
                {
                    "idx": idx,
                    "cue": cue,
                    "groundtruth": gt_str or None,
                    "has_pose_gt": bool(gt_row),
                    "has_motion_gt": bool(motion_comp),
                    "nominal_pose": nominal,
                    "sim_pose": sim_pose,
                    "xyz_rank1_pose": xyz_pose,
                    "vlm_pick_pose": vlm_pose,
                    "bank_dir": bank.get("bank_dir"),
                    "bank_orient": bank.get("bank_orient"),
                    "n_bank_tiles": len(entries) if entries else ranked.get("n_hand_ok"),
                    "gemini_tile": (rec.get("gemini") or {}).get("best_tile_index"),
                    "generation_pose_correct": gen_ok,
                    "sim_pose_correct": sim_ok,
                    "xyz_rank1_pose_correct": xyz_ok,
                    "vlm_pick_pose_correct": vlm_ok,
                    "generation_movement_correct": motion_ok,
                    "motion_component_gt": motion_comp,
                }
            )
    finally:
        env.close()

    summary = {
        "generation_pose_nominal": _rate(rows_out, "generation_pose_correct"),
        "generation_pose_sim": _rate(rows_out, "sim_pose_correct"),
        "xyz_rank1_pose": _rate(rows_out, "xyz_rank1_pose_correct"),
        "vlm_pick_pose": _rate(rows_out, "vlm_pick_pose_correct"),
        "generation_movement": _rate(rows_out, "generation_movement_correct"),
    }

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "shots_json": str(Path(args.shots_json).relative_to(_REPO)),
        "pick_json": str(pick_path.relative_to(_REPO)) if pick_path.is_file() else None,
        "consolidated_gt": str(Path(args.consolidated_json).relative_to(_REPO)),
        "n_shots": len(rows_out),
        "summary": summary,
        "rows": rows_out,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tsv_path = out_path.with_suffix(".tsv")
    cols = [
        "idx",
        "cue",
        "groundtruth",
        "nominal_pose",
        "sim_pose",
        "xyz_rank1_pose",
        "generation_pose_correct",
        "sim_pose_correct",
        "xyz_rank1_pose_correct",
        "generation_movement_correct",
    ]
    lines = ["\t".join(cols)]
    for r in rows_out:
        lines.append(
            "\t".join(
                [
                    str(r["idx"]),
                    r["cue"],
                    str(r.get("groundtruth") or ""),
                    str(r.get("nominal_pose") or ""),
                    str(r.get("sim_pose") or ""),
                    str(r.get("xyz_rank1_pose") or ""),
                    str(r.get("generation_pose_correct")),
                    str(r.get("sim_pose_correct")),
                    str(r.get("xyz_rank1_pose_correct")),
                    str(r.get("generation_movement_correct")),
                ]
            )
        )
    tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Tiago pilot-40 vs manipulator GT ===", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v['pct']}", flush=True)
    print(f"Wrote {out_path}", flush=True)
    print(f"Wrote {tsv_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pick-json", default=str(DEFAULT_PICK))
    p.add_argument("--shots-json", default=str(DEFAULT_SHOTS))
    p.add_argument("--consolidated-json", default=str(CONSOLIDATED))
    p.add_argument("--out-json", default=str(DEFAULT_OUT))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
