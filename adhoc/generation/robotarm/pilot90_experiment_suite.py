"""Pilot-90 experiment map — unified paths for Gemini and Qwen (see README_PILOT.md)."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pilot40_experiment_suite as p40
from pilot40_experiment_suite import _parse_gt_poses, human_gt_pose_ok, pose_generation_correct  # noqa: F401
from pilot90_paths import (  # noqa: F401
    EXPERIMENT_TITLES,
    GENERATION_EXPS,
    GT_PATH,
    MANIFEST_TSV,
    MOTION_MANIFEST,
    MOTION_PAIRWISE_DIR,
    MULTITILE_IMG_DIR,
    N_CUES,
    PAIRWISE_IMG_DIR,
    PROMPT_EXP_DIR,
    RESULT_CFG_DIR,
    SHOTS,
    TILE_DIR,
    TILE_PICK,
    VERIFY_EXP_DIR,
    config_for_experiment,
    load_config_list,
    load_gt_by_cue,
    manifest90_cue_names,
    model_to_tag,
    prompt_exp_path,
    result_config_path,
    score_result_path,
    html_result_path,
    verify_result_path,
)

_REPO = Path(__file__).resolve().parent.parents[2]

# Re-export for backward compat (CONSOLIDATED rows use groundtruth = pose_gt)
CONSOLIDATED = GT_PATH  # noqa: F811 — same path, compatible row shape
MOTION_COMPONENT_GT = GT_PATH
POSE_CFG_LEGACY = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"


def manifest90_cues_csv() -> str:
    return ",".join(manifest90_cue_names())


def manifest90_rows_from_cfg(cfg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = set(manifest90_cue_names())
    return [r for r in cfg_rows if r.get("cue") in manifest]


def _first_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return row.get("gt_fixed_first_pose") or {}


def pose_generation_correct_first(row: dict[str, Any], pose_gt: str) -> bool | None:
    """Task 1: first pose step vs human pose GT."""
    return pose_generation_correct(_first_pose(row), pose_gt)


def experiment_specs_all(model_tag: str = "qwen32b") -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for eid, title in EXPERIMENT_TITLES.items():
        kind = _kind_for_exp(eid)
        spec: dict[str, Any] = {
            "id": eid,
            "title": title,
            "kind": kind,
            "prompt": str(prompt_exp_path(eid).relative_to(_REPO)),
            "prompt_file": f"prompt_exp{eid}.txt",
            "model_tag": model_tag,
            "shots": str(SHOTS.relative_to(_REPO)),
        }
        spec["html"] = str(html_result_path(eid, model_tag).relative_to(_REPO))
        if eid in GENERATION_EXPS:
            spec["result_config"] = str(result_config_path(eid, model_tag).relative_to(_REPO))
            spec["score_json"] = str(score_result_path(eid, model_tag).relative_to(_REPO))
            spec["out_name"] = score_result_path(eid, model_tag).name
        else:
            spec["out_name"] = verify_result_path(eid, model_tag).name
            spec["verify_json"] = str(verify_result_path(eid, model_tag).relative_to(_REPO))
        if eid in {"1", "2", "3", "4", "5", "6"}:
            spec["input_config"] = str(config_for_experiment(eid, model_tag).relative_to(_REPO))
        elif eid in {"7", "8", "9", "10"}:
            spec["input_config"] = str(config_for_experiment(eid, model_tag).relative_to(_REPO))
        if eid == "5":
            spec["grid_sizes"] = "6"
        if eid == "6":
            spec["grid_sizes"] = "12"
        if eid == "8":
            spec["media_dir"] = str(MOTION_MANIFEST.parent / "mp4")
        specs.append(spec)
    return specs


def _kind_for_exp(eid: str) -> str:
    return {
        "1": "pose_generation_score",
        "2": "pose_verify_vlm",
        "3": "pose_verify_text",
        "4": "pose_pairwise",
        "5": "multitile",
        "6": "multitile",
        "7": "motion_generation_score",
        "8": "motion_verify_vlm",
        "9": "motion_verify_text",
        "10": "motion_pairwise_mp4",
    }[eid]


def score_exp1(config_path: Path, out_path: Path) -> dict[str, Any]:
    gt = load_gt_by_cue()
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in manifest90_rows_from_cfg(load_config_list(config_path)):
        cue = row.get("cue")
        ev = gt.get(cue or "")
        if not ev or not ev.get("pose_gt"):
            continue
        correct = pose_generation_correct_first(row, ev["pose_gt"])
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": cue,
                "pose_gt": ev["pose_gt"],
                "generation_correct": correct,
                "scoring": "first_pose_vs_gt",
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "exp1_pose_generation_vs_gt",
        "config_json": str(config_path),
        "groundtruth": str(GT_PATH),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def score_exp7(config_path: Path, out_path: Path) -> dict[str, Any]:
    from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps  # noqa: WPS433

    gt = load_gt_by_cue()
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in manifest90_rows_from_cfg(load_config_list(config_path)):
        cue = row.get("cue")
        ev = gt.get(cue or "") or {}
        mg = ev.get("movement_gt") or {}
        comp = mg.get("component")
        raw = (mg.get("annotation_raw") or "").strip()
        if not comp and not mg.get("always_correct") and raw.lower() != "none":
            continue
        tail = _tail_steps(row.get("movements") or [])
        if mg.get("always_correct") or (comp and comp.get("kind") == "any"):
            match = True
        elif comp:
            match, _ = _tail_matches_component(tail, comp)
        else:
            continue
        n += 1
        if match:
            ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": cue,
                "annotation_raw": raw,
                "component_match": match,
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "exp7_motion_generation_vs_component_gt",
        "config_json": str(config_path),
        "groundtruth_json": str(GT_PATH),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def metrics_from_json(path: Path, spec: dict[str, Any], *, motion_cfg: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}

    kind = spec["kind"]
    if kind in {"pose_generation_score", "motion_generation_score"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        ok = int(data.get("n_correct", 0))
        n = int(data.get("n", 0))
        acc = ok / n if n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }

    if kind in {"pose_verify_vlm", "pose_verify_text"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results") or []
        gt = load_gt_by_cue()
        agree_ok = agree_n = 0
        for r in results:
            cue = r.get("cue")
            if not cue or cue not in gt:
                continue
            model_ok = r.get("result", {}).get("pose_is_appropriate")
            if model_ok is None:
                continue
            agree_n += 1
            human_ok = human_gt_pose_ok(gt[cue].get("pose_gt", ""))
            if bool(model_ok) == human_ok:
                agree_ok += 1
        acc = agree_ok / agree_n if agree_n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": agree_ok,
            "n": agree_n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"human-agree {agree_ok}/{agree_n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps  # noqa: WPS433

        gt = load_gt_by_cue()
        cfg_path = motion_cfg or Path(spec.get("input_config", ""))
        cfg_by = {int(r["idx"]): r for r in load_config_list(cfg_path)}
        det_ok = det_n = 0
        for vr in data.get("rows") or []:
            cue = vr.get("cue")
            row = cfg_by.get(int(vr.get("cue_idx", 0))) or {}
            if not cue:
                cue = row.get("cue")
            ev = gt.get(str(cue or ""), {})
            comp = (ev.get("movement_gt") or {}).get("component")
            if not comp or not row:
                continue
            gen_tail = _tail_steps(row.get("movements") or [])
            gen_match, _ = _tail_matches_component(gen_tail, comp)
            appropriate = vr.get("movement_is_appropriate")
            if appropriate is None:
                continue
            det_n += 1
            if bool(appropriate) == bool(gen_match):
                det_ok += 1
        acc = det_ok / det_n if det_n else None
        return {
            "status": "ok",
            "path": str(path),
            "ok": det_ok,
            "n": det_n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"detection {det_ok}/{det_n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    old_motion = p40.MOTION_CFG
    try:
        if motion_cfg:
            p40.MOTION_CFG = motion_cfg
        return p40.metrics_from_json(path, spec)
    finally:
        p40.MOTION_CFG = old_motion


def print_summary_table(
    specs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    *,
    n_cues: int = N_CUES,
    model_tag: str = "",
) -> None:
    label = model_tag or "model"
    print("\n" + "=" * 92)
    print(f"PILOT-90 ({n_cues} cues, 10 steps) — {label} ACCURACY SUMMARY")
    print("=" * 92)
    print(f"{'#':<4}  {'experiment':<46}  {'result'}")
    print("-" * 92)
    for spec, m in zip(specs, metrics):
        headline = m.get("headline", m.get("status", "?"))
        if m.get("status") == "missing":
            headline = "MISSING JSON"
        elif m.get("status") == "error":
            headline = f"ERROR: {m.get('error', '?')[:40]}"
        print(f"{spec['id']:<4}  {spec['title']:<46}  {headline}")
    print("=" * 92)


# backward compat
DEFAULT_QWEN_OUT = VERIFY_EXP_DIR
POSE_CFG = POSE_CFG_LEGACY
MOTION_CFG = POSE_CFG_LEGACY


def qwen_out_dir(model_tag: str = "qwen32b") -> Path:
    return VERIFY_EXP_DIR


def pose_generation_correct_any(row: dict[str, Any], groundtruth: str) -> bool | None:
    poses: list[dict[str, Any]] = []
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose") or {}
        if pose.get("dir") and pose.get("gripper_orientation"):
            poses.append(pose)
    if not groundtruth or not poses:
        return None
    targets = _parse_gt_poses(groundtruth.strip())
    if not targets:
        return None
    gen_set = {
        (str(p.get("dir", "")).strip(), str(p.get("gripper_orientation", "")).strip()) for p in poses
    }
    if groundtruth.strip().lower().startswith("o"):
        return targets[0] in gen_set
    return any(t in gen_set for t in targets)
