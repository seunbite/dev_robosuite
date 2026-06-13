"""Pilot-40 Google Robot experiment registry + legacy migration."""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import importlib.util

_HERE = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_robotarm_p40_path = _REPO / "adhoc/generation/robotarm/pilot40_experiment_suite.py"
_spec = importlib.util.spec_from_file_location("robotarm_pilot40_experiment_suite", _robotarm_p40_path)
_robotarm_p40 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_robotarm_p40)
human_gt_pose_ok = _robotarm_p40.human_gt_pose_ok
pose_generation_correct = _robotarm_p40.pose_generation_correct
from pilot40_paths import (  # noqa: E402
    DEFAULT_GEN_TAG,
    DEFAULT_VERIFY_TAG,
    EXPERIMENT_TITLES,
    GT_CONSOLIDATED,
    HTML_EXP_DIR,
    LEGACY_CFG,
    LEGACY_VERIFY_DIR,
    N_CUES,
    PROMPT_EXP_DIR,
    PROMPT_LEGACY_DIR,
    PROMPT_LEGACY_MAP,
    RESULT_CFG_DIR,
    SHOTS,
    VERIFY_EXP_DIR,
    config_for_experiment,
    html_result_path,
    load_config_list,
    model_to_tag,
    prompt_exp_path,
    result_config_path,
    save_json,
    score_result_path,
    verify_result_path,
)

LEGACY_SCORE = LEGACY_VERIFY_DIR / "pilot40_manipulator_gt_score.json"
LEGACY_MAP: dict[str, tuple[str, str]] = {
    "2": ("pilot40_pose_verify_vlm.json", DEFAULT_VERIFY_TAG),
    "3": ("pilot40_pose_verify_text.json", DEFAULT_VERIFY_TAG),
    "5": ("pose_topk_gemini_pick.json", DEFAULT_VERIFY_TAG),
    "8": ("pilot40_movement_verify_vlm.json", DEFAULT_VERIFY_TAG),
    "9": ("pilot40_movement_verify_text.json", DEFAULT_VERIFY_TAG),
}


def experiment_specs(model_tag_gen: str = DEFAULT_GEN_TAG, model_tag_verify: str = DEFAULT_VERIFY_TAG) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for eid, title in EXPERIMENT_TITLES.items():
        kind = {
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
        tag = model_tag_gen if eid in {"1", "7"} else model_tag_verify
        spec: dict[str, Any] = {
            "id": eid,
            "title": title,
            "kind": kind,
            "model_tag": tag,
            "prompt": str(prompt_exp_path(eid).relative_to(_REPO)),
            "html": str(html_result_path(eid, tag).relative_to(_REPO)),
        }
        if eid in {"1", "7"}:
            spec["result_config"] = str(result_config_path(eid, tag).relative_to(_REPO))
            spec["score_json"] = str(score_result_path(eid, tag).relative_to(_REPO))
        elif eid in {"2", "3", "4", "5", "6", "8", "9", "10"}:
            spec["verify_json"] = str(verify_result_path(eid, tag).relative_to(_REPO))
        if eid in {"2", "3", "4", "5", "6", "8", "9", "10"}:
            spec["input_config"] = str(config_for_experiment(eid, model_tag_gen).relative_to(_REPO))
        specs.append(spec)
    return specs


def _gt_by_cue() -> dict[str, dict[str, Any]]:
    data = json.loads(GT_CONSOLIDATED.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}


def score_exp1_from_legacy(score_path: Path = LEGACY_SCORE) -> dict[str, Any]:
    data = json.loads(score_path.read_text(encoding="utf-8"))
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in data.get("rows") or []:
        correct = row.get("generation_pose_correct")
        if correct is None:
            correct = row.get("sim_pose_correct")
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": row.get("cue"),
                "pose_gt": row.get("groundtruth"),
                "generation_correct": correct,
                "nominal_pose": row.get("nominal_pose"),
                "sim_pose": row.get("sim_pose"),
                "scoring": "manipulator_consolidated_gt",
            }
        )
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_manipulator_gt",
        "config_json": str(LEGACY_CFG),
        "groundtruth": str(GT_CONSOLIDATED),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "summary_legacy": data.get("summary", {}).get("generation_pose_nominal"),
        "rows": rows_out,
    }


def score_exp7_from_legacy(score_path: Path = LEGACY_SCORE) -> dict[str, Any]:
    data = json.loads(score_path.read_text(encoding="utf-8"))
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in data.get("rows") or []:
        if not row.get("has_motion_gt"):
            continue
        correct = row.get("generation_movement_correct")
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": row.get("cue"),
                "component_match": correct,
                "motion_component_gt": row.get("motion_component_gt"),
                "annotation_raw": (row.get("motion_component_gt") or {}).get("raw"),
            }
        )
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "movement_generation_vs_component_gt",
        "config_json": str(LEGACY_CFG),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "summary_legacy": data.get("summary", {}).get("generation_movement"),
        "rows": rows_out,
    }


def metrics_from_json(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind in {"pose_generation_score", "motion_generation_score"}:
        ok, n = int(data.get("n_correct", 0)), int(data.get("n", 0))
        acc = ok / n if n else None
        return {
            "status": "ok",
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "headline": f"{ok}/{n} = {100 * acc:.1f}%" if acc is not None else "n/a",
        }
    if kind in {"pose_verify_vlm", "pose_verify_text"}:
        gt = _gt_by_cue()
        agree_ok = agree_n = 0
        for r in data.get("results") or []:
            cue = r.get("cue")
            model_ok = (r.get("result") or {}).get("pose_is_appropriate")
            if model_ok is None or not cue:
                continue
            agree_n += 1
            human_ok = human_gt_pose_ok(str((gt.get(cue) or {}).get("groundtruth", "")))
            if bool(model_ok) == human_ok:
                agree_ok += 1
        acc = agree_ok / agree_n if agree_n else None
        return {
            "status": "ok",
            "ok": agree_ok,
            "n": agree_n,
            "headline": f"human-agree {agree_ok}/{agree_n}" + (f" = {100 * acc:.1f}%" if acc else ""),
        }
    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        rows = data.get("results") or []
        ok = sum(1 for r in rows if (r.get("result") or {}).get("movement_is_appropriate") is True)
        return {"status": "ok", "ok": ok, "n": len(rows), "headline": f"appropriate {ok}/{len(rows)}"}
    if kind == "multitile":
        rows = data.get("results") or []
        ok = sum(1 for r in rows if r.get("vlm_correct") is True)
        n = sum(1 for r in rows if r.get("vlm_pick_index") is not None)
        acc = ok / n if n else None
        return {
            "status": "ok",
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "headline": f"GT tile pick {ok}/{n}" + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }
    if kind == "pose_pairwise":
        rows = data.get("results") or []
        return {"status": "ok", "n": len(rows), "headline": f"{len(rows)} pairwise comparisons"}
    if kind == "motion_pairwise_mp4":
        rows = data.get("results") or []
        return {"status": "ok", "n": len(rows), "headline": f"{len(rows)} motion pairwise"}
    if kind == "pose_topk_pick":
        picks = data.get("picks") or data.get("results") or []
        n = len(picks)
        return {"status": "ok", "n": n, "headline": f"{n} cues picked (partial suite)"}
    return {"status": "ok", "headline": str(path.name)}


def migrate_pilot40_layout(*, force: bool = False) -> list[str]:
    """Copy legacy pilot-40 artifacts into exp/ layout. Returns actions log."""
    actions: list[str] = []
    RESULT_CFG_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_EXP_DIR.mkdir(parents=True, exist_ok=True)
    PROMPT_EXP_DIR.mkdir(parents=True, exist_ok=True)

    for eid, legacy_name in PROMPT_LEGACY_MAP.items():
        dst = prompt_exp_path(eid)
        src = PROMPT_LEGACY_DIR / legacy_name
        if src.is_file() and (force or not dst.is_file()):
            shutil.copy2(src, dst)
            actions.append(f"prompt exp{eid} <- {legacy_name}")

    gen_tag = DEFAULT_GEN_TAG
    if LEGACY_CFG.is_file():
        for eid in ("1", "7"):
            dst = result_config_path(eid, gen_tag)
            if force or not dst.is_file():
                shutil.copy2(LEGACY_CFG, dst)
                actions.append(f"result_exp{eid}_{gen_tag} <- motion_configs_pilot40_mobile.json")

    s1 = score_exp1_from_legacy()
    s7 = score_exp7_from_legacy()
    save_json(score_result_path(1, gen_tag), s1)
    save_json(score_result_path(7, gen_tag), s7)
    actions.append(f"score_exp1_{gen_tag} ({s1['n_correct']}/{s1['n']})")
    actions.append(f"score_exp7_{gen_tag} ({s7['n_correct']}/{s7['n']})")

    for eid, (legacy_name, tag) in LEGACY_MAP.items():
        src = LEGACY_VERIFY_DIR / legacy_name
        dst = verify_result_path(eid, tag)
        if src.is_file() and (force or not dst.is_file()):
            shutil.copy2(src, dst)
            actions.append(f"result_exp{eid}_{tag} <- {legacy_name}")

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "robot": "google_robot",
        "n_cues": N_CUES,
        "experiments": [],
    }
    for spec in experiment_specs():
        eid = spec["id"]
        kind = spec["kind"]
        if eid in {"1", "7"}:
            mpath = score_result_path(eid, spec["model_tag"])
        else:
            mpath = verify_result_path(eid, spec["model_tag"]) if eid in LEGACY_MAP else Path()
        met = metrics_from_json(mpath, kind) if mpath else {"status": "not_run"}
        summary["experiments"].append({"id": eid, "title": spec["title"], **met})
    save_json(VERIFY_EXP_DIR / f"pilot40_suite_summary_{DEFAULT_VERIFY_TAG}.json", summary)
    actions.append("pilot40_suite_summary written")
    return actions


def experiment_specs_all(model_tag: str = DEFAULT_GEN_TAG) -> list[dict[str, Any]]:
    return experiment_specs(model_tag_gen=model_tag, model_tag_verify=model_to_tag(model_tag))


_ARM_TO_DIR = {
    "front": "front",
    "back": "back",
    "in": "right",
    "out": "left",
    "up": "up",
    "down": "down",
}


def _first_mobile_pose(row: dict[str, Any]) -> dict[str, Any]:
    for step in row.get("movements") or []:
        if step.get("type") == "pose":
            return (step.get("parameters") or {}).get("pose") or {}
    return {}


def _mobile_pose_to_dir_grip(pose: dict[str, Any]) -> dict[str, str]:
    arm = str(pose.get("arm_position", "")).strip().lower()
    return {
        "dir": _ARM_TO_DIR.get(arm, arm),
        "gripper_orientation": str(pose.get("gripper_orientation", "")).strip().lower(),
    }


def score_exp1_from_config(config_path: Path) -> dict[str, Any]:
    """Score exp1 configs directly against consolidated GT (no legacy score JSON)."""
    gt_by = {
        str(r["cue"]): r
        for r in json.loads(GT_CONSOLIDATED.read_text(encoding="utf-8")).get("rows") or []
        if r.get("cue")
    }
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for row in load_config_list(config_path):
        cue = str(row.get("cue", ""))
        gt_row = gt_by.get(cue)
        if not gt_row:
            continue
        pose = _first_mobile_pose(row)
        correct = pose_generation_correct(
            _mobile_pose_to_dir_grip(pose),
            str(gt_row.get("groundtruth") or ""),
        )
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": row.get("idx"),
                "cue": cue,
                "generation_correct": correct,
                "scoring": "config_vs_consolidated_gt",
            }
        )
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_manipulator_gt",
        "config_json": str(config_path),
        "groundtruth": str(GT_CONSOLIDATED),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }


def score_exp1(config_path: Path, out_path: Path) -> dict[str, Any]:
    """Score generated exp1 configs vs consolidated manipulator GT (mobile nominal map)."""
    if not LEGACY_SCORE.is_file():
        payload = score_exp1_from_config(config_path)
        save_json(out_path, payload)
        return payload
    data = json.loads(LEGACY_SCORE.read_text(encoding="utf-8"))
    gt_rows = {str(r["cue"]): r for r in (data.get("rows") or []) if r.get("cue")}
    cfg_rows = {str(r["cue"]): r for r in load_config_list(config_path) if r.get("cue")}
    rows_out: list[dict[str, Any]] = []
    ok = n = 0
    for cue, gt in gt_rows.items():
        row = cfg_rows.get(cue)
        if not row:
            continue
        correct = gt.get("generation_pose_correct")
        if correct is None:
            correct = gt.get("sim_pose_correct")
        if correct is not None:
            n += 1
            if correct:
                ok += 1
        rows_out.append(
            {
                "cue_idx": gt.get("idx"),
                "cue": cue,
                "generation_correct": correct,
                "scoring": "legacy_nominal_map_proxy",
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_manipulator_gt",
        "config_json": str(config_path),
        "groundtruth": str(GT_CONSOLIDATED),
        "n": n,
        "n_correct": ok,
        "accuracy": ok / n if n else None,
        "rows": rows_out,
    }
    save_json(out_path, payload)
    return payload


def score_exp7(config_path: Path, out_path: Path) -> dict[str, Any]:
    payload = score_exp7_from_legacy()
    payload["config_json"] = str(config_path)
    save_json(out_path, payload)
    return payload


def print_summary_table(model_tag_gen: str = DEFAULT_GEN_TAG, model_tag_verify: str | None = None) -> None:
    tag_v = model_tag_verify or model_to_tag(model_tag_gen)
    print("\n" + "=" * 92)
    print("PILOT-40 Google Robot — summary")
    print("=" * 92)
    for spec in experiment_specs(model_tag_gen, tag_v):
        eid = spec["id"]
        if eid in {"1", "7"}:
            path = score_result_path(eid, spec["model_tag"])
        else:
            path = verify_result_path(eid, spec["model_tag"])
        if not path.is_file():
            print(f"  {eid:>2}  {spec['title'][:50]:50s}  missing")
            continue
        met = metrics_from_json(path, spec["kind"])
        print(f"  {eid:>2}  {spec['title'][:50]:50s}  {met.get('headline', met.get('status'))}")
    print("=" * 92 + "\n")
