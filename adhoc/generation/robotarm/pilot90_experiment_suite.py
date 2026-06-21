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

VERIFY_SCORING = "gt_match"
VERIFY_SCORING_VERSION = 1

# Re-export for backward compat (CONSOLIDATED rows use groundtruth = pose_gt)
CONSOLIDATED = GT_PATH  # noqa: F811 — same path, compatible row shape
MOTION_COMPONENT_GT = GT_PATH
POSE_CFG_LEGACY = _REPO / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_sophisticated_ee_pilot90_non_essence.json"


def manifest90_cues_csv() -> str:
    return ",".join(manifest90_cue_names())


def manifest90_rows_from_cfg(cfg_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from pilot90_paths import row_generation_done  # noqa: WPS433

    manifest = set(manifest90_cue_names())
    return [
        r for r in cfg_rows
        if r.get("cue") in manifest and row_generation_done(r)
    ]


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
        "11": "motion_rep_pairwise_input",
        "12": "context_variation",
        "13": "baseline_fewshot_score",
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
        correct = pose_generation_correct_any(row, ev["pose_gt"])
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
                "scoring": "any_pose_in_config",
            }
        )
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "mode": "pose_generation_vs_human_gt_any_pose",
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


def motion_generation_match(row: dict[str, Any], movement_gt: dict[str, Any]) -> bool | None:
    """Exp7 rule: generated tail vs movement_gt component."""
    from score_pilot40_motion_gt_components import _tail_matches_component, _tail_steps  # noqa: WPS433

    comp = movement_gt.get("component")
    raw = (movement_gt.get("annotation_raw") or "").strip()
    if not comp and not movement_gt.get("always_correct") and raw.lower() != "none":
        return None
    tail = _tail_steps(row.get("movements") or [])
    if movement_gt.get("always_correct") or (comp and comp.get("kind") == "any"):
        return True
    if comp:
        match, _ = _tail_matches_component(tail, comp)
        return match
    return None


def _normalize_recommended_component(raw: dict[str, Any] | None) -> dict[str, Any] | None:
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


def _recommended_component_matches_gt(
    rec: dict[str, Any] | None, comp: dict[str, Any] | None
) -> bool | None:
    if not comp:
        return None
    from motion_gt_tail_builder import build_tail_from_component  # noqa: WPS433
    from score_pilot40_motion_gt_components import _tail_matches_component  # noqa: WPS433

    norm = _normalize_recommended_component(rec)
    if not norm:
        return False
    built = build_tail_from_component(norm)
    if not built:
        return False
    match, _ = _tail_matches_component(built, comp)
    return match


def _model_tag_from_result_path(path: Path) -> str | None:
    m = re.search(r"(?:result|score)_exp\d+_(.+)\.json$", path.name)
    return m.group(1) if m else None


def _generation_scores_from_sidecar(score_path: Path, key: str) -> dict[str, bool | None]:
    if not score_path.is_file():
        return {}
    data = json.loads(score_path.read_text(encoding="utf-8"))
    return {
        str(r["cue"]): r.get(key)
        for r in (data.get("rows") or [])
        if r.get("cue") is not None
    }


def _resolve_generation_config(
    *candidates: str | Path | None,
    fallback: Path | None = None,
) -> Path:
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_file():
            return p
        local = RESULT_CFG_DIR / p.name
        if local.is_file():
            return local
    if fallback and fallback.is_file():
        return fallback
    raise FileNotFoundError("generation config not found")


def _try_resolve_generation_config(
    *candidates: str | Path | None,
    fallback: Path | None = None,
    score_sidecar: Path | None = None,
) -> Path | None:
    try:
        return _resolve_generation_config(*candidates, fallback=fallback)
    except FileNotFoundError:
        pass
    if score_sidecar and score_sidecar.is_file():
        side = json.loads(score_sidecar.read_text(encoding="utf-8"))
        try:
            return _resolve_generation_config(side.get("config_json"), fallback=fallback)
        except FileNotFoundError:
            return None
    return None


def _score_pose_verify_row(
    row: dict[str, Any],
    pose_gt: str,
    *,
    cfg_row: dict[str, Any] | None = None,
    generation_correct: bool | None = None,
) -> dict[str, Any] | None:
    result = row.get("result") or {}
    appropriate = result.get("pose_is_appropriate")
    if appropriate is None or not pose_gt:
        return None
    if generation_correct is None and cfg_row:
        generation_correct = pose_generation_correct_any(cfg_row, pose_gt)
    recommended_matches_gt: bool | None = None
    if appropriate:
        if generation_correct is None:
            return None
        verify_correct = bool(generation_correct)
    else:
        na = result.get("if_not_appropriate") or {}
        rec_pose = {
            "dir": na.get("recommended_dir"),
            "gripper_orientation": na.get("recommended_gripper_orientation"),
        }
        recommended_matches_gt = pose_generation_correct(rec_pose, pose_gt)
        verify_correct = bool(recommended_matches_gt)
    return {
        "pose_is_appropriate": bool(appropriate),
        "generation_correct": generation_correct,
        "recommended_matches_gt": recommended_matches_gt,
        "verify_correct": verify_correct,
        "scoring_mode": VERIFY_SCORING,
    }


def _score_motion_verify_row(
    row: dict[str, Any],
    movement_gt: dict[str, Any],
    *,
    cfg_row: dict[str, Any] | None = None,
    generation_match: bool | None = None,
) -> dict[str, Any] | None:
    parsed = row.get("verify_result") or row.get("parsed") or row.get("result") or {}
    appropriate = row.get("movement_is_appropriate")
    if appropriate is None:
        appropriate = parsed.get("movement_is_appropriate")
    if appropriate is None:
        return None
    comp = movement_gt.get("component")
    if generation_match is None and cfg_row:
        generation_match = motion_generation_match(cfg_row, movement_gt)
    recommended_matches_gt: bool | None = None
    if appropriate:
        if generation_match is None:
            return None
        verify_correct = bool(generation_match)
    else:
        rec = row.get("recommended_component")
        if not rec:
            rec = (parsed.get("if_not_appropriate") or {}).get("recommended_component")
        recommended_matches_gt = _recommended_component_matches_gt(rec, comp)
        verify_correct = bool(recommended_matches_gt)
    return {
        "movement_is_appropriate": bool(appropriate),
        "generation_match": generation_match,
        "recommended_matches_gt": recommended_matches_gt,
        "verify_correct": verify_correct,
        "scoring_mode": VERIFY_SCORING,
    }


def score_verify_pose_json(
    verify_path: Path,
    pose_cfg_path: Path | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """GT verify score for exp2/3: app→exp1 gen match; inapp→recommended pose vs pose_gt."""
    if not verify_path.is_file():
        raise FileNotFoundError(verify_path)
    data = json.loads(verify_path.read_text(encoding="utf-8"))
    model_tag = _model_tag_from_result_path(verify_path) or "qwen32b"
    score_sidecar = score_result_path("1", model_tag)
    gen_by_cue = _generation_scores_from_sidecar(score_sidecar, "generation_correct")
    cfg_path = _try_resolve_generation_config(
        pose_cfg_path,
        data.get("config_json"),
        data.get("generation_config"),
        fallback=pose_cfg_path,
        score_sidecar=score_sidecar,
    )
    cfg_by: dict[int, dict[str, Any]] = {}
    if cfg_path and cfg_path.is_file():
        cfg_by = {int(r["idx"]): r for r in load_config_list(cfg_path)}
    gt = load_gt_by_cue()
    ok = n = 0
    for row in data.get("results") or []:
        cue = str(row.get("cue") or "")
        ev = gt.get(cue) or {}
        pose_gt = str(ev.get("pose_gt") or "")
        cfg_row = cfg_by.get(int(row.get("idx", 0)))
        generation_correct = gen_by_cue.get(cue)
        scored = _score_pose_verify_row(
            row,
            pose_gt,
            cfg_row=cfg_row,
            generation_correct=generation_correct,
        )
        if scored is None:
            row.pop("verify_scoring", None)
            continue
        row["verify_scoring"] = scored
        n += 1
        if scored["verify_correct"]:
            ok += 1
    data["verify_scoring"] = VERIFY_SCORING
    data["verify_scoring_version"] = VERIFY_SCORING_VERSION
    if cfg_path and cfg_path.is_file():
        data["generation_config"] = str(cfg_path)
    elif score_sidecar.is_file():
        data["generation_score_json"] = str(score_sidecar)
    data["groundtruth"] = str(GT_PATH)
    data["n"] = n
    data["n_correct"] = ok
    data["accuracy"] = ok / n if n else None
    if write:
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def score_verify_motion_json(
    verify_path: Path,
    motion_cfg_path: Path | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """GT verify score for exp8/9: app→exp7 gen match; inapp→recommended component vs movement_gt."""
    if not verify_path.is_file():
        raise FileNotFoundError(verify_path)
    data = json.loads(verify_path.read_text(encoding="utf-8"))
    model_tag = _model_tag_from_result_path(verify_path) or "qwen32b"
    score_sidecar = score_result_path("7", model_tag)
    gen_by_cue = _generation_scores_from_sidecar(score_sidecar, "component_match")
    cfg_path = _try_resolve_generation_config(
        motion_cfg_path,
        data.get("config"),
        data.get("config_json"),
        data.get("generation_config"),
        fallback=motion_cfg_path,
        score_sidecar=score_sidecar,
    )
    cfg_by: dict[int, dict[str, Any]] = {}
    if cfg_path and cfg_path.is_file():
        cfg_by = {int(r["idx"]): r for r in load_config_list(cfg_path)}
    gt = load_gt_by_cue()
    ok = n = 0
    for row in data.get("rows") or data.get("results") or []:
        cue = str(row.get("cue") or "")
        cfg_row = cfg_by.get(int(row.get("cue_idx", row.get("idx", 0))))
        if not cue and cfg_row:
            cue = str(cfg_row.get("cue") or "")
        ev = gt.get(cue) or {}
        movement_gt = ev.get("movement_gt") or {}
        generation_match = gen_by_cue.get(cue)
        scored = _score_motion_verify_row(
            row,
            movement_gt,
            cfg_row=cfg_row,
            generation_match=generation_match,
        )
        if scored is None:
            row.pop("verify_scoring", None)
            continue
        row["verify_scoring"] = scored
        n += 1
        if scored["verify_correct"]:
            ok += 1
    data["verify_scoring"] = VERIFY_SCORING
    data["verify_scoring_version"] = VERIFY_SCORING_VERSION
    if cfg_path and cfg_path.is_file():
        data["generation_config"] = str(cfg_path)
    elif score_sidecar.is_file():
        data["generation_score_json"] = str(score_sidecar)
    data["groundtruth"] = str(GT_PATH)
    data["n"] = n
    data["n_correct"] = ok
    data["accuracy"] = ok / n if n else None
    if write:
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def score_exp7(config_path: Path, out_path: Path) -> dict[str, Any]:
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
        match = motion_generation_match(row, mg)
        if match is None:
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


def metrics_from_json(
    path: Path,
    spec: dict[str, Any],
    *,
    pose_cfg: Path | None = None,
    motion_cfg: Path | None = None,
    rescore_json: bool = True,
) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing", "path": str(path)}

    kind = spec["kind"]
    model_tag = str(spec.get("model_tag") or "qwen32b")
    if pose_cfg is None:
        pose_cfg = config_for_experiment("1", model_tag)
    if motion_cfg is None:
        motion_cfg = config_for_experiment("7", model_tag)

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
        try:
            if rescore_json:
                data = score_verify_pose_json(path, pose_cfg if pose_cfg.is_file() else None, write=True)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = json.loads(path.read_text(encoding="utf-8"))
        ok = int(data.get("n_correct", 0))
        n = int(data.get("n", 0))
        acc = data.get("accuracy")
        if acc is None and n:
            acc = ok / n
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"verify-gt {ok}/{n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    if kind in {"motion_verify_vlm", "motion_verify_text"}:
        try:
            if rescore_json:
                data = score_verify_motion_json(path, motion_cfg if motion_cfg.is_file() else None, write=True)
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data = json.loads(path.read_text(encoding="utf-8"))
        ok = int(data.get("n_correct", 0))
        n = int(data.get("n", 0))
        acc = data.get("accuracy")
        if acc is None and n:
            acc = ok / n
        return {
            "status": "ok",
            "path": str(path),
            "ok": ok,
            "n": n,
            "accuracy": acc,
            "accuracy_pct": None if acc is None else round(100 * acc, 1),
            "headline": f"verify-gt {ok}/{n}"
            + (f" = {100 * acc:.1f}%" if acc is not None else ""),
        }

    old_motion = p40.MOTION_CFG
    try:
        if motion_cfg:
            p40.MOTION_CFG = motion_cfg
        return p40.metrics_from_json(path, spec)
    finally:
        p40.MOTION_CFG = old_motion


def _result_json_path(spec: dict[str, Any], model_tag: str) -> Path:
    eid = str(spec["id"])
    kind = spec["kind"]
    if kind in {"pose_generation_score", "motion_generation_score", "context_variation", "baseline_fewshot_score"}:
        return score_result_path(eid, model_tag)
    return verify_result_path(eid, model_tag)


def print_qwen_series_summary() -> None:
    from qwen_cross_summary import print_qwen_cross_summary  # noqa: WPS433

    def _metrics(path: Path, spec: dict[str, Any], **kw: Any) -> dict[str, Any]:
        tag = str(kw.get("model_tag") or spec.get("model_tag") or "qwen32b")
        return metrics_from_json(
            path,
            {**spec, "model_tag": tag},
            pose_cfg=config_for_experiment("1", tag),
            motion_cfg=config_for_experiment("7", tag),
            rescore_json=True,
        )

    specs = experiment_specs_all("qwen32b")
    print_qwen_cross_summary(
        suite_label=f"Manipulator pilot-90 ({N_CUES} cues, tasks 1–13)",
        specs=specs,
        result_path_for=_result_json_path,
        metrics_for=_metrics,
        repo=_REPO,
    )


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
    # any_pose_in_config: any listed GT (dir, orientation) may appear in any pose step
    return any(t in gen_set for t in targets)
