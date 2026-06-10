"""Build GT vs neg-axis pairwise MP4s (shared by Gemini + Qwen exp10)."""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _REPO / "adhoc" / "vlm_test", _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from PIL import Image  # noqa: E402

import testset_utils  # noqa: E402
from motion_gt_tail_builder import (  # noqa: E402
    apply_single_element_variant,
    build_config_from_gt_pose_and_component,
)
from motion_media_paths import pose_jsonl  # noqa: E402
from motion_neg_axis_pick import primary_axis_from_component  # noqa: E402
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR  # noqa: E402
from pilot90_experiment_suite import MOTION_CFG  # noqa: E402
from score_pilot40_motion_gt_components import (  # noqa: E402
    _build_annotation_map,
    _tail_matches_component,
    _tail_steps,
)
from verify_pose_pairwise_12_gemini import _stitch_pair  # noqa: E402
from verify_pose_tiles_gemini import _movement_summary  # noqa: E402

ROBOT = "IIWA"
HZ = 10
N_OUT_FRAMES = 9
PAIRWISE_SPECS_NAME = "pairwise_specs_pilot90.json"
PROMPT_TEMPLATE = _REPO / "data/seed/prompt/pilot40/exp10_motion_pairwise_mp4.txt"


def _rows_by_idx() -> dict[int, dict[str, Any]]:
    data = json.loads(MOTION_CFG.read_text(encoding="utf-8"))
    return {int(r["idx"]): r for r in data}


def _pairwise_specs() -> list[dict[str, Any]]:
    ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    by_idx = _rows_by_idx()
    specs: list[dict[str, Any]] = []
    for idx, row in sorted(by_idx.items()):
        entry = ann.get(idx) or {}
        if entry.get("always_correct") or (entry.get("component") or {}).get("kind") == "any":
            continue
        comp = entry.get("component")
        if not comp:
            continue
        specs.append(
            {
                "idx": idx,
                "cue": str(row["cue"]),
                "component": comp,
                "row": row,
            }
        )
    return specs


def pairwise_gt_side(idx: int, cue: str) -> str:
    """Deterministic 50/50 GT on left vs right (reproducible across Gemini/Qwen)."""
    rng = random.Random(int(idx) * 1009 + hash(cue) % 10007)
    return "left" if rng.random() < 0.5 else "right"


def _movement_step_label(step: dict[str, Any]) -> str | None:
    p = step.get("parameters") or {}
    joint = p.get("joint", "?")
    dirs = p.get("directions") or []
    if not dirs:
        return f"movement {joint}"
    parts: list[str] = []
    for d in dirs:
        deg = d.get("degrees") or {}
        for k, v in deg.items():
            try:
                fv = float(v)
                parts.append(f"{k}{'+' if fv >= 0 else ''}{int(fv) if fv == int(fv) else fv}")
            except (TypeError, ValueError):
                parts.append(f"{k}{v}")
    ax_txt = ",".join(parts) if parts else "?"
    return f"movement {joint} ({ax_txt})"


def compact_tail_label(cfg: dict[str, Any]) -> str:
    """Short symmetric side hint (Gemini-style)."""
    labels: list[str] = []
    for step in cfg.get("movements") or []:
        t = step.get("type")
        p = step.get("parameters") or {}
        if t == "pose":
            pose = p.get("pose") or {}
            labels.append(
                f"pose({pose.get('dir', '?')},{pose.get('gripper_orientation', '?')})"
            )
        elif t == "movement":
            lbl = _movement_step_label(step)
            if lbl:
                labels.append(lbl)
        elif t == "path":
            shape = p.get("shape", "?")
            if shape == "line":
                labels.append(f"path line ({p.get('axis', '?')})")
            else:
                labels.append(f"path arc ({p.get('plane', '?')})")
    if not labels:
        return "(static)"
    if len(labels) == 1:
        return labels[0]
    return " → ".join(labels)


def build_gt_neg_configs(
    row: dict[str, Any],
    component: dict[str, Any],
    *,
    same_joint: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    pax = primary_axis_from_component(component) or "z"
    gt = build_config_from_gt_pose_and_component(row, component, state_tag="pair_gt_positive")
    neg = (
        apply_single_element_variant(gt, "axis", primary_axis=pax, same_joint=same_joint)
        if gt
        else None
    )
    if not gt or not neg:
        return None
    neg["state"] = "pair_gt_neg_axis"
    meta = dict(neg.get("neg_axis_meta") or {})
    gt_joint = meta.get("gt_joint") or meta.get("neg_joint")
    for gs in gt.get("movements") or []:
        if gs.get("type") == "movement":
            gt_joint = str((gs.get("parameters") or {}).get("joint") or gt_joint or "shoulder")
            break
    neg_joint = gt_joint
    for ns in neg.get("movements") or []:
        if ns.get("type") == "movement":
            neg_joint = str((ns.get("parameters") or {}).get("joint") or neg_joint)
            break
    meta.update(
        {
            "true_axis": meta.get("true_axis") or pax,
            "neg_axis": meta.get("neg_axis"),
            "gt_joint": gt_joint,
            "neg_joint": neg_joint,
            "same_joint": gt_joint == neg_joint,
        }
    )
    return gt, neg, meta


def build_pairwise_prompt(
    *,
    row: dict[str, Any],
    gt_cfg: dict[str, Any],
    neg_cfg: dict[str, Any],
    gt_side: str,
    left_summary: str,
    right_summary: str,
    template: str | None = None,
) -> str:
    tpl = template or PROMPT_TEMPLATE.read_text(encoding="utf-8")
    fixed = gt_cfg.get("gt_fixed_first_pose") or row.get("gt_fixed_first_pose") or {}
    return (
        tpl.replace("{{CUE}}", str(row.get("cue", "")))
        .replace("{{DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{FIXED_DIR}}", str(fixed.get("dir", "?")))
        .replace("{{FIXED_GRIPPER_ORIENTATION}}", str(fixed.get("gripper_orientation", "?")))
        .replace("{{LEFT_TAIL_SUMMARY}}", left_summary)
        .replace("{{RIGHT_TAIL_SUMMARY}}", right_summary)
    )


def _write_single_cfg(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps([cfg], indent=2, ensure_ascii=False), encoding="utf-8")


def _sample_for_cfg(cfg: dict[str, Any], cfg_path: Path, idx: int, cue: str, tag: str) -> dict[str, Any]:
    gfp = cfg.get("gt_fixed_first_pose") or {}
    pid = gfp.get("pose_id")
    return {
        "sample_id": testset_utils._safe_name(f"pilot90_pair_{idx}_{cue}_{tag}"),
        "testset": "iconic",
        "cue_idx": idx,
        "cue": cue,
        "config_path": str(cfg_path),
        "gif_path": str(cfg_path),
        "selected_pose_id": int(pid) if pid is not None else 0,
        "meta": {},
    }


def _panel_at_progress(bundle: dict[str, Any], progress: float) -> Image.Image:
    frames: list[np.ndarray] = bundle["frames"]
    nf = len(frames)
    if nf <= 0:
        raise RuntimeError("empty sim bundle")
    end_idx = int(round(progress * (nf - 1)))
    end_idx = max(0, min(end_idx, nf - 1))
    sub = frames[: end_idx + 1]
    stack_count = max(2, min(10, len(sub)))
    under = testset_utils._build_alpha_stack_from_numpy_frames(sub, stack_count=stack_count)
    layers = testset_utils._sim_trajectory_layers(
        bundle, bundle["cam_pos"], bundle["cam_rot"], bundle["fovy"], int(bundle["width"])
    )
    ee = (layers.get("ee") or [])[: end_idx + 1]
    layers_trunc = dict(layers)
    layers_trunc["ee"] = ee
    base = Image.fromarray(np.asarray(frames[end_idx], dtype=np.uint8)).convert("RGB")
    path_layer = testset_utils._image_with_trajectory_layers(base, layers_trunc)
    if under.size != path_layer.size:
        under = under.resize(path_layer.size, Image.Resampling.LANCZOS)
    return Image.blend(under, path_layer, 0.64).convert("RGB")


def _frames_to_mp4(
    frames: list[Image.Image],
    out_mp4: Path,
    *,
    fps: int = 10,
    scratch: Path | None = None,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    td_path: Path
    if scratch is not None:
        td_path = scratch / f"_ffmpeg_{out_mp4.stem}"
        if td_path.is_dir():
            shutil.rmtree(td_path, ignore_errors=True)
        td_path.mkdir(parents=True, exist_ok=True)
    else:
        td_path = Path(tempfile.mkdtemp(prefix="pair_mp4_"))
    try:
        for i, im in enumerate(frames):
            im.save(td_path / f"frame_{i:03d}.png")
        cmd = [
            ff,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(td_path / "frame_%03d.png"),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or "ffmpeg failed").strip())
    finally:
        if scratch is not None:
            shutil.rmtree(td_path, ignore_errors=True)
        else:
            shutil.rmtree(td_path, ignore_errors=True)


def build_pair_axis_mp4(
    *,
    idx: int,
    cue: str,
    row: dict[str, Any],
    component: dict[str, Any],
    out_mp4: Path,
    scratch: Path,
    gt_side: str | None = None,
    same_joint: bool = True,
) -> dict[str, Any] | None:
    built = build_gt_neg_configs(row, component, same_joint=same_joint)
    if not built:
        return None
    gt_cfg, neg_cfg, meta = built
    gt_side = (gt_side or pairwise_gt_side(idx, cue)).lower()
    if gt_side not in ("left", "right"):
        gt_side = "left"

    scratch.mkdir(parents=True, exist_ok=True)
    gt_path = scratch / f"{idx:03d}_{cue}_gt.json"
    neg_path = scratch / f"{idx:03d}_{cue}_neg.json"
    _write_single_cfg(gt_cfg, gt_path)
    _write_single_cfg(neg_cfg, neg_path)

    gt_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(gt_cfg, gt_path, idx, cue, "gt"),
        ROBOT,
        HZ,
        force=True,
        disk_cache=False,
    )
    neg_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(neg_cfg, neg_path, idx, cue, "neg"),
        ROBOT,
        HZ,
        force=True,
        disk_cache=False,
    )

    gt_label = compact_tail_label(gt_cfg)
    neg_label = compact_tail_label(neg_cfg)
    if gt_side == "left":
        left_cfg, right_cfg = gt_cfg, neg_cfg
        left_lbl, right_lbl = gt_label, neg_label
        left_kind, right_kind = "gt", "neg_axis"
    else:
        left_cfg, right_cfg = neg_cfg, gt_cfg
        left_lbl, right_lbl = neg_label, gt_label
        left_kind, right_kind = "neg_axis", "gt"

    out_frames: list[Image.Image] = []
    for k in range(N_OUT_FRAMES):
        progress = k / max(1, N_OUT_FRAMES - 1)
        if gt_side == "left":
            lp, rp = _panel_at_progress(gt_bundle, progress), _panel_at_progress(neg_bundle, progress)
        else:
            lp, rp = _panel_at_progress(neg_bundle, progress), _panel_at_progress(gt_bundle, progress)
        out_frames.append(
            _stitch_pair(lp, rp, cue=cue, left_label=left_lbl, right_label=right_lbl)
        )

    _frames_to_mp4(out_frames, out_mp4, scratch=scratch)
    if not out_mp4.is_file():
        return None

    prompt = build_pairwise_prompt(
        row=row,
        gt_cfg=gt_cfg,
        neg_cfg=neg_cfg,
        gt_side=gt_side,
        left_summary=left_lbl,
        right_summary=right_lbl,
    )
    rel_mp4 = str(out_mp4.relative_to(_REPO))
    start_pose = gt_cfg.get("gt_fixed_first_pose") or {}
    return {
        "idx": idx,
        "cue": cue,
        "start_pose_source": gt_cfg.get("start_pose_source"),
        "start_pose_dir": start_pose.get("dir"),
        "start_pose_gripper": start_pose.get("gripper_orientation"),
        "start_pose_id": start_pose.get("pose_id"),
        "gt_side": gt_side,
        "left": left_kind,
        "right": right_kind,
        "true_axis": meta.get("true_axis"),
        "neg_axis": meta.get("neg_axis"),
        "gt_joint": meta.get("gt_joint"),
        "neg_joint": meta.get("neg_joint"),
        "same_joint": meta.get("same_joint"),
        "left_tail_summary": left_lbl,
        "right_tail_summary": right_lbl,
        "gt_tail_summary": gt_label,
        "neg_tail_summary": neg_label,
        "neg_axis_meta": meta,
        "pair_mp4": rel_mp4,
        "prompt": prompt,
    }


def write_pairwise_specs(
    entries: list[dict[str, Any]],
    out_dir: Path | None = None,
    *,
    specs_name: str | None = None,
) -> Path:
    od = out_dir or MOTION_PAIRWISE_DIR
    od.mkdir(parents=True, exist_ok=True)
    out = od / (specs_name or PAIRWISE_SPECS_NAME)
    payload = {
        "version": 5,
        "layout": "gt_vs_neg_component_from_generation_pose",
        "gt_side": "random_per_cue",
        "n": len(entries),
        "n_with_mp4": sum(1 for e in entries if e.get("pair_mp4")),
        "n_gt_left": sum(1 for e in entries if e.get("gt_side") == "left"),
        "n_gt_right": sum(1 for e in entries if e.get("gt_side") == "right"),
        "mp4": sorted(entries, key=lambda e: int(e["idx"])),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_idx_subset(path: Path | None) -> set[int] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    idxs = data.get("idxs") or data.get("indices") or []
    if not idxs and data.get("mp4"):
        idxs = [int(r["idx"]) for r in data["mp4"] if r.get("vlm_correct")]
    return {int(i) for i in idxs}


def score_pairwise_motion_gt() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score 88 pairwise-eligible configs vs movement component GT (task-7 metric)."""
    by_idx = _rows_by_idx()
    ann = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    correct: list[dict[str, Any]] = []
    wrong: list[dict[str, Any]] = []
    for spec in _pairwise_specs():
        idx = int(spec["idx"])
        cue = str(spec["cue"])
        row = by_idx[idx]
        entry = ann.get(idx) or {}
        comp = entry.get("component")
        if entry.get("always_correct") or (comp and comp.get("kind") == "any"):
            match, kind = True, "any"
        elif comp:
            tail = _tail_steps(row.get("movements") or [])
            match, kind = _tail_matches_component(tail, comp)
        else:
            match, kind = False, None
        rec = {
            "idx": idx,
            "cue": cue,
            "component_match": match,
            "matched_kind": kind,
            "annotation_raw": entry.get("annotation_raw"),
        }
        (correct if match else wrong).append(rec)
    return correct, wrong


def write_motion_gt_correct_subset(*, out_path: Path | None = None) -> Path:
    correct, wrong = score_pairwise_motion_gt()
    od = out_path or (MOTION_PAIRWISE_DIR / "motion_gt_correct_subset.json")
    payload = {
        "source": str(MOTION_CFG.relative_to(_REPO)),
        "filter": "motion_component_gt_match",
        "groundtruth_json": "data/results/verify/pilot40_motion_component_gt.json",
        "n_pairwise": len(correct) + len(wrong),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "accuracy": len(correct) / (len(correct) + len(wrong)) if (correct or wrong) else None,
        "idxs": [int(r["idx"]) for r in correct],
        "cues": [str(r["cue"]) for r in correct],
        "entries": correct,
        "wrong_entries": wrong,
    }
    od.parent.mkdir(parents=True, exist_ok=True)
    od.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return od


def write_gemini_correct_subset(
    exp10_json: Path,
    *,
    out_path: Path | None = None,
) -> Path:
    data = json.loads(exp10_json.read_text(encoding="utf-8"))
    correct = [r for r in data.get("mp4", []) if r.get("vlm_correct")]
    od = out_path or (MOTION_PAIRWISE_DIR / "gemini_correct_subset.json")
    payload = {
        "source": str(exp10_json.relative_to(_REPO)) if exp10_json.is_relative_to(_REPO) else str(exp10_json),
        "filter": "vlm_correct",
        "n_scored": len(data.get("mp4", [])),
        "n_correct": len(correct),
        "idxs": [int(r["idx"]) for r in correct],
        "cues": [str(r["cue"]) for r in correct],
        "entries": [{"idx": int(r["idx"]), "cue": str(r["cue"])} for r in correct],
    }
    od.parent.mkdir(parents=True, exist_ok=True)
    od.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return od


def prepare_pilot90_pairwise_mp4s(
    *,
    out_dir: Path | None = None,
    scratch: Path | None = None,
    force: bool = False,
    limit: int = 0,
    idx_subset: set[int] | None = None,
    same_joint: bool = True,
    write_specs: bool = True,
) -> tuple[int, list[str]]:
    jpath = pose_jsonl(_REPO)
    os.environ["MOTION_POSE_JSONL"] = str(jpath)
    od = out_dir or MOTION_PAIRWISE_DIR
    sc = scratch or (od / "_build_scratch")
    od.mkdir(parents=True, exist_ok=True)
    ready = 0
    failures: list[str] = []
    spec_entries: list[dict[str, Any]] = []
    specs = _pairwise_specs()
    if idx_subset:
        specs = [s for s in specs if int(s["idx"]) in idx_subset]
    if limit:
        specs = specs[:limit]

    for spec in specs:
        idx = int(spec["idx"])
        cue = str(spec["cue"])
        out_mp4 = od / f"{idx:03d}_{cue}_pair_axis.mp4"
        side = pairwise_gt_side(idx, cue)

        if out_mp4.is_file() and not force:
            # Reload spec from sidecar if present
            sidecar = od / f"{idx:03d}_{cue}_pair_spec.json"
            if sidecar.is_file():
                spec_entries.append(json.loads(sidecar.read_text(encoding="utf-8")))
            else:
                spec_entries.append(
                    {
                        "idx": idx,
                        "cue": cue,
                        "gt_side": side,
                        "pair_mp4": str(out_mp4.relative_to(_REPO)),
                    }
                )
            ready += 1
            continue

        print(f"[pairwise] c{idx} {cue} gt_side={side} ...", flush=True)
        try:
            entry = build_pair_axis_mp4(
                idx=idx,
                cue=cue,
                row=spec["row"],
                component=spec["component"],
                out_mp4=out_mp4,
                scratch=sc,
                gt_side=side,
                same_joint=same_joint,
            )
            if entry:
                ready += 1
                spec_entries.append(entry)
                (od / f"{idx:03d}_{cue}_pair_spec.json").write_text(
                    json.dumps(entry, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            else:
                failures.append(f"{cue}: build returned false")
        except Exception as e:
            failures.append(f"{cue}: {e}")
            print(f"[pairwise fail] {cue}: {e}", flush=True)

    if write_specs and spec_entries:
        path = write_pairwise_specs(spec_entries, od)
        print(f"[specs] wrote {len(spec_entries)} -> {path}", flush=True)

    return ready, failures
