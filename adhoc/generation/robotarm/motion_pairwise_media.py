"""Build GT vs neg-axis side-by-side pairwise MP4s for pilot-90 step 10."""
from __future__ import annotations

import json
import os
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
from pilot90_experiment_suite import MOTION_PAIRWISE_DIR, MOTION_CFG  # noqa: E402
from score_pilot40_motion_gt_components import _build_annotation_map  # noqa: E402
from verify_pose_pairwise_12_gemini import _stitch_pair  # noqa: E402
from verify_pose_tiles_gemini import _movement_summary  # noqa: E402

ROBOT = "IIWA"
HZ = 10
N_OUT_FRAMES = 9


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


def _frames_to_mp4(frames: list[Image.Image], out_mp4: Path, *, fps: int = 10) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg not found on PATH")
    with tempfile.TemporaryDirectory(prefix="pair_mp4_") as td:
        td_path = Path(td)
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


def build_pair_axis_mp4(
    *,
    idx: int,
    cue: str,
    row: dict[str, Any],
    component: dict[str, Any],
    out_mp4: Path,
    scratch: Path,
) -> bool:
    pos = build_config_from_gt_pose_and_component(row, component, state_tag="pair_gt_positive")
    if not pos:
        return False
    pax = primary_axis_from_component(component)
    neg = apply_single_element_variant(pos, "axis", primary_axis=pax)
    if not neg:
        return False
    neg["state"] = "pair_gt_neg_axis"

    scratch.mkdir(parents=True, exist_ok=True)
    gt_cfg = scratch / f"{idx:03d}_{cue}_gt.json"
    neg_cfg = scratch / f"{idx:03d}_{cue}_neg.json"
    _write_single_cfg(pos, gt_cfg)
    _write_single_cfg(neg, neg_cfg)

    gt_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(pos, gt_cfg, idx, cue, "gt"), ROBOT, HZ, force=True
    )
    neg_bundle = testset_utils.get_sim_bundle(
        _sample_for_cfg(neg, neg_cfg, idx, cue, "neg"), ROBOT, HZ, force=True
    )

    left_label = f"GT — {_movement_summary(pos)[:80]}"
    right_label = "neg axis — wrong-axis control"
    out_frames: list[Image.Image] = []
    for k in range(N_OUT_FRAMES):
        progress = k / max(1, N_OUT_FRAMES - 1)
        left = _panel_at_progress(gt_bundle, progress)
        right = _panel_at_progress(neg_bundle, progress)
        out_frames.append(
            _stitch_pair(
                left,
                right,
                cue=cue,
                left_label=left_label,
                right_label=right_label,
            )
        )

    _frames_to_mp4(out_frames, out_mp4)
    return out_mp4.is_file()


def prepare_pilot90_pairwise_mp4s(
    *,
    out_dir: Path | None = None,
    scratch: Path | None = None,
    force: bool = False,
    limit: int = 0,
) -> tuple[int, list[str]]:
    jpath = pose_jsonl(_REPO)
    os.environ["MOTION_POSE_JSONL"] = str(jpath)
    od = out_dir or MOTION_PAIRWISE_DIR
    sc = scratch or (od / "_build_scratch")
    od.mkdir(parents=True, exist_ok=True)
    ready = 0
    failures: list[str] = []
    specs = _pairwise_specs()
    if limit:
        specs = specs[:limit]

    for spec in specs:
        idx = int(spec["idx"])
        cue = str(spec["cue"])
        out_mp4 = od / f"{idx:03d}_{cue}_pair_axis.mp4"
        if out_mp4.is_file() and not force:
            ready += 1
            continue
        print(f"[pairwise] c{idx} {cue} ...", flush=True)
        try:
            ok = build_pair_axis_mp4(
                idx=idx,
                cue=cue,
                row=spec["row"],
                component=spec["component"],
                out_mp4=out_mp4,
                scratch=sc,
            )
            if ok:
                ready += 1
            else:
                failures.append(f"{cue}: build returned false")
        except Exception as e:
            failures.append(f"{cue}: {e}")
            print(f"[pairwise fail] {cue}: {e}", flush=True)

    return ready, failures
