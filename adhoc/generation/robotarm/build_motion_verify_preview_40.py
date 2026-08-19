#!/usr/bin/env python3
"""
Compare preview: positive motion from your component GT annotations; negatives flip one element each.

VLM verify (mp4 + alpha_frame_trajectory on LLM-generated motion) lives in
verify_motion_gt_fixed_pilot40_vlm.py — run that first for accuracy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _REPO / "adhoc" / "vlm_test", _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import testset_utils  # noqa: E402
from motion_gt_tail_builder import (  # noqa: E402
    apply_single_element_variant,
    build_config_from_gt_pose_and_component,
)
from score_pilot40_motion_gt_components import _build_annotation_map  # noqa: E402

BASE_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator/motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)
OUT_ROOT = _REPO / "data/results/render/manipulator/verify_preview_40"
CFG_DIR = OUT_ROOT / "configs"
MANIFEST = OUT_ROOT / "manifest_preview_40.json"
OUT_HTML = _REPO / "data/results/html/manipulator/motion_verify_preview_40.html"
ROBOT = "IIWA"
HZ = 10


def _rows() -> list[dict[str, Any]]:
    return sorted(json.loads(BASE_CFG.read_text(encoding="utf-8")), key=lambda r: int(r["idx"]))


def _write_gt_compare_configs(rows: list[dict[str, Any]]) -> dict[str, Path]:
    ann_by_idx = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    positive: list[dict[str, Any]] = []
    neg_axis: list[dict[str, Any]] = []
    neg_joint: list[dict[str, Any]] = []
    neg_direction: list[dict[str, Any]] = []

    for r in rows:
        idx = int(r["idx"])
        comp = (ann_by_idx.get(idx) or {}).get("component")
        pos = build_config_from_gt_pose_and_component(r, comp, state_tag="compare_gt_positive")
        if not pos:
            continue
        positive.append(pos)
        for kind, bucket in (
            ("axis", neg_axis),
            ("joint", neg_joint),
            ("direction", neg_direction),
        ):
            v = apply_single_element_variant(pos, kind)
            if v:
                v["state"] = f"compare_gt_neg_{kind}"
                bucket.append(v)

    CFG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "positive": CFG_DIR / "pilot40_gt_positive.json",
        "axis": CFG_DIR / "pilot40_gt_neg_axis.json",
        "joint": CFG_DIR / "pilot40_gt_neg_joint.json",
        "direction": CFG_DIR / "pilot40_gt_neg_direction.json",
    }
    paths["positive"].write_text(json.dumps(positive, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["axis"].write_text(json.dumps(neg_axis, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["joint"].write_text(json.dumps(neg_joint, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["direction"].write_text(
        json.dumps(neg_direction, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return paths


def _render_variant(cfg: Path, out_dir: Path, cue_indices: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "micromamba",
        "run",
        "-n",
        "robosuite",
        "python",
        str(_HERE / "render.py"),
        f"--config_json={cfg}",
        f"--output_dir={out_dir}",
        f"--sim_robot={ROBOT}",
        f"--hz={HZ}",
        f"--cue_indices={','.join(str(i) for i in cue_indices)}",
        "--skip_existing=True",
        "--do_html=False",
        "--auto_generate_if_missing=False",
    ]
    subprocess.run(cmd, check=False)


def _pick_latest_gif(dir_path: Path, cue: str) -> Path | None:
    if not dir_path.is_dir():
        return None
    cands = sorted(
        [p for p in dir_path.glob("*.gif") if f"_{ROBOT}_{cue}_" in p.name],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _gif_to_mp4(gif: Path, mp4: Path) -> bool:
    ff = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(gif), "-pix_fmt", "yuv420p", str(mp4)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def _alpha_frame_trajectory(cfg_path: Path, idx: int, cue: str, out_png: Path) -> bool:
    try:
        pose_id = 0
        for row in json.loads(cfg_path.read_text(encoding="utf-8")):
            if int(row["idx"]) == idx:
                pose_id = int((row.get("gt_fixed_first_pose") or {}).get("pose_id") or 0)
                break
        sample = {
            "sample_id": testset_utils._safe_name(f"compare_{idx}_{cue}"),
            "testset": "iconic",
            "cue_idx": idx,
            "cue": cue,
            "gif_path": str(cfg_path),
            "config_path": str(cfg_path),
            "selected_pose_id": pose_id,
            "meta": {},
        }
        img, _ = testset_utils.build_tile_figure_sim_trajectory_panel(
            sample,
            ROBOT,
            HZ,
            canonical="alpha_frame_trajectory",
            force=False,
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_png, format="PNG", optimize=True)
        return True
    except Exception as e:
        print(f"[alpha fail] {cue}: {e}", flush=True)
        return False


def _build_media(
    gif: Path | None,
    out_sub: Path,
    key: str,
    *,
    cfg_path: Path | None,
    idx: int,
    cue: str,
) -> dict[str, str | None]:
    if not gif or not gif.is_file():
        return {"gif": None, "mp4": None, "alpha": None}
    mp4 = out_sub / "mp4" / f"{key}.mp4"
    alpha = out_sub / "alpha_frame_trajectory" / f"{key}.png"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    alpha.parent.mkdir(parents=True, exist_ok=True)
    if not mp4.is_file():
        _gif_to_mp4(gif, mp4)
    if cfg_path and (not alpha.is_file()):
        _alpha_frame_trajectory(cfg_path, idx, cue, alpha)
    return {
        "gif": str(gif),
        "mp4": str(mp4) if mp4.is_file() else None,
        "alpha": str(alpha) if alpha.is_file() else None,
    }


def _rel(p: str | None) -> str:
    if not p:
        return ""
    return Path(os.path.relpath(p, OUT_HTML.parent)).as_posix()


def _write_html(rows: list[dict[str, Any]]) -> None:
    cards = []
    for r in rows:
        cards.append(
            f"""
<article class="card">
  <h2>{r['idx']}. {r['cue']}</h2>
  <p class="ann">GT annotation: <code>{r.get('annotation_raw','')}</code></p>
  <p class="desc">{r['description']}</p>
  <h3>Compare — GT-positive vs single-element negatives</h3>
  <p class="note">Positive = your component GT tail on GT-fixed first pose. Negatives flip axis / joint / direction only.</p>
  <div class="grid4">
    <div><h4>GT positive</h4>
      <video src="{_rel(r['positive']['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(r['positive']['alpha'])}" alt="alpha trajectory"/>
    </div>
    <div><h4>axis only</h4>
      <video src="{_rel(r['neg_axis']['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(r['neg_axis']['alpha'])}" alt="alpha trajectory"/>
    </div>
    <div><h4>joint only</h4>
      <video src="{_rel(r['neg_joint']['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(r['neg_joint']['alpha'])}" alt="alpha trajectory"/>
    </div>
    <div><h4>direction only</h4>
      <video src="{_rel(r['neg_direction']['mp4'])}" controls loop muted playsinline></video>
      <img src="{_rel(r['neg_direction']['alpha'])}" alt="alpha trajectory"/>
    </div>
  </div>
  <h3>VLM verify (LLM tail) — see separate page</h3>
  <p>Run <code>verify_motion_gt_fixed_pilot40_vlm.py</code> on generated motion; review at
  <code>pilot40_motion_vlm_verify_gt_fixed.html</code>.</p>
</article>
"""
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>Motion compare preview 40 (GT component)</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f5f7fb;padding:20px;margin:0}}
.card{{background:#fff;border:1px solid #dce1ea;border-radius:10px;padding:14px;margin-bottom:14px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
video,img{{width:100%;background:#fafafa;border:1px solid #ddd;border-radius:8px}}
.ann,.note{{font-size:13px;color:#555}}
</style></head><body>
<h1>Motion compare preview (40 cues)</h1>
<p>Positive motion follows <code>pilot40_motion_component_gt.json</code>. Negatives change one element.</p>
{''.join(cards)}
</body></html>"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    rows = _rows()
    ann_by_idx = {int(a["cue_idx"]): a for a in _build_annotation_map()}
    cue_indices = [int(r["idx"]) for r in rows]
    cfgs = _write_gt_compare_configs(rows)

    pos_dir = OUT_ROOT / "gt_positive"
    neg_axis_dir = OUT_ROOT / "gt_neg_axis"
    neg_joint_dir = OUT_ROOT / "gt_neg_joint"
    neg_dir_dir = OUT_ROOT / "gt_neg_direction"

    _render_variant(cfgs["positive"], pos_dir, cue_indices)
    _render_variant(cfgs["axis"], neg_axis_dir, cue_indices)
    _render_variant(cfgs["joint"], neg_joint_dir, cue_indices)
    _render_variant(cfgs["direction"], neg_dir_dir, cue_indices)

    media_root = OUT_ROOT / "derived_media_gt"
    out_rows = []
    for r in rows:
        idx = int(r["idx"])
        cue = r["cue"]
        ann = ann_by_idx.get(idx) or {}
        out_rows.append(
            {
                "idx": idx,
                "cue": cue,
                "description": r.get("description", ""),
                "annotation_raw": ann.get("annotation_raw", ""),
                "positive": _build_media(
                    _pick_latest_gif(pos_dir / ROBOT, cue),
                    media_root / "positive",
                    f"{idx:03d}_{cue}_pos",
                    cfg_path=cfgs["positive"],
                    idx=idx,
                    cue=cue,
                ),
                "neg_axis": _build_media(
                    _pick_latest_gif(neg_axis_dir / ROBOT, cue),
                    media_root / "neg_axis",
                    f"{idx:03d}_{cue}_axis",
                    cfg_path=cfgs["axis"],
                    idx=idx,
                    cue=cue,
                ),
                "neg_joint": _build_media(
                    _pick_latest_gif(neg_joint_dir / ROBOT, cue),
                    media_root / "neg_joint",
                    f"{idx:03d}_{cue}_joint",
                    cfg_path=cfgs["joint"],
                    idx=idx,
                    cue=cue,
                ),
                "neg_direction": _build_media(
                    _pick_latest_gif(neg_dir_dir / ROBOT, cue),
                    media_root / "neg_direction",
                    f"{idx:03d}_{cue}_dir",
                    cfg_path=cfgs["direction"],
                    idx=idx,
                    cue=cue,
                ),
            }
        )
        print(f"[media] {idx} {cue}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"rows": out_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(out_rows)
    print(f"wrote {MANIFEST}")
    print(f"wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
