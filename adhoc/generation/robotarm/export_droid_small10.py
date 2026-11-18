#!/usr/bin/env python3
"""Dump 10 small-amplitude paper-30 cues, human + LLM, at DROID 15 Hz.

    python adhoc/generation/robotarm/export_droid_small10.py
"""
from __future__ import annotations

import html as html_mod
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
JSONL = _REPO / "data/seed/_remainder/closest_poses_results.jsonl"
GT_JSON = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
LLM_JSON = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
OUT = _PAPER / "droid_replay" / "small10"

# Compact EE / joint gestures on *both* GT and Gemini codes.
# Skipped large paths (r≥0.12, dist≥0.2, pose leaps like front→back).
SMALL10: list[tuple[int, str]] = [
    (1, "nod_yes"),
    (2, "shake_head_no"),
    (4, "gaze_shift_avoid"),
    (6, "tilt_head_curious"),
    (8, "rub_eye_tired"),
    (12, "stop_palm_out"),
    (15, "highfive_invite"),
    (18, "air_quotes"),
    (25, "facepalm"),
    (29, "depict_small_object"),
]


def _load_single_arm(path: Path, idx: int, cue: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data:
        plat = (entry.get("platforms") or {}).get("single_arm") or {}
        if int(entry.get("idx") or plat.get("idx") or 0) != idx:
            continue
        if str(entry.get("cue") or plat.get("cue") or "") != cue:
            continue
        if not plat.get("movements"):
            raise ValueError(f"empty movements: {path.name} {idx} {cue}")
        return {
            "idx": idx,
            "cue": cue,
            "category": str(entry.get("category") or ""),
            "description": plat.get("description") or "",
            "movements": plat["movements"],
            "parse_ok": bool(plat.get("parse_ok", True)),
            "source_code": plat.get("source_code") or "",
            "platform": "single_arm",
            "domain": "manipulator",
            "notes": plat.get("notes") or [],
        }
    raise ValueError(f"not found: {path.name} {idx} {cue}")


def _import_generate():
    robotarm = _HERE
    google_legacy = _REPO / "adhoc/generation/tiago/legacy"
    sys.path = [x for x in sys.path if Path(x).resolve() != google_legacy.resolve()]
    if str(robotarm) in sys.path:
        sys.path.remove(str(robotarm))
    sys.path.insert(0, str(robotarm))
    for mod in list(sys.modules):
        if mod == "legacy" or mod.startswith("legacy."):
            del sys.modules[mod]
    from legacy.motion_generation_core import generate  # noqa: WPS433

    return generate


def _traj_stats(npz_path: Path) -> dict[str, Any]:
    z = np.load(npz_path)
    q = np.asarray(z["joints_rad"], dtype=np.float64)
    ee = np.asarray(z["ee_xyz"], dtype=np.float64)
    hz = float(np.asarray(z["hz"]).reshape(-1)[0])
    dt = 1.0 / max(hz, 1e-9)
    dq = np.diff(q, axis=0)
    jspan = q.max(0) - q.min(0)
    ee_d = np.diff(ee, axis=0) if len(ee) > 1 else np.zeros((0, 3))
    ee_span = ee.max(0) - ee.min(0) if len(ee) else np.zeros(3)
    ee_speed = np.linalg.norm(ee_d, axis=1) / dt if len(ee_d) else np.array([0.0])
    active = int(np.argmax(jspan)) if q.size else 0
    return {
        "hz": hz,
        "n_frames": int(q.shape[0]),
        "duration_s": float(q.shape[0] / max(hz, 1e-9)),
        "play_duration_s": float(q.shape[0] / max(hz * 0.5, 1e-9)),
        "max_qdot_deg_s": float(np.degrees(np.max(np.abs(dq)) / dt)) if len(dq) else 0.0,
        "max_joint_span_deg": float(np.degrees(jspan.max())) if q.size else 0.0,
        "active_joint": f"q{active}",
        "ee_span_cm": (ee_span * 100.0).tolist(),
        "max_ee_speed_m_s": float(ee_speed.max()) if len(ee_speed) else 0.0,
        "ee_path_cm": float(np.sum(np.linalg.norm(ee_d, axis=1)) * 100.0) if len(ee_d) else 0.0,
    }


def _dump_one(generate, row: dict[str, Any], src: str, force: bool) -> dict[str, Any]:
    stem = f"{row['idx']:02d}_{row['cue']}"
    dest = OUT / stem / src
    dest.mkdir(parents=True, exist_ok=True)
    npz = dest / f"{stem}_{src}.npz"
    rec: dict[str, Any] = {
        "idx": row["idx"],
        "cue": row["cue"],
        "src": src,
        "code": row.get("source_code") or "",
        "npz": str(npz),
        "gif": str(dest / "preview.gif"),
        "error": "",
    }
    if npz.is_file() and not force:
        rec["stats"] = _traj_stats(npz)
        return rec
    scratch = dest / "scratch"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    tmp_cfg = scratch / "cfg.json"
    payload = {k: v for k, v in row.items() if not str(k).startswith("_")}
    tmp_cfg.write_text(json.dumps([payload], indent=2), encoding="utf-8")
    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    os.environ["MANIPULATOR_CUBE_CLIP"] = "1"
    os.environ["MOTION_FRAME_SCALE"] = "1.0"
    traj_stem = str(dest / f"{stem}_{src}")
    try:
        generate(
            robot="FR3",
            cue=str(row["cue"]),
            cue_idx=int(row["idx"]),
            pose_index=None,
            jsonl_path=str(JSONL),
            config_path=str(tmp_cfg),
            output_dir=str(scratch),
            hz=15,
            top_k=1,
            ignore_xyz=False,
            overlay_progress_bar=False,
            record_joints=True,
            traj_out_path=traj_stem,
        )
        gifs = sorted(scratch.rglob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if gifs:
            shutil.copy2(gifs[0], dest / "preview.gif")
        rec["stats"] = _traj_stats(npz)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)
    return rec


def _cell(label: str, rec: dict[str, Any], html_path: Path) -> str:
    gif = Path(rec["gif"])
    if gif.is_file() and not rec.get("error"):
        rel = os.path.relpath(gif, html_path.parent)
        bust = int(gif.stat().st_mtime)
        media = f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
    else:
        media = f'<p class="miss">FAIL<br>{html_mod.escape(rec.get("error") or "missing")}</p>'
    st = rec.get("stats") or {}
    if st:
        span = st.get("ee_span_cm") or [0, 0, 0]
        stats_html = (
            f'<ul class="stats">'
            f'<li>dump <strong>{st["duration_s"]:.2f}s</strong> @ 15 Hz '
            f'({st["n_frames"]} frames) · replay --slow 0.5 → {st["play_duration_s"]:.2f}s</li>'
            f'<li>peak joint <strong>{st["max_qdot_deg_s"]:.0f} °/s</strong> · '
            f'{st["active_joint"]} span <strong>{st["max_joint_span_deg"]:.1f}°</strong></li>'
            f'<li>EE box Δxyz {span[0]:.1f} / {span[1]:.1f} / {span[2]:.1f} cm · '
            f'path {st["ee_path_cm"]:.1f} cm · peak {st["max_ee_speed_m_s"]:.2f} m/s</li>'
            f"</ul>"
        )
    else:
        stats_html = ""
    return (
        f'<div class="col"><h3>{html_mod.escape(label)}</h3>{media}'
        f'<pre>{html_mod.escape(rec.get("code") or "")}</pre>{stats_html}</div>'
    )


def write_html(pairs: list[dict[str, Any]], out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(
        1
        for p in pairs
        for side in ("human", "llm")
        if p[side].get("stats") and not p[side].get("error")
    )
    sections = []
    for p in pairs:
        sections.append(
            f'<section><h2>{p["idx"]:02d}. {html_mod.escape(p["cue"])}</h2>'
            f'<div class="grid">'
            f'{_cell("human-authored", p["human"], out_html)}'
            f'{_cell("LLM-authored (Gemini 2.5 Pro)", p["llm"], out_html)}'
            f"</div></section>"
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DROID small-10 · human vs LLM</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6;max-width:1100px}}
section{{margin:16px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.col{{background:#1a1d24;border-radius:10px;padding:8px}}
.col h3{{margin:0 0 6px;font-size:.85rem}}
h2{{font-size:1rem;margin:0 0 8px}}
img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:6px;border-radius:6px}}
.miss{{color:#f28b82;font-size:.8rem}}
.stats{{font-size:.78rem;color:#c4c7cc;padding-left:18px;margin:6px 0 0}}
@media (max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<h1>DROID small-10 · FR3 Arm v2.0 dump @ 15 Hz</h1>
<p class="meta">{n_ok}/20 dumps · EE fitted into ±15 cm cube (center z+40 cm) · DROID joint_position · replay --slow 0.5</p>
<p class="meta">{html_mod.escape(datetime.now().isoformat(timespec="seconds"))}</p>
{''.join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def _parse_only() -> set[int] | None:
    if "--only" not in sys.argv:
        return None
    i = sys.argv.index("--only")
    return {int(x) for x in sys.argv[i + 1].split(",")}


def rebuild_gallery(index: dict | None = None) -> Path:
    """Rewrite gallery.html for all SMALL10 dumps using current preview.gif mtimes."""
    codes: dict[tuple[int, str], str] = {}
    if index:
        codes.update(index)
    idx_path = OUT / "index.json"
    if idx_path.is_file():
        for p in json.loads(idx_path.read_text()):
            for src in ("human", "llm"):
                codes[(p["idx"], src)] = (p.get(src) or {}).get("code") or codes.get((p["idx"], src), "")
    pairs: list[dict[str, Any]] = []
    for idx, cue in SMALL10:
        pair: dict[str, Any] = {"idx": idx, "cue": cue}
        for src in ("human", "llm"):
            stem = f"{idx:02d}_{cue}"
            dest = OUT / stem / src
            npz = dest / f"{stem}_{src}.npz"
            gif = dest / "preview.gif"
            rec: dict[str, Any] = {
                "idx": idx,
                "cue": cue,
                "src": src,
                "code": codes.get((idx, src), ""),
                "npz": str(npz),
                "gif": str(gif),
                "error": "" if npz.is_file() else f"missing {npz}",
            }
            if npz.is_file():
                rec["stats"] = _traj_stats(npz)
            pair[src] = rec
        pairs.append(pair)
    html_path = OUT / "gallery.html"
    write_html(pairs, html_path)
    return html_path


def main() -> None:
    force = "--force" in sys.argv
    only = _parse_only()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_HERE / "replay_droid.py", OUT / "replay_droid.py")
    generate = _import_generate()
    pairs: list[dict[str, Any]] = []
    for idx, cue in SMALL10:
        if only is not None and idx not in only:
            continue
        pair: dict[str, Any] = {"idx": idx, "cue": cue}
        for src, jpath in (("human", GT_JSON), ("llm", LLM_JSON)):
            print(f"\n==== {idx:02d} {cue} {src} ====")
            row = _load_single_arm(jpath, idx, cue)
            pair[src] = _dump_one(generate, row, src, force)
        pairs.append(pair)
    if only is None:
        write_html(pairs, OUT / "gallery.html")
        (OUT / "index.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        html_path = OUT / "gallery.html"
    else:
        extra = {(p["idx"], src): (p.get(src) or {}).get("code") or "" for p in pairs for src in ("human", "llm")}
        html_path = rebuild_gallery(extra)
    print(f"gallery: {html_path}")


if __name__ == "__main__":
    main()
