#!/usr/bin/env python3
"""Render paper-30 single-arm Motion Codes on Franka (Panda).

Human-authored GT vs Gemini 2.5 Pro, side by side.

    python adhoc/generation/robotarm/run_franka_paper30.py
    python adhoc/generation/robotarm/run_franka_paper30.py --force
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
OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_franka_paper30"
)
HTML_COPY = _PAPER / "franka_paper30" / "gallery.html"


def _load_single_arm(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for entry in data:
        plat = (entry.get("platforms") or {}).get("single_arm") or {}
        if not plat.get("movements"):
            continue
        rows.append(
            {
                "idx": int(entry.get("idx") or plat.get("idx") or 0),
                "cue": str(entry.get("cue") or plat.get("cue") or ""),
                "category": str(entry.get("category") or ""),
                "description": plat.get("description") or "",
                "movements": plat["movements"],
                "parse_ok": bool(plat.get("parse_ok", True)),
                "source_code": plat.get("source_code") or "",
                "platform": "single_arm",
                "domain": "manipulator",
                "notes": plat.get("notes") or [],
            }
        )
    rows.sort(key=lambda r: r["idx"])
    return rows


def _render_panda(row: dict[str, Any], gif_path: Path, *, force: bool) -> str:
    if gif_path.is_file() and not force:
        return ""
    robotarm = _HERE
    google_legacy = _REPO / "adhoc/generation/tiago/legacy"
    sys.path = [p for p in sys.path if Path(p).resolve() != google_legacy.resolve()]
    if str(robotarm) in sys.path:
        sys.path.remove(str(robotarm))
    sys.path.insert(0, str(robotarm))
    for mod in list(sys.modules):
        if mod == "legacy" or mod.startswith("legacy."):
            del sys.modules[mod]
    from legacy.motion_generation_core import generate  # noqa: WPS433

    os.environ.setdefault("MANIPULATOR_POSE_XYZ_MODE", "trim")
    os.environ["MOTION_FRAME_SCALE"] = "1.0"
    scratch = OUT / "scratch" / f"{row['idx']:02d}_{row['cue']}_{row['_src']}"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    tmp_cfg = scratch / "cfg.json"
    payload = {k: v for k, v in row.items() if not str(k).startswith("_")}
    tmp_cfg.write_text(json.dumps([payload], indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        generate(
            robot="Panda",
            cue=str(row["cue"]),
            cue_idx=int(row["idx"]),
            pose_index=None,
            jsonl_path=str(JSONL),
            config_path=str(tmp_cfg),
            output_dir=str(scratch),
            hz=12,
            top_k=1,
            ignore_xyz=False,
            overlay_progress_bar=False,
        )
        robot_dir = scratch / "Panda"
        gifs = sorted(robot_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not gifs:
            return "no gif written"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(gifs[0], gif_path)
        return ""
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _cell(label: str, rec: dict[str, Any], out_html: Path) -> str:
    gif = Path(rec["gif"]) if rec.get("gif") else None
    if gif is not None and gif.is_file():
        rel = os.path.relpath(gif, out_html.parent)
        bust = int(gif.stat().st_mtime)
        media = f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
    else:
        media = f'<p class="miss">FAIL<br>{html_mod.escape(rec.get("error") or "missing")}</p>'
    return (
        f'<div class="col"><h3>{html_mod.escape(label)}</h3>{media}'
        f'<pre>{html_mod.escape(rec.get("code") or "")}</pre>'
        f'<p class="notes">{html_mod.escape(rec.get("error") or rec.get("notes") or "")}</p></div>'
    )


def write_html(pairs: list[dict[str, Any]], out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    n_ok = sum(
        1
        for p in pairs
        for side in ("human", "llm")
        if p[side].get("gif") and not p[side].get("error")
    )
    n_all = len(pairs) * 2
    sections = []
    for p in pairs:
        sections.append(
            f'<section><h2>{p["idx"]:02d}. {html_mod.escape(p["cue"])} '
            f'<span class="tag">{html_mod.escape(p["category"])}</span></h2>'
            f'<div class="grid">'
            f'{_cell("human-authored", p["human"], out_html)}'
            f'{_cell("LLM-authored (Gemini 2.5 Pro)", p["llm"], out_html)}'
            f"</div></section>"
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Franka paper-30 · human vs LLM</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6;max-width:1100px}}
section{{margin:16px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.col{{background:#1a1d24;border-radius:10px;padding:8px}}
.col h3{{margin:0 0 6px;font-size:.85rem}}
h2{{font-size:1rem;margin:0 0 8px}}
.tag{{font-size:.75rem;color:#8ab4f8;font-weight:500}}
img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:6px;border-radius:6px}}
.miss{{color:#f28b82;font-size:.8rem}}
.notes{{font-size:.75rem;color:#9aa0a6}}
@media (max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body>
<h1>Franka (Panda) · paper-30 Motion Codes</h1>
<p class="meta">{n_ok}/{n_all} GIFs · human-authored GT vs Gemini 2.5 Pro on robosuite <code>Panda</code>.
Same single-arm Motion Code as the IIWA paper-30 bank.</p>
<p class="meta">{html_mod.escape(datetime.now().isoformat(timespec="seconds"))}</p>
{"".join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    gt_rows = _load_single_arm(GT_JSON)
    llm_rows = {r["cue"]: r for r in _load_single_arm(LLM_JSON)}
    OUT.mkdir(parents=True, exist_ok=True)
    gif_dir = OUT / "gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    for human in gt_rows:
        llm = llm_rows.get(human["cue"])
        recs: dict[str, Any] = {}
        for src, row in (("human", human), ("llm", llm)):
            if row is None:
                recs[src] = {"gif": "", "error": "missing cue", "code": "", "notes": ""}
                continue
            row = dict(row)
            row["_src"] = src
            gif_path = gif_dir / f"{row['idx']:02d}_{row['cue']}_{src}.gif"
            print(f"[franka30] {src} {row['idx']:02d} {row['cue']} …", flush=True)
            err = _render_panda(row, gif_path, force=args.force)
            if err:
                print(f"[franka30] FAIL {src} {row['cue']}: {err}", flush=True)
            else:
                print(f"[franka30] ok {gif_path.name}", flush=True)
            recs[src] = {
                "gif": str(gif_path) if gif_path.is_file() else "",
                "error": err,
                "code": row.get("source_code") or "",
                "notes": " ".join(row.get("notes") or []),
            }
        pairs.append(
            {
                "idx": human["idx"],
                "cue": human["cue"],
                "category": human.get("category") or "",
                "human": recs["human"],
                "llm": recs["llm"],
            }
        )
    (OUT / "results.json").write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    html_out = OUT / "gallery.html"
    write_html(pairs, html_out)
    write_html(pairs, HTML_COPY)
    n_ok = sum(
        1
        for p in pairs
        for side in ("human", "llm")
        if p[side].get("gif") and not p[side].get("error")
    )
    print(f"done {n_ok}/{len(pairs)*2}\n{html_out}\n{HTML_COPY}", flush=True)


if __name__ == "__main__":
    main()
