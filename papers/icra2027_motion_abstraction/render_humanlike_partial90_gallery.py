#!/usr/bin/env python3
"""Render whatever human-like pilot75 Motion Codes exist, then build an HTML gallery.

Covers Unitree G1 / Tiago for paper30 (already rendered) + pilot75-so-far.
"""
from __future__ import annotations

import html as html_mod
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
ADHOC = REPO / "adhoc/generation/robotarm"
sys.path[:0] = [str(REPO), str(ADHOC)]

from run_embodiment_exp0_gemini import gif_path_for, render_pending_once  # noqa: E402

PAPER_CFG = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
PAPER_RENDER = (
    REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
)
PILOT_CFG = (
    REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
    / "embodiment_motion_code_all.json"
)
PILOT_RENDER = (
    REPO
    / "data/results/render"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike"
)
OUT_HTML = (
    REPO
    / "papers/icra2027_motion_abstraction"
    / "humanlike_partial90_gallery.html"
)
PLATS = ("humanoid", "tiago")


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _drain(configs: list[dict], render_root: Path, *, batch: int = 4) -> None:
    render_root.mkdir(parents=True, exist_ok=True)
    # reset skip set between drains
    if hasattr(render_pending_once, "_skip"):
        delattr(render_pending_once, "_skip")
    total = 0
    while True:
        n = render_pending_once(
            configs,
            render_root=render_root,
            platforms=list(PLATS),
            max_items=batch,
        )
        total += n
        if n == 0:
            break
        print(f"[drain] attempted +{n} (running total {total})", flush=True)
    print(f"[drain] done attempted={total}", flush=True)


def _cells(configs: list[dict], render_root: Path, *, source: str) -> list[dict]:
    rows = []
    for entry in sorted(configs, key=lambda e: int(e.get("idx") or 0)):
        for plat in PLATS:
            cfg = (entry.get("platforms") or {}).get(plat) or {}
            if not cfg.get("movements"):
                continue
            gif = gif_path_for(render_root, plat, int(entry["idx"]), str(entry["cue"]))
            rows.append(
                {
                    "source": source,
                    "plat": plat,
                    "idx": int(entry["idx"]),
                    "cue": str(entry["cue"]),
                    "gif": gif if gif.is_file() else None,
                    "code": str(cfg.get("source_code") or "")[:160],
                }
            )
    return rows


def _build_html(rows: list[dict]) -> None:
    by_plat: dict[str, list[dict]] = {p: [] for p in PLATS}
    for r in rows:
        by_plat[r["plat"]].append(r)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Human-like partial renders (paper30 + pilot75 so far)</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}",
        "h1,h2{font-weight:600} .meta{color:#aaa;margin-bottom:18px}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
        ".card{background:#1c1c1c;border:1px solid #333;border-radius:10px;overflow:hidden}",
        ".card img{width:100%;aspect-ratio:1;object-fit:cover;background:#000;display:block}",
        ".card .miss{width:100%;aspect-ratio:1;display:flex;align-items:center;justify-content:center;",
        "background:#2a2a2a;color:#888;font-size:13px}",
        ".card .cap{padding:8px 10px;font-size:12px;line-height:1.35}",
        ".tag{display:inline-block;padding:1px 6px;border-radius:999px;font-size:10px;",
        "background:#333;margin-right:4px}",
        ".tag.paper{background:#1f4d3a} .tag.pilot{background:#3a2f1f}",
        ".code{color:#9ad;font-family:ui-monospace,monospace;font-size:10px;word-break:break-all}",
        "</style></head><body>",
        "<h1>Human-like robots — rendered so far toward ~90 cues</h1>",
    ]
    n_ok = sum(1 for r in rows if r["gif"])
    n_paper = sum(1 for r in rows if r["source"] == "paper30")
    n_pilot = sum(1 for r in rows if r["source"] == "pilot75")
    parts.append(
        f"<div class='meta'>cells={len(rows)} · with GIF={n_ok} · "
        f"paper30 cells={n_paper} · pilot75 cells={n_pilot} "
        f"(humanoid codes in pilot75 ≈ 52/75; tiago ≈ 1/75)</div>"
    )

    for plat in PLATS:
        subset = by_plat[plat]
        ok = sum(1 for r in subset if r["gif"])
        parts.append(f"<h2>{plat} ({ok}/{len(subset)} GIFs)</h2><div class='grid'>")
        for r in subset:
            tag = "paper" if r["source"] == "paper30" else "pilot"
            if r["gif"]:
                # file:// absolute path for local open
                uri = r["gif"].resolve().as_uri()
                media = f"<img src='{html_mod.escape(uri)}' loading='lazy'>"
            else:
                media = "<div class='miss'>not rendered</div>"
            parts.append(
                "<div class='card'>"
                + media
                + "<div class='cap'>"
                + f"<span class='tag {tag}'>{r['source']}</span>"
                + f"<b>#{r['idx']}</b> {html_mod.escape(r['cue'])}"
                + f"<div class='code'>{html_mod.escape(r['code'])}</div>"
                + "</div></div>"
            )
        parts.append("</div>")

    parts.append("</body></html>")
    OUT_HTML.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT_HTML)


def main() -> None:
    pilot = _load(PILOT_CFG)
    print(f"pilot75 entries={len(pilot)} → render to {PILOT_RENDER}", flush=True)
    _drain(pilot, PILOT_RENDER, batch=4)

    paper = _load(PAPER_CFG)
    rows = _cells(paper, PAPER_RENDER, source="paper30") + _cells(
        pilot, PILOT_RENDER, source="pilot75"
    )
    _build_html(rows)


if __name__ == "__main__":
    main()
