#!/usr/bin/env python3
"""Build humanlike_partial90_gallery.html with cue-spec Orient/Loc/Move badges."""
from __future__ import annotations

import html as html_mod
import json
import sys
from pathlib import Path

REPO = Path("/Users/sb/Downloads/workspace/dev_robosuite")
sys.path[:0] = [str(REPO), str(REPO / "adhoc/generation/robotarm")]
from run_embodiment_exp0_gemini import gif_path_for  # noqa: E402

PAPER_CFG = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
PAPER_RENDER = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
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
SCORE_JSON = REPO / "data/results/evaluation/humanlike_cuespec_scores.json"
OUT_HTML = REPO / "papers/icra2027_motion_abstraction/humanlike_partial90_gallery.html"
PLATS = ("humanoid", "tiago")


def _score_index(data: dict) -> dict[tuple[str, str, str], dict]:
    out = {}
    for row in data.get("rows") or []:
        out[(row["source"], row["platform"], row["cue"])] = row
    return out


def _badge(attr: str, block: dict | None) -> str:
    if not block or not block.get("applicable"):
        return f"<span class='b na'>({attr})</span>"
    ok = block.get("ok")
    cls = "ok" if ok else "bad"
    mark = "✓" if ok else "✗"
    return f"<span class='b {cls}'>({attr}{mark})</span>"


def _cells(configs, render_root, source):
    rows = []
    for entry in sorted(configs, key=lambda e: int(e.get("idx") or 0)):
        for plat in PLATS:
            cfg = (entry.get("platforms") or {}).get(plat) or {}
            if not (cfg.get("movements") or cfg.get("source_code")):
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


def main() -> None:
    scores = _score_index(json.loads(SCORE_JSON.read_text())) if SCORE_JSON.is_file() else {}
    paper = json.loads(PAPER_CFG.read_text())
    pilot = json.loads(PILOT_CFG.read_text())
    rows = _cells(paper, PAPER_RENDER, "paper30") + _cells(pilot, PILOT_RENDER, "pilot75")

    # summary rates from score file
    summary_bits = []
    if SCORE_JSON.is_file():
        raw = json.loads(SCORE_JSON.read_text())
        for source, plats in (raw.get("by_source") or {}).items():
            for plat, block in plats.items():
                o, l, m = block["orientation"], block["location"], block["movement"]

                def fr(b):
                    return f"{b['ok']}/{b['n']}" if b["n"] else "—"

                summary_bits.append(
                    f"{source}/{plat}: O {fr(o)} · L {fr(l)} · M {fr(m)}"
                )

    by_plat = {p: [] for p in PLATS}
    for r in rows:
        by_plat[r["plat"]].append(r)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Human-like partial renders + cue-spec scores</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}",
        "h1,h2{font-weight:600} .meta{color:#aaa;margin-bottom:12px;line-height:1.5}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
        ".card{background:#1c1c1c;border:1px solid #333;border-radius:10px;overflow:hidden;position:relative}",
        ".card img{width:100%;aspect-ratio:1;object-fit:cover;background:#000;display:block}",
        ".card .miss{width:100%;aspect-ratio:1;display:flex;align-items:center;justify-content:center;background:#2a2a2a;color:#888;font-size:13px}",
        ".card .cap{padding:8px 10px;font-size:12px;line-height:1.35}",
        ".tag{display:inline-block;padding:1px 6px;border-radius:999px;font-size:10px;background:#333;margin-right:4px}",
        ".tag.paper{background:#1f4d3a} .tag.pilot{background:#3a2f1f}",
        ".code{color:#9ad;font-family:ui-monospace,monospace;font-size:10px;word-break:break-all}",
        ".badges{position:absolute;top:6px;right:6px;display:flex;flex-direction:column;gap:3px;align-items:flex-end}",
        ".b{font-size:10px;font-weight:700;padding:2px 6px;border-radius:6px;color:#fff}",
        ".b.ok{background:rgba(34,140,70,.92)} .b.bad{background:rgba(200,55,55,.92)}",
        ".b.na{background:rgba(90,90,90,.85)}",
        "</style></head><body>",
        "<h1>Human-like robots — renders + cue-spec auto scores</h1>",
        "<div class='meta'>Discrete GT from canvas requiredComponents "
        "(Orient=dir/grip/head…, Loc=x/y/z bands, Move=axis/sign keys). "
        + " · ".join(summary_bits)
        + "</div>",
    ]

    for plat in PLATS:
        subset = by_plat[plat]
        ok_gif = sum(1 for r in subset if r["gif"])
        parts.append(f"<h2>{plat} ({ok_gif}/{len(subset)} GIFs)</h2><div class='grid'>")
        for r in subset:
            tag = "paper" if r["source"] == "paper30" else "pilot"
            sc = scores.get((r["source"], r["plat"], r["cue"]))
            badges = (
                "<div class='badges'>"
                + _badge("Orient", sc.get("orientation") if sc else None)
                + _badge("Loc", sc.get("location") if sc else None)
                + _badge("Move", sc.get("movement") if sc else None)
                + "</div>"
            )
            if r["gif"]:
                media = f"<img src='{html_mod.escape(r['gif'].resolve().as_uri())}' loading='lazy'>"
            else:
                media = "<div class='miss'>not rendered</div>"
            parts.append(
                "<div class='card'>"
                + badges
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
    print("wrote", OUT_HTML, "cards", len(rows), "scores", len(scores))


if __name__ == "__main__":
    main()
