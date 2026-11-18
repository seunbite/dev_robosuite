#!/usr/bin/env python3
"""Build the 50-cue × 5-morphology gallery used for paper inspection."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/html/embodiment_extension20_gemini.html"
HARD_NEGATIVE_TRIALS = (
    REPO / "papers/icra2027_motion_abstraction/humanlike90_4afc_eval/trials.json"
)

PLATFORMS = [
    ("humanoid", "Unitree G1"),
    ("tiago", "TIAGo"),
    ("single_arm", "Single Arm (IIWA)"),
    ("stretch", "Stretch"),
    ("roomba", "Roomba"),
]

OLD_LLM_JSON = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
NEW_LLM_JSON = REPO / "data/results/motion_configs/embodiment_extension20_gemini/embodiment_motion_code_all.json"
HUMANLIKE_GEMINI_JSON = REPO / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike/embodiment_motion_code_all.json"
HUMANLIKE_QWEN_JSON = REPO / "data/results/motion_configs/embodiment_motion_code_qwen_qwen32b_humanlike90/embodiment_motion_code_all.json"
TRANSFER_AUDIT = REPO / "papers/icra2027_motion_abstraction/transfer_outcomes_50.json"


def load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def by_id(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["idx"]): row for row in load(path)}


OLD_LLM = by_id(OLD_LLM_JSON)
NEW_LLM = by_id(NEW_LLM_JSON)
HUMANLIKE_GEMINI = {row["cue"]: row for row in load(HUMANLIKE_GEMINI_JSON)}
HUMANLIKE_QWEN = {row["cue"]: row for row in load(HUMANLIKE_QWEN_JSON)}
AUDIT = load(TRANSFER_AUDIT)
AUDIT_BY_CELL = {
    (str(row["cue"]), str(row["platform"])): row for row in AUDIT["details"]
}


def audit_summary_markup() -> str:
    labels = {"single_arm": "IIWA", "stretch": "Stretch", "roomba": "Roomba"}
    cards = []
    for platform in ("single_arm", "stretch", "roomba"):
        row = AUDIT["summary"][platform]
        s, c = row["strategy"], row["consequence"]
        cards.append(
            '<div class="audit-card">'
            f'<h3>{labels[platform]}</h3>'
            '<div class="audit-lines">'
            f'<span>Preserved <b>{s.get("Preserved", 0)}</b></span>'
            f'<span>Transferred <b>{s.get("Carrier-transferred", 0)}</b></span>'
            f'<span>Rematerialized <b>{s.get("Rematerialized", 0)}</b></span>'
            f'<span class="aug">Augmented <b>{c.get("Augmented", 0)}</b></span>'
            f'<span class="maint">Maintained <b>{c.get("Maintained", 0)}</b></span>'
            f'<span class="lost">Lost <b>{c.get("Lost", 0)}</b></span>'
            '</div></div>'
        )
    return (
        '<section class="audit-summary"><div><h2>Non-anthropomorphic structural audit</h2>'
        '<p>50 LLM-authored programs per robot. Strategy and evidence consequence are scored separately; '
        'the labels describe program structure, not human recognition.</p></div>'
        f'<div class="audit-cards">{"".join(cards)}</div></section>'
    )


def load_afc_trials() -> dict[str, dict[str, Any]]:
    trials = load(HARD_NEGATIVE_TRIALS)
    return {str(trial["cue"]): trial for trial in trials}


def afc_markup(trial: dict[str, Any]) -> str:
    answer = str(trial["answer_id"])
    pills = []
    for i, choice in enumerate(trial["choices"]):
        cue = str(choice["id"])
        label = str(choice.get("label") or cue.replace("_", " "))
        correct = cue == answer
        cls = "choice correct" if correct else "choice"
        answer_tag = '<span class="answer-tag">answer</span>' if correct else ""
        pills.append(
            f'<span class="{cls}" title="{html.escape(cue)}">'
            f'<b>{chr(65 + i)}</b>{html.escape(label)}{answer_tag}</span>'
        )
    return (
        '<div class="afc"><span class="afc-label">4AFC hard-negative options</span>'
        f'<div class="choices">{"".join(pills)}</div></div>'
    )


def source_for(idx: int, cue: str, platform: str) -> tuple[dict[str, Any], str, str, int]:
    """Return config row, model label, render experiment, and render-local ID."""
    if idx <= 30:
        return OLD_LLM[idx], "LLM · Gemini", "embodiment_motion_code_gemini_gemini-2.5-pro", idx
    if platform in {"humanoid", "tiago"}:
        gemini = HUMANLIKE_GEMINI.get(cue)
        if gemini and platform in (gemini.get("platforms") or {}):
            return (
                gemini,
                "LLM · Gemini",
                "embodiment_motion_code_gemini_gemini-2.5-pro_pilot75_humanoidlike",
                int(gemini["idx"]),
            )
        qwen = HUMANLIKE_QWEN[cue]
        return (
            qwen,
            "LLM · Qwen-32B",
            "embodiment_motion_code_qwen_qwen32b_humanlike90",
            int(qwen["idx"]),
        )
    return NEW_LLM[idx], "LLM · Gemini", "embodiment_extension20_gemini", idx


def gif_for(experiment: str, platform: str, idx: int, cue: str) -> Path:
    if platform == "single_arm":
        rel = Path("manipulator/gif/IIWA")
    else:
        rel = Path(platform) / "gif"
    return REPO / "data/results/render" / experiment / rel / f"emb_g{idx:02d}_{cue}.gif"


def cell(idx: int, cue: str, platform: str, label: str) -> tuple[str, bool]:
    row, authorship, experiment, render_idx = source_for(idx, cue, platform)
    cfg = (row.get("platforms") or {}).get(platform) or {}
    gif = gif_for(experiment, platform, render_idx, cue)
    exists = gif.is_file()
    if exists:
        rel = os.path.relpath(gif, OUT.parent)
        media = (
            f'<img src="{html.escape(rel)}?v={int(gif.stat().st_mtime)}" '
            f'loading="lazy" alt="{html.escape(cue)} on {html.escape(label)}">'
        )
    else:
        media = '<div class="missing">Missing render</div>'
    code = str(cfg.get("source_code") or "").strip() or "(source code unavailable)"
    movements = json.dumps(cfg.get("movements") or [], ensure_ascii=False, indent=2)
    audit = AUDIT_BY_CELL.get((cue, platform))
    audit_html = ""
    if audit:
        strategy = str(audit["strategy"])
        consequence = str(audit["consequence"])
        reason = f'{audit["strategy_reason"]}; {audit["consequence_reason"]}'
        audit_html = (
            '<div class="audit-badges">'
            f'<span class="score strategy">{html.escape(strategy)}</span>'
            f'<span class="score consequence {consequence.lower()}">{html.escape(consequence)}</span>'
            '</div>'
            f'<p class="audit-reason">{html.escape(reason)}</p>'
        )
    return (
        f'<article class="cell" data-platform="{platform}">'
        f'<header><h3>{html.escape(label)}</h3><span class="badge llm">{authorship}</span></header>'
        f'{media}{audit_html}<details><summary>Motion program</summary><pre>{html.escape(code)}</pre></details>'
        f'<details><summary>Parsed config</summary><pre>{html.escape(movements)}</pre></details></article>',
        exists,
    )


def main() -> None:
    cue_rows = [OLD_LLM[i] for i in sorted(OLD_LLM)] + [NEW_LLM[i] for i in sorted(NEW_LLM)]
    afc_trials = load_afc_trials()
    sections: list[str] = []
    gif_count = 0
    for number, row in enumerate(cue_rows, start=1):
        idx = int(row["idx"])
        cue = str(row["cue"])
        category = str(row.get("category") or "extension")
        trial = afc_trials.get(cue)
        if trial is None:
            raise KeyError(f"missing hard-negative trial for {cue}")
        afc = afc_markup(trial)
        option_search = " ".join(str(choice["id"]) for choice in trial["choices"])
        cells: list[str] = []
        for platform, label in PLATFORMS:
            rendered, exists = cell(idx, cue, platform, label)
            cells.append(rendered)
            gif_count += int(exists)
        sections.append(
            f'<section class="cue" data-search="{html.escape((cue + " " + category + " " + option_search).lower())}">'
            f'<div class="cue-title"><div><span class="seq">{number:02d}</span>'
            f'<h2>ID {idx:02d} · {html.escape(cue)}</h2></div><span class="category">{html.escape(category)}</span></div>{afc}'
            f'<div class="grid">{"".join(cells)}</div></section>'
        )

    updated = datetime.now().isoformat(timespec="seconds")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>50 Cues × 5 Morphologies</title>
<style>
:root{{--bg:#0d1016;--panel:#171b24;--cell:#10141c;--line:#303744;--text:#eef1f6;--muted:#aab2bf;--blue:#76a9ff;--gold:#efc46b}}
*{{box-sizing:border-box}} body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
.top{{position:sticky;top:0;z-index:4;background:rgba(13,16,22,.96);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:18px 22px 14px}}
.topline{{display:flex;gap:18px;align-items:end;justify-content:space-between}} h1{{font-size:1.45rem;margin:0 0 4px}} .meta{{color:var(--muted);font-size:.88rem}}
input{{width:min(420px,42vw);border:1px solid var(--line);background:#080b10;color:var(--text);border-radius:8px;padding:10px 12px;font-size:.92rem}}
.legend{{display:flex;gap:12px;margin-top:10px;color:var(--muted);font-size:.78rem}} .dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px}}
main{{padding:6px 14px 40px}} .cue{{margin:12px 0;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}}
.cue-title{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}} .cue-title>div{{display:flex;gap:10px;align-items:center}} .seq{{color:var(--muted);font-variant-numeric:tabular-nums}}
h2{{font-size:1rem;margin:0}} .category{{font-size:.72rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:4px 8px}}
.afc{{display:flex;gap:10px;align-items:center;margin:-2px 0 10px;padding:8px 10px;border:1px solid #2a3342;border-radius:8px;background:#111722}}
.afc-label{{flex:0 0 auto;color:var(--muted);font-size:.72rem;font-weight:650}} .choices{{display:flex;gap:6px;flex-wrap:wrap}}
.choice{{display:inline-flex;gap:6px;align-items:center;padding:4px 7px;border:1px solid #354052;border-radius:6px;background:#0b1018;color:#cbd2dc;font-size:.72rem}}
.choice b{{color:#8da2bd}} .choice.correct{{border-color:#4778b5;background:#13253c;color:#9bc4ff}} .choice.correct b{{color:#9bc4ff}} .answer-tag{{font-size:.58rem;text-transform:uppercase;letter-spacing:.04em;color:#0d1016;background:var(--blue);border-radius:999px;padding:2px 5px}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}} .cell{{min-width:0;background:var(--cell);border-radius:9px;padding:8px;border:1px solid #222936}}
.cell header{{display:flex;justify-content:space-between;gap:6px;align-items:center;margin-bottom:7px}} h3{{font-size:.78rem;margin:0}} .badge{{font-size:.62rem;border-radius:999px;padding:3px 6px;white-space:nowrap}}
.llm{{color:var(--blue);background:#182d4b}} img,.missing{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#05070a;display:block}}
.missing{{display:grid;place-items:center;color:#ff8e8e}} details{{margin-top:6px}} summary{{cursor:pointer;color:var(--muted);font-size:.7rem}} pre{{font:10px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;max-height:150px;overflow:auto;background:#070a0f;padding:7px;border-radius:5px;color:#d9dee7}}
.audit-summary{{margin:12px 0;padding:14px;border:1px solid #344157;border-radius:12px;background:#121925;display:grid;grid-template-columns:minmax(220px,.8fr) minmax(0,2fr);gap:14px;align-items:center}}
.audit-summary h2{{font-size:1rem;margin:0 0 5px}} .audit-summary p{{font-size:.75rem;line-height:1.4;color:var(--muted);margin:0}} .audit-cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.audit-card{{padding:9px;border:1px solid #2d3748;border-radius:8px;background:#0c1119}} .audit-card h3{{font-size:.78rem;margin:0 0 7px;color:var(--blue)}} .audit-lines{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}}
.audit-lines span{{font-size:.62rem;color:var(--muted);padding:4px;border-radius:5px;background:#151d29}} .audit-lines b{{display:block;font-size:.9rem;color:var(--text);margin-top:2px}} .audit-lines .aug b{{color:#86c5ff}} .audit-lines .maint b{{color:#89d3a4}} .audit-lines .lost b{{color:#ff9b9b}}
.audit-badges{{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}} .score{{font-size:.62rem;padding:3px 6px;border-radius:999px;border:1px solid #45536a;color:#cbd5e1}} .score.strategy{{background:#1a2230}} .score.augmented{{border-color:#397bb5;color:#91caff;background:#12283c}} .score.maintained{{border-color:#3b8058;color:#94dfac;background:#122b1d}} .score.lost{{border-color:#994b4b;color:#ffaaaa;background:#321919}}
.audit-reason{{min-height:2.5em;margin:5px 1px 0;color:#8f9bad;font-size:.61rem;line-height:1.3}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(5,minmax(180px,1fr));overflow-x:auto}}}} @media(max-width:700px){{.topline{{align-items:stretch;flex-direction:column}}input{{width:100%}}}}
</style></head><body>
<div class="top"><div class="topline"><div><h1>50 Cues × 5 Morphologies</h1><div class="meta">250 motion cells · {gif_count} GIFs found · updated {updated}</div></div><input id="search" placeholder="Filter by cue or category…"></div>
<div class="legend"><span><i class="dot" style="background:var(--blue)"></i>All five morphologies are LLM-authored (Gemini 2.5 Pro; Qwen-32B fallback for four previously unrendered cells).</span></div></div>
<main>{audit_summary_markup()}{''.join(sections)}</main>
<script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.trim().toLowerCase();document.querySelectorAll('.cue').forEach(x=>x.hidden=s&&!x.dataset.search.includes(s))}});</script>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    if len(cue_rows) != 50 or gif_count != 250:
        raise SystemExit(f"gallery incomplete: cues={len(cue_rows)}, gifs={gif_count}")
    print(f"wrote {OUT}")
    print(f"cues={len(cue_rows)} morphologies={len(PLATFORMS)} gifs={gif_count}")


if __name__ == "__main__":
    main()
