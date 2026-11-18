#!/usr/bin/env python3
"""Build a separate gallery for the 30 human-authored cues on five robots."""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
RENDER = REPO / "data/results/render/embodiment_motion_code_gt"
OUT = REPO / "data/results/html/human_authored_30x5.html"
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


def load_afc_trials() -> dict[str, dict[str, Any]]:
    trials = json.loads(HARD_NEGATIVE_TRIALS.read_text(encoding="utf-8"))
    return {str(trial["cue"]): trial for trial in trials}


def afc_markup(trial: dict[str, Any]) -> str:
    choices = trial["choices"]
    answer = str(trial["answer_id"])
    pills = []
    for i, choice in enumerate(choices):
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


def gif_path(platform: str, idx: int, cue: str) -> Path:
    rel = Path("manipulator/gif/IIWA") if platform == "single_arm" else Path(platform) / "gif"
    return RENDER / rel / f"emb_g{idx:02d}_{cue}.gif"


def card(row: dict[str, Any], platform: str, label: str) -> tuple[str, bool]:
    idx, cue = int(row["idx"]), str(row["cue"])
    cfg = (row.get("platforms") or {}).get(platform) or {}
    gif = gif_path(platform, idx, cue)
    if gif.is_file():
        rel = os.path.relpath(gif, OUT.parent)
        media = f'<img src="{html.escape(rel)}?v={int(gif.stat().st_mtime)}" loading="lazy" alt="{html.escape(cue)} on {html.escape(label)}">'
    else:
        media = '<div class="missing">Missing render</div>'
    code = str(cfg.get("source_code") or "").strip() or "(source code unavailable)"
    movements = json.dumps(cfg.get("movements") or [], ensure_ascii=False, indent=2)
    return (
        f'<article class="cell" data-platform="{platform}"><header><h3>{html.escape(label)}</h3>'
        f'<span class="badge">Human-authored</span></header>{media}'
        f'<details><summary>Motion program</summary><pre>{html.escape(code)}</pre></details>'
        f'<details><summary>Parsed config</summary><pre>{html.escape(movements)}</pre></details></article>',
        gif.is_file(),
    )


def main() -> None:
    rows = sorted(json.loads(SOURCE.read_text(encoding="utf-8")), key=lambda x: int(x["idx"]))
    afc_trials = load_afc_trials()
    sections: list[str] = []
    gifs = 0
    for row in rows:
        idx, cue = int(row["idx"]), str(row["cue"])
        category = str(row.get("category") or "")
        trial = afc_trials.get(cue)
        if trial is None:
            raise KeyError(f"missing hard-negative trial for {cue}")
        afc = afc_markup(trial)
        option_search = " ".join(str(choice["id"]) for choice in trial["choices"])
        cards: list[str] = []
        for platform, label in PLATFORMS:
            markup, exists = card(row, platform, label)
            cards.append(markup)
            gifs += int(exists)
        sections.append(
            f'<section class="cue" data-search="{html.escape((cue + " " + category + " " + option_search).lower())}">'
            f'<div class="cue-title"><div><span class="seq">{idx:02d}</span><h2>{html.escape(cue)}</h2></div>'
            f'<span class="category">{html.escape(category)}</span></div>{afc}'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    updated = datetime.now().isoformat(timespec="seconds")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human-authored 30 Cues × 5 Morphologies</title>
<style>
:root{{--bg:#0d1016;--panel:#171b24;--cell:#10141c;--line:#303744;--text:#eef1f6;--muted:#aab2bf;--gold:#efc46b}}
*{{box-sizing:border-box}}body{{font-family:Inter,ui-sans-serif,system-ui,sans-serif;margin:0;background:var(--bg);color:var(--text)}}
.top{{position:sticky;top:0;z-index:4;background:rgba(13,16,22,.96);backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:18px 22px 14px}}
.topline{{display:flex;gap:18px;align-items:end;justify-content:space-between}}h1{{font-size:1.45rem;margin:0 0 4px}}.meta{{color:var(--muted);font-size:.88rem}}
input{{width:min(420px,42vw);border:1px solid var(--line);background:#080b10;color:var(--text);border-radius:8px;padding:10px 12px;font-size:.92rem}}
.legend{{margin-top:10px;color:var(--muted);font-size:.78rem}}.dot{{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:5px;background:var(--gold)}}
main{{padding:6px 14px 40px}}.cue{{margin:12px 0;padding:12px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}}
.cue-title{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}.cue-title>div{{display:flex;gap:10px;align-items:center}}.seq{{color:var(--muted);font-variant-numeric:tabular-nums}}
h2{{font-size:1rem;margin:0}}.category{{font-size:.72rem;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:4px 8px}}
.afc{{display:flex;gap:10px;align-items:center;margin:-2px 0 10px;padding:8px 10px;border:1px solid #2a3342;border-radius:8px;background:#111722}}
.afc-label{{flex:0 0 auto;color:var(--muted);font-size:.72rem;font-weight:650}}.choices{{display:flex;gap:6px;flex-wrap:wrap}}
.choice{{display:inline-flex;gap:6px;align-items:center;padding:4px 7px;border:1px solid #354052;border-radius:6px;background:#0b1018;color:#cbd2dc;font-size:.72rem}}
.choice b{{color:#8da2bd}}.choice.correct{{border-color:#9a7b35;background:#29220f;color:#f3d88d}}.choice.correct b{{color:#f3d88d}}.answer-tag{{font-size:.58rem;text-transform:uppercase;letter-spacing:.04em;color:#10141c;background:var(--gold);border-radius:999px;padding:2px 5px}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}}.cell{{min-width:0;background:var(--cell);border-radius:9px;padding:8px;border:1px solid #222936}}
.cell header{{display:flex;justify-content:space-between;gap:6px;align-items:center;margin-bottom:7px}}h3{{font-size:.78rem;margin:0}}.badge{{font-size:.62rem;border-radius:999px;padding:3px 6px;white-space:nowrap;color:var(--gold);background:#3a2d12}}
img,.missing{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#05070a;display:block}}.missing{{display:grid;place-items:center;color:#ff8e8e}}
details{{margin-top:6px}}summary{{cursor:pointer;color:var(--muted);font-size:.7rem}}pre{{font:10px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;word-break:break-word;max-height:150px;overflow:auto;background:#070a0f;padding:7px;border-radius:5px;color:#d9dee7}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(5,minmax(180px,1fr));overflow-x:auto}}}}@media(max-width:700px){{.topline{{align-items:stretch;flex-direction:column}}input{{width:100%}}}}
</style></head><body><div class="top"><div class="topline"><div><h1>Human-authored 30 Cues × 5 Morphologies</h1>
<div class="meta">150 motion cells · {gifs} GIFs found · updated {updated}</div></div><input id="search" placeholder="Filter by cue or category…"></div>
<div class="legend"><span><i class="dot"></i>All five morphologies use the human-authored ground-truth Motion Programs.</span></div></div>
<main>{''.join(sections)}</main><script>const q=document.querySelector('#search');q.addEventListener('input',()=>{{const s=q.value.trim().toLowerCase();document.querySelectorAll('.cue').forEach(x=>x.hidden=s&&!x.dataset.search.includes(s))}});</script>
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    if len(rows) != 30 or gifs != 150:
        raise SystemExit(f"gallery incomplete: cues={len(rows)}, gifs={gifs}")
    print(f"wrote {OUT}")
    print(f"cues={len(rows)} morphologies={len(PLATFORMS)} gifs={gifs}")


if __name__ == "__main__":
    main()
