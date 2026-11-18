#!/usr/bin/env python3
"""Build a self-contained-index HTML gallery for selecting Fig. 5 examples."""
from __future__ import annotations

import html
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
REPO = PAPER.parent.parent
OUT = PAPER / "fig5_candidate_gallery.html"
ASSETS = PAPER / "fig5_gallery_assets"
FEASIBILITY = PAPER / "embodiment_physical_difficulty_90.json"
CUES = REPO / "data/seed/agent_fill/embodiment_short/cues.json"
GEMINI = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"

ROBOTS = (
    ("Single Arm", GEMINI / "manipulator/gif/IIWA"),
    ("Stretch", GEMINI / "stretch/gif"),
    ("Roomba", GEMINI / "roomba/gif"),
    ("Paddleclaw", PAPER / "synthetic_morphology/renders/cues30/gcr_alpha1"),
    ("Trileg Crown", PAPER / "synthetic_morphology/renders/cues30/gcr_alpha0"),
    ("Spine Crab", PAPER / "synthetic_morphology/renders/cues30/gcr_alpha05"),
)


def local_asset(robot: str, cue: str, path: Path) -> str:
    """Expose only gallery GIFs beneath the served paper directory."""
    folder = ASSETS / robot.lower().replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{cue}.gif"
    if not dest.exists():
        dest.hardlink_to(path)
    return dest.relative_to(PAPER).as_posix()


def main() -> None:
    cues = json.loads(CUES.read_text())
    feasibility = json.loads(FEASIBILITY.read_text())["by_robot"]
    sections = []
    for row in cues:
        idx, cue = int(row["idx"]), str(row["cue"])
        cards = []
        for robot, folder in ROBOTS:
            if robot in {"Paddleclaw", "Trileg Crown", "Spine Crab"}:
                path = folder / f"{cue}.gif"
            else:
                path = folder / f"emb_g{idx:02d}_{cue}.gif"
            status = "feasible" if cue in feasibility[robot]["feasible"] else "difficult"
            src = local_asset(robot, cue, path)
            cards.append(f'''<article class="card {status}" data-robot="{html.escape(robot)}" data-status="{status}" data-key="{html.escape(robot)}::{html.escape(cue)}">
  <button class="pick" type="button" title="Select this tile">☆</button>
  <div class="meta"><b>{html.escape(robot)}</b><span class="badge">{status}</span></div>
  <img loading="lazy" src="{src}" alt="{html.escape(robot)} - {html.escape(cue)}">
</article>''')
        sections.append(f'''<section class="cue" data-cue="{html.escape(cue)}">
<h2><span>{idx:02d}</span> {html.escape(cue)}</h2>
<div class="grid">{''.join(cards)}</div>
</section>''')

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fig. 5 candidate selector</title>
<style>
:root{{--ink:#181818;--muted:#6b7280;--line:#d8dce2;--paper:#f5f6f8;--blue:#1f6feb;--red:#b42318}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.35 Arial,sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line);padding:12px 22px;box-shadow:0 2px 10px #00000010}}
.top{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}} h1{{font-size:18px;margin:0 12px 0 0}} input{{width:260px;padding:8px 10px;border:1px solid #b8bec8;border-radius:6px}}
.filters button{{padding:7px 10px;border:1px solid #b8bec8;background:white;cursor:pointer}} .filters button.active{{background:var(--ink);color:white}}
#selected{{margin-left:auto;color:var(--muted)}} main{{max-width:1500px;margin:auto;padding:14px 22px 70px}}
.cue{{background:white;border:1px solid var(--line);border-radius:9px;margin:0 0 15px;overflow:hidden}} h2{{font:700 14px ui-monospace,monospace;margin:0;padding:9px 12px;border-bottom:1px solid var(--line)}} h2 span{{color:var(--muted);margin-right:5px}}
.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line)}} .card{{position:relative;background:white;padding:8px}}
.meta{{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}} .badge{{font-size:11px;text-transform:uppercase;color:#35705a}} .difficult .badge{{color:var(--red)}}
img{{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;background:#eee;border:1px solid #e1e3e6}}
.pick{{position:absolute;right:10px;bottom:10px;z-index:2;width:31px;height:31px;border-radius:50%;border:1px solid #999;background:#fffffff0;font-size:20px;cursor:pointer}}
.card.selected{{outline:4px solid var(--blue);outline-offset:-4px}} .card.selected .pick{{background:var(--blue);color:white;border-color:var(--blue)}}
.cue.hidden,.card.hidden{{display:none}} footer{{position:fixed;bottom:0;left:0;right:0;background:#181818;color:white;padding:9px 22px;font:12px ui-monospace,monospace;white-space:nowrap;overflow:auto}}
@media(max-width:1200px){{.grid{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:700px){{.grid{{grid-template-columns:repeat(2,1fr)}}#selected{{width:100%}}}} 
</style></head><body>
<header><div class="top"><h1>Fig. 5 candidate selector</h1><input id="search" placeholder="Search cue or robot"><div class="filters"><button class="active" data-filter="all">All</button><button data-filter="feasible">Feasible</button><button data-filter="difficult">Difficult</button></div><div id="selected">0 selected</div></div></header>
<main>{''.join(sections)}</main><footer id="selection">Click ☆ to collect candidate tiles.</footer>
<script>
const chosen=new Set(); let filter='all'; const q=document.querySelector('#search');
function refresh(){{
 const term=q.value.trim().toLowerCase();
 document.querySelectorAll('.cue').forEach(sec=>{{let visible=0;sec.querySelectorAll('.card').forEach(card=>{{
  const matchFilter=filter==='all'||card.dataset.status===filter;
  const matchText=(sec.dataset.cue+' '+card.dataset.robot).toLowerCase().includes(term);
  card.classList.toggle('hidden',!(matchFilter&&matchText)); if(matchFilter&&matchText)visible++;
 }});sec.classList.toggle('hidden',visible===0)}});
 document.querySelector('#selected').textContent=chosen.size+' selected';
 document.querySelector('#selection').textContent=chosen.size?[...chosen].join('  |  '):'Click ☆ to collect candidate tiles.';
}}
document.querySelectorAll('.filters button').forEach(b=>b.onclick=()=>{{document.querySelector('.filters .active').classList.remove('active');b.classList.add('active');filter=b.dataset.filter;refresh()}});
document.querySelectorAll('.pick').forEach(b=>b.onclick=()=>{{const c=b.closest('.card'),k=c.dataset.key;if(chosen.has(k))chosen.delete(k);else chosen.add(k);c.classList.toggle('selected',chosen.has(k));b.textContent=chosen.has(k)?'★':'☆';refresh()}});
q.oninput=refresh;
</script></body></html>'''
    OUT.write_text(document)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
