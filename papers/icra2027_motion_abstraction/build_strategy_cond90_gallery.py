#!/usr/bin/env python3
"""HTML gallery: 90 strategy-conditioned renders (IIWA / Stretch / Roomba x 30),
each beside its unconditioned baseline, labelled with the automatic strategy code
from classify_rematerialization.py."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
OUT = HERE / "strategy_cond90_gallery.html"

BODIES = [
    ("single_arm", "IIWA", BASE / "manipulator/gif/IIWA",
     REPO / "data/results/single_arm_strategy_fewshot/render/manipulator/gif/IIWA"),
    ("stretch", "Stretch", BASE / "stretch/gif",
     REPO / "data/results/stretch_strategy_fewshot/render"),
    ("roomba", "Roomba", BASE / "roomba/gif",
     REPO / "data/results/roomba_strategy_fewshot/render"),
]
LABEL = {"carrier": "Carrier", "trace": "Tracing", "additive": "Additive", "none": "None"}


def gif(d: Path, cue: str) -> str | None:
    hits = sorted(d.glob(f"emb_g*_{cue}.gif"))
    return os.path.relpath(hits[0], HERE) if hits else None


def main() -> None:
    data = json.loads((HERE / "rematerialization_strategy_counts.json").read_text())
    cell = {(c["body"], c["cue"]): c for c in data["cells"]}
    cues = list(dict.fromkeys(c["cue"] for c in data["cells"]))
    counts = data["counts"]

    rows_html = []
    for key, name, _, _ in BODIES:
        for cond, tag in ((key, name), (f"{key}_strategy", f"{name}, strategy-cond.")):
            c = counts.get(cond, {})
            n = sum(c.values()) or 1
            rows_html.append(
                f"<tr><td>{tag}</td>" + "".join(
                    f"<td>{round(100*c.get(s,0)/n)}% <span class=n>({c.get(s,0)})</span></td>"
                    for s in LABEL) + "</tr>")

    sections = []
    for key, name, bdir, sdir in BODIES:
        cards = []
        for i, cue in enumerate(cues, 1):
            b, s = cell.get((key, cue)), cell.get((f"{key}_strategy", cue))
            if not s:
                continue
            bg, sg = gif(bdir, cue), gif(sdir, cue)
            def media(p, cap):
                inner = f'<img loading="lazy" src="{html.escape(p)}">' if p else '<div class=miss>not rendered</div>'
                return f'<figure>{inner}<figcaption>{cap}</figcaption></figure>'
            bs = b["strategy"] if b else "none"
            cards.append(f"""
<article class="card" data-s="{s['strategy']}" data-b="{bs}">
  <h3><span class=idx>{i:02d}</span> {html.escape(cue)}</h3>
  <div class=pair>{media(sg, 'strategy-cond.')}{media(bg, 'baseline')}</div>
  <div class=tags><span class="tag {bs}">{LABEL[bs]}</span><span class=arrow>→</span><span class="tag {s['strategy']}">{LABEL[s['strategy']]}</span>{' <span class=chg>changed</span>' if bs != s['strategy'] else ''}</div>
  <details><summary>programs</summary>
    <p class=lab>strategy-cond.</p><code>{html.escape(s['code'])}</code>
    <p class=lab>baseline</p><code>{html.escape(b['code'] if b else '')}</code>
  </details>
</article>""")
        sections.append(f'<section data-body="{key}"><h2>{name} <span class=n>30 cues</span></h2><div class=grid>{"".join(cards)}</div></section>')

    OUT.write_text(f"""<!doctype html><html><head><meta charset=utf-8>
<title>Strategy-conditioned renders</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;
--carrier:#2f6fb0;--trace:#3a8a55;--additive:#b5642a;--none:#8a8a85}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif}}
header{{padding:28px 32px 8px;max-width:1400px;margin:auto}}
h1{{font-size:22px;margin:0 0 6px}} h2{{font-size:17px;margin:28px 0 12px}}
.n{{color:var(--mut);font-weight:400;font-size:12px}}
main{{max-width:1400px;margin:auto;padding:0 32px 60px}}
table{{border-collapse:collapse;margin:12px 0;font-variant-numeric:tabular-nums}}
td,th{{padding:4px 14px 4px 0;text-align:left;border-bottom:1px solid var(--line)}}
.rules{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;max-width:900px}}
.rules ol{{margin:6px 0;padding-left:20px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
button{{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:4px 12px;cursor:pointer;font:inherit}}
button.on{{background:var(--fg);color:var(--bg)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}}
.card h3{{font-size:13px;margin:0 0 8px;font-family:ui-monospace,monospace}} .idx{{color:var(--mut)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
figure{{margin:0}} img{{width:100%;border-radius:4px;background:#eee;display:block}}
figcaption{{font-size:11px;color:var(--mut);text-align:center}}
.miss{{aspect-ratio:1;display:grid;place-items:center;color:var(--mut);background:#f0f0ec;font-size:12px;border-radius:4px}}
.tags{{margin:8px 0 4px;display:flex;gap:6px;align-items:center;font-size:12px}}
.tag{{color:#fff;border-radius:10px;padding:1px 9px}} .arrow{{color:var(--mut)}}
.carrier{{background:var(--carrier)}} .trace{{background:var(--trace)}} .additive{{background:var(--additive)}} .none{{background:var(--none)}}
.chg{{color:var(--additive);font-size:11px}}
details{{font-size:11px}} code{{display:block;white-space:pre-wrap;word-break:break-word;background:#f4f4f0;padding:6px;border-radius:4px}}
.lab{{margin:6px 0 2px;color:var(--mut)}}
</style></head><body>
<header><h1>Strategy-conditioned renders — 90 programs</h1>
<p class=n>IIWA / Stretch / Roomba × 30 cues · Gemini 2.5 Pro · left = strategy-conditioned, right = unconditioned baseline · labels are the automatic code, baseline → conditioned</p>
<table><tr><th></th>{''.join(f'<th>{v}</th>' for v in LABEL.values())}</tr>{''.join(rows_html)}</table>
<div class=rules><b>How each program is coded</b> (first rule that fires wins; reference = human-authored humanoid program for the same cue)
<ol><li><b>None</b> — no movement and no path beyond a single forward advance (pose-only, or advance-and-hold).</li>
<li><b>Tracing</b> — draws a real envelope: an arc with radius ≥ 0.08 m, or a line path in ≥ 3 distinct directions. In-place spins (r&lt;0.08) do not count.</li>
<li><b>Carrier</b> — some stroke shares the reference's visible plane (sagittal: tilt/pitch/x/rock · lateral: pan/yaw/y/strafe · vertical: z/lift/height · roll) and a compatible sign.</li>
<li><b>Additive</b> — has motion, but no stroke matches the reference.</li></ol></div>
</header>
<main><div class=bar>
<b>Body</b><button class=on data-fb=all>All</button><button data-fb=single_arm>IIWA</button><button data-fb=stretch>Stretch</button><button data-fb=roomba>Roomba</button>
<b style="margin-left:14px">Conditioned as</b><button class=on data-fs=all>All</button>{''.join(f'<button data-fs={k}>{v}</button>' for k,v in LABEL.items())}
<button data-fs=changed style="margin-left:8px">Changed only</button></div>
{''.join(sections)}</main>
<script>
let fb='all',fs='all';
function apply(){{document.querySelectorAll('section').forEach(s=>s.hidden=fb!=='all'&&s.dataset.body!==fb);
document.querySelectorAll('.card').forEach(c=>c.hidden=fs==='all'?false:fs==='changed'?c.dataset.s===c.dataset.b:c.dataset.s!==fs);}}
document.querySelectorAll('[data-fb]').forEach(b=>b.onclick=()=>{{fb=b.dataset.fb;document.querySelectorAll('[data-fb]').forEach(x=>x.classList.toggle('on',x===b));apply();}});
document.querySelectorAll('[data-fs]').forEach(b=>b.onclick=()=>{{fs=b.dataset.fs;document.querySelectorAll('[data-fs]').forEach(x=>x.classList.toggle('on',x===b));apply();}});
</script></body></html>""", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
