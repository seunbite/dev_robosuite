#!/usr/bin/env python3
"""Render all few-shots + all generated configs, build combined HTML."""
import json, os, sys, subprocess
from html import escape

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path[:0] = [ROOT, _SCRIPT_DIR]
from render_mobile_config import _make_env, render_config

SHOT_JSON = os.path.join(ROOT, "data", "seed", "shots", "google_robot", "shot_configs_19_mobile.json")
GEN_JSON = os.path.join(
    ROOT, "data", "results", "motion_configs", "google_robot", "motion_configs_19_mobile.json"
)
GIF_DIR = os.path.join(ROOT, "data", "results", "render", "prompt19_mobile_all")
os.makedirs(GIF_DIR, exist_ok=True)

def load_json(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)

shots = load_json(SHOT_JSON)
gens = load_json(GEN_JSON)

print(f"Few-shots: {len(shots)}, Generated: {len(gens)}", flush=True)

env = _make_env()
all_cfgs = []

def render_one(cfg, prefix, idx, cue):
    safe = cue.replace("/","_").replace(" ","_")
    gif_name = f"{prefix}{idx:02d}_{safe}.gif"
    gif_path = os.path.join(GIF_DIR, gif_name)
    try:
        frames = render_config(cfg, env=env)
        if frames:
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
            print(f"  OK {gif_name} ({len(frames)}f)", flush=True)
            return gif_name
    except Exception as e:
        print(f"  ERR {gif_name}: {e}", flush=True)
    return None

try:
    print("\n=== Few-shots ===", flush=True)
    for s in shots:
        idx = s["idx"]
        cue = s["cue"]
        gif = render_one(s, "mm19_c", idx, cue)
        all_cfgs.append({"cfg": s, "gif": gif, "kind": "fewshot"})

    print("\n=== Generated ===", flush=True)
    for g in gens:
        idx = g.get("idx", -1)
        cue = g.get("cue", "?")
        gif = render_one(g, "mm19_g", idx, cue)
        all_cfgs.append({"cfg": g, "gif": gif, "kind": "generated"})
finally:
    env.close()

# HTML
cards = []
for item in all_cfgs:
    cfg = item["cfg"]
    gif = item["gif"]
    kind = item["kind"]
    cue = cfg.get("cue", "?")
    idx_v = cfg.get("idx", -1)
    tag = f"c{idx_v:02d}" if kind == "fewshot" else f"g{idx_v:02d}"
    movements = cfg.get("movements", [])
    steps = []
    for m in movements:
        mt = m.get("type", "?")
        steps.append(f'<span class="step {mt}">{mt}</span>')
    reasoning = cfg.get("reasoning", "")
    display = {k: v for k, v in cfg.items() if k not in ("state","time","model","reasoning","validation_warnings","description","cue_text","group")}
    gif_html = f'<img src="{gif}" loading="lazy">' if gif else '<div class="na">No render</div>'
    badge = f'<span class="badge {kind}">{kind}</span>'
    cards.append(f"""<article class="card">
<div class="card-header">{badge}<span class="idx">{tag}</span> {escape(cue)}</div>
<div class="card-body">
<div class="gif">{gif_html}</div>
<div class="steps">{' &rarr; '.join(steps)}</div>
<div class="label">Reasoning</div>
<pre class="cot">{escape(reasoning or 'N/A')}</pre>
<div class="label">Config</div>
<pre class="json">{escape(json.dumps(display, indent=2, ensure_ascii=False))}</pre>
</div></article>""")

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Mobile Manip - All Configs</title>
<style>
:root{{--bg:#f6f8fb;--s:#fff;--b:#d0d7de;--t:#1f2328;--m:#59636e;--a:#0969da;--g:#1a7f37;--p:#8250df;--y:#9a6700}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui;background:var(--bg);color:var(--t)}}
.w{{max-width:1500px;margin:0 auto;padding:24px}}h1{{font-size:26px}}
.sub{{color:var(--m);margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}}
.card{{background:var(--s);border:1px solid var(--b);border-radius:12px;overflow:hidden}}
.card-header{{padding:12px;border-bottom:1px solid var(--b);background:#eef2f7;font-weight:600;display:flex;align-items:center;gap:8px}}
.card-header .idx{{color:var(--a)}}
.badge{{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700;text-transform:uppercase}}
.badge.fewshot{{background:#ddf4ff;color:var(--a)}}.badge.generated{{background:#fff8c5;color:var(--y)}}
.card-body{{padding:14px}}.gif img{{display:block;max-width:100%;border:1px solid var(--b);border-radius:8px;margin:0 0 10px}}
.na{{color:var(--m);font-style:italic}}.steps{{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}}
.step{{padding:3px 8px;border-radius:999px;font-size:13px;background:#1f6feb22;color:var(--a)}}
.step.movement{{background:#2ea04322;color:var(--g)}}.step.path{{background:#bc8cff22;color:var(--p)}}
.step.pose_to_pose{{background:#fff8c522;color:var(--y)}}
.label{{font-size:12px;font-weight:700;color:var(--m);text-transform:uppercase;margin:12px 0 4px}}
.cot{{padding:10px;border-left:3px solid var(--p);background:#eef2f7;border-radius:6px;white-space:pre-wrap;font-size:13px;max-height:150px;overflow:auto}}
.json{{padding:10px;background:#eef2f7;border-radius:6px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:300px;overflow:auto}}
</style></head><body><div class="w">
<h1>Mobile Manipulator Configs</h1>
<p class="sub">{len(shots)} few-shots + {len(gens)} generated (contextual)</p>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""

html_path = os.path.join(GIF_DIR, "index.html")
with open(html_path, "w") as f:
    f.write(html)
print(f"\nHTML: {html_path}", flush=True)
subprocess.Popen(["open", html_path])
print("DONE", flush=True)
