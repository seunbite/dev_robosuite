#!/usr/bin/env python3
"""Re-render failed GIFs and rebuild HTML."""
import json, os, sys, subprocess
from html import escape

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path[:0] = [ROOT, _SCRIPT_DIR]

from render_mobile_config import _make_env, render_config

def _load_json_list(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

CONFIG_JSON = os.path.join(
    ROOT, "data", "results", "motion_configs", "google_robot", "motion_configs_19_mobile.json"
)
GIF_DIR = os.path.join(ROOT, "data", "results", "render", "prompt19_mobile_generated")

configs = _load_json_list(CONFIG_JSON)
os.makedirs(GIF_DIR, exist_ok=True)

# Find which GIFs are missing
to_render = []
for cfg in configs:
    cue = cfg.get("cue", "?")
    idx = cfg.get("idx", -1)
    safe = cue.replace("/", "_").replace(" ", "_")
    gif_path = os.path.join(GIF_DIR, f"mm19_g{idx:02d}_{safe}.gif")
    if not os.path.exists(gif_path):
        to_render.append((idx, cue, safe, gif_path, cfg))

print(f"Total configs: {len(configs)}, missing GIFs: {len(to_render)}", flush=True)

env = _make_env()
ok = 0
try:
    for idx, cue, safe, gif_path, cfg in to_render:
        try:
            frames = render_config(cfg, env=env)
            if frames:
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
                ok += 1
                print(f"  OK g{idx:02d} {cue} ({len(frames)}f)", flush=True)
        except Exception as e:
            print(f"  ERR g{idx:02d} {cue}: {e}", flush=True)
finally:
    env.close()

print(f"\nRe-rendered {ok}/{len(to_render)}", flush=True)

# Rebuild HTML with ALL configs
cards = []
for cfg in sorted(configs, key=lambda c: c.get("idx", 0)):
    cue = cfg.get("cue", "?")
    idx_v = cfg.get("idx", -1)
    safe = cue.replace("/", "_").replace(" ", "_")
    gif_name = f"mm19_g{idx_v:02d}_{safe}.gif"
    gif_path = os.path.join(GIF_DIR, gif_name)
    movements = cfg.get("movements", [])
    steps = []
    for m in movements:
        mt = m.get("type", "?")
        steps.append(f'<span class="step {mt}">{mt}</span>')
    reasoning = cfg.get("reasoning", "")
    display = {k: v for k, v in cfg.items() if k not in ("state", "time", "model", "reasoning", "validation_warnings")}
    gif_html = f'<img src="{gif_name}" loading="lazy">' if os.path.exists(gif_path) else '<div class="na">No render</div>'
    cards.append(f"""<article class="card">
<div class="card-header"><span class="idx">g{idx_v}</span> {escape(cue)}</div>
<div class="card-body">
<div class="gif">{gif_html}</div>
<div class="steps">{' &rarr; '.join(steps)}</div>
<div class="label">Reasoning</div>
<pre class="cot">{escape(reasoning or 'N/A')}</pre>
<div class="label">Config</div>
<pre class="json">{escape(json.dumps(display, indent=2, ensure_ascii=False))}</pre>
</div></article>""")

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Mobile Manip Generated (contextual)</title>
<style>
:root{{--bg:#f6f8fb;--s:#fff;--b:#d0d7de;--t:#1f2328;--m:#59636e;--a:#0969da;--g:#1a7f37;--p:#8250df}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui;background:var(--bg);color:var(--t)}}
.w{{max-width:1500px;margin:0 auto;padding:24px}}h1{{font-size:26px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}}
.card{{background:var(--s);border:1px solid var(--b);border-radius:12px;overflow:hidden}}
.card-header{{padding:12px;border-bottom:1px solid var(--b);background:#eef2f7;font-weight:600}}
.card-header .idx{{color:var(--a);margin-right:8px}}
.card-body{{padding:14px}}.gif img{{display:block;max-width:100%;border:1px solid var(--b);border-radius:8px;margin:0 0 10px}}
.na{{color:var(--m);font-style:italic}}.steps{{display:flex;gap:6px;margin:0 0 12px}}
.step{{padding:3px 8px;border-radius:999px;font-size:13px;background:#1f6feb22;color:var(--a)}}
.step.movement{{background:#2ea04322;color:var(--g)}}.step.path{{background:#bc8cff22;color:var(--p)}}
.label{{font-size:12px;font-weight:700;color:var(--m);text-transform:uppercase;margin:12px 0 4px}}
.cot{{padding:10px;border-left:3px solid var(--p);background:#eef2f7;border-radius:6px;white-space:pre-wrap;font-size:13px;max-height:150px;overflow:auto}}
.json{{padding:10px;background:#eef2f7;border-radius:6px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:300px;overflow:auto}}
</style></head><body><div class="w">
<h1>Generated Mobile Manipulator Configs (contextual)</h1>
<p style="color:var(--m)">{len(configs)} configs total</p>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""

html_path = os.path.join(GIF_DIR, "index.html")
with open(html_path, "w") as f:
    f.write(html)
print(f"HTML: {html_path}", flush=True)
subprocess.Popen(["open", html_path])
print("DONE", flush=True)
