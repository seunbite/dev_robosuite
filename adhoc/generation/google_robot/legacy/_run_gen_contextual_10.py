#!/usr/bin/env python3
"""Generate first 10 contextual cues, render, rebuild HTML with all configs."""
import json, os, sys, time

# API key — use last valid key (matches `source APIKEY.sh` behavior)
with open(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "APIKEY.sh")) as f:
    for line in f:
        if "GOOGLE_API_KEY" in line and "=" in line:
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key and not key.startswith("$"):
                os.environ["GOOGLE_API_KEY"] = key

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path[:0] = [ROOT, _SCRIPT_DIR]

import yaml
from config_gen_single_mobile import generate_motion_config, _load_json_list
from render_mobile_config import _make_env, render_config
from html import escape

SHOTS_JSON = os.path.join(ROOT, "data", "seed", "shots", "google_robot", "shot_configs_19_mobile.json")
CONFIG_JSON = os.path.join(
    ROOT, "data", "results", "motion_configs", "google_robot", "motion_configs_19_mobile.json"
)
PROMPT_FILE = os.path.join(ROOT, "data", "seed", "prompt", "google_robot", "prompt_19_mobile.txt")
YAML_PATH = os.path.join(ROOT, "data", "seed", "yml", "cues_new.yml")
GIF_DIR = os.path.join(ROOT, "data", "results", "render", "prompt19_mobile_all")

with open(YAML_PATH) as f:
    cues = yaml.safe_load(f)
with open(SHOTS_JSON) as f:
    fewshot_cues = {s["cue"] for s in json.load(f)}

existing_configs = _load_json_list(CONFIG_JSON) if os.path.exists(CONFIG_JSON) else []
already_generated = {c["cue"] for c in existing_configs}

all_items = []
idx = 61
for group in ("contextual", "iconic"):
    for cue_name, cue_desc in cues.get(group, {}).items():
        if cue_name not in fewshot_cues:
            if cue_name not in already_generated:
                all_items.append((idx, cue_name, cue_desc))
            idx += 1

target = all_items
print(f"Will generate {len(target)} cues:", flush=True)
for i, n, d in target:
    print(f"  {i}: {n}", flush=True)

failed, success = [], []
for ci, (cue_idx, cue_name, cue_desc) in enumerate(target):
    print(f"\n[{ci+1}/{len(target)}] {cue_name} (idx={cue_idx})", flush=True)
    ok = False
    for retry in range(3):
        try:
            generate_motion_config(
                cue_name=cue_name, cue_idx=cue_idx,
                prompt_file=PROMPT_FILE, shots_json=SHOTS_JSON,
                config_json=CONFIG_JSON, yaml_path=YAML_PATH,
                model_name="gemini-2.5-pro",
            )
            success.append((cue_idx, cue_name))
            ok = True
            break
        except Exception as e:
            err = str(e)
            if "503" in err or "429" in err:
                wait = 60 * (retry + 1)
                print(f"  Retry {retry+1} in {wait}s... ({err[:200]})", flush=True)
                time.sleep(wait)
            else:
                print(f"  FAIL: {err[:80]}", flush=True)
                break
    if not ok:
        failed.append(cue_name)
    time.sleep(30)

print(f"\n=== Gen done: {len(success)} ok, {len(failed)} fail ===", flush=True)

# Render only newly generated
configs = _load_json_list(CONFIG_JSON)
configs_by_idx = {c["idx"]: c for c in configs}
os.makedirs(GIF_DIR, exist_ok=True)

print(f"\nRendering {len(success)} new configs...", flush=True)
env = _make_env()
try:
    for cue_idx, cue_name in success:
        cfg = configs_by_idx.get(cue_idx)
        if cfg is None:
            continue
        safe = cue_name.replace("/", "_").replace(" ", "_")
        gif_path = os.path.join(GIF_DIR, f"mm19_g{cue_idx}_{safe}.gif")
        try:
            frames = render_config(cfg, env=env)
            if frames:
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
                print(f"  OK {os.path.basename(gif_path)} ({len(frames)}f)", flush=True)
        except Exception as e:
            print(f"  RENDER ERR {cue_name}: {e}", flush=True)
finally:
    env.close()

# Rebuild combined HTML (fewshots + all generated)
shots = _load_json_list(SHOTS_JSON)
all_configs = shots + sorted(configs, key=lambda c: c.get("idx", 0))

cards = []
for cfg in all_configs:
    cue = cfg.get("cue", "?")
    idx_v = cfg.get("idx", -1)
    state = cfg.get("state", "generated")
    safe = cue.replace("/", "_").replace(" ", "_")
    prefix = "mm19_c" if state == "fewshot" else "mm19_g"
    gif_name = f"{prefix}{idx_v:02d}_{safe}.gif"
    gif_path_full = os.path.join(GIF_DIR, gif_name)
    movements = cfg.get("movements", [])
    steps = []
    for m in movements:
        mt = m.get("type", "?")
        steps.append(f'<span class="step {mt}">{mt}</span>')
    reasoning = cfg.get("reasoning", "")
    display = {k: v for k, v in cfg.items() if k not in ("state", "time", "model", "reasoning", "validation_warnings")}
    gif_html = f'<img src="{gif_name}" loading="lazy">' if os.path.exists(gif_path_full) else '<div class="na">No render</div>'
    badge = f'<span class="badge {state}">{state}</span>'
    cards.append(f"""<article class="card">
<div class="card-header">{badge} <span class="idx">{prefix[5]}{idx_v}</span> {escape(cue)}</div>
<div class="card-body">
<div class="gif">{gif_html}</div>
<div class="steps">{' &rarr; '.join(steps)}</div>
<div class="label">Reasoning</div>
<pre class="cot">{escape(reasoning or 'N/A')}</pre>
<div class="label">Config</div>
<pre class="json">{escape(json.dumps(display, indent=2, ensure_ascii=False))}</pre>
</div></article>""")

html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Mobile Manip All Configs</title>
<style>
:root{{--bg:#f6f8fb;--s:#fff;--b:#d0d7de;--t:#1f2328;--m:#59636e;--a:#0969da;--g:#1a7f37;--p:#8250df}}
*{{box-sizing:border-box}}body{{margin:0;font-family:system-ui;background:var(--bg);color:var(--t)}}
.w{{max-width:1500px;margin:0 auto;padding:24px}}h1{{font-size:26px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}}
.card{{background:var(--s);border:1px solid var(--b);border-radius:12px;overflow:hidden}}
.card-header{{padding:12px;border-bottom:1px solid var(--b);background:#eef2f7;font-weight:600}}
.card-header .idx{{color:var(--a);margin-right:8px}}
.badge{{padding:2px 8px;border-radius:999px;font-size:11px;font-weight:700}}
.badge.fewshot{{background:#ddf4ff;color:#0969da}}.badge.generated{{background:#dafbe1;color:#1a7f37}}
.card-body{{padding:14px}}.gif img{{display:block;max-width:100%;border:1px solid var(--b);border-radius:8px;margin:0 0 10px}}
.na{{color:var(--m);font-style:italic}}.steps{{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 12px}}
.step{{padding:3px 8px;border-radius:999px;font-size:13px;background:#1f6feb22;color:var(--a)}}
.step.movement{{background:#2ea04322;color:var(--g)}}.step.path{{background:#bc8cff22;color:var(--p)}}
.step.pose_to_pose{{background:#cf222e22;color:#cf222e}}
.label{{font-size:12px;font-weight:700;color:var(--m);text-transform:uppercase;margin:12px 0 4px}}
.cot{{padding:10px;border-left:3px solid var(--p);background:#eef2f7;border-radius:6px;white-space:pre-wrap;font-size:13px;max-height:150px;overflow:auto}}
.json{{padding:10px;background:#eef2f7;border-radius:6px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:300px;overflow:auto}}
</style></head><body><div class="w">
<h1>Mobile Manip Configs ({len(shots)} fewshot + {len(configs)} generated)</h1>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""

html_path = os.path.join(GIF_DIR, "index.html")
with open(html_path, "w") as f:
    f.write(html)
print(f"\nHTML: {html_path}", flush=True)

import subprocess
subprocess.Popen(["open", html_path])
print("DONE", flush=True)
