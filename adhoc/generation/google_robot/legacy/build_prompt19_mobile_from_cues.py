#!/usr/bin/env python3
"""Assemble the mobile-manipulator prompt from JSON shots + txt template.

Reads:
  data/seed/shots/google_robot/shot_configs_19_mobile.json
  data/seed/prompt/google_robot/prompt_19_mobile.txt (template)
  data/seed/yml/cues_new.yml

Outputs:
  data/seed/prompt/google_robot/prompt_19_mobile.txt (assembled)
  data/seed/prompt/google_robot/prompt_19_mobile_preview.html
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CUES_PATH = ROOT / "data" / "seed" / "yml" / "cues_new.yml"
_P = ROOT / "data" / "seed" / "prompt" / "google_robot"
_S = ROOT / "data" / "seed" / "shots" / "google_robot"

SHOT_PATH = _S / "shot_configs_19_mobile.json"
PROMPT_TEMPLATE_PATH = _P / "prompt_19_mobile.txt"
PROMPT_ASSEMBLED_PATH = _P / "prompt_19_mobile.txt"
HTML_PATH = _P / "prompt_19_mobile_preview.html"


# ── YAML parser (no pyyaml dependency) ────────────────────────────────────

def parse_simple_grouped_yaml(path: Path) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    current_group: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not line.startswith("  "):
            current_group = line.rstrip(":")
            groups[current_group] = {}
            continue
        if current_group is None:
            continue
        key, value = line.strip().split(":", 1)
        groups[current_group][key.strip()] = value.strip()
    return groups


# ── Build logic ───────────────────────────────────────────────────────────

def main():
    _P.mkdir(parents=True, exist_ok=True)
    _S.mkdir(parents=True, exist_ok=True)

    # 1. Read few-shot configs
    shots: list[dict[str, Any]] = json.loads(SHOT_PATH.read_text(encoding="utf-8"))
    print(f"Read {len(shots)} few-shot configs ← {SHOT_PATH}")

    # 2. Read prompt template
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    print(f"Read prompt template ← {PROMPT_TEMPLATE_PATH}")

    # 3. Build cue catalog
    cue_catalog = parse_simple_grouped_yaml(CUES_PATH)
    catalog_parts: list[str] = []
    for group_name in ("iconic", "contextual"):
        if group_name not in cue_catalog:
            continue
        catalog_parts.append(f"[Available {group_name} cues]")
        for cue, text in cue_catalog[group_name].items():
            catalog_parts.append(f"- {cue}: {text}")
        catalog_parts.append("")
    catalog_str = "\n".join(catalog_parts)

    # 4. Assemble prompt (catalog injected, examples/cue left as placeholders)
    assembled = template.replace("{{CUE_CATALOG}}", catalog_str)
    PROMPT_ASSEMBLED_PATH.write_text(assembled, encoding="utf-8")
    print(f"Wrote assembled prompt → {PROMPT_ASSEMBLED_PATH}")

    # 5. HTML preview
    _build_html_preview(shots, template, catalog_str)
    print(f"Wrote HTML preview → {HTML_PATH}")


def _build_html_preview(records: list[dict], template: str, catalog: str):
    cards = []
    for r in records:
        cue = r["cue"]
        idx = r["idx"]
        reasoning = r.get("reasoning", "")
        movements = r.get("movements", [])

        steps_html = []
        for m in movements:
            mtype = m.get("type", "?")
            steps_html.append(f'<span class="step {mtype}">{mtype}</span>')

        display = {k: v for k, v in r.items() if k not in ("state", "time", "model", "reasoning")}

        cards.append(f"""
    <article class="card">
      <div class="card-header"><span class="idx">c{idx:02d}</span> {escape(cue)}</div>
      <div class="card-body">
        <div class="steps">{' → '.join(steps_html)}</div>
        <div class="label">Reasoning</div>
        <pre class="cot">{escape(reasoning)}</pre>
        <div class="label">Config</div>
        <pre class="json">{escape(json.dumps(display, indent=2, ensure_ascii=False))}</pre>
      </div>
    </article>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Mobile Manip Few-shots</title>
<style>
body{{margin:0;font-family:system-ui;background:#f6f8fb;color:#1f2328}}
.wrap{{max-width:1400px;margin:0 auto;padding:24px}}
h1{{font-size:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:12px;overflow:hidden}}
.card-header{{padding:12px;border-bottom:1px solid #d0d7de;background:#eef2f7;font-weight:600}}
.card-header .idx{{color:#0969da;margin-right:8px}}
.card-body{{padding:14px}}
.steps{{display:flex;gap:6px;margin:0 0 12px}}
.step{{padding:3px 8px;border-radius:999px;font-size:13px;background:#1f6feb22;color:#0969da}}
.step.movement{{background:#2ea04322;color:#1a7f37}}
.step.path{{background:#bc8cff22;color:#8250df}}
.label{{font-size:12px;font-weight:700;color:#59636e;text-transform:uppercase;margin:12px 0 4px}}
.cot{{padding:10px;border-left:3px solid #8250df;background:#eef2f7;border-radius:6px;white-space:pre-wrap;font-size:13px;max-height:150px;overflow:auto}}
.json{{padding:10px;background:#eef2f7;border-radius:6px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:300px;overflow:auto}}
</style></head><body>
<div class="wrap">
<h1>Mobile Manipulator Few-Shot Configs ({len(records)} examples)</h1>
<div class="grid">{''.join(cards)}</div>
</div></body></html>"""

    HTML_PATH.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
