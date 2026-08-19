#!/usr/bin/env python3
"""Build HTML review page for Tiago few-shot renders + configs."""
from __future__ import annotations

import html
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]

SHOTS_PATH = _REPO / "data/seed/shots/tiago/diverse_shots_mobile.json"
PROMPT_PATH = _REPO / "data/seed/prompt/tiago/exp/prompt_exp1.txt"
OUT_DIR = _REPO / "data/results/visualize/tiago/new_shots_render"
OUT_HTML = OUT_DIR / "index.html"
OUT_CONFIGS = OUT_DIR / "shots_configs.json"


def _step_tags(movements: list[dict]) -> str:
    tags = []
    for step in movements:
        st = step.get("type", "?")
        tags.append(f'<span class="tag tag-step">{html.escape(st)}</span>')
    return "".join(tags)


def build_html() -> Path:
    shots = json.loads(SHOTS_PATH.read_text(encoding="utf-8"))
    prompt_text = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.is_file() else ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CONFIGS.write_text(json.dumps(shots, indent=2, ensure_ascii=False), encoding="utf-8")

    cards = []
    for shot in shots:
        cue = shot["cue"]
        legacy_rel = f"{cue}.gif"
        new_rel = f"new_render/{cue}.gif"
        legacy_path = OUT_DIR / legacy_rel
        new_path = OUT_DIR / new_rel

        def _gif_block(rel: str, path: Path, label: str) -> str:
            if path.is_file():
                return (
                    f'<figure><figcaption>{html.escape(label)}</figcaption>'
                    f'<img src="{html.escape(rel)}" alt="{html.escape(cue)}" loading="lazy"/></figure>'
                )
            return (
                f'<figure><figcaption>{html.escape(label)}</figcaption>'
                f'<p class="missing">GIF missing — run render_new_shots.py</p></figure>'
            )

        media = (
            f'<div class="media-grid">'
            f'{_gif_block(legacy_rel, legacy_path, "Legacy render (ignore_xyz)")}'
            f'{_gif_block(new_rel, new_path, "New render (xyz-trim)")}'
            f"</div>"
        )
        reasoning = shot.get("reasoning", "")
        reasoning_html = "<br/>".join(html.escape(line) for line in reasoning.splitlines())
        config_json = json.dumps(shot, indent=2, ensure_ascii=False)
        movements = shot.get("movements", [])

        cards.append(
            f"""<article class="card">
  <header>
    <h2>{html.escape(cue)}</h2>
    <div class="tags">{_step_tags(movements)}</div>
  </header>
  <p class="desc">{html.escape(shot.get("description", ""))}</p>
    <div class="media-config">
    <div class="media">{media}</div>
    <div class="config">
      <h3>Config</h3>
      <pre class="json">{html.escape(config_json)}</pre>
    </div>
  </div>
  <details class="reasoning">
    <summary>Reasoning</summary>
    <p>{reasoning_html}</p>
  </details>
</article>"""
        )

    prompt_block = html.escape(prompt_text)
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Tiago — Few-shot Shots Review</title>
  <style>
    :root {{
      --bg: #0f172a; --panel: #111827; --border: #334155;
      --text: #e2e8f0; --muted: #94a3b8; --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: system-ui, -apple-system, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.45;
    }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 1.6rem; margin: 0 0 8px; }}
    .summary {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; padding: 14px 16px; margin: 16px 0 24px;
      color: var(--muted); font-size: 0.95rem;
    }}
    .summary a {{ color: var(--accent); }}
    details.prompt {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 10px; margin-bottom: 24px; overflow: hidden;
    }}
    details.prompt summary {{
      cursor: pointer; padding: 12px 16px; font-weight: 600;
      background: #1e293b;
    }}
    details.prompt pre {{
      margin: 0; padding: 16px; white-space: pre-wrap; word-break: break-word;
      font-size: 12px; color: #cbd5e1; max-height: 420px; overflow: auto;
    }}
    .grid {{ display: flex; flex-direction: column; gap: 24px; }}
    .card {{
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 12px; padding: 16px;
    }}
    .card h2 {{ margin: 0; font-size: 1.2rem; }}
    .card h3 {{ margin: 0 0 8px; font-size: 0.85rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .tags {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
    .tag {{
      font-size: 10px; padding: 3px 8px; border-radius: 999px;
      font-weight: 600; text-transform: uppercase;
    }}
    .tag-step {{ background: #1e3a5f; color: #7dd3fc; border: 1px solid #0c4a6e; }}
    .desc {{ color: var(--muted); margin: 12px 0; }}
    .media-config {{
      display: grid; grid-template-columns: minmax(280px, 420px) 1fr; gap: 16px;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .media-config {{ grid-template-columns: 1fr; }}
    }}
    .media-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    }}
    @media (max-width: 900px) {{
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
    .media figure {{ margin: 0; }}
    .media figcaption {{
      font-size: 0.78rem; color: var(--muted); margin-bottom: 6px;
      text-transform: uppercase; letter-spacing: 0.03em;
    }}
    .media img {{
      width: 100%; height: auto; border-radius: 8px;
      border: 1px solid var(--border); background: #000;
    }}
    .missing {{ color: #f87171; font-size: 0.9rem; }}
    pre.json {{
      margin: 0; padding: 12px; border-radius: 8px;
      background: #0b1220; border: 1px solid var(--border);
      font-size: 11px; line-height: 1.35; overflow: auto;
      max-height: 520px; color: #a5f3fc;
    }}
    details.reasoning {{
      margin-top: 12px; border-top: 1px solid var(--border); padding-top: 10px;
    }}
    details.reasoning summary {{ cursor: pointer; color: var(--muted); font-size: 0.9rem; }}
    details.reasoning p {{ font-size: 0.85rem; color: #cbd5e1; margin: 8px 0 0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Tiago — Few-shot Shots</h1>
    <div class="summary">
      <p>{len(shots)} shots from <code>diverse_shots_mobile.json</code> (FEWSHOT_SHOTS for exp0/exp1 correction).
      Legacy = ignore_xyz · New render = xyz-trim (<code>MOBILE_POSE_XYZ_MODE=trim</code>).</p>
      <p>Prompt: <code>data/seed/prompt/tiago/exp/prompt_exp1.txt</code></p>
    </div>

    <details class="prompt" open>
      <summary>prompt_exp1.txt</summary>
      <pre>{prompt_block}</pre>
    </details>

    <section class="grid">
{"".join(cards)}
    </section>
  </div>
</body>
</html>
"""
    OUT_HTML.write_text(page, encoding="utf-8")
    return OUT_HTML


if __name__ == "__main__":
    out = build_html()
    print(f"Wrote {out}")
