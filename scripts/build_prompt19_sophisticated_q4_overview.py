from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
FILES = [
    ("iconic", ROOT / "data/seed/motion_configs_prompt_v19_sophisticated.json"),
    ("contextual", ROOT / "data/seed/motion_configs_prompt_v19_sophisticated_contextual.json"),
]
OUT = ROOT / "data/seed/q4_analysis/prompt19_sophisticated_q4_overview_20260404_ko.html"
MOTION_DIRS = {
    "iconic": ROOT / "data/motions/v19_sophisticated/IIWA",
    "contextual": ROOT / "data/motions/v19_sophisticated_contextual/IIWA",
}


def parse_q4(reasoning: str) -> tuple[str, str, str]:
    q4 = reasoning.split("Q4:", 1)[1].replace("\n", " ").strip()
    q4 = re.sub(r"\s+", " ", q4)
    m = re.search(r"options?=(.*?);\s*winner=(.*)", q4, re.I)
    if not m:
        return q4, "", ""
    return q4, m.group(1).strip(), m.group(2).strip()


def split_options(options_text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(C\d)\)?\s*", options_text))
    if not matches:
        return []
    out = []
    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(options_text)
        text = options_text[start:end].strip(" ,;")
        out.append((key, text))
    return out


def classify_design(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["recoil", "dip", "gather", "wind-up", "upbeat", "preparatory", "anticipation", "pre-action", "pre-lift", "pre-sweep", "pre-circle", "pre-drop"]):
        return "Anticipation / Prep"
    if any(k in t for k in ["hold", "lock", "pause", "linger", "freeze", "wait", "static pose"]):
        return "Hold / Lock"
    if any(k in t for k in ["sag", "settle", "return", "release pose", "relaxation", "falling-away", "lean-back"]):
        return "Follow-through / Settle"
    if any(k in t for k in ["wave", "double-tap", "jab", "pump", "poke", "wiggle", "tremor", "scan"]):
        return "Semantic Accent"
    if any(k in t for k in ["lean", "tilt", "drift", "forward extension", "look back", "torso-turn", "arc", "sway"]):
        return "Context-Shaping Posture"
    return "Other"


def html_escape(text: str) -> str:
    return html.escape(text, quote=True)


def find_gif(dataset: str, cue: str) -> str | None:
    base = MOTION_DIRS[dataset]
    single = sorted(base.glob(f"*_{cue}_p*.gif"))
    if single:
        return single[-1].resolve().as_uri()
    tiled = sorted(base.glob(f"*_{cue}_c*_tiled.gif"))
    if tiled:
        return tiled[-1].resolve().as_uri()
    any_match = sorted(base.glob(f"*_{cue}_*.gif"))
    if any_match:
        return any_match[-1].resolve().as_uri()
    return None


rows = []
winner_counter: Counter[str] = Counter()
type_counter: Counter[str] = Counter()
set_counter: Counter[str] = Counter()

for dataset, path in FILES:
    data = json.loads(path.read_text())
    for row in data:
        reasoning = row.get("reasoning", "")
        if "Q4:" not in reasoning:
            continue
        raw_q4, options_text, winner_text = parse_q4(reasoning)
        options = split_options(options_text)
        winner_match = re.search(r"\b(C\d(?:\s+and\s+C\d)?)\b", winner_text, re.I)
        winner = winner_match.group(1).upper() if winner_match else "NA"
        design_type = classify_design(f"{options_text} {winner_text}")
        rows.append(
            {
                "dataset": dataset,
                "idx": row["idx"],
                "cue": row["cue"],
                "options": options,
                "winner": winner,
                "winner_text": winner_text,
                "design_type": design_type,
                "raw_q4": raw_q4,
                "gif": find_gif(dataset, row["cue"]),
            }
        )
        winner_counter[winner] += 1
        type_counter[design_type] += 1
        set_counter[dataset] += 1

rows.sort(key=lambda r: (0 if r["dataset"] == "iconic" else 1, r["idx"]))

summary_cards = [
    ("Q4 Total", str(len(rows))),
    ("Iconic", str(set_counter["iconic"])),
    ("Contextual", str(set_counter["contextual"])),
    ("Winner C1", str(winner_counter["C1"])),
    ("Winner C2", str(winner_counter["C2"])),
]

summary_html = "".join(
    f'<div class="stat"><div class="stat-label">{html_escape(label)}</div><div class="stat-value">{html_escape(value)}</div></div>'
    for label, value in summary_cards
)

type_html = "".join(
    f'<div class="chip">{html_escape(k)} <strong>{v}</strong></div>'
    for k, v in type_counter.most_common()
)

card_html_parts = []
for r in rows:
    options_html = "".join(
        f'<div class="option"><span class="opt-key">{html_escape(key)}</span><span class="opt-text">{html_escape(text)}</span></div>'
        for key, text in r["options"]
    )
    card_html_parts.append(
        f"""
        <article class="card">
          <div class="card-top">
            <div class="meta-row">
              <span class="dataset {r['dataset']}">{html_escape(r['dataset'])}</span>
              <span class="idx">c{r['idx']}</span>
              <span class="type">{html_escape(r['design_type'])}</span>
            </div>
            <h3>{html_escape(r['cue'])}</h3>
          </div>
          <div class="gif-wrap">
            {f'<img class="gif" src="{html_escape(r["gif"])}" alt="{html_escape(r["cue"])}" />' if r["gif"] else '<div class="gif-missing">GIF not found</div>'}
          </div>
          <div class="section">
            <div class="section-label">Goal Cue</div>
            <div class="goal">{html_escape(r['cue'])}</div>
          </div>
          <div class="section">
            <div class="section-label">Q4 Considered</div>
            <div class="options">{options_html}</div>
          </div>
          <div class="section">
            <div class="section-label">Chosen</div>
            <div class="chosen"><span class="winner">{html_escape(r['winner'])}</span> {html_escape(r['winner_text'])}</div>
          </div>
          <details class="raw">
            <summary>Raw Q4</summary>
            <pre>{html_escape(r['raw_q4'])}</pre>
          </details>
        </article>
        """
    )

html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt 19 Sophisticated Q4 Overview</title>
  <style>
    :root {{
      --bg: #ffffff;
      --panel: #ffffff;
      --line: #d8dde3;
      --muted: #68707c;
      --text: #111418;
      --soft: #f5f7f9;
      --iconic: #d9ecff;
      --contextual: #e7f7e7;
      --accent: #1e293b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1700px;
      margin: 0 auto;
      padding: 24px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .lead {{
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px 14px;
    }}
    .stat-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .stat-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: var(--soft);
      padding: 6px 10px;
      font-size: 12px;
      border-radius: 999px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 14px;
    }}
    .card-top h3 {{
      margin: 8px 0 0;
      font-size: 18px;
      line-height: 1.25;
    }}
    .meta-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 12px;
    }}
    .dataset, .idx, .type {{
      padding: 4px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
    }}
    .dataset.iconic {{ background: var(--iconic); }}
    .dataset.contextual {{ background: var(--contextual); }}
    .section {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid #edf1f4;
    }}
    .gif-wrap {{
      margin-top: 12px;
      border: 1px solid #edf1f4;
      background: #fff;
      padding: 8px;
    }}
    .gif {{
      width: 100%;
      display: block;
      background: #fff;
    }}
    .gif-missing {{
      min-height: 180px;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 13px;
      background: var(--soft);
    }}
    .section-label {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .goal {{
      font-weight: 700;
      font-size: 15px;
    }}
    .options {{
      display: grid;
      gap: 8px;
    }}
    .option {{
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 8px;
      align-items: start;
      padding: 8px 10px;
      background: var(--soft);
      border: 1px solid #edf1f4;
    }}
    .opt-key {{
      font-weight: 700;
      color: var(--accent);
    }}
    .opt-text, .chosen {{
      font-size: 14px;
      line-height: 1.45;
    }}
    .winner {{
      display: inline-block;
      min-width: 52px;
      font-weight: 700;
      color: var(--accent);
    }}
    details.raw {{
      margin-top: 12px;
      border-top: 1px solid #edf1f4;
      padding-top: 10px;
    }}
    details.raw summary {{
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
    }}
    details.raw pre {{
      margin: 10px 0 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      color: #3a4350;
      background: var(--soft);
      padding: 10px;
      border: 1px solid #edf1f4;
    }}
    @media (max-width: 1200px) {{
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    }}
    @media (max-width: 760px) {{
      .grid, .stats {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 16px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated: Q4 Design Overview</h1>
    <p class="lead">Q4가 있는 69개만 모았습니다. 각 카드에는 목표 cue, Q4에서 비교한 옵션, 실제 최종 채택안, 그리고 그 채택이 어떤 설계 유형인지가 들어 있습니다.</p>
    <section class="stats">{summary_html}</section>
    <section class="chips">{type_html}</section>
    <section class="grid">
      {''.join(card_html_parts)}
    </section>
  </main>
</body>
</html>
"""

OUT.write_text(html_text)
print(f"Wrote {OUT}")
