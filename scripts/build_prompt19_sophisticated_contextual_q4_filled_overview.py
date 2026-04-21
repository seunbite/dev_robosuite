from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"

TARGET = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
MANIFEST = SEED / "q4_fill_sophisticated_contextual" / "manifest.json"
OUT_DIR = SEED / "q4_fill_sophisticated_contextual"
OUT_HTML = OUT_DIR / "prompt19_sophisticated_contextual_q4_filled_overview_20260405_ko.html"


def _load_json(path: Path):
    return json.loads(path.read_text())


def _compact_steps(row: dict) -> str:
    parts = []
    for mv in row.get("movements", []):
        t = mv.get("type")
        p = mv.get("parameters", {})
        if t == "pose":
            pose = p.get("pose", {})
            parts.append(
                f"pose({pose.get('dir')}, {pose.get('gripper_orientation')}, "
                f"x{pose.get('x')}, y{pose.get('y')}, z{pose.get('z')})"
            )
        elif t == "movement":
            parts.append(f"movement({p.get('joint')}, rep={p.get('repetition')})")
        elif t == "path":
            shape = p.get("shape")
            if shape == "line":
                parts.append(f"path(line {p.get('axis')} {p.get('distance')})")
            elif shape == "arc":
                parts.append(f"path(arc {p.get('plane')} r{p.get('radius')} s{p.get('sweep')})")
            else:
                parts.append(f"path({shape})")
        else:
            parts.append(str(t))
    return " -> ".join(parts)


def main() -> None:
    rows = _load_json(TARGET)
    manifest = _load_json(MANIFEST)
    regenerated = {(int(item["idx"]), item["cue"]) for item in manifest.get("missing", [])}

    cards = []
    for row in sorted(rows, key=lambda x: int(x["idx"])):
        idx = int(row["idx"])
        cue = row["cue"]
        is_regenerated = (idx, cue) in regenerated
        badge = "Regenerated Q4" if is_regenerated else "Existing Q4"
        badge_class = "regen" if is_regenerated else "existing"
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="row">
                  <h2>c{idx} {html.escape(cue)}</h2>
                  <span class="badge {badge_class}">{badge}</span>
                </div>
                <div class="summary">{html.escape(_compact_steps(row))}</div>
              </div>
              <div class="body">
                <div class="label">Reasoning</div>
                <pre>{html.escape(row.get("reasoning", ""))}</pre>
                <div class="label">Config</div>
                <pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>
              </div>
            </article>
            """
        )

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt 19 Sophisticated Contextual Q4 Filled</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#dde4ea; --muted:#64707b; --ink:#111; --regen:#0f766e; --regen-bg:#ecfdf5; --existing:#475569; --existing-bg:#f8fafc; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1700px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    .lead {{ margin:0 0 8px; color:var(--muted); }}
    .meta {{ margin:0 0 20px; color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
    .row {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
    h2 {{ margin:0; font-size:18px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; font-weight:700; }}
    .badge.regen {{ color:var(--regen); background:var(--regen-bg); }}
    .badge.existing {{ color:var(--existing); background:var(--existing-bg); }}
    .summary {{ margin-top:8px; font-size:13px; color:var(--muted); }}
    .body {{ padding:14px 16px 18px; }}
    .label {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    pre {{ margin:0 0 14px; white-space:pre-wrap; word-break:break-word; background:#f8fafb; border:1px solid #edf1f4; padding:10px 12px; font-size:12px; line-height:1.45; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated Contextual Q4 Filled</h1>
    <p class="lead">`motion_configs_prompt_v19_sophisticated_contextual.json`의 45개 전부가 이제 Q4를 갖습니다.</p>
    <p class="meta">regenerated: {len(regenerated)} / existing: {len(rows) - len(regenerated)} / total: {len(rows)}</p>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(text, encoding="utf-8")
    print(f"Wrote html: {OUT_HTML}")


if __name__ == "__main__":
    main()
