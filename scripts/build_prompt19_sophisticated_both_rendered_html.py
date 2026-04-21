from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"

ICONIC_JSON = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_JSON = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
CONTEXTUAL_MANIFEST = SEED / "q4_fill_sophisticated_contextual" / "manifest.json"

ICONIC_MOTION_DIR = MOTIONS / "v19_sophisticated" / "IIWA"
CONTEXTUAL_MOTION_DIR = MOTIONS / "v19_sophisticated_contextual_q4filled" / "IIWA"

OUT_DIR = SEED / "q4_fill_sophisticated_contextual"
OUT_HTML = OUT_DIR / "prompt19_sophisticated_both_rendered_20260405_ko.html"


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


def _find_gif(motion_dir: Path, cue: str, cue_idx: int) -> Path | None:
    single = sorted(motion_dir.glob(f"*_{cue}_p*.gif"))
    if single:
        return single[-1]
    tiled = sorted(motion_dir.glob(f"*_{cue}_c{cue_idx}_tiled.gif"))
    if tiled:
        return tiled[-1]
    any_match = sorted(motion_dir.glob(f"*_{cue}_*.gif"))
    return any_match[-1] if any_match else None


def _build_cards(rows: list[dict], motion_dir: Path, badge_map: dict[tuple[int, str], tuple[str, str]]) -> str:
    cards = []
    for row in sorted(rows, key=lambda x: int(x["idx"])):
        idx = int(row["idx"])
        cue = row["cue"]
        badge_text, badge_class = badge_map.get((idx, cue), ("Q4 Present", "present"))
        gif = _find_gif(motion_dir, cue, idx)
        media = (
            f'<img src="{gif.resolve().as_uri()}" alt="{html.escape(cue)}" loading="lazy">'
            if gif
            else '<div class="missing">Render missing</div>'
        )
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="row">
                  <h3>c{idx} {html.escape(cue)}</h3>
                  <span class="badge {badge_class}">{badge_text}</span>
                </div>
                <div class="summary">{html.escape(_compact_steps(row))}</div>
              </div>
              <div class="media">{media}</div>
              <div class="body">
                <div class="label">Reasoning</div>
                <pre>{html.escape(row.get("reasoning", ""))}</pre>
                <div class="label">Config</div>
                <pre>{html.escape(json.dumps(row, ensure_ascii=False, indent=2))}</pre>
              </div>
            </article>
            """
        )
    return "".join(cards)


def main() -> None:
    iconic_rows = _load_json(ICONIC_JSON)
    contextual_rows = _load_json(CONTEXTUAL_JSON)
    manifest = _load_json(CONTEXTUAL_MANIFEST)

    regenerated = {(int(item["idx"]), item["cue"]) for item in manifest.get("missing", [])}
    contextual_badges = {}
    for row in contextual_rows:
        key = (int(row["idx"]), row["cue"])
        if key in regenerated:
            contextual_badges[key] = ("Regenerated Q4", "regen")
        else:
            contextual_badges[key] = ("Existing Q4", "existing")

    iconic_badges = {(int(r["idx"]), r["cue"]): ("Existing Q4", "existing") for r in iconic_rows}

    iconic_cards = _build_cards(iconic_rows, ICONIC_MOTION_DIR, iconic_badges)
    contextual_cards = _build_cards(contextual_rows, CONTEXTUAL_MOTION_DIR, contextual_badges)

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt 19 Sophisticated Rendered Overview</title>
  <style>
    :root {{
      --bg:#fff;
      --surface:#fff;
      --line:#dde4ea;
      --muted:#64707b;
      --ink:#111;
      --section:#f8fafc;
      --regen:#0f766e;
      --regen-bg:#ecfdf5;
      --existing:#475569;
      --existing-bg:#f8fafc;
      --present:#1d4ed8;
      --present-bg:#eff6ff;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1720px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    .lead {{ margin:0 0 8px; color:var(--muted); }}
    .meta {{ margin:0 0 22px; color:var(--muted); font-size:13px; }}
    .section {{ margin-top:28px; }}
    .section-head {{ position:sticky; top:0; z-index:3; background:rgba(255,255,255,.96); backdrop-filter: blur(4px); padding:12px 0 10px; border-bottom:1px solid var(--line); }}
    h2 {{ margin:0; font-size:24px; }}
    .section-meta {{ margin-top:4px; color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; gap:18px; padding-top:16px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
    .row {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
    h3 {{ margin:0; font-size:18px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; font-weight:700; }}
    .badge.regen {{ color:var(--regen); background:var(--regen-bg); }}
    .badge.existing {{ color:var(--existing); background:var(--existing-bg); }}
    .badge.present {{ color:var(--present); background:var(--present-bg); }}
    .summary {{ margin-top:8px; font-size:13px; color:var(--muted); }}
    .media {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
    .media img {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }}
    .missing {{ min-height:260px; display:grid; place-items:center; background:#f8fafb; border:1px solid var(--line); color:var(--muted); }}
    .body {{ padding:14px 16px 18px; }}
    .label {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    pre {{ margin:0 0 14px; white-space:pre-wrap; word-break:break-word; background:#f8fafb; border:1px solid #edf1f4; padding:10px 12px; font-size:12px; line-height:1.45; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated Rendered Overview</h1>
    <p class="lead">iconic 59개와 contextual 45개를 한 페이지에서 같이 봅니다. 둘 다 렌더 GIF, reasoning, config를 함께 붙였습니다.</p>
    <p class="meta">iconic motion dir: {html.escape(str(ICONIC_MOTION_DIR))} | contextual motion dir: {html.escape(str(CONTEXTUAL_MOTION_DIR))}</p>

    <section class="section">
      <div class="section-head">
        <h2>Sophisticated Iconic</h2>
        <div class="section-meta">total: {len(iconic_rows)} / q4 present: {len(iconic_rows)}</div>
      </div>
      <div class="grid">{iconic_cards}</div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Sophisticated Contextual</h2>
        <div class="section-meta">total: {len(contextual_rows)} / regenerated q4: {len(regenerated)} / existing q4: {len(contextual_rows) - len(regenerated)}</div>
      </div>
      <div class="grid">{contextual_cards}</div>
    </section>
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(text, encoding="utf-8")
    print(f"Wrote html: {OUT_HTML}")


if __name__ == "__main__":
    main()
