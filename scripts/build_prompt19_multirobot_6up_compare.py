from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

import fire


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
OUT_DIR = SEED
ROBOTS = ["IIWA", "Panda", "XArm7"]

ICONIC_PATH = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_PATH = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
NO_REASONING_ICONIC = SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json"
NO_REASONING_CONTEXTUAL = SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json"


def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _latest_single_gif(base: Path, cue: str) -> Path | None:
    safe = _safe_name(cue)
    matches = sorted(base.rglob(f"*_{safe}_p*.gif"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _img_or_missing(path: Path | None, alt: str) -> str:
    if path is None:
        return "<div class='missing'>No image</div>"
    return f"<img src='{path.resolve().as_uri()}' alt='{html.escape(alt)}' loading='lazy'>"


def _build_section(title: str, ref_rows: list[dict], no_reason_rows: list[dict], dataset: str) -> str:
    no_reason_map = {int(r["idx"]): r for r in no_reason_rows}
    cards: list[str] = []
    for row in sorted(ref_rows, key=lambda x: int(x["idx"])):
        idx = int(row["idx"])
        cue = row["cue"]
        no_reason_row = no_reason_map.get(idx)

        soph_cells = []
        for robot in ROBOTS:
            gif = _latest_single_gif(
                (MOTIONS / ("v19_sophisticated" if dataset == "iconic" else "v19_sophisticated_contextual") / robot),
                cue,
            )
            soph_cells.append(
                f"<div class='cell'><div class='robot'>{robot}</div>{_img_or_missing(gif, f'sophisticated {robot} {cue}')}</div>"
            )

        no_reason_cells = []
        for robot in ROBOTS:
            gif = _latest_single_gif(
                MOTIONS / "baseline_prompt19_full_no_reasoning" / f"no_reasoning_{dataset}" / robot,
                cue,
            )
            no_reason_cells.append(
                f"<div class='cell'><div class='robot'>{robot}</div>{_img_or_missing(gif, f'no reasoning {robot} {cue}')}</div>"
            )

        cards.append(
            f"""
            <article class='card'>
              <div class='hdr'>
                <div class='cue'>c{idx} {html.escape(cue)}</div>
                <div class='meta'>{html.escape(row.get('description', ''))}</div>
              </div>
              <div class='sixup'>
                {''.join(soph_cells)}
                {''.join(no_reason_cells)}
              </div>
            </article>
            """
        )

    return f"""
    <section class='section'>
      <div class='section-head'>
        <h2>{html.escape(title)}</h2>
        <div class='section-meta'>{len(cards)} cues</div>
      </div>
      <div class='stack'>{''.join(cards)}</div>
    </section>
    """


def build(
    iconic_only: bool = False,
    contextual_only: bool = False,
    output_name: str | None = None,
) -> str:
    if iconic_only and contextual_only:
        raise ValueError("Choose at most one of iconic_only/contextual_only")
    iconic_rows = _load_json(ICONIC_PATH)
    contextual_rows = _load_json(CONTEXTUAL_PATH)
    no_reason_iconic_rows = _load_json(NO_REASONING_ICONIC)
    no_reason_contextual_rows = _load_json(NO_REASONING_CONTEXTUAL)

    if output_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if iconic_only:
            output_name = f"prompt19_multirobot_iconic_sophisticated_vs_no_reasoning_6up_{stamp}_ko.html"
        elif contextual_only:
            output_name = f"prompt19_multirobot_contextual_sophisticated_vs_no_reasoning_6up_{stamp}_ko.html"
        else:
            output_name = f"prompt19_multirobot_sophisticated_vs_no_reasoning_6up_{stamp}_ko.html"

    sections = []
    if not contextual_only:
        sections.append(_build_section("Iconic", iconic_rows, no_reason_iconic_rows, "iconic"))
    if not iconic_only:
        sections.append(_build_section("Contextual", contextual_rows, no_reason_contextual_rows, "contextual"))

    page = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prompt19 Multirobot 6-Up Compare</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#d9e0e7; --ink:#111; --muted:#66717d; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width: 2400px; margin:0 auto; padding: 24px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    h2 {{ margin:0; font-size:24px; }}
    .lead {{ margin:0 0 10px; color:var(--muted); }}
    .meta {{ margin:0 0 24px; color:var(--muted); font-size:13px; }}
    .section {{ margin-top:28px; }}
    .section-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid var(--line); }}
    .section-meta {{ color:var(--muted); font-size:13px; }}
    .stack {{ display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:12px 14px; border-bottom:1px solid var(--line); }}
    .cue {{ font-size:18px; font-weight:700; }}
    .meta {{ font-size:13px; color:var(--muted); }}
    .sixup {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; padding:12px 14px 14px; }}
    .cell {{ min-width:0; }}
    .robot {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    img, .missing {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }}
    .missing {{ min-height:180px; display:grid; place-items:center; color:var(--muted); }}
    @media (max-width: 1800px) {{ .sixup {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
    @media (max-width: 980px) {{ .sixup {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width: 720px) {{ .sixup {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Multirobot 6-Up Compare</h1>
    <p class="lead">왼쪽 3개는 Sophisticated, 오른쪽 3개는 No Reasoning입니다. 각 cue는 IIWA / Panda / XArm7 순서로 고정돼 있습니다.</p>
    <p class="meta">generated={html.escape(datetime.now().isoformat(timespec='seconds'))} | iconic_only={str(iconic_only).lower()} | contextual_only={str(contextual_only).lower()}</p>
    {''.join(sections)}
  </main>
</body>
</html>
"""

    out_path = OUT_DIR / output_name
    out_path.write_text(page, encoding="utf-8")
    print(out_path)
    return str(out_path)


if __name__ == "__main__":
    fire.Fire({"build": build})
