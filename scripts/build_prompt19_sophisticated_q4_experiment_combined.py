from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
ORIG_CONFIGS = {
    "iconic": SEED / "motion_configs_prompt_v19_sophisticated.json",
    "contextual": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
}

SOURCES = [
    (
        "Contrastive Pilot",
        SEED / "q4_contrastive_experiment" / "manifest.json",
        "confusable cue를 먼저 잡고 cue-specific accent를 우선한 5개 실험입니다.",
    ),
    (
        "Generic Recoil / Hold Retest",
        SEED / "q4_generic_recoil_hold_experiment" / "manifest.json",
        "원래 recoil / hold로 수렴하던 Q4를 semantic accent 쪽으로 다시 설계한 5개 실험입니다.",
    ),
]

OUT = SEED / "q4_experiment_combined" / "prompt19_sophisticated_q4_experiment_combined_20260404_ko.html"


def _gif_uri(path_str: str) -> str:
    if not path_str:
        return ""
    return Path(path_str).resolve().as_uri()


def _load_original_row(dataset: str, idx: int) -> dict:
    path = ORIG_CONFIGS[dataset]
    rows = json.loads(path.read_text())
    for row in rows:
        if int(row["idx"]) == int(idx):
            return row
    return {}


def _build_card(row: dict) -> str:
    before_uri = _gif_uri(row.get("before_gif", ""))
    after_uri = _gif_uri(row.get("after_gif", ""))
    confusable = ", ".join(row.get("confusable_with", []))
    feature = row.get("discriminative_feature", "")
    reasoning = row.get("reasoning", "")
    dataset = row.get("dataset", "")
    before_config = _load_original_row(dataset, row.get("idx"))
    after_config = row.get("config") or {}
    return f"""
    <article class="card">
      <div class="hdr">
        <div class="meta"><span class="dataset {html.escape(dataset)}">{html.escape(dataset)}</span><span>c{row.get('idx')}</span></div>
        <h3>{html.escape(row.get('cue', ''))}</h3>
        <div class="sub">confusable_with: {html.escape(confusable)}</div>
        <div class="sub">discriminative_feature: <strong>{html.escape(feature)}</strong></div>
      </div>
      <div class="media-grid">
        <div class="media-card">
          <div class="label">Before</div>
          {f'<img src="{before_uri}" alt="before {html.escape(row.get("cue", ""))}">' if before_uri else '<div class="missing">missing</div>'}
        </div>
        <div class="media-card">
          <div class="label">After</div>
          {f'<img src="{after_uri}" alt="after {html.escape(row.get("cue", ""))}">' if after_uri else '<div class="missing">missing</div>'}
        </div>
      </div>
      <div class="text-block">
        <div class="label">Reasoning</div>
        <pre>{html.escape(reasoning)}</pre>
      </div>
      <div class="text-block">
        <div class="label">Before Config</div>
        <pre>{html.escape(json.dumps(before_config, ensure_ascii=False, indent=2))}</pre>
      </div>
      <div class="text-block">
        <div class="label">After Config</div>
        <pre>{html.escape(json.dumps(after_config, ensure_ascii=False, indent=2))}</pre>
      </div>
    </article>
    """


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    total = 0
    for title, manifest_path, description in SOURCES:
        data = json.loads(manifest_path.read_text())
        cues = data.get("cues", [])
        total += len(cues)
        cards = "".join(_build_card(row) for row in cues)
        sections.append(
            f"""
            <section class="section">
              <div class="section-head">
                <h2>{html.escape(title)}</h2>
                <p>{html.escape(description)}</p>
              </div>
              <div class="grid">{cards}</div>
            </section>
            """
        )

    html_text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt 19 Sophisticated Q4 Experiment Combined</title>
  <style>
    body {{ margin: 0; font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background: #fff; color: #111; }}
    .wrap {{ max-width: 1640px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .lead {{ margin: 0 0 20px; color: #5d6670; }}
    .summary {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 24px; }}
    .chip {{ border: 1px solid #d9e0e6; background: #f7f9fb; padding: 6px 10px; font-size: 13px; border-radius: 999px; }}
    .section {{ margin-top: 28px; }}
    .section-head h2 {{ margin: 0 0 6px; font-size: 24px; }}
    .section-head p {{ margin: 0 0 14px; color: #5d6670; }}
    .grid {{ display: grid; gap: 18px; }}
    .card {{ border: 1px solid #dde3e8; background: #fff; }}
    .hdr {{ padding: 14px 16px; border-bottom: 1px solid #eef2f5; }}
    .meta {{ display: flex; gap: 8px; font-size: 12px; color: #5d6670; margin-bottom: 6px; }}
    .dataset {{ padding: 2px 8px; border: 1px solid #d9e0e6; border-radius: 999px; }}
    .dataset.iconic {{ background: #e9f3ff; }}
    .dataset.contextual {{ background: #ebf8eb; }}
    .hdr h3 {{ margin: 0 0 6px; font-size: 20px; }}
    .sub {{ font-size: 13px; color: #5d6670; margin-top: 2px; }}
    .media-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 14px 16px; }}
    .media-card {{ border: 1px solid #eef2f5; padding: 8px; background: #fff; }}
    .media-card img {{ width: 100%; display: block; }}
    .label {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: #5d6670; margin-bottom: 6px; }}
    .text-block {{ padding: 0 16px 16px; }}
    .text-block pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #f7f9fb; border: 1px solid #eef2f5; padding: 10px 12px; font-size: 13px; line-height: 1.45; }}
    .missing {{ min-height: 180px; display: grid; place-items: center; background: #f7f9fb; color: #6c7680; }}
    @media (max-width: 960px) {{ .media-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated: Q4 Experiment Combined</h1>
    <p class="lead">지금까지 다시 렌더한 Q4 실험 10개를 한 페이지에 모았습니다. 각 카드에서 기존 GIF와 새 GIF를 바로 비교할 수 있습니다.</p>
    <div class="summary">
      <span class="chip">total experiments: {total}</span>
      <span class="chip">sets: 2</span>
      <span class="chip">format: before / after / reasoning</span>
    </div>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
