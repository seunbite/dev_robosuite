from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"

ORIGINALS = {
    "iconic": SEED / "motion_configs_prompt_v19_sophisticated.json",
    "contextual": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
}
ORIGINAL_MOTIONS = {
    "iconic": MOTIONS / "v19_sophisticated" / "IIWA",
    "contextual": MOTIONS / "v19_sophisticated_contextual" / "IIWA",
}

EXPERIMENTS = [
    {
        "label": "Prompt Refresh 10",
        "seed_root": SEED / "q4_prompt_refresh_experiment",
        "motion_root": MOTIONS / "q4_prompt_refresh_experiment" / "IIWA" / "IIWA",
        "iconic_config": SEED / "q4_prompt_refresh_experiment" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_iconic_10cue.json",
        "contextual_config": SEED / "q4_prompt_refresh_experiment" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_contextual_10cue.json",
    },
    {
        "label": "Prompt Refresh Extra 5",
        "seed_root": SEED / "q4_prompt_refresh_experiment_extra5",
        "motion_root": MOTIONS / "q4_prompt_refresh_experiment_extra5" / "IIWA" / "IIWA",
        "iconic_config": SEED / "q4_prompt_refresh_experiment_extra5" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_extra5_iconic.json",
        "contextual_config": SEED / "q4_prompt_refresh_experiment_extra5" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_extra5_contextual.json",
    },
    {
        "label": "Prompt Refresh Extra 5B",
        "seed_root": SEED / "q4_prompt_refresh_experiment_extra5_b",
        "motion_root": MOTIONS / "q4_prompt_refresh_experiment_extra5_b" / "IIWA" / "IIWA",
        "iconic_config": SEED / "q4_prompt_refresh_experiment_extra5_b" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_extra5_b_iconic.json",
        "contextual_config": SEED / "q4_prompt_refresh_experiment_extra5_b" / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_extra5_b_contextual.json",
    },
]

OUT_DIR = SEED / "q4_prompt_refresh_combined20"
OUT_HTML = OUT_DIR / "prompt19_sophisticated_q4_prompt_refresh_combined20_20260404_ko.html"


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _find_any_gif(base: Path, cue: str, cue_idx: int | None = None) -> Path | None:
    if cue_idx is not None:
        tiled = sorted(base.rglob(f"*_{cue}_c{cue_idx}_tiled.gif"))
        if tiled:
            return tiled[-1]
    single = sorted(base.rglob(f"*_{cue}_p*.gif"))
    if single:
        return single[-1]
    any_match = sorted(base.rglob(f"*_{cue}_*.gif"))
    return any_match[-1] if any_match else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    before_rows = {name: {int(r["idx"]): r for r in _load_rows(path)} for name, path in ORIGINALS.items()}
    cards: list[str] = []
    total = 0

    for exp in EXPERIMENTS:
        manifest = json.loads((exp["seed_root"] / "manifest.json").read_text())
        iconic_after = {int(r["idx"]): r for r in _load_rows(exp["iconic_config"])}
        contextual_after = {int(r["idx"]): r for r in _load_rows(exp["contextual_config"])}
        after_maps = {"iconic": iconic_after, "contextual": contextual_after}

        for spec in manifest["targets"]:
            dataset = spec["dataset"]
            idx = int(spec["idx"])
            cue = spec["cue"]
            before_row = before_rows[dataset][idx]
            after_row = after_maps[dataset][idx]
            before_gif = _find_any_gif(ORIGINAL_MOTIONS[dataset], cue, idx)
            after_gif = _find_any_gif(exp["motion_root"], cue)
            before_media = f'<img src="{before_gif.resolve().as_uri()}" alt="before">' if before_gif else '<div class="missing">missing</div>'
            after_media = f'<img src="{after_gif.resolve().as_uri()}" alt="after">' if after_gif else '<div class="missing">missing</div>'
            cards.append(
                f"""
                <article class="card">
                  <div class="hdr">
                    <div class="eyebrow">{html.escape(exp["label"])}</div>
                    <div class="title">{html.escape(dataset)} · c{idx} · {html.escape(cue)}</div>
                  </div>
                  <div class="media-grid">
                    <section><div class="label">Before</div>{before_media}</section>
                    <section><div class="label">After</div>{after_media}</section>
                  </div>
                  <div class="body">
                    <div class="label">Before Reasoning</div>
                    <pre>{html.escape(before_row.get("reasoning", ""))}</pre>
                    <div class="label">After Reasoning</div>
                    <pre>{html.escape(after_row.get("reasoning", ""))}</pre>
                    <div class="label">Before Config</div>
                    <pre>{html.escape(json.dumps(before_row, ensure_ascii=False, indent=2))}</pre>
                    <div class="label">After Config</div>
                    <pre>{html.escape(json.dumps(after_row, ensure_ascii=False, indent=2))}</pre>
                  </div>
                </article>
                """
            )
            total += 1

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt19 Sophisticated Q4 Prompt Refresh Combined 20</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#dfe5ea; --muted:#62707c; --ink:#111; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1720px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    .lead {{ margin:0 0 8px; color:var(--muted); }}
    .meta {{ margin:0 0 20px; font-size:13px; color:var(--muted); }}
    .grid {{ display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
    .eyebrow {{ font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; }}
    .title {{ font-size:18px; font-weight:700; }}
    .media-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; padding:14px 16px; border-bottom:1px solid var(--line); }}
    .media-grid img {{ width:100%; display:block; border:1px solid var(--line); }}
    .body {{ padding:14px 16px 18px; }}
    .label {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    pre {{ margin:0 0 14px; white-space:pre-wrap; word-break:break-word; background:#f8fafb; border:1px solid #edf1f4; padding:10px 12px; font-size:12px; line-height:1.45; }}
    .missing {{ min-height:240px; display:grid; place-items:center; background:#f8fafb; border:1px solid var(--line); color:var(--muted); }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Sophisticated Q4 Prompt Refresh Combined 20</h1>
    <p class="lead">지금까지 새 prompt로 다시 생성한 20개 cue를 원본 sophisticated 결과와 한 페이지에서 비교합니다.</p>
    <p class="meta">cards: {total}</p>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    OUT_HTML.write_text(text, encoding="utf-8")
    print(f"Wrote html: {OUT_HTML}")


if __name__ == "__main__":
    main()
