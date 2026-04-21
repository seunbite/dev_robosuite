from __future__ import annotations

import html
import json
import random
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
OUT_ROOT = SEED / "q4_prompt_refresh_experiment"
MOTION_OUT = MOTIONS / "q4_prompt_refresh_experiment" / "IIWA" / "IIWA"

PROMPT_PATH = SEED / "q4_contrastive_experiment" / "prompt_v19_sophisticated_q4_contrastive_full.txt"
SHOTS_PATH = SEED / "q4_contrastive_experiment" / "shot_configs_v19_sophisticated_q4_contrastive.json"
ICONIC_SRC = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_SRC = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
ICONIC_OUT = OUT_ROOT / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_iconic_10cue.json"
CONTEXTUAL_OUT = OUT_ROOT / "motion_configs_prompt_v19_sophisticated_q4_prompt_refresh_contextual_10cue.json"
HTML_OUT = OUT_ROOT / "prompt19_sophisticated_q4_prompt_refresh_compare_20260404_ko.html"
MANIFEST_OUT = OUT_ROOT / "manifest.json"


TARGETS = [
    {"dataset": "iconic", "idx": 1, "cue": "raising_hand_greeting"},
    {"dataset": "iconic", "idx": 13, "cue": "point_self"},
    {"dataset": "iconic", "idx": 16, "cue": "stop_palm_out"},
    {"dataset": "iconic", "idx": 24, "cue": "cover_mouth_gasp"},
    {"dataset": "iconic", "idx": 25, "cue": "cheers_toast"},
    {"dataset": "iconic", "idx": 36, "cue": "visor_search"},
    {"dataset": "contextual", "idx": 36, "cue": "prepare_action_raise_hold"},
    {"dataset": "contextual", "idx": 38, "cue": "hesitation_pause_hold"},
    {"dataset": "contextual", "idx": 39, "cue": "commit_action_fast_reach"},
    {"dataset": "iconic", "idx": 45, "cue": "slow_down_request_palm_down"},
]


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(rows, key=lambda x: int(x["idx"])), ensure_ascii=False, indent=2), encoding="utf-8")


def _find_any_gif(base: Path, cue: str, cue_idx: int | None = None) -> Path | None:
    safe = cue
    if cue_idx is not None:
        tiled = sorted(base.rglob(f"*_{safe}_c{cue_idx}_tiled.gif"))
        if tiled:
            return tiled[-1]
    single = sorted(base.rglob(f"*_{safe}_p*.gif"))
    if single:
        return single[-1]
    any_match = sorted(base.rglob(f"*_{safe}_*.gif"))
    return any_match[-1] if any_match else None


def generate() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
    from config_gen_single import generate_motion_config  # noqa: E402

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    random.seed(19)
    iconic_rows = []
    contextual_rows = []
    for spec in TARGETS:
        out_path = ICONIC_OUT if spec["dataset"] == "iconic" else CONTEXTUAL_OUT
        generate_motion_config(
            cue_name=spec["cue"],
            cue_idx=int(spec["idx"]),
            model_name="gemini-2.5-pro",
            prompt_file=str(PROMPT_PATH),
            shots_json=str(SHOTS_PATH),
            config_json=str(out_path),
            max_handmade_examples=10,
            max_correction_examples=10,
            temperature=None,
            use_shots=True,
            require_reasoning=True,
        )
        rows = _load_rows(out_path)
        row = next(r for r in rows if int(r["idx"]) == int(spec["idx"]) and r["cue"] == spec["cue"])
        if spec["dataset"] == "iconic":
            iconic_rows = [r for r in rows if int(r["idx"]) in {1, 13, 16, 24, 25, 36, 45}]
        else:
            contextual_rows = [r for r in rows if int(r["idx"]) in {36, 38, 39}]
    _write_rows(ICONIC_OUT, iconic_rows)
    _write_rows(CONTEXTUAL_OUT, contextual_rows)

    manifest = {
        "prompt_path": str(PROMPT_PATH),
        "shots_path": str(SHOTS_PATH),
        "iconic_out": str(ICONIC_OUT),
        "contextual_out": str(CONTEXTUAL_OUT),
        "targets": TARGETS,
        "model_name": "gemini-2.5-pro",
    }
    MANIFEST_OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote iconic configs: {ICONIC_OUT}")
    print(f"Wrote contextual configs: {CONTEXTUAL_OUT}")
    print(f"Wrote manifest: {MANIFEST_OUT}")


def build_html() -> None:
    iconic_before = {int(r["idx"]): r for r in _load_rows(ICONIC_SRC)}
    contextual_before = {int(r["idx"]): r for r in _load_rows(CONTEXTUAL_SRC)}
    iconic_after = {int(r["idx"]): r for r in _load_rows(ICONIC_OUT)}
    contextual_after = {int(r["idx"]): r for r in _load_rows(CONTEXTUAL_OUT)}

    source_dirs = {
        "iconic": MOTIONS / "v19_sophisticated" / "IIWA",
        "contextual": MOTIONS / "v19_sophisticated_contextual" / "IIWA",
    }

    cards = []
    for spec in TARGETS:
        before_row = iconic_before[spec["idx"]] if spec["dataset"] == "iconic" else contextual_before[spec["idx"]]
        after_row = iconic_after[spec["idx"]] if spec["dataset"] == "iconic" else contextual_after[spec["idx"]]
        before_gif = _find_any_gif(source_dirs[spec["dataset"]], spec["cue"], spec["idx"])
        after_gif = _find_any_gif(MOTION_OUT, spec["cue"])
        before_html = f'<img src="{before_gif.resolve().as_uri()}" alt="before">' if before_gif else '<div class="missing">missing</div>'
        after_html = f'<img src="{after_gif.resolve().as_uri()}" alt="after">' if after_gif else '<div class="missing">missing</div>'
        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="title">{html.escape(spec["dataset"])} · c{spec["idx"]} · {html.escape(spec["cue"])}</div>
              </div>
              <div class="media-grid">
                <section><div class="label">Before</div>{before_html}</section>
                <section><div class="label">After</div>{after_html}</section>
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

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prompt19 Sophisticated Q4 Prompt Refresh Compare</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#dfe5ea; --muted:#62707c; --ink:#111; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1700px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    .lead {{ margin:0 0 16px; color:var(--muted); }}
    .meta {{ margin:0 0 24px; font-size:13px; color:var(--muted); }}
    .grid {{ display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
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
    <h1>Prompt 19 Sophisticated Q4 Prompt Refresh</h1>
    <p class="lead">새로 수정한 Q4 프롬프트로 다시 생성한 10개 cue를 원본 sophisticated 결과와 비교합니다.</p>
    <p class="meta">prompt: {html.escape(str(PROMPT_PATH))}<br>shots: {html.escape(str(SHOTS_PATH))}<br>after renders: {html.escape(str(MOTION_OUT))}</p>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(text, encoding="utf-8")
    print(f"Wrote html: {HTML_OUT}")


if __name__ == "__main__":
    import fire

    fire.Fire({"generate": generate, "build_html": build_html})
