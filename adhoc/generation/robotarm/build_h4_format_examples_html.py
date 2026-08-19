#!/usr/bin/env python3
"""Build a short HTML of concrete H4 MCQ input/output format examples."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data/results/evaluation"
MODEL_DIR = EVAL / "h4_mcq_models"
OUT = REPO / "data/results/html/h4_mcq_format_examples.html"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _find(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    for row in rows:
        if ".reversed." in row["question_id"]:
            continue
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    raise KeyError(kwargs)


def _options_block(row: dict[str, Any]) -> str:
    return "\n".join(
        f"{option['id']}. {option['text']}" for option in row["options"]
    )


def _full_user_text(row: dict[str, Any]) -> str:
    instruction = row.get("answer_instruction", "Answer only A or B.")
    return f"{row['prompt']}\n\n{_options_block(row)}\n\n{instruction}"


def _card(
    *,
    title: str,
    subtitle: str,
    why: str,
    row: dict[str, Any],
    default_system: str,
) -> str:
    system = row.get("system_prompt") or default_system
    prediction = row.get("prediction")
    raw = row.get("raw_output", "")
    correct = bool(row.get("correct"))
    verdict = "CORRECT" if correct else "WRONG"
    verdict_class = "ok" if correct else "bad"
    mode = row.get("response_mode", "option")
    return f"""
<article class="card">
  <header>
    <div>
      <h2>{html.escape(title)}</h2>
      <p class="sub">{html.escape(subtitle)}</p>
    </div>
    <div class="meta">
      <span class="tag">{html.escape(row['cue'])}</span>
      <span class="tag">{html.escape(row['platform'])}</span>
      <span class="tag">{html.escape(row['attribute'])}</span>
      <span class="tag">mode={html.escape(str(mode))}</span>
    </div>
  </header>
  <p class="why"><strong>이 예시가 보여주는 것:</strong> {html.escape(why)}</p>
  <div class="grid">
    <section>
      <h3>INPUT — system</h3>
      <pre>{html.escape(system)}</pre>
      <h3>INPUT — user</h3>
      <pre>{html.escape(_full_user_text(row))}</pre>
    </section>
    <section>
      <h3>OUTPUT — model</h3>
      <div class="outbox">
        <div class="line"><span class="k">raw_output</span><code>{html.escape(str(raw))}</code></div>
        <div class="line"><span class="k">parsed</span><code>{html.escape(str(prediction))}</code></div>
        <div class="line"><span class="k">GT option</span><code>{html.escape(str(row['correct_option']))} = {html.escape(json.dumps(row['correct_value'], ensure_ascii=False))}</code></div>
        <div class="line"><span class="k">foil</span><code>{html.escape(json.dumps(row['foil_value'], ensure_ascii=False))}</code></div>
        <div class="line"><span class="k">verdict</span><span class="{verdict_class}">{verdict}</span></div>
        <div class="line"><span class="k">model</span><code>{html.escape(str(row.get('model', '')))}</code></div>
      </div>
      <h3>옵션 목록 (모델이 본 A/B)</h3>
      <pre>{html.escape(_options_block(row))}</pre>
    </section>
  </div>
</article>
"""


def main() -> None:
    cue_only = _find(
        _load_jsonl(MODEL_DIR / "h4_qwen_3b_predictions.jsonl"),
        cue="nod_yes",
        platform="humanoid",
        attribute="repetition",
    )
    blank = _find(
        _load_jsonl(MODEL_DIR / "h4blankorig_qwen_3b_predictions.jsonl"),
        cue="nod_yes",
        platform="humanoid",
        attribute="repetition",
    )
    glossary = _find(
        _load_jsonl(MODEL_DIR / "h4iter1_qwen_3b_predictions.jsonl"),
        experiment_id="glossary_value",
        cue="nod_yes",
        platform="humanoid",
        attribute="repetition",
    )
    targeted = _find(
        _load_jsonl(MODEL_DIR / "h4iter2_qwen_3b_predictions.jsonl"),
        experiment_id="targeted_ab",
        cue="nod_yes",
        platform="humanoid",
        attribute="repetition",
    )
    routed = _find(
        _load_jsonl(MODEL_DIR / "h4hold_qwen_3b_predictions.jsonl"),
        experiment_id="routed_filtered",
        cue="stretch_arms_up",
        platform="humanoid",
        attribute="direction",
    )
    routed_32b = _find(
        _load_jsonl(MODEL_DIR / "h4hold_qwen_32b_predictions.jsonl"),
        experiment_id="routed_filtered",
        cue="stretch_arms_up",
        platform="humanoid",
        attribute="direction",
    )

    default_system = (
        "You are an expert in robot gesture motion codes. "
        "Select the correct attribute value. Answer only A or B."
    )
    cards = [
        _card(
            title="Example 1. Cue-only baseline (초기 실험)",
            subtitle="전체 GT config 없음. cue 이름 + component 번호 + attribute 이름 + A/B만 전달.",
            why="이전 '빈칸 채우기'가 아니었던 형태. 모델이 볼 config가 없어서 pose/sign을 추측해야 했다.",
            row=cue_only,
            default_system=default_system,
        ),
        _card(
            title="Example 2. Blank-config baseline",
            subtitle="전체 movements JSON을 주고, 목표 field만 <BLANK>로 대체. A/B 선택.",
            why="진짜 빈칸 채우기 형태. nod_yes인데도 3B가 1을 고른 실패 예시.",
            row=blank,
            default_system=default_system,
        ),
        _card(
            title="Example 3. Iteration 1 — glossary_value",
            subtitle="blank-config + gesture meaning + motion-code glossary. 값(value) 출력 유도.",
            why="프롬프트에 '이 제스처가 무엇인지'와 convention 설명을 추가한 형태.",
            row=glossary,
            default_system=default_system,
        ),
        _card(
            title="Example 4. Iteration 2 — targeted_ab",
            subtitle="blank-config + gesture meaning + attribute별 guidance. A/B 응답.",
            why="repetition 필드에만 해당하는 힌트('반복 제스처면 1보다 큰 값')를 붙인 형태.",
            row=targeted,
            default_system=default_system,
        ),
        _card(
            title="Example 5. Final holdout — routed_filtered (3B)",
            subtitle="필터된 attribute만 평가. direction에는 meaning + field guidance + synthetic example.",
            why="최종 70점대 protocol. stretch_arms_up → direction=up 을 맞춘 성공 예시.",
            row=routed,
            default_system=default_system,
        ),
        _card(
            title="Example 5b. 같은 final prompt — 32B output",
            subtitle="입력은 Example 5와 동일. 모델만 Qwen2.5-VL-32B.",
            why="같은 입력에서 더 큰 모델의 raw output이 어떻게 다른지 비교.",
            row=routed_32b,
            default_system=default_system,
        ),
    ]

    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H4 MCQ Format Examples</title>
<style>
:root {{
  --bg: #f3f4f6; --panel: #ffffff; --ink: #172033; --muted: #667085;
  --line: #d0d5dd; --ok: #17643a; --bad: #a52a2a; --accent: #2457a6;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 28px 20px 64px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
.lead {{ color: var(--muted); max-width: 920px; margin: 0 0 22px; }}
.toc {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px; margin-bottom: 24px;
}}
.toc a {{
  display: block; background: var(--panel); border: 1px solid var(--line);
  padding: 12px; color: inherit; text-decoration: none;
}}
.toc a strong {{ display: block; margin-bottom: 4px; }}
.card {{
  background: var(--panel); border: 1px solid var(--line); margin-bottom: 22px;
}}
.card > header {{
  display: flex; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 16px 18px; border-bottom: 1px solid var(--line);
}}
.card h2 {{ margin: 0 0 4px; font-size: 18px; }}
.sub {{ margin: 0; color: var(--muted); }}
.meta {{ display: flex; gap: 6px; flex-wrap: wrap; align-content: flex-start; }}
.tag {{
  border: 1px solid var(--line); padding: 2px 7px; border-radius: 3px;
  color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 12px;
}}
.why {{ padding: 0 18px; color: var(--ink); }}
.grid {{ display: grid; grid-template-columns: 1.35fr 0.85fr; }}
.grid section {{ padding: 14px 18px 18px; min-width: 0; }}
.grid section + section {{ border-left: 1px solid var(--line); }}
h3 {{ margin: 14px 0 8px; font-size: 13px; color: var(--accent); text-transform: uppercase; letter-spacing: 0.04em; }}
pre {{
  margin: 0; white-space: pre-wrap; word-break: break-word;
  background: #f8fafc; border: 1px solid var(--line); padding: 12px;
  font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  max-height: 520px; overflow: auto;
}}
.outbox {{ border: 1px solid var(--line); background: #f8fafc; padding: 10px 12px; }}
.line {{ display: grid; grid-template-columns: 110px 1fr; gap: 10px; padding: 6px 0; border-top: 1px solid var(--line); }}
.line:first-child {{ border-top: 0; }}
.k {{ color: var(--muted); }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
.ok {{ color: var(--ok); font-weight: 700; }}
.bad {{ color: var(--bad); font-weight: 700; }}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: 1fr; }}
  .grid section + section {{ border-left: 0; border-top: 1px solid var(--line); }}
}}
</style>
</head>
<body>
<div class="wrap">
  <h1>H4 Binary MCQ — 구체적인 Input / Output 예시</h1>
  <p class="lead">
    같은 종류의 질문을 실험 단계별로 하나씩 보여준다. 왼쪽이 모델에 실제로 들어간 텍스트이고,
    오른쪽이 모델 raw output과 채점 결과다. A/B reversed 짝은 생략하고 original 쪽만 실었다.
  </p>
  <div class="toc">
    <a href="#"><strong>1. Cue-only</strong>config 없이 cue+attribute만</a>
    <a href="#"><strong>2. Blank-config</strong>전체 GT + &lt;BLANK&gt;</a>
    <a href="#"><strong>3. Glossary</strong>meaning + convention</a>
    <a href="#"><strong>4. Targeted</strong>attribute별 guidance</a>
    <a href="#"><strong>5. Final routed</strong>70점대 protocol</a>
  </div>
  {''.join(cards)}
</div>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
