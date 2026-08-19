#!/usr/bin/env python3
"""Build English blank-config MCQ with per-embodiment Motion Code schema YAML.

Replaces the short shared CONVENTIONS glossary with the full platform block from
data/seed/yml/motion_code_embodiment.yaml (anchors resolved).
"""
from __future__ import annotations

import copy
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from build_h4_prompt_iterations import DESCRIPTIONS

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / "data/results/evaluation"
SCHEMA_YAML = REPO / "data/seed/yml/motion_code_embodiment.yaml"
ORIGINAL = EVAL / "h4_gt_blank_config_mcq.json"
REVERSED = EVAL / "h4_gt_blank_config_mcq_counterbalanced.json"
OUT = EVAL / "h4_glossary_value_english.json"
OUT_REV = EVAL / "h4_glossary_value_english_counterbalanced.json"
OUT_HTML = REPO / "data/results/html/h4_blank_schema_yaml_examples.html"

SYSTEM = (
    "You are an expert in embodied robot gesture motion codes. "
    "Recover the missing value from cue semantics, the embodiment schema, "
    "and the full motion config context."
)
ANSWER = (
    "Output only the exact selected candidate VALUE, with no A/B label and no explanation."
)
# Stretch replaces quadruped in the blank/binary paper eval set.
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")


@lru_cache(maxsize=1)
def _schema_root() -> dict[str, Any]:
    return yaml.safe_load(SCHEMA_YAML.read_text(encoding="utf-8"))


def embodiment_schema_block(platform: str) -> str:
    """Return the full resolved YAML block for one embodiment."""
    root = _schema_root()
    if platform not in root:
        raise KeyError(f"Unknown platform in schema YAML: {platform}")
    payload = {platform: root[platform]}
    dumped = yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )
    return (
        f"[EMBODIMENT SCHEMA — {platform}]\n"
        f"# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} section, anchors resolved)\n"
        f"{dumped.rstrip()}"
    )


def _english_body(row: dict[str, Any]) -> str:
    return (
        f"The full Human GT motion config for gesture '{row['cue']}' on the "
        f"{row['platform']} embodiment is shown below. One field, "
        f"'{row['attribute']}' of the {row['component_number']}th "
        f"{row['component_type']} component, has been replaced with a blank token.\n"
        "Use the gesture meaning, the embodiment schema, and the full config context "
        "to choose the exact attribute value that belongs in the blank from the two "
        "candidates.\n\n"
        "[FULL GT MOVEMENTS CONFIG]\n"
        + _extract_config_json(row["prompt"])
    )


def _extract_config_json(prompt: str) -> str:
    marker = "[FULL GT MOVEMENTS CONFIG]\n"
    if marker not in prompt:
        raise ValueError("missing config block")
    return prompt.split(marker, 1)[1].strip()


def _options_block(row: dict[str, Any]) -> str:
    return "\n".join(f"{option['id']}. {option['text']}" for option in row["options"])


def _full_user_text(row: dict[str, Any]) -> str:
    return (
        f"{row['prompt']}\n\n{_options_block(row)}\n\n"
        f"{row.get('answer_instruction', ANSWER)}"
    )


def _glossary_row(source: dict[str, Any], *, reversed_pair: bool) -> dict[str, Any]:
    row = copy.deepcopy(source)
    base_id = re.sub(r"\.reversed$", "", row["question_id"])
    if not base_id.endswith(".glossary_en"):
        base_id = f"{base_id}.glossary_en"
    if reversed_pair:
        row["question_id"] = f"{base_id}.reversed"
        row["paired_question_id"] = base_id
    else:
        row["question_id"] = base_id
        row.pop("paired_question_id", None)
    row["experiment_id"] = "schema_yaml_en"
    row["system_prompt"] = SYSTEM
    row["response_mode"] = "value"
    row["answer_instruction"] = ANSWER
    row["schema_source"] = str(SCHEMA_YAML.relative_to(REPO))
    row["prompt"] = (
        f"[GESTURE MEANING]\n{DESCRIPTIONS[row['cue']]}\n\n"
        f"{embodiment_schema_block(str(row['platform']))}\n\n"
        f"{_english_body(row)}"
    )
    if re.search(r"[가-힣]", row["prompt"]):
        raise RuntimeError(f"Korean remains in {row['question_id']}")
    if "[CONVENTIONS]" in row["prompt"]:
        raise RuntimeError(f"Legacy CONVENTIONS block still present in {row['question_id']}")
    return row


def _payload(questions: list[dict[str, Any]], *, reversed_set: bool) -> dict[str, Any]:
    return {
        "definition": {
            "format": (
                "english blank-config MCQ with full per-embodiment "
                "motion_code_embodiment.yaml schema (no shared CONVENTIONS glossary)"
            ),
            "option_order": (
                "exact A/B reversal of h4_glossary_value_english.json"
                if reversed_set
                else "same as h4_gt_blank_config_mcq.json"
            ),
            "language": "en",
            "variant": "schema_yaml_en",
            "schema_yaml": str(SCHEMA_YAML.relative_to(REPO)),
        },
        "summary": {
            "questions": len(questions),
            "by_platform": dict(Counter(row["platform"] for row in questions)),
            "by_attribute": dict(Counter(row["attribute"] for row in questions)),
        },
        "questions": questions,
    }


def _pick_examples(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One example per platform, preferring nod_yes / a common attribute."""
    picked: list[dict[str, Any]] = []
    for platform in PLATFORMS:
        candidates = [q for q in questions if q["platform"] == platform]
        preferred = [
            q
            for q in candidates
            if q["cue"] == "nod_yes" and q["attribute"] in ("repetition", "gripper_orientation", "direction")
        ]
        pool = preferred or candidates
        if not pool:
            continue
        picked.append(pool[0])
    return picked


def _write_html(examples: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for row in examples:
        schema_chars = len(embodiment_schema_block(str(row["platform"])))
        prompt_chars = len(row["prompt"])
        cards.append(
            f"""
<article class="card">
  <header>
    <h2>{_esc(row['platform'])} — {_esc(row['cue'])} / {_esc(row['attribute'])}</h2>
    <div class="meta">
      <span class="tag">{_esc(row['question_id'])}</span>
      <span class="tag">schema≈{schema_chars:,} chars</span>
      <span class="tag">prompt≈{prompt_chars:,} chars</span>
      <span class="tag">GT={_esc(json.dumps(row['correct_value'], ensure_ascii=False))}</span>
      <span class="tag">foil={_esc(json.dumps(row['foil_value'], ensure_ascii=False))}</span>
    </div>
  </header>
  <section>
    <h3>INPUT — system</h3>
    <pre>{_esc(row['system_prompt'])}</pre>
    <h3>INPUT — user</h3>
    <pre>{_esc(_full_user_text(row))}</pre>
  </section>
</article>
"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H4 Blank-Config + Embodiment Schema YAML — Example Inputs</title>
<style>
:root {{
  --bg: #f4f1ea; --panel: #fffdf8; --ink: #1c1917; --muted: #57534e;
  --line: #d6d3d1; --accent: #0f766e;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  background: radial-gradient(circle at top left, #efe6d5, var(--bg));
  color: var(--ink); line-height: 1.45;
}}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px; }}
h1 {{ font-size: 1.6rem; margin: 0 0 8px; }}
.lead {{ color: var(--muted); margin: 0 0 28px; max-width: 70ch; }}
.card {{
  background: var(--panel); border: 1px solid var(--line); border-radius: 14px;
  padding: 18px 18px 8px; margin: 0 0 22px;
  box-shadow: 0 10px 30px rgba(28,25,23,.04);
}}
.card h2 {{ margin: 0 0 8px; font-size: 1.15rem; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
.tag {{
  display: inline-block; background: #ecfdf5; color: var(--accent);
  border: 1px solid #99f6e4; border-radius: 999px; padding: 2px 10px;
  font-size: .78rem;
}}
h3 {{ margin: 16px 0 6px; font-size: .95rem; color: var(--accent); }}
pre {{
  white-space: pre-wrap; word-break: break-word;
  background: #1c1917; color: #f5f5f4; border-radius: 10px;
  padding: 14px; font-size: 12px; line-height: 1.4;
  max-height: 520px; overflow: auto;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
}}
</style>
</head>
<body>
<main>
  <h1>H4 blank-config fill-in — example inputs</h1>
  <p class="lead">
    Protocol: English blank-config MCQ. Shared <code>CONVENTIONS</code> glossary removed.
    Each prompt includes the full per-embodiment block from
    <code>data/seed/yml/motion_code_embodiment.yaml</code> (anchors resolved),
    plus gesture meaning and the full GT movements JSON with one <code>&lt;BLANK&gt;</code>.
    One example per embodiment below.
  </p>
  {''.join(cards)}
</main>
</body>
</html>
"""
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(page, encoding="utf-8")


def _esc(text: str) -> str:
    import html

    return html.escape(text)


def main() -> None:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))["questions"]
    reversed_rows = json.loads(REVERSED.read_text(encoding="utf-8"))["questions"]
    out_rows = [_glossary_row(row, reversed_pair=False) for row in original]
    rev_rows = [_glossary_row(row, reversed_pair=True) for row in reversed_rows]
    OUT.write_text(
        json.dumps(_payload(out_rows, reversed_set=False), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    OUT_REV.write_text(
        json.dumps(_payload(rev_rows, reversed_set=True), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    examples = _pick_examples(out_rows)
    _write_html(examples)
    lengths = {
        platform: len(embodiment_schema_block(platform))
        for platform in PLATFORMS
    }
    print(json.dumps(_payload(out_rows, reversed_set=False)["summary"], indent=2))
    print("schema_chars_by_platform:", json.dumps(lengths, indent=2))
    print("sample_prompt_chars:", len(out_rows[0]["prompt"]))
    print(OUT)
    print(OUT_REV)
    print(OUT_HTML)
    print("--- sample prompt head ---")
    print(out_rows[0]["prompt"][:900])


if __name__ == "__main__":
    main()
