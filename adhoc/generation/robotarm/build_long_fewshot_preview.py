#!/usr/bin/env python3
"""Render Long 1-4 few-shot examples and build a review HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from embodiment_renderers import render_roomba  # noqa: E402

_DEFAULT_SHOT = (
    _REPO
    / "data/seed/shots/embodiment_long/roomba_rub_eye_tired_long_1_4.json"
)
_DEFAULT_RENDER = _REPO / "data/results/render/embodiment_long_fewshot"
_DEFAULT_HTML = _REPO / "data/results/html/embodiment_long_fewshot.html"


def _row(idx: int, cue: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "idx": idx,
        "cue": cue,
        "description": cue,
        "movements": config["movements"],
        "source_code": config.get("source_code", ""),
        "platform": "roomba",
    }


def _component_listing(example: dict[str, Any]) -> str:
    movements = example["configs"]["result"]["movements"]
    start = example["main_component_range"]["start"]
    end = example["main_component_range"]["end"]
    labels = {
        item["component_index"]: item["position"].replace("_core", "").upper()
        for item in example["added_submotions"]
    }
    lines = []
    for index, movement in enumerate(movements, start=1):
        role = "CORE" if start <= index <= end else labels.get(index, "CONTEXT")
        lines.append(
            f"{index}. [{role}] "
            + json.dumps(movement, ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(lines)


def build_preview(
    shot_path: Path, render_root: Path, out_html: Path, *, force: bool = False
) -> Path:
    data = json.loads(shot_path.read_text(encoding="utf-8"))
    examples = data["examples"]
    original = examples[0]["configs"]["original"]
    rows = [_row(0, "rub_eye_tired_original", original)]
    rows.extend(
        _row(i, f"rub_eye_tired_{ex['level']}", ex["configs"]["result"])
        for i, ex in enumerate(examples, start=1)
    )
    gif_dir = render_root / "roomba" / "gif"
    render_roomba(rows, gif_dir, force=force)

    cards = []
    all_items: list[tuple[str, dict[str, Any] | None, dict[str, Any]]] = [
        ("Original / Short core", None, original)
    ]
    all_items.extend(
        (str(ex["level"]).upper(), ex, ex["configs"]["result"]) for ex in examples
    )
    for idx, (label, example, config) in enumerate(all_items):
        cue = "rub_eye_tired_original" if idx == 0 else f"rub_eye_tired_long{idx}"
        gif = gif_dir / f"emb_g{idx:02d}_{cue}.gif"
        rel = os.path.relpath(gif, out_html.parent)
        if example is None:
            description = "Unchanged one-component core used as the independent starting point for every Long level."
            target = "Added context: 0 · Core range: 1–1"
            listing = "1. [CORE] " + json.dumps(
                config["movements"][0], ensure_ascii=False, sort_keys=True
            )
            added = "<li>None</li>"
        else:
            description = example["description"]
            target_cfg = example["added_submotion_target"]
            target = (
                f"Added context target: {target_cfg['min']}–{target_cfg['max']} "
                f"({target_cfg['policy']}) · Core range: "
                f"{example['main_component_range']['start']}–"
                f"{example['main_component_range']['end']}"
            )
            listing = _component_listing(example)
            added = "".join(
                "<li>"
                f"#{item['component_index']} · {html.escape(item['position'])}: "
                f"{html.escape(item['description'])}"
                "</li>"
                for item in example["added_submotions"]
            )
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(label)}</h2>
              <p class="meta">{html.escape(target)}</p>
              <img src="{html.escape(rel)}" loading="lazy" alt="{html.escape(label)} render">
              <p>{html.escape(description)}</p>
              <h3>Added submotions</h3><ol>{added}</ol>
              <h3>Component roles</h3><pre>{html.escape(listing)}</pre>
              <h3>Motion Code</h3><pre>{html.escape(config.get("source_code", ""))}</pre>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Long 1–4 few-shot review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; padding: 24px; background: #111318; color: #e8eaf0; }}
    header {{ max-width: 1100px; margin: 0 auto 20px; }}
    .note {{ color: #b8becb; line-height: 1.5; }}
    main {{ max-width: 1100px; margin: auto; display: grid; grid-template-columns: repeat(auto-fit,minmax(330px,1fr)); gap: 16px; }}
    .card {{ border: 1px solid #353a46; border-radius: 10px; padding: 16px; background: #191c23; }}
    img {{ width: 100%; border-radius: 8px; background: #fff; }}
    h1,h2,h3 {{ margin-top: 0; }}
    h3 {{ margin-top: 18px; font-size: 14px; }}
    .meta {{ color: #9fb7ff; font-size: 13px; }}
    p,li {{ line-height: 1.45; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 10px; border-radius: 6px; background: #0f1116; font-size: 11px; }}
  </style>
</head>
<body>
  <header>
    <h1>Long 1–4 few-shot · Roomba / rub_eye_tired</h1>
    <p class="note">Every condition starts independently from the same Original core. Long 1 is the corrected natural contextual baseline; Long 2–4 use controlled added-submotion ranges 1–2, 3–4, and 5–6.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shot", type=Path, default=_DEFAULT_SHOT)
    parser.add_argument("--render-root", type=Path, default=_DEFAULT_RENDER)
    parser.add_argument("--html", type=Path, default=_DEFAULT_HTML)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(build_preview(args.shot, args.render_root, args.html, force=args.force))


if __name__ == "__main__":
    main()
