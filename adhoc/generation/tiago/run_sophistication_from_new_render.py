#!/usr/bin/env python3
"""Run a one-shot sophistication experiment from original new_render GIF."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _ROBOTARM, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from correction_lib import (  # noqa: E402
    _ensure_mp4,
    _gif_first_frame_png,
    cue_description,
    default_config_path,
    load_config_list,
    render_instance,
    write_config_snapshot,
)
from render_paths import render_gif_path, render_exp_dir, render_media_paths  # noqa: E402
from correction_prompts import (  # noqa: E402
    build_inner_prompt_google,
    format_prior_attempts_block,
    row_from_parsed_config,
)

DEFAULT_MODEL_TAG = "gemini-2.5-pro"
DEFAULT_EXP_ID = "1"
DEFAULT_INNER_EXP_ID = "0"
_WRAPPER_PATH = _REPO / "data/seed/prompt/_shared_sophistication_wrapper.txt"

_Q1_RE = re.compile(r"^\s*#\s*Sophistication\s*Q1\s*:\s*(.+)$", re.I | re.M)
_Q2_RE = re.compile(r"^\s*#\s*Sophistication\s*Q2\s*:\s*(.+)$", re.I | re.M)


def _generation_gif(model_tag: str, exp_id: str, cue: str, idx: int) -> Path:
    return render_gif_path("tiago", model_tag, exp_id, idx, cue, "gen")


def _record_dir(model_tag: str, exp_id: str, cue: str) -> Path:
    return (
        _REPO
        / "data/results/verify/tiago/sophistication"
        / f"exp{exp_id}"
        / model_tag
        / cue
    )


def _record_path(model_tag: str, exp_id: str, cue: str) -> Path:
    return _record_dir(model_tag, exp_id, cue) / "orig_new_render_sophistication.json"


def _snapshot_path(model_tag: str, exp_id: str, cue: str) -> Path:
    return (
        _REPO
        / "data/results/motion_configs/tiago/exp/sophistication"
        / model_tag
        / f"exp{exp_id}"
        / cue
        / "orig_new_render_sophistication_config.json"
    )


def _render_dir(model_tag: str, exp_id: str, cue: str) -> Path:
    del cue
    return render_exp_dir("tiago", model_tag, exp_id)


def _html_path(model_tag: str, exp_id: str, cue: str, idx: int) -> Path:
    return (
        _REPO
        / "data/results/visualize/tiago/sophistication"
        / f"exp{exp_id}"
        / model_tag
        / f"g{idx:02d}_new_render_sophistication.html"
    )


def _log_path(model_tag: str, exp_id: str) -> Path:
    return (
        _REPO
        / "data/results/verify/tiago/sophistication"
        / f"exp{exp_id}"
        / model_tag
        / "sophistication_from_new_render_log.jsonl"
    )


def _media_kind() -> str:
    return "animated GIF of the full rendered motion sequence"


def _load_wrapper_template() -> str:
    return _WRAPPER_PATH.read_text(encoding="utf-8")


def _build_prompt(
    *,
    cue: str,
    row: dict[str, Any],
    description: str,
    backend: str = "gemini",
    inner_exp_id: str = DEFAULT_INNER_EXP_ID,
) -> str:
    inner = build_inner_prompt_google(
        inner_exp_id,  # type: ignore[arg-type]
        cue=cue,
        row=row,
        description=description,
        backend=backend,
    )
    wrapper = _load_wrapper_template()
    attempts = [("Prior attempt (original exp1 config / new_render input)", row)]
    return (
        wrapper.replace("{{MEDIA_KIND}}", _media_kind())
        .replace("{{CUMULATIVE_MEDIA_NOTE}}", "")
        .replace("{{INNER_PROMPT}}", inner.strip())
        .replace("{{CUE}}", cue)
        .replace("{{DESCRIPTION}}", description)
        .replace("{{PRIOR_ATTEMPTS_BLOCK}}", format_prior_attempts_block(attempts))
    )


def _is_q_prompt_echo(which: int, ans: str) -> bool:
    low = ans.lower().strip()
    if not low:
        return True
    if which == 1:
        return (
            "what single auxiliary motion" in low
            or ("would most strengthen this cue" in low and "state where it occurs" in low)
        )
    return (
        "how should the full `movements` sequence change" in low
        or ("incorporate that auxiliary motion" in low and "write \"n/a\"" in low)
    )


def _extract_q_line(raw_text: str, which: int) -> str | None:
    pat = _Q1_RE if which == 1 else _Q2_RE
    lines = (raw_text or "").splitlines()
    candidates: list[str] = []
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            continue
        ans = m.group(1).strip()
        if _is_q_prompt_echo(which, ans):
            for j in range(i + 1, min(i + 4, len(lines))):
                nxt = lines[j].strip()
                if not nxt.startswith("#"):
                    continue
                alt = nxt.lstrip("#").strip()
                if alt and not _is_q_prompt_echo(which, alt):
                    candidates.append(alt)
                    break
            continue
        candidates.append(ans)
    return candidates[0] if candidates else None


def _extract_reasoning_and_json(raw_text: str) -> tuple[str, dict[str, Any]]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    reasoning = text[: m.start()].strip()
    return reasoning, obj


def _generate_with_gemini(*, prompt: str, gif_path: Path, model: str) -> str:
    from google import genai
    from google.genai import types

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=gif_path.read_bytes(), mime_type="image/gif"),
            prompt,
        ],
    )
    return (resp.text or "").strip()


def _rel(out_html: Path, path: Path) -> str:
    if not path.is_file():
        return ""
    rel = os.path.relpath(path, out_html.parent).replace("\\", "/")
    return html.escape(f"{rel}?v={int(path.stat().st_mtime)}")


def _config_html(row: dict[str, Any] | None, *, summary: str = "Config") -> str:
    if not row:
        return '<div class="cfg-missing">Config missing</div>'
    payload = {
        "idx": row.get("idx"),
        "cue": row.get("cue"),
        "description": row.get("description"),
        "movements": row.get("movements"),
        "state": row.get("state"),
        "sophistication_q1": row.get("sophistication_q1"),
        "sophistication_q2": row.get("sophistication_q2"),
    }
    return (
        f'<details class="cfg-toggle"><summary>{html.escape(summary)}</summary>'
        f"<pre>{html.escape(json.dumps(payload, indent=2, ensure_ascii=False))}</pre></details>"
    )


def _write_review_html(
    *,
    out_html: Path,
    cue: str,
    idx: int,
    prompt: str,
    raw_text: str,
    input_row: dict[str, Any],
    output_row: dict[str, Any],
    orig_gif: Path,
    orig_mp4: Path,
    input_gif: Path,
    input_mp4: Path,
    output_gif: Path,
    output_mp4: Path,
    q1: str | None,
    q2: str | None,
) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)

    def _fig(title: str, gif: Path, mp4: Path, row: dict[str, Any]) -> str:
        gif_r = _rel(out_html, gif)
        mp4_r = _rel(out_html, mp4)
        body = (
            f'<img src="{gif_r}" loading="lazy" alt="{html.escape(title)}">'
            if gif_r
            else '<div class="missing">GIF missing</div>'
        )
        if mp4_r:
            body += f'<p class="mp4"><a href="{mp4_r}">MP4</a></p>'
        body += _config_html(row)
        return f"<figure><figcaption>{html.escape(title)}</figcaption>{body}</figure>"

    page = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>g{idx:02d} sophistication — {html.escape(cue)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f6f7fb; color: #1a1a1a; }}
h1 {{ margin-bottom: 8px; }}
.meta {{ color: #555; margin-bottom: 14px; line-height: 1.45; }}
.media-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
@media (max-width: 1100px) {{ .media-grid {{ grid-template-columns: 1fr; }} }}
figure {{ margin: 0; background: #fff; border: 1px solid #dde3ee; border-radius: 10px; padding: 12px; }}
figcaption {{ font-size: 0.85rem; color: #666; margin-bottom: 6px; font-weight: 600; }}
img {{ width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }}
.mp4 {{ font-size: 0.75rem; margin: 4px 0 0; }}
.cfg-toggle {{ margin-top: 8px; font-size: 0.8rem; }}
.cfg-toggle summary {{ cursor: pointer; color: #444; font-weight: 600; user-select: none; }}
.cfg-toggle pre {{
  margin: 6px 0 0; padding: 8px; background: #f7f8fc; border: 1px solid #e8ecf4;
  border-radius: 6px; font-size: 0.72rem; line-height: 1.35; white-space: pre-wrap;
  word-break: break-word; max-height: 300px; overflow: auto;
}}
.cfg-missing {{ color: #999; font-size: 0.78rem; margin-top: 6px; }}
.block {{ margin-top: 16px; background: #fff; border: 1px solid #dde3ee; border-radius: 10px; padding: 14px; }}
.block h2 {{ margin: 0 0 8px; font-size: 0.95rem; }}
pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 0.78rem; line-height: 1.45; }}
.missing {{ color: #c00; font-size: 0.85rem; }}
</style></head>
<body>
<h1>g{idx:02d} <code>{html.escape(cue)}</code> — new_render sophistication</h1>
<div class="meta">
  <b>Input to VLM:</b> original exp1 config rendered with the new renderer.<br>
  <b>Experiment goal:</b> keep the core cue, and add a supporting / auxiliary motion that makes it read more iconically.
</div>
<div class="media-grid">
  {_fig("Original legacy render", orig_gif, orig_mp4, input_row)}
  {_fig("New render input (original config)", input_gif, input_mp4, input_row)}
  {_fig("Sophistication output", output_gif, output_mp4, output_row)}
</div>
<div class="block"><h2>Sophistication Q1</h2><pre>{html.escape(q1 or "")}</pre></div>
<div class="block"><h2>Sophistication Q2</h2><pre>{html.escape(q2 or "")}</pre></div>
<div class="block"><h2>Prompt</h2><pre>{html.escape(prompt)}</pre></div>
<div class="block"><h2>Raw Model Output</h2><pre>{html.escape(raw_text)}</pre></div>
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")
    return out_html


def run(
    *,
    cue: str,
    model_tag: str = DEFAULT_MODEL_TAG,
    exp_id: str = DEFAULT_EXP_ID,
    inner_exp_id: str = DEFAULT_INNER_EXP_ID,
    force: bool = False,
) -> dict[str, Any]:
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")

    cfg_path = default_config_path("tiago", model_tag, exp_id)  # type: ignore[arg-type]
    rows = load_config_list(cfg_path)
    input_row = next((r for r in rows if str(r.get("cue", "")) == cue), None)
    if input_row is None:
        raise SystemExit(f"cue not found: {cue}")

    idx = int(input_row.get("idx", -1))
    description = cue_description("tiago", input_row)
    input_gif = _generation_gif(model_tag, exp_id, cue, idx)
    if not input_gif.is_file():
        raise SystemExit(f"generation input missing: {input_gif}")
    input_mp4 = input_gif.with_suffix(".mp4")
    if not input_mp4.is_file():
        input_mp4 = _ensure_mp4(input_gif, input_mp4) or input_mp4

    record_path = _record_path(model_tag, exp_id, cue)
    snapshot_path = _snapshot_path(model_tag, exp_id, cue)
    render_dir = _render_dir(model_tag, exp_id, cue)
    html_path = _html_path(model_tag, exp_id, cue, idx)
    if force:
        for p in (record_path, snapshot_path, html_path):
            if p.is_file():
                p.unlink()
        if render_dir.is_dir():
            shutil.rmtree(render_dir)

    prompt = _build_prompt(
        cue=cue,
        row=input_row,
        description=description,
        backend="gemini",
        inner_exp_id=inner_exp_id,
    )

    t0 = time.time()
    raw_text = _generate_with_gemini(prompt=prompt, gif_path=input_gif, model=model_tag)
    reasoning, config_parsed = _extract_reasoning_and_json(raw_text)
    q1 = _extract_q_line(raw_text, 1)
    q2 = _extract_q_line(raw_text, 2)

    if not config_parsed.get("movements"):
        raise SystemExit("model output JSON has no movements")

    output_row = row_from_parsed_config(
        "tiago",
        exp_id,  # type: ignore[arg-type]
        input_row,
        config_parsed,
        model=model_tag,
    )
    output_row["state"] = "exp1_pose_sophistication"
    output_row["sophistication_q1"] = q1
    output_row["sophistication_q2"] = q2
    if reasoning:
        output_row["reasoning"] = reasoning

    write_config_snapshot(snapshot_path, output_row)
    media = render_instance(
        "tiago",
        output_row,
        config_path=snapshot_path,
        render_dir=render_dir,
        model_tag=model_tag,
        exp_id=exp_id,  # type: ignore[arg-type]
        render_role="soph",
    )

    orig_gif, orig_mp4, _ = render_media_paths("tiago", model_tag, exp_id, idx, cue, "gen")
    _write_review_html(
        out_html=html_path,
        cue=cue,
        idx=idx,
        prompt=prompt,
        raw_text=raw_text,
        input_row=input_row,
        output_row=output_row,
        orig_gif=orig_gif,
        orig_mp4=orig_mp4,
        input_gif=input_gif,
        input_mp4=input_mp4,
        output_gif=media.gif or Path(),
        output_mp4=media.mp4 or Path(),
        q1=q1,
        q2=q2,
    )

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cue": cue,
        "idx": idx,
        "model_tag": model_tag,
        "exp_id": exp_id,
        "inner_exp_id": inner_exp_id,
        "prompt_path": str(_WRAPPER_PATH),
        "base_config_json": str(cfg_path),
        "input_new_render_gif": str(input_gif),
        "config_snapshot_json": str(snapshot_path),
        "html": str(html_path),
        "elapsed_s": round(time.time() - t0, 1),
        "sophistication_q1": q1,
        "sophistication_q2": q2,
        "raw_text": raw_text,
        "output_row": output_row,
        "rendered_media": {
            "gif": str(media.gif) if media.gif else None,
            "mp4": str(media.mp4) if media.mp4 else None,
            "pose_png": str(media.pose_png) if media.pose_png else None,
        },
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    log = _log_path(model_tag, exp_id)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "cue": cue,
                    "idx": idx,
                    "elapsed_s": payload["elapsed_s"],
                    "input_new_render_gif": str(input_gif),
                    "sophistication_gif": str(media.gif) if media.gif else None,
                    "record_json": str(record_path),
                    "html": str(html_path),
                    "sophistication_q1": q1,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cue", required=True)
    ap.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    ap.add_argument("--exp", default=DEFAULT_EXP_ID)
    ap.add_argument("--inner-exp", default=DEFAULT_INNER_EXP_ID)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run(
        cue=args.cue,
        model_tag=args.model_tag,
        exp_id=args.exp,
        inner_exp_id=args.inner_exp,
        force=args.force,
    )


if __name__ == "__main__":
    main()
