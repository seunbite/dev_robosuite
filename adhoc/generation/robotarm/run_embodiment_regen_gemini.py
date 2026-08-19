#!/usr/bin/env python3
"""Regenerate Motion Codes for Gemini-made videos that humans judged NO.

Input per cell: MP4 + current Motion Code + cue + embodiment YAML schema
(same yaml_cue block used at first generation).

Asks Gemini whether / how to revise the Motion Code so the cue becomes
identifiable, then parses → JSON → render and analyzes edit patterns.

Example::

    source APIKEY.sh
    python adhoc/generation/robotarm/run_embodiment_regen_gemini.py \\
      --model gemini-2.5-pro --workers 4
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    embodiment_schema_yaml,
    extract_motion_code,
    gif_path_for,
    render_pending_once,
    _extract_json_obj,
)

HUMAN_NO_JSON = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "cueid_human_binary_acc_by_platform_gemini_made.json"
)
GEMINI_MASTER = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)
RENDER_SRC = (
    _REPO / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro"
)
# Preferred visual input for regen (bank + EE translate). Falls back to RENDER_SRC.
RENDER_SRC_EE_TRANSLATE = (
    _REPO
    / "data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro_ee_translate"
)
RENDER_GT = _REPO / "data/results/render/embodiment_motion_code_gt"
GT_MASTER = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
CUES_JSON = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
DEFAULT_OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gemini25pro_regen_human_no"
)

_STEP_RE = re.compile(
    r"\b(pose|path|move|hold|movement)\s*\(", re.IGNORECASE
)
_ATTR_RE = re.compile(
    r"\b(dir|grip|gripper|z|y|x|spd|speed|hold|rep|axis|dist|degrees|r|shape|mode|torso|elbow|head|base)\b",
    re.IGNORECASE,
)


def _client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
    return genai.Client(api_key=key)


def _load_human_no() -> list[dict[str, Any]]:
    data = json.loads(HUMAN_NO_JSON.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for plat, block in (data.get("yesno") or {}).get("by_platform", {}).items():
        for w in block.get("wrong_details") or []:
            rows.append(
                {
                    "platform": plat,
                    "idx": int(w["idx"]),
                    "human_note": str(w.get("note") or ""),
                }
            )
    rows.sort(key=lambda r: (r["platform"], r["idx"]))
    return rows


def _index_master(master: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in master:
        idx = int(entry["idx"])
        cue = str(entry["cue"])
        desc = str(entry.get("description") or "")
        for plat, cell in (entry.get("platforms") or {}).items():
            out[(idx, plat)] = {
                "idx": idx,
                "cue": cue,
                "description": desc,
                "cell": cell,
                "entry": entry,
            }
    return out


def _mp4_path(platform: str, idx: int, cue: str) -> Path:
    # On-disk layout: single_arm → manipulator/mp4/IIWA/...
    if platform == "single_arm":
        return (
            RENDER_SRC
            / "manipulator"
            / "mp4"
            / "IIWA"
            / f"emb_g{idx:02d}_{cue}.mp4"
        )
    return RENDER_SRC / platform / "mp4" / f"emb_g{idx:02d}_{cue}.mp4"


def build_regen_prompt(
    *,
    platform: str,
    cue: str,
    description: str,
    motion_code: str,
    strategy: str = "default",
) -> str:
    schema = embodiment_schema_yaml(platform)
    if strategy == "cot_dynamic":
        return f"""You are reviewing a rendered robot gesture that a human judge marked as NOT identifiable as the target cue.

You are given:
1) the rendered video (attached) — the current Gemini-made motion,
2) the Motion Code that produced it,
3) the target cue,
4) the embodiment schema YAML (same schema used at generation time).

Task:
- Revise the Motion Code so the cue becomes identifiable on this embodiment.
- Stay inside the schema. Every Motion Code MUST start with pose(...).
- Total steps: 2~8. Join steps with +.

Before writing Motion Code, reason about embodiment constraints:

[Q1 — Dynamic meaning plan]
Even if this embodiment lacks the ideal body parts for the cue, consider how you
can still convey the cue's meaning by stacking dynamic movements (paths/moves/holds
with timing, repetition, direction, and spatial emphasis) that this body *can* do.
Look at the attached video and current code: what is missing for the cue, and how
stacked dynamics could fix it. Write ONE short prose sentence (not Motion Code).

[Q2 — Revised Motion Code]
Implement that plan as ONE complete revised Motion Code line.

[CUE]
name: {cue}
description: {description}

[CURRENT MOTION CODE]
{motion_code}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Write exactly these two lines (and nothing else):
# Q1: <one sentence: how stacked dynamic movement conveys the cue on this body>
# Q2: <pose(...)+... revised Motion Code only>
Do not invent attributes or component forms outside the embodiment schema above.
""".strip()

    if strategy == "cot_contrast":
        return f"""You are reviewing a rendered robot gesture that a human judge marked as NOT identifiable as the target cue.

You are given:
1) the rendered video (attached) — the current Gemini-made motion,
2) the Motion Code that produced it,
3) the target cue,
4) the embodiment schema YAML (same schema used at generation time).

Task:
- First read what the current video looks like.
- Then revise the Motion Code so the motion reads as the TARGET cue on this embodiment.
- Stay inside the schema. Every Motion Code MUST start with pose(...).
- Total steps: 2~8. Join steps with +.

[Q1 — Perceived cue]
Watch the attached video (you may also use the current Motion Code as a hint).
Which cue does this motion currently look like?
Answer with a short cue name only (snake_case if possible).

[Q2 — Contrastive edit plan]
The TARGET cue is given below.
- If your Q1 cue is different from the TARGET cue: explain in ONE sentence how to change the Motion Code so the result does NOT look like the Q1 cue, but DOES look like the TARGET cue.
- If your Q1 cue already matches the TARGET cue: explain in ONE sentence how to strengthen the TARGET reading (make it clearer / less ambiguous).

[Q3 — Revised Motion Code]
Implement that plan as ONE complete revised Motion Code line.

[TARGET CUE]
name: {cue}
description: {description}

[CURRENT MOTION CODE]
{motion_code}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Write exactly these three lines (and nothing else):
# Q1: <perceived_cue_name>
# Q2: <one-sentence contrastive edit plan>
# Q3: <pose(...)+... revised Motion Code only>
Do not invent attributes or component forms outside the embodiment schema above.
""".strip()

    if strategy == "cot_contrast_nocode":
        return f"""You are reviewing a rendered robot gesture that a human judge marked as NOT identifiable as the target cue.

You are given:
1) the rendered video (attached) — the current Gemini-made motion,
2) the target cue,
3) the embodiment schema YAML (same schema used at generation time).

You are NOT given the Motion Code that produced the video. Infer from the video alone what the motion looks like, then write a new Motion Code for the TARGET cue.

Task:
- First read what the current video looks like.
- Then write a Motion Code so the motion reads as the TARGET cue on this embodiment.
- Stay inside the schema. Every Motion Code MUST start with pose(...).
- Total steps: 2~8. Join steps with +.

[Q1 — Perceived cue]
Watch the attached video.
Which cue does this motion currently look like?
Answer with a short cue name only (snake_case if possible).

[Q2 — Contrastive edit plan]
The TARGET cue is given below.
- If your Q1 cue is different from the TARGET cue: explain in ONE sentence how to write Motion Code so the result does NOT look like the Q1 cue, but DOES look like the TARGET cue.
- If your Q1 cue already matches the TARGET cue: explain in ONE sentence how to strengthen the TARGET reading (make it clearer / less ambiguous).

[Q3 — Revised Motion Code]
Implement that plan as ONE complete Motion Code line (write from scratch; you do not have the old code).

[TARGET CUE]
name: {cue}
description: {description}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Write exactly these three lines (and nothing else):
# Q1: <perceived_cue_name>
# Q2: <one-sentence contrastive edit plan>
# Q3: <pose(...)+... Motion Code only>
Do not invent attributes or component forms outside the embodiment schema above.
""".strip()

    if strategy == "structural":
        strategy_block = """
[REVISION STRATEGY — STRUCTURAL (required)]
Do NOT rely on tiny numeric value tweaks alone (e.g. only changing z/spd/hold by a little).
Prefer structural edits that make the cue readable on this embodiment:
1) ADD components before and/or after the core gesture (prep pose, accent path/move, hold, settle).
2) SPECIFY more attributes on existing steps (dir/grip/xyz/elbow/torso/head, path axis/shape/dist/rep, joint channel) when the schema allows — make underspecified steps explicit.
3) You may still adjust values, but only together with (1) and/or (2).
4) Keep everything inside the schema. Every code MUST start with pose(...).
Typical good pattern: pose(...) + <prep/move/path> + <core move/path> + hold(...) + optional settle.
""".strip()
    else:
        strategy_block = """
[REVISION STRATEGY]
Revise the Motion Code so the cue becomes identifiable. Stay inside the schema.
Every Motion Code MUST start with pose(...).
""".strip()

    return f"""You are reviewing a rendered robot gesture that a human judge marked as NOT identifiable as the target cue.

You are given:
1) the rendered video (attached),
2) the Motion Code that produced it,
3) the target cue,
4) the embodiment schema YAML (same schema used at generation time).

Task:
- Decide if revising the Motion Code can make this cue identifiable on this embodiment.
- If yes, output a revised complete Motion Code that stays inside the schema.
- If the embodiment fundamentally cannot express the cue, say so and still give your best approximate Motion Code.

{strategy_block}

[CUE]
name: {cue}
description: {description}

[CURRENT MOTION CODE]
{motion_code}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Write exactly these lines first:
# IdentifiableQ: <yes/no — can a Motion Code revision make this cue identifiable?>
# ChangeSummary: <one sentence: what you change and why>
# ChangeTags: <comma-separated short tags from: dir/facing, grip, xyz/percentile, path_shape, path_axis, speed, hold_time, repetition, joint/channel, add_component, remove_component, specify_attr, embodiment_limit, other>

Then output ONE revised Motion Code line only (steps joined by +). No JSON, no extra commentary.
Every Motion Code MUST start with pose(...). Do not invent attributes outside the schema.
""".strip()


def _parse_meta(raw: str) -> dict[str, str]:
    meta = {
        "identifiable_q": "",
        "change_summary": "",
        "change_tags": "",
        "q1": "",
        "q2": "",
        "q3": "",
    }
    for line in raw.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("# identifiableq:"):
            meta["identifiable_q"] = s.split(":", 1)[1].strip()
        elif low.startswith("# changesummary:"):
            meta["change_summary"] = s.split(":", 1)[1].strip()
        elif low.startswith("# changetags:"):
            meta["change_tags"] = s.split(":", 1)[1].strip()
        elif low.startswith("# q1:"):
            meta["q1"] = s.split(":", 1)[1].strip()
        elif low.startswith("# q2:"):
            meta["q2"] = s.split(":", 1)[1].strip()
        elif low.startswith("# q3:"):
            meta["q3"] = s.split(":", 1)[1].strip()
    return meta


def _extract_revised_code(raw: str, obj: dict[str, Any] | None = None) -> str:
    """Prefer ``# Q3:`` / ``# Q2:`` Motion Code lines; fall back to generic extract."""
    lines = raw.splitlines()
    for tag in ("# q3:", "# q2:"):
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.lower().startswith(tag):
                continue
            code = s.split(":", 1)[1].strip().strip("`")
            if re.match(r"^(pose|path|move|hold|movement)\s*\(", code, re.I):
                return code
            for nxt in lines[i + 1 :]:
                t = nxt.strip().strip("`")
                if not t or t.startswith("#"):
                    continue
                if re.match(r"^(pose|path|move|hold|movement)\s*\(", t, re.I):
                    return t
                break
    return extract_motion_code(raw, obj)


def _call_regen(
    *,
    client: Any,
    model: str,
    prompt: str,
    video: Path | None = None,
    videos: list[Path] | None = None,
) -> tuple[str, dict[str, Any]]:
    from google.genai import types

    config_kwargs: dict[str, Any] = {
        "temperature": 0.2,
        "max_output_tokens": 2048,
    }
    try:
        budget = 128 if "pro" in model.lower() else 0
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        config_kwargs["max_output_tokens"] = max(2048, budget + 1024)
    except Exception:
        pass
    config = types.GenerateContentConfig(**config_kwargs)
    paths = list(videos) if videos else ([video] if video is not None else [])
    if not paths:
        raise ValueError("video or videos required")
    parts = [
        types.Part.from_bytes(data=p.read_bytes(), mime_type="video/mp4") for p in paths
    ]
    resp = client.models.generate_content(
        model=model,
        contents=[*parts, prompt],
        config=config,
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        try:
            for cand in getattr(resp, "candidates", None) or []:
                content = getattr(cand, "content", None)
                for p in getattr(content, "parts", None) or []:
                    piece = getattr(p, "text", None)
                    if piece:
                        text += piece
        except Exception:
            pass
        text = text.strip()
    usage: dict[str, Any] = {}
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None:
        usage = {
            "prompt_token_count": getattr(meta, "prompt_token_count", None),
            "candidates_token_count": getattr(meta, "candidates_token_count", None),
            "thoughts_token_count": getattr(meta, "thoughts_token_count", None),
        }
    return text, usage


def _steps(code: str) -> list[str]:
    if not code.strip():
        return []
    # split on + outside parens roughly
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in code:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "+" and depth == 0:
            s = "".join(cur).strip()
            if s:
                parts.append(s)
            cur = []
            continue
        cur.append(ch)
    s = "".join(cur).strip()
    if s:
        parts.append(s)
    return parts


def _step_types(code: str) -> list[str]:
    out = []
    for step in _steps(code):
        m = _STEP_RE.match(step.strip())
        out.append(m.group(1).lower() if m else "?")
    return out


def analyze_edits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tag_c: Counter[str] = Counter()
    id_q: Counter[str] = Counter()
    type_add: Counter[str] = Counter()
    type_rem: Counter[str] = Counter()
    len_delta: list[int] = []
    by_plat: dict[str, Counter[str]] = defaultdict(Counter)
    changed = 0
    same = 0
    for r in rows:
        if r.get("error"):
            continue
        meta = r.get("meta") or {}
        iq = (meta.get("identifiable_q") or "").lower()
        if iq.startswith("yes"):
            id_q["yes"] += 1
        elif iq.startswith("no"):
            id_q["no"] += 1
        else:
            id_q["other"] += 1
        for tag in re.split(r"[,;/|]+", meta.get("change_tags") or ""):
            t = tag.strip().lower()
            if t:
                tag_c[t] += 1
                by_plat[r["platform"]][t] += 1
        old = str(r.get("old_code") or "").strip()
        new = str(r.get("new_code") or "").strip()
        if old == new or not new:
            same += 1
        else:
            changed += 1
        ot, nt = _step_types(old), _step_types(new)
        len_delta.append(len(nt) - len(ot))
        for t in set(nt) - set(ot):
            type_add[t] += 1
        for t in set(ot) - set(nt):
            type_rem[t] += 1
        # attr keyword presence delta
        old_attrs = set(_ATTR_RE.findall(old.lower()))
        new_attrs = set(_ATTR_RE.findall(new.lower()))
        for a in new_attrs - old_attrs:
            tag_c[f"+attr:{a}"] += 1
        for a in old_attrs - new_attrs:
            tag_c[f"-attr:{a}"] += 1

    return {
        "n_rows": len(rows),
        "n_ok": sum(1 for r in rows if not r.get("error")),
        "identifiable_q": dict(id_q),
        "code_changed": changed,
        "code_same_or_empty": same,
        "change_tags": tag_c.most_common(40),
        "change_tags_by_platform": {
            p: c.most_common(15) for p, c in sorted(by_plat.items())
        },
        "step_type_added": dict(type_add),
        "step_type_removed": dict(type_rem),
        "mean_step_delta": (sum(len_delta) / len(len_delta)) if len_delta else 0.0,
        "parse_ok": sum(1 for r in rows if r.get("parse_ok")),
    }


def write_analysis_md(analysis: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Regeneration analysis — human-NO Gemini-made set",
        "",
        f"- N cells: **{analysis['n_rows']}** (API ok={analysis['n_ok']}, parse_ok={analysis['parse_ok']})",
        f"- Model says revision can make identifiable: {analysis['identifiable_q']}",
        f"- Code changed: **{analysis['code_changed']}** · same/empty: {analysis['code_same_or_empty']}",
        f"- Mean step-count delta (new−old): **{analysis['mean_step_delta']:+.2f}**",
        "",
        "## Top ChangeTags (model self-report)",
        "",
        "| Tag | Count |",
        "|---|---:|",
    ]
    for tag, n in analysis["change_tags"][:25]:
        if tag.startswith("+attr:") or tag.startswith("-attr:"):
            continue
        lines.append(f"| {tag} | {n} |")
    lines += [
        "",
        "## Attribute keyword adds/removes (heuristic from code text)",
        "",
        "| Tag | Count |",
        "|---|---:|",
    ]
    for tag, n in analysis["change_tags"]:
        if tag.startswith("+attr:") or tag.startswith("-attr:"):
            lines.append(f"| `{tag}` | {n} |")
    lines += [
        "",
        "## Step type set changes",
        "",
        f"- Added types: `{analysis['step_type_added']}`",
        f"- Removed types: `{analysis['step_type_removed']}`",
        "",
        "## By platform (top tags)",
        "",
    ]
    for plat, tags in analysis["change_tags_by_platform"].items():
        clean = [(t, n) for t, n in tags if not t.startswith(("+attr", "-attr"))]
        lines.append(f"- **{plat}**: " + ", ".join(f"{t} ({n})" for t, n in clean[:8]))
    lines += ["", "## Sample ChangeSummary", ""]
    for r in rows[:20]:
        if r.get("error"):
            continue
        s = (r.get("meta") or {}).get("change_summary") or ""
        if s:
            lines.append(
                f"- `{r['platform']}` g{int(r['idx']):02d} {r['cue']}: {s}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gallery_html(
    rows: list[dict[str, Any]],
    *,
    out_html: Path,
    render_root: Path,
    src_render: Path,
    gt_render: Path = RENDER_GT,
    gt_codes: dict[tuple[int, str], str] | None = None,
    extra_render: Path | None = None,
    extra_codes: dict[tuple[str, int], str] | None = None,
    extra_meta: dict[tuple[str, int], dict[str, str]] | None = None,
    extra_label: str = "After (regen structural)",
    extra_code_label: str = "New Motion Code (regen v2)",
    title: str | None = None,
    subtitle: str | None = None,
) -> None:
    def _gif_under(root: Path, platform: str, idx: int, cue: str) -> Path:
        if platform == "single_arm":
            return root / "manipulator" / "gif" / "IIWA" / f"emb_g{idx:02d}_{cue}.gif"
        return root / platform / "gif" / f"emb_g{idx:02d}_{cue}.gif"

    n_cols = 4 if extra_render is not None else 3
    grid_cols = " ".join(["1fr"] * n_cols)
    sections = []
    for r in rows:
        plat, idx, cue = r["platform"], int(r["idx"]), r["cue"]
        gt_gif = _gif_under(gt_render, plat, idx, cue)
        old_gif = _gif_under(src_render, plat, idx, cue)
        new_gif = gif_path_for(render_root, plat, idx, cue)
        if plat == "single_arm" and not new_gif.is_file():
            new_gif = _gif_under(render_root, plat, idx, cue)
        extra_gif = None
        if extra_render is not None:
            extra_gif = gif_path_for(extra_render, plat, idx, cue)
            if plat == "single_arm" and not extra_gif.is_file():
                extra_gif = _gif_under(extra_render, plat, idx, cue)
        old_mp4 = _mp4_path(plat, idx, cue)
        gt_code = ""
        if gt_codes is not None:
            gt_code = gt_codes.get((idx, plat), "") or ""
        xcode = ""
        if extra_codes is not None:
            xcode = extra_codes.get((plat, idx), "") or ""
        xmeta = (extra_meta or {}).get((plat, idx), {})

        def media(path: Path | None, label: str) -> str:
            if path is not None and path.is_file():
                rel = os.path.relpath(path, out_html.parent)
                bust = int(path.stat().st_mtime)
                return (
                    f'<div class="col"><h3>{html_mod.escape(label)}</h3>'
                    f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy"></div>'
                )
            return (
                f'<div class="col"><h3>{html_mod.escape(label)}</h3>'
                f'<p class="miss">missing</p></div>'
            )

        def code_col(label: str, code: str) -> str:
            return (
                f'<div><div class="lab">{html_mod.escape(label)}</div>'
                f"<pre>{html_mod.escape(code or '')}</pre></div>"
            )

        meta = r.get("meta") or {}
        note = (
            f'<p class="note">human note: {html_mod.escape(r.get("human_note") or "")} · '
            f'v1 IdentifiableQ: {html_mod.escape(meta.get("identifiable_q") or "")} · '
            f'v1 tags: {html_mod.escape(meta.get("change_tags") or "")}'
        )
        if extra_render is not None:
            note += (
                f' · v2 IdentifiableQ: {html_mod.escape(xmeta.get("identifiable_q") or "")} · '
                f'v2 tags: {html_mod.escape(xmeta.get("change_tags") or "")}'
            )
        note += "</p>"
        sum_bits = [html_mod.escape(meta.get("change_summary") or "")]
        if extra_render is not None and xmeta.get("change_summary"):
            sum_bits.append(
                "v2: " + html_mod.escape(xmeta.get("change_summary") or "")
            )
        media_cols = (
            media(gt_gif, "GT (human-authored)")
            + media(old_gif, "Original (input · ee-translate)")
            + media(new_gif, "After (regen R1 · ee-translate)")
        )
        code_cols = (
            code_col("GT Motion Code", gt_code)
            + code_col("Old Motion Code (Gemini)", r.get("old_code") or "")
            + code_col("New Motion Code (regen R1)", r.get("new_code") or "")
        )
        if extra_render is not None:
            media_cols += media(extra_gif, extra_label)
            code_cols += code_col(extra_code_label, xcode)
        sections.append(
            f'<section class="cue"><h2>{plat} · g{idx:02d} {html_mod.escape(cue)}</h2>'
            f"{note}"
            f'<p class="sum">{" · ".join(s for s in sum_bits if s)}</p>'
            f'<div class="grid">{media_cols}</div>'
            f'<div class="codes">{code_cols}</div>'
            f'<p class="path">mp4: {html_mod.escape(str(old_mp4.relative_to(_REPO)))}</p>'
            f"</section>"
        )
    page_title = title or "Regen · human-NO set"
    page_sub = subtitle or (
        f"n={len(rows)} · GT · Gemini · regen v1"
        + (" · regen R2" if extra_render is not None else "")
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_mod.escape(page_title)}</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:{grid_cols};gap:12px}}
.col h3{{font-size:.85rem;margin:0 0 6px;font-weight:600}}
.col img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}}
.miss{{color:#f28b82}}
.note,.sum,.path{{font-size:.85rem;color:#c4c7ce}}
.codes{{display:grid;grid-template-columns:{grid_cols};gap:10px;margin-top:10px}}
.lab{{font-size:.7rem;color:#9aa0a6;text-transform:uppercase}}
pre{{font-size:11px;white-space:pre-wrap;background:#0a0c10;padding:8px;border-radius:6px;max-height:160px;overflow:auto}}
@media (max-width:1100px){{.grid,.codes{{grid-template-columns:1fr 1fr}}}}
@media (max-width:700px){{.grid,.codes{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html_mod.escape(page_title)}</h1>
<p class="meta">{html_mod.escape(page_sub)}</p>
{"".join(sections)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all human-NO cells")
    ap.add_argument("--platforms", default="", help="optional comma filter")
    ap.add_argument(
        "--strategy",
        default="default",
        choices=(
            "default",
            "structural",
            "cot_dynamic",
            "cot_contrast",
            "cot_contrast_nocode",
        ),
        help="default | structural | cot_dynamic | cot_contrast | cot_contrast_nocode (video only, no Motion Code)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--visual-render-root",
        type=Path,
        default=None,
        help="MP4/GIF root used as visual input (default: *_ee_translate if present else RENDER_SRC)",
    )
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    args = ap.parse_args()

    global RENDER_SRC
    vis = args.visual_render_root
    if vis is None:
        # Prefer ee-translate renders as the canonical visual input going forward.
        if (RENDER_SRC_EE_TRANSLATE / "humanoid" / "gif").is_dir() or (
            RENDER_SRC_EE_TRANSLATE / "tiago" / "gif"
        ).is_dir():
            vis = RENDER_SRC_EE_TRANSLATE
        else:
            vis = RENDER_SRC
    RENDER_SRC = Path(vis).expanduser().resolve()
    print(f"[visual] RENDER_SRC={RENDER_SRC}", flush=True)

    out_dir: Path = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    master_out = out_dir / "regen_motion_code_all.json"
    render_root = out_dir / "render"
    html_out = out_dir / "regen_gallery.html"
    analysis_json = out_dir / "regen_analysis.json"
    analysis_md = out_dir / "regen_analysis.md"

    human_no = _load_human_no()
    if args.platforms:
        allow = {p.strip() for p in args.platforms.split(",") if p.strip()}
        human_no = [r for r in human_no if r["platform"] in allow]
    if args.max_items > 0:
        human_no = human_no[: args.max_items]

    gemini_master = json.loads(GEMINI_MASTER.read_text(encoding="utf-8"))
    index = _index_master(gemini_master)
    cues = {int(c["idx"]): c for c in json.loads(CUES_JSON.read_text(encoding="utf-8"))}

    done: dict[str, dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["job_id"]] = row

    jobs = []
    for r in human_no:
        job_id = f"{r['platform']}.g{r['idx']:02d}"
        if job_id in done:
            continue
        jobs.append(r)

    print(
        f"regen: resume={len(done)} pending={len(jobs)}/{len(human_no)} "
        f"model={args.model} workers={args.workers} strategy={args.strategy}",
        flush=True,
    )

    client = _client() if jobs else None

    def work(r: dict[str, Any]) -> dict[str, Any]:
        plat, idx = r["platform"], int(r["idx"])
        job_id = f"{plat}.g{idx:02d}"
        info = index.get((idx, plat))
        if not info:
            return {
                **r,
                "job_id": job_id,
                "error": "missing_gemini_config",
            }
        cue = info["cue"]
        desc = str(cues.get(idx, {}).get("description") or info["description"] or "")
        old_code = str((info["cell"] or {}).get("source_code") or "").strip()
        video = _mp4_path(plat, idx, cue)
        if not video.is_file():
            return {
                **r,
                "job_id": job_id,
                "cue": cue,
                "old_code": old_code,
                "error": f"missing_video:{video}",
            }
        prompt = build_regen_prompt(
            platform=plat,
            cue=cue,
            description=desc,
            motion_code=old_code,
            strategy=args.strategy,
        )
        last_err = None
        for attempt in range(args.retries + 1):
            try:
                raw, usage = _call_regen(
                    client=client, model=args.model, prompt=prompt, video=video
                )
                meta = _parse_meta(raw)
                obj = _extract_json_obj(raw)
                new_code = _extract_revised_code(raw, obj)
                if meta.get("q2") and not meta.get("change_summary"):
                    meta["change_summary"] = meta["q2"]
                elif meta.get("q1") and not meta.get("change_summary"):
                    meta["change_summary"] = meta["q1"]
                converted = convert_code(new_code, platform=plat) if new_code else {
                    "movements": [],
                    "parse_ok": False,
                    "notes": ["empty_extract"],
                    "source_code": "",
                }
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return {
                    **r,
                    "job_id": job_id,
                    "cue": cue,
                    "description": desc,
                    "old_code": old_code,
                    "new_code": converted.get("source_code") or new_code,
                    "movements": converted.get("movements") or [],
                    "parse_ok": bool(converted.get("parse_ok")),
                    "notes": converted.get("notes") or [],
                    "meta": meta,
                    "raw_output": raw,
                    "usage": usage,
                    "model": args.model,
                    "strategy": args.strategy,
                    "mp4": str(video.relative_to(_REPO)),
                }
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 20))
        return {
            **r,
            "job_id": job_id,
            "cue": info["cue"],
            "old_code": old_code,
            "error": last_err or "unknown",
        }

    completed = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(work, r): r for r in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                with pred_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["job_id"]] = row
                completed += 1
                if completed % 5 == 0 or completed == len(jobs):
                    print(
                        f"regen: {len(done)}/{len(human_no)} (+{completed}/{len(jobs)})",
                        flush=True,
                    )

    # rebuild ordered rows for all human_no
    ordered = []
    for r in human_no:
        job_id = f"{r['platform']}.g{r['idx']:02d}"
        ordered.append(done.get(job_id) or {**r, "job_id": job_id, "error": "missing"})

    # master config for render
    by_cue: dict[int, dict[str, Any]] = {}
    for row in ordered:
        if row.get("error") or not row.get("parse_ok"):
            continue
        idx = int(row["idx"])
        plat = row["platform"]
        entry = by_cue.setdefault(
            idx,
            {
                "idx": idx,
                "cue": row["cue"],
                "category": "",
                "description": row.get("description") or "",
                "platforms": {},
            },
        )
        domain = {
            "single_arm": "manipulator",
            "tiago": "tiago",
            "stretch": "stretch",
            "humanoid": "unitree_g1",
            "roomba": "roomba",
        }.get(plat, plat)
        entry["platforms"][plat] = {
            "idx": idx,
            "cue": row["cue"],
            "description": row.get("description") or "",
            "movements": row.get("movements") or [],
            "state": "motion_code_regen",
            "model": args.model,
            "time": datetime.now().isoformat(timespec="seconds"),
            "experiment": f"embodiment_regen_human_no_{args.strategy}",
            "generation_valid": bool(row.get("movements")),
            "platform": plat,
            "domain": domain,
            "parse_ok": True,
            "notes": row.get("notes") or [],
            "source_code": row.get("new_code") or "",
        }
    configs = [by_cue[i] for i in sorted(by_cue)]
    master_out.write_text(json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_render and configs:
        print("[render] draining revised Motion Codes…", flush=True)
        # fresh skip set for this run
        render_pending_once._skip = set()  # type: ignore[attr-defined]
        # render in batches until done or stalled
        for _ in range(40):
            n = render_pending_once(
                configs,
                render_root=render_root,
                out_json=master_out,
                platforms=sorted({r["platform"] for r in ordered}),
                force=args.force_render,
                max_items=6,
            )
            if n == 0:
                break
            print(f"[render] attempted {n}", flush=True)

    analysis = analyze_edits(ordered)
    analysis_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_analysis_md(analysis, ordered, analysis_md)
    gt_codes = {
        (idx, plat): str((cell or {}).get("source_code") or "")
        for (idx, plat), cell in (
            ((int(e["idx"]), p), c)
            for e in json.loads(GT_MASTER.read_text(encoding="utf-8"))
            for p, c in (e.get("platforms") or {}).items()
        )
    }
    write_gallery_html(
        ordered,
        out_html=html_out,
        render_root=render_root,
        src_render=RENDER_SRC,
        gt_render=RENDER_GT,
        gt_codes=gt_codes,
        title="HL Human-NO · regen R1 (ee-translate input + render)",
        subtitle=(
            f"n={len(ordered)} · platforms={args.platforms or 'all'} · "
            f"visual={RENDER_SRC.name} · strategy={args.strategy}"
        ),
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "n_human_no": len(human_no),
                "analysis": analysis,
                "outputs": {
                    "predictions": str(pred_path.relative_to(_REPO)),
                    "master_json": str(master_out.relative_to(_REPO)),
                    "gallery": str(html_out.relative_to(_REPO)),
                    "analysis_md": str(analysis_md.relative_to(_REPO)),
                    "render_root": str(render_root.relative_to(_REPO)),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2)[:2000], flush=True)
    print(f"gallery → {html_out}", flush=True)
    print(f"analysis → {analysis_md}", flush=True)


if __name__ == "__main__":
    main()
