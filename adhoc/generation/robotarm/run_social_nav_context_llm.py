#!/usr/bin/env python3
"""LLM restaging of social-navigation cues under 10 contexts, on Roomba and Stretch.

Grid: 19 human-authored cues x 10 contexts x 2 platforms = 380 Motion Programs.

The prompt is deliberately starved of the numeric deformation fields in
``social_nav_context_10.yml`` (spd/hold/amp/rep/approach/height). The model gets
the situation prose plus the hard constraints (channel budget, forbidden tokens)
and has to decide the delivery itself, so this arm stays independent of the
deterministic transform it is meant to be compared against.

    source APIKEY.sh
    python adhoc/generation/robotarm/run_social_nav_context_llm.py --platform roomba,stretch
    python adhoc/generation/robotarm/run_social_nav_context_llm.py --render --gallery
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    _extract_json_obj,
    embodiment_schema_yaml,
    extract_motion_code,
    gemini_generate,
)

CONTEXT_YML = REPO / "data/seed/yml/social_nav_context_10.yml"
PROMPT_DIR = REPO / "data/seed/prompt/social_nav"
PROMPT_TPL = PROMPT_DIR / "prompt_context_variation.txt"
GT_DIR = REPO / "data/results/motion_configs/embodiment_motion_code_gt"
OUT_DIR = REPO / "data/results/motion_configs/social_nav_context_llm"
RENDER_DIR = REPO / "data/results/render/social_nav_context_llm"

PLATFORMS = ("roomba", "stretch")

# ---------------------------------------------------------------- constraints

_FORBID_TEXT = {
    "sharp_strike": "No step may carry spd >= 3.0. A sharp strike is wrong here.",
    "full_spin": "No path(base, arc, ...) with |deg| >= 300. Full rotations are out.",
    "advance_large": "No path(base, line, ...) with x >= +0.30. Any advance stays under 30 cm.",
    "circle": "No path(base, arc, ...) with r >= 0.20. Wide circling loops are out.",
}

_ARM_TOKENS = re.compile(r"path\s*\(\s*ee\b|move\s*\(\s*(shoulder|elbow|wrist|torso)\b", re.I)
_HEAD_TOKENS = re.compile(r"move\s*\(\s*head\b|head\s*=", re.I)
_BASE_TOKENS = re.compile(r"path\s*\(\s*base\b", re.I)
_POSE_ARM_ATTR = re.compile(r"\b(dir|grip|torso|elbow|x|y|z)\s*=", re.I)


def _spd_values(code: str) -> list[float]:
    return [float(v) for v in re.findall(r"spd\s*=\s*([0-9.]+)", code)]


def _base_arc_degs(code: str) -> list[float]:
    out = []
    for step in re.findall(r"path\s*\(\s*base\s*,\s*arc[^)]*\)", code, re.I):
        out += [abs(float(v)) for v in re.findall(r"deg\s*=\s*\[?\s*([+-]?[0-9.]+)", step)]
    return out


def _base_arc_radii(code: str) -> list[float]:
    out = []
    for step in re.findall(r"path\s*\(\s*base\s*,\s*arc[^)]*\)", code, re.I):
        out += [float(v) for v in re.findall(r"\br\s*=\s*([0-9.]+)", step)]
    return out


def _base_line_x(code: str) -> list[float]:
    out = []
    for step in re.findall(r"path\s*\(\s*base\s*,\s*line[^)]*\)", code, re.I):
        out += [float(v) for v in re.findall(r"\bx\s*=\s*\[?\s*([+-]?[0-9.]+)", step)]
    return out


def check_forbid(code: str, forbid: list[str]) -> list[str]:
    bad = []
    for key in forbid:
        if key == "sharp_strike" and any(v >= 3.0 for v in _spd_values(code)):
            bad.append("sharp_strike: a step still uses spd >= 3.0")
        elif key == "full_spin" and any(v >= 300 for v in _base_arc_degs(code)):
            bad.append("full_spin: a base arc still sweeps >= 300 degrees")
        elif key == "advance_large" and any(v >= 0.30 for v in _base_line_x(code)):
            bad.append("advance_large: a base line still advances x >= +0.30")
        elif key == "circle" and any(v >= 0.20 for v in _base_arc_radii(code)):
            bad.append("circle: a base arc still uses r >= 0.20")
    return bad


def check_channels(code: str, channels: list[str], platform: str) -> list[str]:
    if platform == "roomba":
        return []  # the base is the only carrier Roomba has
    bad = []
    if "arm" not in channels:
        if _ARM_TOKENS.search(code):
            bad.append("arm channel is unavailable but the program still drives the arm")
        for pose in re.findall(r"pose\s*\(([^)]*)\)", code, re.I):
            attrs = {m.group(1).lower() for m in _POSE_ARM_ATTR.finditer(pose)}
            attrs -= {"head"}
            if "dir" in attrs and not re.search(r"dir\s*=\s*still", pose, re.I):
                bad.append("arm channel is unavailable: pose(dir=...) must be dir=still")
            if attrs & {"grip", "torso", "elbow", "x", "y", "z"}:
                bad.append("arm channel is unavailable: drop grip/torso/elbow/x/y/z from pose(...)")
    if "head" not in channels and _HEAD_TOKENS.search(code):
        bad.append("head channel is unavailable but the program still drives the head")
    if "base" not in channels and _BASE_TOKENS.search(code):
        bad.append("base channel is unavailable but the program still drives the base")
    return bad


def constraint_block(ctx: dict[str, Any], platform: str) -> str:
    lines: list[str] = []
    channels = list(ctx.get("channels") or ["base", "arm", "head"])
    if platform == "stretch":
        if "arm" not in channels:
            lines.append(
                "The arm is committed and cannot be used. Emit no path(ee, ...) and no "
                "move(shoulder|elbow|wrist|torso, ...). In pose(...) use dir=still and set no "
                "grip/torso/elbow/x/y/z. The head and the base are all you have — the same "
                "channel budget a Roomba always has."
            )
        if "head" not in channels:
            lines.append("The head cannot be used. Emit no move(head, ...) and no head= in pose(...).")
        if "base" not in channels:
            lines.append("The base cannot move. Emit no path(base, ...).")
    for key in ctx.get("forbid") or []:
        if key in _FORBID_TEXT:
            lines.append(_FORBID_TEXT[key])
    if not lines:
        lines.append("None beyond the platform schema.")
    return "\n".join(f"- {ln}" for ln in lines)


_RULE_KEYS = (
    "spd",
    "hold",
    "amp",
    "amp_arm",
    "rep",
    "approach",
    "height",
    "channels",
    "pre",
    "post",
    "forbid",
)


def rules_yaml(ctx: dict[str, Any], platform: str) -> str:
    """Dump the catalog fields a human author would read, plus hard channel/forbid notes."""
    body = {k: ctx[k] for k in _RULE_KEYS if k in ctx}
    extra = constraint_block(ctx, platform)
    dumped = yaml.safe_dump(body, sort_keys=False, allow_unicode=True).rstrip()
    defs = {
        "sharp_strike": "any step with spd >= 3.0",
        "full_spin": "path(base, arc) with |deg| >= 300",
        "advance_large": "path(base, line) with x >= +0.30",
        "circle": "path(base, arc) with r >= 0.20",
    }
    used = [k for k in (ctx.get("forbid") or []) if k in defs]
    if used:
        dumped += "\n" + yaml.safe_dump(
            {"forbid_means": {k: defs[k] for k in used}},
            sort_keys=False,
        ).rstrip()
    if extra and extra != "None beyond the platform schema.":
        dumped += "\n# hard constraints\n" + extra
    return dumped


# ---------------------------------------------------------------- inputs


def load_contexts() -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    spec = yaml.safe_load(CONTEXT_YML.read_text(encoding="utf-8"))
    cues = list(spec["cues_19"])
    order = list(spec["grid"]["contexts"])
    contexts = [(k, spec["contexts"][k]) for k in order]
    return cues, contexts


def load_gt(platform: str) -> dict[int, dict[str, Any]]:
    raw = json.loads((GT_DIR / f"result_{platform}.json").read_text(encoding="utf-8"))
    rows = raw["rows"] if isinstance(raw, dict) else raw
    return {int(r["idx"]): r for r in rows}


def cue_glosses() -> dict[int, str]:
    """Stretch GT carries the semantic description; Roomba's is boilerplate."""
    out: dict[int, str] = {}
    for platform in ("stretch", "roomba"):
        for idx, row in load_gt(platform).items():
            desc = str(row.get("description") or "").strip()
            if desc and not desc.lower().startswith("motion-code embodiment"):
                out.setdefault(idx, desc)
    return out


def _one_line(code: str) -> str:
    return " ".join((code or "").split())


# ---------------------------------------------------------------- prompt


def build_prompt(
    *,
    platform: str,
    cue: str,
    gloss: str,
    base_code: str,
    ctx_key: str,
    ctx: dict[str, Any],
    schema: str,
    example: str = "",
    hints: str = "",
) -> str:
    tpl = PROMPT_TPL.read_text(encoding="utf-8")
    label = f"{ctx_key} — {str(ctx.get('label') or '').strip()}"
    desc = " ".join(str(ctx.get("desc") or "").split())
    return (
        tpl.replace("{{PLATFORM}}", platform)
        .replace("{{CUE}}", cue)
        .replace("{{GLOSS}}", gloss or cue.replace("_", " "))
        .replace("{{BASE_CODE}}", base_code)
        .replace("{{CONTEXT_LABEL}}", label)
        .replace("{{CONTEXT_DESC}}", desc)
        .replace("{{RULES_YAML}}", rules_yaml(ctx, platform))
        .replace("{{SCHEMA}}", schema)
        .replace("{{EXAMPLE}}", example or "(none)")
        .replace("{{HINTS}}", hints or "None beyond the platform schema.")
    )


def repair_suffix(code: str, problems: list[str]) -> str:
    return (
        "\n\n[REPAIR]\nYour previous answer was:\n"
        f"{code}\n\nIt violated:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\nFix only these and keep everything else, including the identity beat. "
        "Return the same JSON object shape.\n"
    )


# ---------------------------------------------------------------- generation

_PRINT_LOCK = threading.Lock()


def generate_one(
    *,
    platform: str,
    cue_row: dict[str, Any],
    ctx_key: str,
    ctx: dict[str, Any],
    base_row: dict[str, Any],
    gloss: str,
    schema: str,
    example: str,
    hints: str,
    model: str,
    retries: int,
) -> dict[str, Any]:
    idx = int(cue_row["idx"])
    cue = str(cue_row["cue"])
    base_code = _one_line(base_row.get("source_code") or "")
    prompt = build_prompt(
        platform=platform,
        cue=cue,
        gloss=gloss,
        base_code=base_code,
        ctx_key=ctx_key,
        ctx=ctx,
        schema=schema,
        example=example,
        hints=hints,
    )

    raw = ""
    obj: dict[str, Any] | None = None
    code = ""
    converted: dict[str, Any] = {}
    problems: list[str] = []
    attempt_prompt = prompt

    for attempt in range(retries + 1):
        try:
            raw = gemini_generate(attempt_prompt, model=model)
            obj = _extract_json_obj(raw) or {}
            code = _one_line(str(obj.get("motion_code") or "") or extract_motion_code(raw, obj))
            converted = convert_code(code, platform=platform)
            problems = []
            if not (converted.get("parse_ok") and converted.get("movements")):
                problems.append(
                    "the program did not parse against the schema: "
                    + "; ".join(converted.get("notes") or ["unknown parse error"])
                )
            problems += check_channels(code, list(ctx.get("channels") or []), platform)
            problems += check_forbid(code, list(ctx.get("forbid") or []))
            if ctx_key != "routine" and code == base_code:
                problems.append("the program is byte-identical to the routine one; nothing was restaged")
            if not problems:
                break
            attempt_prompt = prompt + repair_suffix(code, problems)
        except Exception as exc:  # noqa: BLE001
            problems = [f"request failed: {exc}"]
            time.sleep(1.5 * (attempt + 1))

    out_code = _one_line(converted.get("source_code") or code)
    row = {
        "platform": platform,
        "idx": idx,
        "cue": cue,
        "context": ctx_key,
        "context_label": str(ctx.get("label") or ""),
        "context_desc": " ".join(str(ctx.get("desc") or "").split()),
        "base_code": base_code,
        "source_code": out_code,
        "identity_beat": str((obj or {}).get("identity_beat") or ""),
        "changes": list((obj or {}).get("changes") or []),
        "notes_llm": str((obj or {}).get("notes") or ""),
        "movements": converted.get("movements") or [],
        "parse_ok": bool(converted.get("parse_ok") and converted.get("movements")),
        "violations": problems,
        "changed": bool(out_code and out_code != base_code),
        "model": model,
        "time": datetime.now().isoformat(timespec="seconds"),
        "raw_response": raw,
    }
    with _PRINT_LOCK:
        flag = "ok " if row["parse_ok"] and not problems else "BAD"
        print(f"[{flag}] {platform:7s} c{idx:02d} {cue:22s} {ctx_key:18s} {out_code[:70]}", flush=True)
    return row


def out_path(platform: str) -> Path:
    return OUT_DIR / f"{platform}_context_programs.json"


def load_existing(platform: str) -> dict[tuple[int, str], dict[str, Any]]:
    p = out_path(platform)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    rows = raw["rows"] if isinstance(raw, dict) else raw
    return {(int(r["idx"]), str(r["context"])): r for r in rows}


def save(platform: str, rows: dict[tuple[int, str], dict[str, Any]], model: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows.values(), key=lambda r: (int(r["idx"]), str(r["context"])))
    payload = {
        "platform": platform,
        "model": model,
        "n": len(ordered),
        "n_ok": sum(1 for r in ordered if r.get("parse_ok") and not r.get("violations")),
        "rows": ordered,
    }
    p = out_path(platform)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def run_platform(
    platform: str,
    *,
    model: str,
    retries: int,
    workers: int,
    only_cues: set[int] | None,
    only_contexts: set[str] | None,
    force: bool,
    dry_run: bool,
) -> None:
    cues, contexts = load_contexts()
    gt = load_gt(platform)
    glosses = cue_glosses()
    schema = embodiment_schema_yaml(platform)
    example = (PROMPT_DIR / "example_wave_hi.txt").read_text(encoding="utf-8").strip()
    hints = (PROMPT_DIR / f"hints_{platform}.txt").read_text(encoding="utf-8").strip()
    existing = {} if force else load_existing(platform)

    jobs = []
    for cue_row in cues:
        idx = int(cue_row["idx"])
        if only_cues and idx not in only_cues:
            continue
        base_row = gt.get(idx)
        if not base_row:
            print(f"[skip] {platform} c{idx} missing from GT", flush=True)
            continue
        for ctx_key, ctx in contexts:
            if only_contexts and ctx_key not in only_contexts:
                continue
            prev = existing.get((idx, ctx_key))
            if prev and prev.get("parse_ok") and not prev.get("violations"):
                continue
            jobs.append((cue_row, base_row, ctx_key, ctx))

    print(f"\n=== {platform}: {len(jobs)} jobs (done {len(existing)}) ===", flush=True)
    if dry_run:
        if jobs:
            cue_row, base_row, ctx_key, ctx = jobs[0]
            print(
                build_prompt(
                    platform=platform,
                    cue=str(cue_row["cue"]),
                    gloss=glosses.get(int(cue_row["idx"]), ""),
                    base_code=_one_line(base_row.get("source_code") or ""),
                    ctx_key=ctx_key,
                    ctx=ctx,
                    schema=schema,
                    example=example,
                    hints=hints,
                )
            )
        return
    if not jobs:
        return

    def work(job):
        cue_row, base_row, ctx_key, ctx = job
        return generate_one(
            platform=platform,
            cue_row=cue_row,
            ctx_key=ctx_key,
            ctx=ctx,
            base_row=base_row,
            gloss=glosses.get(int(cue_row["idx"]), ""),
            schema=schema,
            example=example,
            hints=hints,
            model=model,
            retries=retries,
        )

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(work, jobs):
            existing[(int(row["idx"]), str(row["context"]))] = row
            done += 1
            if done % 10 == 0:
                save(platform, existing, model)
    p = save(platform, existing, model)
    ok = sum(1 for r in existing.values() if r.get("parse_ok") and not r.get("violations"))
    print(f"wrote {p}  ({ok}/{len(existing)} clean)", flush=True)


# ---------------------------------------------------------------- render


def render_platform(platform: str, *, force: bool, fps: int) -> None:
    from embodiment_renderers import render_roomba, render_stretch

    rows = load_existing(platform)
    out = RENDER_DIR / platform / "gif"
    out.mkdir(parents=True, exist_ok=True)
    render_rows = [
        {
            "idx": int(r["idx"]),
            "cue": f"{r['cue']}__{r['context']}",
            "description": r["context_label"],
            "movements": r["movements"],
            "source_code": r["source_code"],
            "parse_ok": True,
            "platform": platform,
        }
        for r in sorted(rows.values(), key=lambda r: (int(r["idx"]), str(r["context"])))
        if r.get("parse_ok")
    ]
    if not render_rows:
        print(f"[render] {platform}: nothing to render", flush=True)
        return
    fn = render_roomba if platform == "roomba" else render_stretch
    fn(render_rows, out, force=force, fps=fps)


def _gif_rel(platform: str, row: dict[str, Any], html_dir: Path) -> str | None:
    gif = RENDER_DIR / platform / "gif" / f"emb_g{int(row['idx']):02d}_{row['cue']}__{row['context']}.gif"
    return os.path.relpath(gif, html_dir) if gif.is_file() else None


GT_RENDER = REPO / "data/results/render" / "embodiment_motion_code_gt"


def _gt_gif_rel(platform: str, idx: int, cue: str, html_dir: Path) -> str | None:
    gif = GT_RENDER / platform / "gif" / f"emb_g{idx:02d}_{cue}.gif"
    return os.path.relpath(gif, html_dir) if gif.is_file() else None


def _gt_code(platform: str, idx: int) -> str:
    row = load_gt(platform).get(idx) or {}
    return _one_line(str(row.get("source_code") or ""))


def build_gallery(platforms: list[str], *, only_cues: set[int] | None = None) -> Path:
    import html as _html

    catalog, contexts = load_contexts()
    ctx_order = [k for k, _ in contexts]
    data = {p: load_existing(p) for p in platforms}
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    out = RENDER_DIR / "index.html"
    bust = int(time.time())

    cue_names = {int(cr["idx"]): str(cr["cue"]) for cr in catalog}
    cue_names.update({int(r["idx"]): r["cue"] for d in data.values() for r in d.values()})
    if only_cues:
        cue_ids = sorted(only_cues)
    else:
        cue_ids = sorted({idx for d in data.values() for (idx, _) in d})

    def _plat_block(p: str, img: str, code: str, changes: list[str] | None = None) -> str:
        chg = "".join(f"<li>{_html.escape(c)}</li>" for c in (changes or []))
        return (
            f"<div class='plat'><div class='pn'>{_html.escape(p)}</div>{img}"
            f"<code>{_html.escape(code)}</code><ul>{chg}</ul></div>"
        )

    def _img(rel: str | None) -> str:
        if rel:
            return f"<img src='{_html.escape(rel)}?v={bust}' loading='lazy'/>"
        return "<div class='nogif'>no gif</div>"

    blocks = []
    for idx in cue_ids:
        cue = cue_names.get(idx, "?")
        gt_plats = []
        for p in platforms:
            gt_plats.append(
                _plat_block(p, _img(_gt_gif_rel(p, idx, cue, RENDER_DIR)), _gt_code(p, idx))
            )
        cells = [
            "<div class='ctx gt'><h4>순정</h4>"
            "<p class='cl'>컨텍스트 없이 작성한 순정 Motion Program</p>"
            f"{''.join(gt_plats)}</div>"
        ]
        for ctx_key in ctx_order:
            per_plat = []
            for p in platforms:
                row = data[p].get((idx, ctx_key))
                if not row:
                    continue
                per_plat.append(
                    _plat_block(
                        p,
                        _img(_gif_rel(p, row, RENDER_DIR)),
                        row["source_code"],
                        list(row.get("changes") or []),
                    )
                )
            label = (data[platforms[0]].get((idx, ctx_key)) or {}).get("context_label") or ""
            cells.append(
                f"<div class='ctx'><h4>{_html.escape(ctx_key)}</h4>"
                f"<p class='cl'>{_html.escape(label)}</p>"
                f"{''.join(per_plat)}</div>"
            )
        blocks.append(
            f"<section><h2>c{idx:02d} {_html.escape(cue)}</h2>"
            f"<div class='row'>{''.join(cells)}</div></section>"
        )

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Social-nav cue x context — LLM restaging</title><style>
body{{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#111}}
h2{{font-size:16px;margin:24px 0 6px;position:sticky;left:0}}
.row{{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}}
.ctx{{min-width:250px;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}}
.ctx.gt{{border-color:#1565c0;background:#f4f8ff}}
.ctx h4{{margin:0 0 2px;font-size:11px;color:#e65100;text-transform:uppercase;letter-spacing:.04em}}
.ctx.gt h4{{color:#1565c0}}
.cl{{margin:0 0 8px;font-size:12px;line-height:1.35;color:#333}}
.plat{{margin-bottom:10px}}
.pn{{font-size:11px;font-weight:600;color:#1565c0}}
img{{width:100%;border-radius:6px;background:#eee}}
.nogif{{height:120px;display:grid;place-items:center;color:#999;font-size:11px;background:#f0f0f0;border-radius:6px}}
code{{display:block;font-size:10px;white-space:pre-wrap;color:#333;margin:4px 0}}
ul{{margin:2px 0 0 14px;padding:0;font-size:10px;color:#555}}
</style></head><body>
<h1>Social navigation: cue × context</h1>
<p>First column is the unmodified human-authored GIF. The rest are the same cue restaged for each situation.</p>
{''.join(blocks)}
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return out


# ---------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="roomba,stretch")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--cues", default="", help="comma-separated cue idx filter")
    ap.add_argument("--contexts", default="", help="comma-separated context key filter")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true", help="regenerate even if a clean row exists")
    ap.add_argument("--dry-run", action="store_true", help="print the first filled prompt and stop")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--gallery", action="store_true")
    args = ap.parse_args()

    platforms = [p.strip() for p in args.platform.split(",") if p.strip()]
    bad = [p for p in platforms if p not in PLATFORMS]
    if bad:
        ap.error(f"unknown platform(s): {bad}; choose from {PLATFORMS}")
    only_cues = {int(v) for v in args.cues.split(",") if v.strip()} or None
    only_contexts = {v.strip() for v in args.contexts.split(",") if v.strip()} or None

    for platform in platforms:
        if not args.skip_generate:
            run_platform(
                platform,
                model=args.model,
                retries=args.retries,
                workers=args.workers,
                only_cues=only_cues,
                only_contexts=only_contexts,
                force=args.force,
                dry_run=args.dry_run,
            )
        if args.render and not args.dry_run:
            render_platform(platform, force=args.force_render, fps=args.fps)
    if args.gallery and not args.dry_run:
        build_gallery(platforms, only_cues=only_cues)


if __name__ == "__main__":
    main()
