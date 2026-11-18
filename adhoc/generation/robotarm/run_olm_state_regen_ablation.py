#!/usr/bin/env python3
"""3 cues × 3 regen methods (simple / video+O-L-M / O-L-M only).

G1 + Tiago. Writes a 3×3 HTML gallery.

    source APIKEY.sh
    python adhoc/generation/robotarm/run_olm_state_regen_ablation.py
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
for p in (_REPO, _HERE, _PAPER):
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
from run_embodiment_regen_gemini import (  # noqa: E402
    _call_regen,
    _extract_revised_code,
    _parse_meta,
    build_regen_prompt,
)
from score_humanoid_loc_dir_move import (  # noqa: E402
    _effective_xyz,
    _first_pose,
    _palm_normal,
    _reach,
    _tiago_effective_xyz,
    PALM_VEC,
    REACH_VEC,
)

DEFAULT_CUES = ("cover_yawn_sleepy", "salute", "arms_open_welcome")
HARD_CUES = ("highfive_invite", "cross_arms_defensive", "stop_palm_out")
METHODS = ("simple", "video_olm", "olm_only")
METHOD_LABEL = {
    "simple": "1) simple (mp4 + YAML)",
    "video_olm": "2) mp4 + simulator O/L/M",
    "olm_only": "3) simulator O/L/M only",
}
PLATFORMS = ("humanoid", "tiago")

GEMINI_MASTER = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gemini_humanlike90"
    / "embodiment_motion_code_all.json"
)
GT_MASTER = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
CUES_JSON = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
RENDER_EE = (
    _REPO
    / "data/results/render"
    / "embodiment_motion_code_gemini_gemini-2.5-pro_ee_translate"
)
RENDER_PRO = (
    _REPO / "data/results/render" / "embodiment_motion_code_gemini_gemini-2.5-pro"
)
RENDER_GT = _REPO / "data/results/render" / "embodiment_motion_code_gt"
DEFAULT_OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_olm_state_regen_3x3"
)
HTML_COPY = _PAPER / "olm_regen_3x3" / "gallery.html"


def _client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
    return genai.Client(api_key=key)


def _call_text(*, client: Any, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    from google.genai import types

    config_kwargs: dict[str, Any] = {"temperature": 0.2, "max_output_tokens": 2048}
    try:
        budget = 128 if "pro" in model.lower() else 0
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
        config_kwargs["max_output_tokens"] = max(2048, budget + 1024)
    except Exception:
        pass
    config = types.GenerateContentConfig(**config_kwargs)
    resp = client.models.generate_content(
        model=model, contents=[prompt], config=config
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


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _interval(val: Any) -> str | None:
    if isinstance(val, (list, tuple)) and len(val) >= 2:
        a, b = val[0], val[-1]
        return f"({a}, {b})"
    if isinstance(val, (int, float)):
        return f"(0, {val})"
    return None


def _movement_lines(converted: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for step in converted.get("movements") or []:
        t = step.get("type")
        p = step.get("parameters") or {}
        if t == "path":
            path = p.get("path") if isinstance(p.get("path"), dict) else p
            axis = str(path.get("axis") or path.get("plane") or "?").lower()
            iv = _interval(path.get("distance") or path.get("degrees"))
            shape = str(path.get("shape") or "line")
            if iv:
                lines.append(f"moved along the {axis} axis {iv} ({shape})")
            else:
                lines.append(f"moved along the {axis} axis (path {shape})")
        elif t == "movement":
            mov = p.get("movement") if isinstance(p.get("movement"), dict) else p
            joints = mov.get("joints") or []
            if not joints and (mov.get("axis") or mov.get("channel")):
                joints = [mov]
            for j in joints:
                axis = str(j.get("axis") or j.get("channel") or "?").lower()
                ch = str(j.get("channel") or j.get("joint") or j.get("part") or "").lower()
                iv = _interval(j.get("degrees") if j.get("degrees") is not None else j.get("distance"))
                who = f"{ch} " if ch else ""
                if iv:
                    lines.append(f"{who}moved along the {axis} axis {iv}".strip())
                elif j.get("height") is not None:
                    lines.append(f"{who}height → {j.get('height')}".strip())
        elif t == "hold":
            continue
    if not lines:
        lines.append("no axis travel (hold / static pose only)")
    return lines


def format_sim_olm(platform: str, converted: dict[str, Any]) -> str:
    """Dump current pose/move as simulator-facing O / L / M text."""
    pose = _first_pose(converted) or {}
    reach = _reach(pose)
    palm = _palm_normal(pose)
    dir_tok = str(pose.get("arm_position") or pose.get("dir") or "unspecified")
    grip_tok = str(pose.get("gripper_orientation") or pose.get("grip") or "unspecified")
    if reach is None:
        reach_s = "unspecified (no reach dir)"
    else:
        reach_s = f"{_angle(reach, REACH_VEC['front']):.0f}° from front (+X); dir={dir_tok}"
    if palm is None:
        palm_s = "unspecified (no palm normal)"
    else:
        palm_s = f"{_angle(palm, PALM_VEC['palmup']):.0f}° from palm-up (+Z); grip={grip_tok}"
    if platform == "tiago":
        xyz, meta = _tiago_effective_xyz(pose) if pose else ({}, {"source": "empty"})
    else:
        xyz, meta = _effective_xyz(pose) if pose else ({}, {"source": "empty"})
    x = xyz.get("x")
    y = xyz.get("y")
    z = xyz.get("z")

    def _n(v: Any) -> str:
        if isinstance(v, (int, float)):
            return f"{v:.0f}"
        return "?"

    loc = f"(x={_n(x)}, y={_n(y)}, z={_n(z)})"
    src = meta.get("source")
    moves = _movement_lines(converted)
    move_txt = "; ".join(moves)
    src_code = str(converted.get("source_code") or "")
    raw_moves = [
        tok.strip()
        for tok in src_code.split("+")
        if tok.strip().lower().startswith(("move(", "path(", "movement("))
    ]
    raw_line = ""
    if raw_moves:
        raw_line = "\n  Motion-code steps: " + " + ".join(raw_moves)
    return (
        "Orientation:\n"
        f"  reach: {reach_s}\n"
        f"  palm: {palm_s}\n"
        f"Location: {loc}  [body %; source={src}]\n"
        f"Movement: {move_txt}{raw_line}"
    )


def build_olm_prompt(
    *,
    platform: str,
    cue: str,
    description: str,
    motion_code: str,
    olm_text: str,
    include_video: bool,
    include_code: bool,
) -> str:
    schema = embodiment_schema_yaml(platform)
    video_line = (
        "1) the rendered video (attached) — current Gemini-made motion,\n"
        if include_video
        else ""
    )
    n = 1 + int(include_video)
    code_block = ""
    if include_code:
        code_block = f"\n[CURRENT MOTION CODE]\n{motion_code}\n"
    given = []
    if include_video:
        given.append("the rendered video")
    given.append("the current orientation / location / movement readout from the simulator")
    if include_code:
        given.append("the Motion Code that produced it")
    given.append("the target cue")
    given.append("the embodiment schema YAML")
    numbered = "\n".join(f"{i}) {g}" for i, g in enumerate(given, start=1))
    video_hint = (
        "Look at the attached video together with the simulator readout.\n"
        if include_video
        else "You are NOT given a video. Use only the simulator readout (and code if present).\n"
    )
    return f"""You are reviewing a robot gesture that a human judge marked as NOT identifiable as the target cue.

You are given:
{numbered}

The simulator readout is what the renderer actually realizes (body-% location, reach/palm angles, axis travel). Use it to correct orientation, location, and movement.

Task:
- Revise the Motion Code so the cue becomes identifiable on this embodiment.
- Stay inside the schema. Every Motion Code MUST start with pose(...).
- Total steps: 2~8. Join steps with +.

{video_hint}
Fix recognition-critical handles:
- Orientation: reach direction and palm facing (the angles below are from the current pose).
- Location: (x, y, z) in body-relative percent.
- Movement: which axis the end-effector / joints actually traveled.

[CUE]
name: {cue}
description: {description}

[CURRENT SIMULATOR STATE]
{olm_text}
{code_block}
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


def _index_by_cue(
    master: list[dict[str, Any]], cues: tuple[str, ...] | list[str]
) -> dict[str, dict[str, Any]]:
    want = set(cues)
    out: dict[str, dict[str, Any]] = {}
    for entry in master:
        cue = str(entry.get("cue") or "")
        if cue not in want:
            continue
        n = len(entry.get("platforms") or {})
        if cue not in out or n > len((out[cue].get("platforms") or {})):
            out[cue] = entry
    return out


def resolve_mp4(platform: str, idx: int, cue: str) -> Path | None:
    name = f"emb_g{idx:02d}_{cue}.mp4"
    for root in (RENDER_EE, RENDER_PRO):
        for rel in (
            Path(platform) / "mp4" / name,
            Path(platform) / "gif" / name,
        ):
            p = root / rel
            if p.is_file():
                return p
    return None


def resolve_gif(root: Path, platform: str, idx: int, cue: str) -> Path | None:
    gif = gif_path_for(root, platform, idx, cue)
    if gif.is_file():
        return gif
    mp4 = gif.with_suffix(".mp4")
    if mp4.is_file():
        return mp4
    return None


def media_tag(path: Path | None, html_parent: Path) -> str:
    if path is None or not path.is_file():
        return '<p class="miss">missing</p>'
    rel = os.path.relpath(path, html_parent)
    bust = int(path.stat().st_mtime)
    src = html_mod.escape(f"{rel}?v={bust}")
    if path.suffix.lower() == ".mp4":
        return (
            f'<video src="{src}" autoplay loop muted playsinline '
            f'style="width:100%;aspect-ratio:1;object-fit:cover;background:#000;border-radius:8px"></video>'
        )
    return f'<img src="{src}" loading="lazy">'


def write_gallery(
    jobs: list[dict[str, Any]],
    *,
    out_html: Path,
    render_root: Path,
    cue_meta: dict[str, dict[str, Any]],
    cues: tuple[str, ...] | list[str],
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    by: dict[tuple[str, str, str], dict[str, Any]] = {
        (j["cue"], j["method"], j["platform"]): j for j in jobs
    }

    def cell(cue: str, method: str) -> str:
        bits = [f'<div class="method"><h3>{html_mod.escape(METHOD_LABEL[method])}</h3>']
        for plat in PLATFORMS:
            j = by.get((cue, method, plat), {})
            idx = int(cue_meta[cue]["idx"])
            orig = resolve_gif(RENDER_EE, plat, idx, cue) or resolve_gif(
                RENDER_PRO, plat, idx, cue
            )
            new = resolve_gif(render_root, plat, idx, cue)
            # method-specific render lives under render_root / method / platform
            new_m = resolve_gif(render_root / method, plat, idx, cue)
            if new_m is not None:
                new = new_m
            err = j.get("error") or ""
            bits.append(
                f'<div class="plat"><div class="plat-h">{plat}</div>'
                f'<div class="pair">'
                f'<div><div class="lab">before</div>{media_tag(orig, out_html.parent)}</div>'
                f'<div><div class="lab">after</div>{media_tag(new, out_html.parent)}</div>'
                f"</div>"
                f'<pre class="code">{html_mod.escape(j.get("new_code") or err or "")}</pre>'
                f'<p class="sum">{html_mod.escape((j.get("meta") or {}).get("change_summary") or "")}</p>'
                f'<pre class="olm">{html_mod.escape(j.get("olm_text") or "")}</pre>'
                f"</div>"
            )
        bits.append("</div>")
        return "".join(bits)

    sections = []
    for cue in cues:
        entry = cue_meta[cue]
        idx = int(entry["idx"])
        desc = str(entry.get("description") or "")
        old_h = ((entry.get("platforms") or {}).get("humanoid") or {}).get("source_code") or ""
        old_t = ((entry.get("platforms") or {}).get("tiago") or {}).get("source_code") or ""
        rows = "".join(cell(cue, m) for m in METHODS)
        sections.append(
            f'<section class="cue"><h2>g{idx:02d} {html_mod.escape(cue)}</h2>'
            f'<p class="desc">{html_mod.escape(desc)}</p>'
            f'<div class="olds"><div><div class="lab">G1 old</div><pre>{html_mod.escape(old_h)}</pre></div>'
            f'<div><div class="lab">Tiago old</div><pre>{html_mod.escape(old_t)}</pre></div></div>'
            f'<div class="methods">{rows}</div></section>'
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>O/L/M regen 3×3</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.cue{{margin:16px;padding:14px;border:1px solid #333;border-radius:12px;background:#1a1d24}}
.desc{{color:#c4c7ce}}
.olds,.methods{{display:grid;gap:12px}}
.methods{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.olds{{grid-template-columns:1fr 1fr;margin-bottom:12px}}
.method{{background:#12141a;border-radius:10px;padding:10px}}
.method h3{{margin:0 0 8px;font-size:.9rem}}
.plat{{margin-top:10px;border-top:1px solid #2a2e38;padding-top:8px}}
.plat-h{{font-weight:700;font-size:.8rem;color:#8ab4f8;text-transform:uppercase}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.lab{{font-size:.7rem;color:#9aa0a6;margin-bottom:4px}}
img,video{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:8px;border-radius:6px;max-height:140px;overflow:auto}}
.olm{{max-height:110px;color:#c5e1a5}}
.sum{{font-size:.8rem;color:#e8eaed}}
.miss{{color:#f28b82}}
@media (max-width:1100px){{.methods{{grid-template-columns:1fr}}}}
</style></head>
<body>
<h1>O/L/M regen · {len(cues)} cues × 3 methods</h1>
<p class="meta">G1 + Tiago · one-shot Gemini · {html_mod.escape(datetime.now().isoformat(timespec="seconds"))}</p>
{"".join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def _cell_to_config(row: dict[str, Any], *, experiment: str) -> dict[str, Any]:
    plat = row["platform"]
    domain = {"humanoid": "unitree_g1", "tiago": "tiago"}.get(plat, plat)
    return {
        "idx": int(row["idx"]),
        "cue": row["cue"],
        "description": row.get("description") or "",
        "movements": row.get("movements") or [],
        "state": "motion_code_regen",
        "model": row.get("model") or "",
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "generation_valid": bool(row.get("movements")),
        "platform": plat,
        "domain": domain,
        "parse_ok": bool(row.get("parse_ok")),
        "notes": row.get("notes") or [],
        "source_code": row.get("new_code") or "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument(
        "--cues",
        default=",".join(HARD_CUES),
        help="Comma-separated cue names to run (default: extra hard set).",
    )
    args = ap.parse_args()
    run_cues = tuple(c.strip() for c in args.cues.split(",") if c.strip())

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    html_out = out_dir / "gallery.html"

    master = json.loads(GEMINI_MASTER.read_text(encoding="utf-8"))
    cues_desc = {
        int(c["idx"]): c for c in json.loads(CUES_JSON.read_text(encoding="utf-8"))
    }

    done: dict[str, dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["job_id"]] = row

    gallery_cues = list(
        dict.fromkeys(
            list(DEFAULT_CUES)
            + [str(r.get("cue") or "") for r in done.values() if r.get("cue")]
            + list(run_cues)
        )
    )
    gallery_cues = [c for c in gallery_cues if c]
    cue_index = _index_by_cue(master, gallery_cues)
    missing = [c for c in run_cues if c not in cue_index]
    if missing:
        raise SystemExit(f"missing cues in master: {missing}")

    cue_meta: dict[str, dict[str, Any]] = {}
    for cue, entry in cue_index.items():
        idx = int(entry["idx"])
        desc = str(
            cues_desc.get(idx, {}).get("description") or entry.get("description") or ""
        )
        cue_meta[cue] = {**entry, "description": desc}

    jobs_spec: list[dict[str, Any]] = []
    for cue in run_cues:
        entry = cue_meta[cue]
        idx = int(entry["idx"])
        desc = str(entry["description"])
        for plat in PLATFORMS:
            cell = (entry.get("platforms") or {}).get(plat) or {}
            old_code = str(cell.get("source_code") or "").strip()
            converted = convert_code(old_code, platform=plat) if old_code else {
                "movements": [],
                "parse_ok": False,
            }
            olm = format_sim_olm(plat, converted)
            video = resolve_mp4(plat, idx, cue)
            for method in METHODS:
                job_id = f"{plat}.{cue}.{method}"
                jobs_spec.append(
                    {
                        "job_id": job_id,
                        "platform": plat,
                        "cue": cue,
                        "idx": idx,
                        "description": desc,
                        "method": method,
                        "old_code": old_code,
                        "olm_text": olm,
                        "mp4": str(video) if video else "",
                    }
                )

    pending = [j for j in jobs_spec if j["job_id"] not in done]
    print(
        f"olm-regen: resume={len(done)} pending={len(pending)}/{len(jobs_spec)} "
        f"model={args.model}",
        flush=True,
    )
    for j in jobs_spec:
        print(f"  [{j['platform']}] {j['cue']} {j['method']}\n{j['olm_text']}\n", flush=True)

    if not args.skip_llm and pending:
        client = _client()

        def work(spec: dict[str, Any]) -> dict[str, Any]:
            plat, cue, method = spec["platform"], spec["cue"], spec["method"]
            video = Path(spec["mp4"]) if spec["mp4"] else None
            if method == "simple":
                prompt = build_regen_prompt(
                    platform=plat,
                    cue=cue,
                    description=spec["description"],
                    motion_code=spec["old_code"],
                    strategy="default",
                )
                use_video = True
            elif method == "video_olm":
                prompt = build_olm_prompt(
                    platform=plat,
                    cue=cue,
                    description=spec["description"],
                    motion_code=spec["old_code"],
                    olm_text=spec["olm_text"],
                    include_video=True,
                    include_code=True,
                )
                use_video = True
            else:
                prompt = build_olm_prompt(
                    platform=plat,
                    cue=cue,
                    description=spec["description"],
                    motion_code=spec["old_code"],
                    olm_text=spec["olm_text"],
                    include_video=False,
                    include_code=True,
                )
                use_video = False
            last_err = None
            for attempt in range(args.retries + 1):
                try:
                    if use_video:
                        if video is None or not video.is_file():
                            raise FileNotFoundError(f"missing mp4 for {spec['job_id']}")
                        raw, usage = _call_regen(
                            client=client, model=args.model, prompt=prompt, video=video
                        )
                    else:
                        raw, usage = _call_text(
                            client=client, model=args.model, prompt=prompt
                        )
                    meta = _parse_meta(raw)
                    obj = _extract_json_obj(raw)
                    new_code = _extract_revised_code(raw, obj)
                    converted = (
                        convert_code(new_code, platform=plat) if new_code else {
                            "movements": [],
                            "parse_ok": False,
                            "notes": ["empty_extract"],
                            "source_code": "",
                        }
                    )
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                    return {
                        **spec,
                        "new_code": converted.get("source_code") or new_code,
                        "movements": converted.get("movements") or [],
                        "parse_ok": bool(converted.get("parse_ok")),
                        "notes": converted.get("notes") or [],
                        "meta": meta,
                        "raw_output": raw,
                        "usage": usage,
                        "model": args.model,
                        "prompt": prompt,
                    }
                except Exception as exc:  # noqa: BLE001
                    last_err = str(exc)
                    time.sleep(min(2**attempt, 20))
            return {**spec, "error": last_err or "unknown"}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(work, spec): spec for spec in pending}
            for fut in as_completed(futs):
                row = fut.result()
                with pred_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["job_id"]] = row
                print(
                    f"done {row['job_id']} parse={row.get('parse_ok')} err={row.get('error','')}",
                    flush=True,
                )

    ordered = [done.get(j["job_id"]) or {**j, "error": "missing"} for j in jobs_spec]

    if not args.skip_render:
        for method in METHODS:
            rows = [r for r in ordered if r.get("method") == method and r.get("parse_ok")]
            if not rows:
                continue
            by_cue: dict[int, dict[str, Any]] = {}
            for row in rows:
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
                entry["platforms"][plat] = _cell_to_config(
                    row, experiment=f"olm_state_regen_{method}"
                )
            configs = [by_cue[i] for i in sorted(by_cue)]
            render_root = out_dir / "render" / method
            master_out = out_dir / f"regen_{method}.json"
            master_out.write_text(
                json.dumps(configs, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[render] {method} ×{len(rows)}", flush=True)
            render_pending_once._skip = set()  # type: ignore[attr-defined]
            for _ in range(20):
                n = render_pending_once(
                    configs,
                    render_root=render_root,
                    out_json=master_out,
                    platforms=list(PLATFORMS),
                    force=args.force_render,
                    max_items=6,
                )
                if n == 0:
                    break
                print(f"[render] {method} attempted {n}", flush=True)

    gallery_jobs = list(ordered)
    seen_ids = {j["job_id"] for j in gallery_jobs}
    for jid, row in done.items():
        if jid not in seen_ids and row.get("cue") in cue_meta:
            gallery_jobs.append(row)
    (out_dir / "jobs.json").write_text(
        json.dumps(gallery_jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_gallery(
        gallery_jobs,
        out_html=html_out,
        render_root=out_dir / "render",
        cue_meta=cue_meta,
        cues=gallery_cues,
    )
    HTML_COPY.parent.mkdir(parents=True, exist_ok=True)
    write_gallery(
        gallery_jobs,
        out_html=HTML_COPY,
        render_root=out_dir / "render",
        cue_meta=cue_meta,
        cues=gallery_cues,
    )
    print(f"html: {html_out}\nhtml copy: {HTML_COPY}", flush=True)


if __name__ == "__main__":
    main()
