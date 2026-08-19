#!/usr/bin/env python3
"""Corr-2 regen for HL (humanoid/tiago) Human-NO cells.

Input per cell (two information sources):
  - Original Gemini render + Motion Code (ee-translate)
  - After corr-1 (R1) render + Motion Code
Videos are attached side-by-side (hstack) and both codes appear in the prompt.

Example::

    source APIKEY.sh
    python adhoc/generation/robotarm/run_embodiment_regen_corr2_hl.py \\
      --model gemini-2.5-pro --workers 4
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_exp0_gemini import (  # noqa: E402
    _extract_json_obj,
    embodiment_schema_yaml,
    gif_path_for,
    render_pending_once,
)
from run_embodiment_regen_gemini import (  # noqa: E402
    CUES_JSON,
    GEMINI_MASTER,
    GT_MASTER,
    RENDER_GT,
    RENDER_SRC_EE_TRANSLATE,
    _call_regen,
    _client,
    _extract_revised_code,
    _index_master,
    _parse_meta,
    analyze_edits,
    write_analysis_md,
)

R1_DIR = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gemini25pro_regen_hl_ee_translate_r1"
)
DEFAULT_OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gemini25pro_regen_hl_ee_translate_r2"
)
ORIG_RENDER = RENDER_SRC_EE_TRANSLATE


def _find_mp4(root: Path, platform: str, idx: int, cue: str) -> Path:
    name = f"emb_g{idx:02d}_{cue}.mp4"
    candidates = [
        root / platform / "mp4" / name,
        root / platform / "gif" / name,
        root / "render" / platform / "mp4" / name,
        root / "render" / platform / "gif" / name,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _hstack_pair(left: Path, right: Path, out: Path) -> Path:
    """Original | After side-by-side MP4 for multimodal input."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_mtime >= max(left.stat().st_mtime, right.stat().st_mtime):
        return out
    ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(left),
        "-i",
        str(right),
        "-filter_complex",
        (
            "[0:v]scale=480:480:force_original_aspect_ratio=decrease,"
            "pad=480:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=20[a];"
            "[1:v]scale=480:480:force_original_aspect_ratio=decrease,"
            "pad=480:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=20[b];"
            "[a][b]hstack=inputs=2:shortest=1[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def build_corr2_prompt(
    *,
    platform: str,
    cue: str,
    description: str,
    original_code: str,
    after_code: str,
) -> str:
    schema = embodiment_schema_yaml(platform)
    return f"""You are revising a robot Motion Code for a cue that a human judge marked NOT identifiable.

You are given TWO information sources for the same cue on the same embodiment:
1) ORIGINAL — left panel of the attached side-by-side video, plus its Motion Code (first Gemini attempt).
2) AFTER (corr1) — right panel of the attached video, plus the corr1 revised Motion Code.

Use BOTH videos and BOTH codes as evidence. Notice what corr1 fixed, what still fails, and what may have gotten worse. Then produce a further revised Motion Code (corr2) so the TARGET cue is identifiable.

Rules:
- Stay inside the embodiment schema. Every Motion Code MUST start with pose(...).
- Total steps: 2~8. Join steps with +.
- Prefer targeted fixes over a full rewrite when corr1 already looks partly right.
- Do not invent attributes outside the schema.

[TARGET CUE]
name: {cue}
description: {description}

[ORIGINAL MOTION CODE]
{original_code}

[AFTER MOTION CODE (corr1)]
{after_code}

[EMBODIMENT SCHEMA — {platform}]
# Source: data/seed/yml/motion_code_embodiment.yaml ({platform} only, anchors resolved)
{schema}

[Output]
Write exactly these lines first:
# IdentifiableQ: <yes/no — can a Motion Code revision make this cue identifiable?>
# ChangeSummary: <one sentence: what you change vs corr1 and why, using both sources>
# ChangeTags: <comma-separated short tags from: dir/facing, grip, xyz/percentile, path_shape, path_axis, speed, hold_time, repetition, joint/channel, add_component, remove_component, specify_attr, embodiment_limit, other>

Then output ONE revised Motion Code line only (steps joined by +). No JSON, no extra commentary.
Every Motion Code MUST start with pose(...).
""".strip()


def write_corr2_gallery(
    rows: list[dict[str, Any]],
    *,
    out_html: Path,
    orig_render: Path,
    r1_render: Path,
    r2_render: Path,
    r1_codes: dict[str, str],
    gt_codes: dict[tuple[int, str], str],
    title: str,
    subtitle: str,
) -> None:
    def _gif(root: Path, plat: str, idx: int, cue: str) -> Path:
        p = gif_path_for(root, plat, idx, cue)
        if p.is_file():
            return p
        alt = root / plat / "gif" / f"emb_g{idx:02d}_{cue}.gif"
        return alt

    def media(path: Path, label: str) -> str:
        if path.is_file():
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

    sections = []
    for r in rows:
        plat, idx, cue = r["platform"], int(r["idx"]), r["cue"]
        jid = r.get("job_id") or f"{plat}.g{idx:02d}"
        meta = r.get("meta") or {}
        sections.append(
            f'<section class="cue"><h2>{plat} · g{idx:02d} {html_mod.escape(cue)}</h2>'
            f'<p class="note">human note: {html_mod.escape(r.get("human_note") or "")} · '
            f'IdentifiableQ: {html_mod.escape(meta.get("identifiable_q") or "")} · '
            f'tags: {html_mod.escape(meta.get("change_tags") or "")}</p>'
            f'<p class="sum">{html_mod.escape(meta.get("change_summary") or "")}</p>'
            f'<div class="grid">'
            f'{media(_gif(RENDER_GT, plat, idx, cue), "GT")}'
            f'{media(_gif(orig_render, plat, idx, cue), "Original")}'
            f'{media(_gif(r1_render, plat, idx, cue), "After (corr1)")}'
            f'{media(_gif(r2_render, plat, idx, cue), "After (regen R2 · ee-translate)")}'
            f"</div>"
            f'<div class="codes">'
            f'{code_col("GT", gt_codes.get((idx, plat), ""))}'
            f'{code_col("Original", r.get("original_code") or r.get("old_code") or "")}'
            f'{code_col("After (corr1)", r1_codes.get(jid, r.get("after_code") or ""))}'
            f'{code_col("New Motion Code (regen R2)", r.get("new_code") or "")}'
            f"</div></section>"
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{html_mod.escape(title)}</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.col h3{{font-size:.85rem;margin:0 0 6px;font-weight:600}}
.col img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}}
.miss{{color:#f28b82}}
.note,.sum{{font-size:.85rem;color:#c4c7ce}}
.codes{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}}
.lab{{font-size:.7rem;color:#9aa0a6;text-transform:uppercase}}
pre{{font-size:11px;white-space:pre-wrap;background:#0a0c10;padding:8px;border-radius:6px;max-height:160px;overflow:auto}}
@media (max-width:1100px){{.grid,.codes{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<h1>{html_mod.escape(title)}</h1>
<p class="meta">{html_mod.escape(subtitle)}</p>
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
    ap.add_argument("--platforms", default="humanoid,tiago")
    ap.add_argument("--r1-dir", type=Path, default=R1_DIR)
    ap.add_argument("--orig-render", type=Path, default=ORIG_RENDER)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    # Also attach both videos as separate parts (in addition to hstack).
    ap.add_argument(
        "--dual-parts",
        action="store_true",
        help="attach original+after as two video parts instead of one hstack",
    )
    args = ap.parse_args()

    # Corr2 renders use EE-translate / trim (same as R1 visual root).
    os.environ.setdefault("G1_POSE_XYZ_MODE", "translate")
    os.environ.setdefault("MOBILE_POSE_XYZ_MODE", "trim")

    allow = {p.strip() for p in args.platforms.split(",") if p.strip()}
    r1_dir = args.r1_dir.expanduser().resolve()
    orig_render = args.orig_render.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    master_out = out_dir / "regen_motion_code_all.json"
    render_root = out_dir / "render"
    pair_dir = out_dir / "input_pairs"
    html_out = out_dir / "regen_gallery.html"
    analysis_json = out_dir / "regen_analysis.json"
    analysis_md = out_dir / "regen_analysis.md"

    r1_rows: dict[str, dict[str, Any]] = {}
    for line in (r1_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("platform") not in allow or r.get("error"):
            continue
        r1_rows[r["job_id"]] = r
    if args.max_items > 0:
        keep = sorted(r1_rows)[: args.max_items]
        r1_rows = {k: r1_rows[k] for k in keep}

    gemini_index = _index_master(
        json.loads(GEMINI_MASTER.read_text(encoding="utf-8"))
    )
    cues = {
        int(c["idx"]): c
        for c in json.loads(CUES_JSON.read_text(encoding="utf-8"))
    }

    done: dict[str, dict[str, Any]] = {}
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["job_id"]] = row

    jobs = [jid for jid in sorted(r1_rows) if jid not in done]
    print(
        f"corr2: resume={len(done)} pending={len(jobs)}/{len(r1_rows)} "
        f"model={args.model} workers={args.workers} dual_parts={args.dual_parts}",
        flush=True,
    )
    client = _client() if jobs else None

    def work(job_id: str) -> dict[str, Any]:
        prev = r1_rows[job_id]
        plat = prev["platform"]
        idx = int(prev["idx"])
        cue = prev["cue"]
        desc = (
            prev.get("description")
            or str(cues.get(idx, {}).get("description") or "")
            or ""
        )
        info = gemini_index.get((idx, plat)) or {}
        original_code = str(
            prev.get("old_code")
            or ((info.get("cell") or {}).get("source_code") or "")
        ).strip()
        after_code = str(prev.get("new_code") or "").strip()
        orig_mp4 = _find_mp4(orig_render, plat, idx, cue)
        after_mp4 = _find_mp4(r1_dir, plat, idx, cue)
        if not orig_mp4.is_file() or not after_mp4.is_file():
            return {
                "platform": plat,
                "idx": idx,
                "cue": cue,
                "description": desc,
                "job_id": job_id,
                "human_note": prev.get("human_note") or "",
                "original_code": original_code,
                "after_code": after_code,
                "old_code": original_code,
                "error": f"missing_video:orig={orig_mp4.is_file()} after={after_mp4.is_file()}",
            }

        prompt = build_corr2_prompt(
            platform=plat,
            cue=cue,
            description=desc,
            original_code=original_code,
            after_code=after_code,
        )
        if args.dual_parts:
            video_args: dict[str, Any] = {"videos": [orig_mp4, after_mp4]}
            pair_path = None
        else:
            pair_path = pair_dir / plat / f"emb_g{idx:02d}_{cue}_orig_after.mp4"
            _hstack_pair(orig_mp4, after_mp4, pair_path)
            video_args = {"video": pair_path}

        last_err = None
        for attempt in range(args.retries + 1):
            try:
                raw, usage = _call_regen(
                    client=client, model=args.model, prompt=prompt, **video_args
                )
                meta = _parse_meta(raw)
                obj = _extract_json_obj(raw)
                new_code = _extract_revised_code(raw, obj)
                converted = (
                    convert_code(new_code, platform=plat)
                    if new_code
                    else {
                        "movements": [],
                        "parse_ok": False,
                        "notes": ["empty_extract"],
                        "source_code": "",
                    }
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return {
                    "platform": plat,
                    "idx": idx,
                    "cue": cue,
                    "description": desc,
                    "job_id": job_id,
                    "human_note": prev.get("human_note") or "",
                    "original_code": original_code,
                    "after_code": after_code,
                    # analyze_edits diffs old→new; baseline for corr2 is corr1.
                    "old_code": after_code,
                    "new_code": converted.get("source_code") or new_code,
                    "movements": converted.get("movements") or [],
                    "parse_ok": bool(converted.get("parse_ok")),
                    "notes": converted.get("notes") or [],
                    "meta": meta,
                    "raw_output": raw,
                    "usage": usage,
                    "model": args.model,
                    "strategy": "corr2_orig_after",
                    "round": 2,
                    "mp4_original": str(orig_mp4.relative_to(_REPO)),
                    "mp4_after": str(after_mp4.relative_to(_REPO)),
                    "mp4_pair": (
                        str(pair_path.relative_to(_REPO)) if pair_path else None
                    ),
                }
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 20))
        return {
            "platform": plat,
            "idx": idx,
            "cue": cue,
            "job_id": job_id,
            "original_code": original_code,
            "after_code": after_code,
            "old_code": original_code,
            "error": last_err or "unknown",
            "round": 2,
        }

    completed = 0
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(work, jid): jid for jid in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                with pred_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["job_id"]] = row
                completed += 1
                if completed % 3 == 0 or completed == len(jobs):
                    print(
                        f"corr2: {len(done)}/{len(r1_rows)} (+{completed}/{len(jobs)})",
                        flush=True,
                    )

    ordered = [done.get(jid) or {**r1_rows[jid], "error": "missing"} for jid in sorted(r1_rows)]

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
            "tiago": "tiago",
            "humanoid": "unitree_g1",
        }.get(plat, plat)
        entry["platforms"][plat] = {
            "idx": idx,
            "cue": row["cue"],
            "description": row.get("description") or "",
            "movements": row.get("movements") or [],
            "state": "motion_code_regen",
            "model": args.model,
            "time": datetime.now().isoformat(timespec="seconds"),
            "experiment": "embodiment_regen_hl_ee_translate_corr2",
            "generation_valid": bool(row.get("movements")),
            "platform": plat,
            "domain": domain,
            "parse_ok": True,
            "notes": row.get("notes") or [],
            "source_code": row.get("new_code") or "",
        }
    configs = [by_cue[i] for i in sorted(by_cue)]
    master_out.write_text(
        json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.skip_render and configs:
        print("[render] corr2 Motion Codes…", flush=True)
        render_pending_once._skip = set()  # type: ignore[attr-defined]
        for _ in range(40):
            n = render_pending_once(
                configs,
                render_root=render_root,
                out_json=master_out,
                platforms=sorted(allow),
                force=args.force_render,
                max_items=6,
            )
            if n == 0:
                break
            print(f"[render] attempted {n}", flush=True)

    analysis = analyze_edits(ordered)
    # analyze_edits diffs old_code vs new_code; store after_code as baseline too
    analysis_json.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_analysis_md(analysis, ordered, analysis_md)

    r1_codes = {jid: str(r.get("new_code") or "") for jid, r in r1_rows.items()}
    gt_codes = {
        (int(e["idx"]), p): str((c or {}).get("source_code") or "")
        for e in json.loads(GT_MASTER.read_text(encoding="utf-8"))
        for p, c in (e.get("platforms") or {}).items()
    }
    write_corr2_gallery(
        ordered,
        out_html=html_out,
        orig_render=orig_render,
        r1_render=r1_dir / "render",
        r2_render=render_root,
        r1_codes=r1_codes,
        gt_codes=gt_codes,
        title="HL Human-NO · corr2 (original + after input)",
        subtitle=(
            f"n={len(ordered)} · input=hstack(original|corr1)+both codes · "
            f"model={args.model} · ee-translate render"
        ),
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "n": len(ordered),
                "strategy": "corr2_orig_after",
                "r1_dir": str(r1_dir.relative_to(_REPO)),
                "analysis": analysis,
                "outputs": {
                    "predictions": str(pred_path.relative_to(_REPO)),
                    "master_json": str(master_out.relative_to(_REPO)),
                    "gallery": str(html_out.relative_to(_REPO)),
                    "render_root": str(render_root.relative_to(_REPO)),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ok = sum(1 for r in ordered if r.get("parse_ok") and not r.get("error"))
    print(f"corr2 done parse_ok={ok}/{len(ordered)} gallery → {html_out}", flush=True)


if __name__ == "__main__":
    main()
