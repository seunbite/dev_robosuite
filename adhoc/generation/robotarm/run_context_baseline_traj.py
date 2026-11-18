#!/usr/bin/env python3
"""Few-shot joint / xyzrpy baselines on the social-nav *context* pipeline.

The model sees (cue, context, embodiment) and must emit a trajectory that
stays the same cue while matching the situation. Ours uses Motion Programs;
these baselines use dense vectors (joint angles or xyzrpy).

    source ../../../APIKEY.sh
    python run_context_baseline_traj.py
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import run_embodiment_baseline_traj as bt  # noqa: E402

CONTEXT_YML = _REPO / "data/seed/yml/social_nav_context_10.yml"
SMP_GIF = _REPO / "data/results/render/social_nav_context_llm"
GT_GIF = _REPO / "data/results/render/embodiment_motion_code_gt"
DEFAULT_OUT = _REPO / "data/results/baseline_traj/context_pilot"

CUES = [
    (4, "gaze_shift_avoid"),
    (11, "beckon_come_here"),
    (20, "hands_up_surrender"),
    (15, "highfive_invite"),
    (25, "facepalm"),
]
CONTEXTS = ("urgent_emergency", "quiet_place")
PLATS = ("single_arm", "stretch", "roomba")


def _contexts() -> dict[str, dict[str, Any]]:
    spec = yaml.safe_load(CONTEXT_YML.read_text(encoding="utf-8"))
    return {k: spec["contexts"][k] for k in CONTEXTS}


def _prompt_with_context(
    mode: bt.Mode,
    *,
    plat: str,
    target: dict[str, Any],
    shots: list[dict[str, Any]],
    ctx_key: str,
    ctx: dict[str, Any],
) -> str:
    base = bt._build_prompt(mode, plat=plat, target=target, shots=shots)
    label = str(ctx.get("label") or ctx_key)
    desc = " ".join(str(ctx.get("desc") or "").split())
    return (
        base
        + "\n\n---\n"
        "CONTEXT (same cue, different delivery):\n"
        f"key: {ctx_key}\n"
        f"{label}\n"
        f"{desc}\n\n"
        "Keep the cue identity: a viewer must still recognize the same emblem.\n"
        "Restage timing, scale, approach, dwell, or add a short prelude/hold so "
        "the motion also fits this situation. Urgent should read immediately; "
        "quiet should be smaller, slower, and less intrusive.\n"
        "Do not emit a different emblem. Do not copy an example trajectory.\n"
    )


def generate(args: argparse.Namespace) -> None:
    from google.genai import types
    from tqdm.auto import tqdm

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    gt = bt._index_gt()
    ctxs = _contexts()
    client = bt._client()
    shots_path = args.fewshot_shots
    modes: list[bt.Mode] = ["joint", "xyzrpy"] if args.mode == "both" else [args.mode]  # type: ignore

    jobs = []
    for mode in modes:
        pred_path = out / f"predictions_{mode}.jsonl"
        done: set[str] = set()
        if pred_path.exists() and not args.force:
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("parse_ok"):
                    done.add(r["job_id"])
        for idx, cue in CUES:
            for plat in PLATS:
                cell = gt.get((idx, plat))
                if not cell:
                    print(f"[skip] no GT {plat} g{idx:02d}", flush=True)
                    continue
                for ctx_key in CONTEXTS:
                    job_id = f"{mode}.{plat}.g{idx:02d}.{ctx_key}"
                    if job_id in done:
                        continue
                    jobs.append((mode, idx, cue, plat, cell, ctx_key, ctxs[ctx_key], pred_path))

    thinking = 128 if args.model == "gemini-2.5-pro" and args.thinking_budget == 0 else args.thinking_budget
    print(f"context-baseline pending={len(jobs)} model={args.model} thinking={thinking}", flush=True)

    def work(item: tuple) -> dict[str, Any]:
        mode, idx, cue, plat, cell, ctx_key, ctx, pred_path = item
        job_id = f"{mode}.{plat}.g{idx:02d}.{ctx_key}"
        shots = bt._fewshot_artifacts(out, plat, gt, shots_path=shots_path)
        prompt = _prompt_with_context(
            mode,
            plat=plat,
            target={"cue": cue, "description": cell.get("description") or ""},
            shots=shots,
            ctx_key=ctx_key,
            ctx=ctx,
        )
        last_err = None
        last_raw = ""
        for attempt in range(args.retries + 1):
            try:
                cfg: dict[str, Any] = {"temperature": 0.3, "max_output_tokens": 16384}
                try:
                    tb = int(args.thinking_budget)
                    if args.model == "gemini-2.5-pro" and tb == 0:
                        tb = 128
                    cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=tb)
                except Exception:
                    pass
                resp = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**cfg),
                )
                raw = (getattr(resp, "text", None) or "").strip()
                if not raw:
                    for cand in getattr(resp, "candidates", None) or []:
                        content = getattr(cand, "content", None)
                        for part in getattr(content, "parts", None) or []:
                            if getattr(part, "text", None):
                                raw += part.text
                    raw = raw.strip()
                last_raw = raw
                schema_joints = shots[0]["joint_names"] if shots else []
                schema_links = (shots[0].get("link_names") if shots else None) or ["end_effector"]
                if plat in bt.PART_JOINTS:
                    schema_joints = bt.flat_joint_names(plat)
                    schema_links = bt.link_order(plat)
                payload = bt._validate(
                    mode,
                    bt._extract_json(raw),
                    cue=cue,
                    joint_names=schema_joints,
                    link_names=schema_links,
                    plat=plat,
                )
                row = {
                    "job_id": job_id,
                    "mode": mode,
                    "platform": plat,
                    "idx": idx,
                    "cue": cue,
                    "context": ctx_key,
                    "context_label": str(ctx.get("label") or ""),
                    "description": cell.get("description") or "",
                    "payload": payload,
                    "raw_output": raw,
                    "model": args.model,
                    "parse_ok": True,
                }
                tqdm.write(f"[infer] {job_id} attempt={attempt + 1} parse=PASS")
                if args.sleep > 0:
                    time.sleep(args.sleep)
                return row
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                tqdm.write(f"[infer] {job_id} attempt={attempt + 1} parse=FAIL error={last_err}")
                time.sleep(min(2**attempt, 15))
        return {
            "job_id": job_id,
            "mode": mode,
            "platform": plat,
            "idx": idx,
            "cue": cue,
            "context": ctx_key,
            "error": last_err or "unknown",
            "raw_output": last_raw[:8000],
            "model": args.model,
            "parse_ok": False,
        }

    def gif_path(row: dict[str, Any]) -> Path:
        slug = bt._model_slug(row.get("model") or args.model)
        ctx = row["context"]
        return (
            out
            / f"render_{row['mode']}"
            / row["platform"]
            / f"emb_g{int(row['idx']):02d}_{row['cue']}__{ctx}_{slug}.gif"
        )

    def render_one(row: dict[str, Any]) -> None:
        if not row.get("parse_ok"):
            return
        plat = row["platform"]
        gif = gif_path(row)
        meta_path = out / "gt_extract" / plat / f"g{int(row['idx']):02d}_{row['cue']}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        try:
            bt.RENDERERS[plat](row["payload"], row["mode"], gif, meta)
            tqdm.write(f"[render] {row['job_id']} PASS -> {gif.name}")
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[render] {row['job_id']} FAIL error={exc}")

    render_pool = ThreadPoolExecutor(max_workers=max(1, args.render_workers))
    with tqdm(total=len(jobs), desc="inference", unit="job") as bar, ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            pred_path = out / f"predictions_{row['mode']}.jsonl"
            with pred_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
            bar.update(1)
            if row.get("parse_ok"):
                render_pool.submit(render_one, row)
    render_pool.shutdown(wait=True)


def _rel(path: Path, html: Path) -> str | None:
    if path.is_file():
        return os.path.relpath(path, html.parent)
    return None


def gallery(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).expanduser().resolve()
    html = out / "gallery_context_baseline.html"
    slug = bt._model_slug(args.model)
    ctxs = _contexts()
    sections = []
    for idx, cue in CUES:
        for plat in PLATS:
            cols = []
            gt = GT_GIF / plat / "gif" / f"emb_g{idx:02d}_{cue}.gif"
            cols.append(("GT (no context)", _rel(gt, html)))
            for ctx_key in CONTEXTS:
                smp = SMP_GIF / plat / "gif" / f"emb_g{idx:02d}_{cue}__{ctx_key}.gif"
                jgif = (
                    out
                    / "render_joint"
                    / plat
                    / f"emb_g{idx:02d}_{cue}__{ctx_key}_{slug}.gif"
                )
                xgif = (
                    out
                    / "render_xyzrpy"
                    / plat
                    / f"emb_g{idx:02d}_{cue}__{ctx_key}_{slug}.gif"
                )
                label = str(ctxs[ctx_key].get("label") or ctx_key)
                cols.append((f"SMP · {ctx_key}", _rel(smp, html)))
                cols.append((f"joint · {ctx_key}", _rel(jgif, html)))
                cols.append((f"xyzrpy · {ctx_key}", _rel(xgif, html)))
                del label
            cells = []
            for title, rel in cols:
                if rel:
                    cells.append(
                        f'<div class="col"><h3>{html_mod.escape(title)}</h3>'
                        f'<img src="{html_mod.escape(rel)}" alt=""></div>'
                    )
                else:
                    cells.append(
                        f'<div class="col"><h3>{html_mod.escape(title)}</h3>'
                        f'<div class="miss">missing</div></div>'
                    )
            sections.append(
                f'<section class="cue"><h2>{html_mod.escape(plat)} · g{idx:02d} {html_mod.escape(cue)}</h2>'
                f'<div class="grid">{"".join(cells)}</div></section>'
            )
    html.write_text(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Context baselines · joint vs xyzrpy</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6}}
.cue{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}
.col h3{{font-size:.78rem;margin:0 0 6px;color:#c4c7c5}}
.col img{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000}}
.miss{{aspect-ratio:1;display:grid;place-items:center;background:#090a0d;border-radius:8px;color:#f28b82}}
</style></head><body>
<h1>Context restaging · Motion Program vs few-shot joint / xyzrpy</h1>
<p class="meta">5 cues × {", ".join(PLATS)} × {", ".join(CONTEXTS)} · model={html_mod.escape(args.model)}.
GT is the unconditioned human program. SMP is Gemini Motion Program restaged for the context.
joint / xyzrpy are few-shot trajectory baselines given the same cue+context.</p>
{"".join(sections)}
</body></html>
""",
        encoding="utf-8",
    )
    print(f"gallery → {html}", flush=True)
    return html


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--fewshot-shots", type=Path, default=bt.DEFAULT_FEWSHOT_SHOTS)
    ap.add_argument("--mode", choices=("joint", "xyzrpy", "both"), default="both")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--thinking-budget", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--render-workers", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--gallery-only", action="store_true")
    args = ap.parse_args()
    if args.gallery_only:
        gallery(args)
        return
    if not args.skip_generate:
        generate(args)
    gallery(args)


if __name__ == "__main__":
    main()
