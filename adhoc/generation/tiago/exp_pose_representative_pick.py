#!/usr/bin/env python3
"""Task B — pick most representative pose from dir/orient/bended pool (exp1 GT-correct cues)."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
# tiago must precede robotarm — both ship a top-level `legacy` package.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROBOTARM) not in sys.path:
    sys.path.insert(1, str(_ROBOTARM))
if str(_REPO) not in sys.path:
    sys.path.insert(2, str(_REPO))

from pose_bank_topk_tiles import (  # noqa: E402
    build_pick_vs_nonpick_montage,
    first_pose_from_row,
    render_plausible_pool_tile,
)
from legacy.render_mobile_config import _make_env  # noqa: E402
from pilot40_paths import (  # noqa: E402
    GT_CONSOLIDATED,
    GT_PATH,
    HTML_EXP_DIR,
    VERIFY_EXP_DIR,
    load_config_list,
    load_gt_by_cue,
    prompt_exp_path,
    result_config_path,
    save_json,
)
from score_mobile_motion_gt import pose_generation_correct_any_mobile  # noqa: E402

DEFAULT_GRID_DIR = _REPO / "data/results/visualize/tiago/expB_representative"
DEFAULT_OUT = VERIFY_EXP_DIR / "result_expB_gemini-2.5-pro.json"
YELLOW_TOKENS = ["point_you", "4", "8", "conditional_accept", "unsure_respose", "interrupt_soft"]
RED_TOKENS = ["11", "13", "uncertainty_wave"]


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _generate_with_fallback(client, *, model: str, img_bytes: bytes, prompt: str) -> tuple[str, str]:
    from google.genai import types

    models = [model, "gemini-2.5-flash"] if model else ["gemini-2.5-pro", "gemini-2.5-flash"]
    seen: set[str] = set()
    models = [m for m in models if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]
    last_err: Exception | None = None
    for m in models:
        try:
            resp = client.models.generate_content(
                model=m,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            return (resp.text or "").strip(), m
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg:
                print(f"[warn] {m} exhausted, trying fallback...", flush=True)
                continue
            raise
    raise RuntimeError(f"All Gemini models failed: {last_err}")


def _gemini_client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    return genai.Client(api_key=key)


def _load_resume(out_json: Path) -> dict[str, dict[str, Any]]:
    if not out_json.is_file():
        return {}
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return {str(r["cue"]): r for r in (data.get("results") or []) if r.get("cue")}


def _fill_prompt(template: str, row: dict[str, Any], pose: dict[str, Any], n_tiles: int) -> str:
    eb = pose.get("elbow_bended")
    eb_s = "true" if eb is True else ("false" if eb is False else "any")
    return (
        template.replace("{{CUE}}", str(row.get("cue", "")))
        .replace("{{DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{N_TILES}}", str(n_tiles))
        .replace("{{ARM_POSITION}}", str(pose.get("arm_position", "")))
        .replace("{{GRIPPER_ORIENTATION}}", str(pose.get("gripper_orientation", "")))
        .replace("{{ELBOW_BENDED}}", eb_s)
    )


def _pose_class_key(pose: dict[str, Any]) -> str:
    arm = str(pose.get("arm_position", "")).strip().lower()
    orient = str(pose.get("gripper_orientation", "")).strip().lower()
    eb = pose.get("elbow_bended")
    eb_s = "b" if eb is True else ("n" if eb is False else "any")
    return f"{arm}_{orient}_{eb_s}"


def _load_correct_cues(config_path: Path, score_path: Path | None) -> list[dict[str, Any]]:
    if score_path and score_path.is_file():
        score = json.loads(score_path.read_text(encoding="utf-8"))
        correct = {str(r["cue"]) for r in score.get("rows") or [] if r.get("generation_correct") is True}
    else:
        gt = load_gt_by_cue()
        correct = set()
        for row in load_config_list(config_path):
            cue = str(row.get("cue", ""))
            ev = gt.get(cue)
            if not ev or not ev.get("pose_gt"):
                continue
            if pose_generation_correct_any_mobile(row, str(ev["pose_gt"])) is True:
                correct.add(cue)

    rows = load_config_list(config_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        cue = str(row.get("cue", ""))
        if cue not in correct:
            continue
        pose = first_pose_from_row(row)
        if not pose:
            continue
        out.append(row)
    return out


def _rel_href(out_html: Path, asset: Path) -> str:
    if not asset.is_file():
        return ""
    return Path(os.path.relpath(asset.resolve(), out_html.parent.resolve())).as_posix()


def _rows_by_cue(config_path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["cue"]): r for r in load_config_list(config_path) if r.get("cue")}


def _multi_tile_cues(results: list[dict[str, Any]]) -> list[str]:
    cues: list[str] = []
    for rec in sorted(results, key=lambda r: int(r.get("idx", 0))):
        if int(rec.get("n_tiles", 0)) <= 1 or rec.get("vlm_skipped") == "single_candidate":
            continue
        cue = str(rec.get("cue", "")).strip()
        if cue:
            cues.append(cue)
    return cues


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _resolve_tokens(tokens: list[str], cues: list[str]) -> set[str]:
    out: set[str] = set()
    cue_norm = {c: _norm(c) for c in cues}
    for tok in tokens:
        t = str(tok).strip()
        if not t:
            continue
        if t.isdigit():
            k = int(t)
            if 1 <= k <= len(cues):
                out.add(cues[k - 1])
            continue
        nt = _norm(t.replace("respose", "response"))
        exact = [c for c, nc in cue_norm.items() if nc == nt]
        if exact:
            out.update(exact)
            continue
        partial = [c for c, nc in cue_norm.items() if nt in nc]
        if len(partial) == 1:
            out.add(partial[0])
    return out


def build_montage_from_payload(
    payload: dict[str, Any],
    *,
    config_path: Path | None = None,
    out_path: Path | None = None,
    cell_size: int = 256,
) -> Path:
    cfg_path = config_path or (_REPO / str(payload.get("config_json", "")))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config json not found: {cfg_path}")
    od = Path(out_path) if out_path else DEFAULT_GRID_DIR / "expB_pick_vs_nonpick_montage.png"
    cues = _multi_tile_cues(payload.get("results") or [])
    yellow_cues = _resolve_tokens(YELLOW_TOKENS, cues)
    red_cues = _resolve_tokens(RED_TOKENS, cues)
    top_marks = {c: "ok" for c in cues}
    bottom_marks: dict[str, str] = {}
    for c in yellow_cues:
        top_marks[c] = "warn"
        bottom_marks[c] = "warn"
    for c in red_cues:
        top_marks[c] = "bad"
        bottom_marks[c] = "ok"

    n_ok = sum(1 for c in cues if top_marks.get(c) == "ok")
    n_warn = sum(1 for c in cues if top_marks.get(c) == "warn")
    n_bad = sum(1 for c in cues if top_marks.get(c) == "bad")
    acc_ok_over_ok_bad = (n_ok / (n_ok + n_bad)) if (n_ok + n_bad) else 0.0
    payload["pick_score_marks"] = top_marks
    payload["pick_score_counts"] = {"ok": n_ok, "warn": n_warn, "bad": n_bad}
    payload["pick_acc_ok_over_ok_plus_bad"] = acc_ok_over_ok_bad

    # subset: 8 ok + 1 warn + 1 bad (in ok>warn>bad order)
    ordered = sorted(cues, key=lambda c: {"ok": 0, "warn": 1, "bad": 2}.get(top_marks.get(c, "ok"), 99))
    ok_list = [c for c in ordered if top_marks.get(c) == "ok"][:8]
    if "point_you" in top_marks and top_marks["point_you"] == "warn":
        warn_pick = ["point_you"]
    else:
        warn_pick = [c for c in ordered if top_marks.get(c) == "warn"][:1]
    bad_list = [c for c in ordered if top_marks.get(c) == "bad"][:1]
    subset_cues = ok_list + warn_pick + bad_list
    payload["montage_subset_cues"] = subset_cues
    env = _make_env()
    try:
        out = build_pick_vs_nonpick_montage(
            env,
            results=payload.get("results") or [],
            rows_by_cue=_rows_by_cue(cfg_path),
            out_path=od,
            cell_size=cell_size,
            pick_score_marks=top_marks,
            nonpick_score_marks=bottom_marks,
        )
        subset_out = build_pick_vs_nonpick_montage(
            env,
            results=payload.get("results") or [],
            rows_by_cue=_rows_by_cue(cfg_path),
            out_path=od.with_name("expB_pick_vs_nonpick_subset_8g1y1r.png"),
            cell_size=cell_size,
            pick_score_marks=top_marks,
            nonpick_score_marks=bottom_marks,
            subset=subset_cues,
        )
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()
    print(f"[montage] wrote {out}", flush=True)
    print(f"[montage] wrote {subset_out}", flush=True)
    return out


def _write_review_html(payload: dict[str, Any], out_path: Path, *, grid_dir: Path) -> None:
    results = payload.get("results") or []
    rows_html: list[str] = []
    cards: list[str] = []
    for r in results:
        cue = str(r.get("cue", ""))
        idx = r.get("idx", "")
        pick = r.get("vlm_pick_index")
        n_tiles = r.get("n_tiles", 0)
        pclass = r.get("pose_class", "")
        grid_rel = r.get("grid_image", "")
        review_rel = r.get("review_image", "")
        grid_path = (_REPO / grid_rel) if grid_rel else grid_dir / ""
        review_path = (_REPO / review_rel) if review_rel else grid_path
        img_href = _rel_href(out_path, review_path) if review_rel else _rel_href(out_path, grid_path)
        vlm = r.get("vlm_result") or {}
        assessment = html.escape(str(vlm.get("assessment", ""))[:400])
        picked = r.get("picked_entry") or {}
        tile_idx = picked.get("tile_idx", "")
        pick_xyz = (
            f"x={picked.get('x', '?'):g} y={picked.get('y', '?'):g} z={picked.get('z', '?'):g}"
            if picked
            else ""
        )
        runner = vlm.get("runner_up_tile_index")
        conf = vlm.get("confidence")
        eb_relaxed = (r.get("pool_meta") or {}).get("elbow_bended_relaxed")
        if r.get("vlm_skipped") == "single_candidate":
            status = "auto (1 tile)"
            pick_s = "1"
        elif pick is not None and not r.get("dry_run"):
            status = "VLM"
            pick_s = str(pick)
        elif r.get("dry_run"):
            status = "dry-run"
            pick_s = "—"
        elif r.get("error"):
            status = "error"
            pick_s = "—"
        else:
            status = "—"
            pick_s = "—"
        rows_html.append(
            f"<tr><td>g{idx}</td><td><b>{html.escape(cue)}</b></td>"
            f"<td>{html.escape(status)}</td><td><b>{html.escape(pick_s)}</b>/{n_tiles}</td>"
            f"<td>{html.escape(str(tile_idx))}</td>"
            f"<td>{html.escape(pick_xyz)}</td></tr>"
        )
        badge = (
            '<span class="ok">VLM pick</span>'
            if pick is not None and not r.get("dry_run") and not r.get("vlm_skipped")
            else (
                '<span class="na">1 tile — auto</span>'
                if r.get("vlm_skipped") == "single_candidate"
                else ('<span class="na">dry-run</span>' if r.get("dry_run") else '<span class="na">—</span>')
            )
        )
        pick_line = ""
        if pick is not None:
            extra = f" plausible_tile={tile_idx}" if tile_idx not in ("", None, -1) else ""
            relaxed = " · elbow filter relaxed" if eb_relaxed else ""
            pick_line = (
                f"<p class='pick'><b>Pick: tile #{pick}</b> / {n_tiles}{extra}"
                f"{(' · xyz ' + pick_xyz) if pick_xyz else ''}{relaxed}</p>"
            )
            if runner is not None:
                pick_line += f"<p class='meta'>runner-up: #{runner}</p>"
            if conf is not None:
                pick_line += f"<p class='meta'>confidence: {html.escape(str(conf))}</p>"
        if img_href and n_tiles > 0:
            img = f'<img src="{html.escape(img_href)}" loading="lazy" alt="{html.escape(cue)}"/>'
        elif n_tiles == 0 or r.get("error"):
            img = f"<p class='missing'>{html.escape(str(r.get('error', 'empty pool')))}</p>"
        else:
            img = "<p class='missing'>grid image missing</p>"
        cards.append(
            f"<section class='card'><h2>g{idx} · {html.escape(cue)} {badge}</h2>"
            f"<p class='meta'>class={html.escape(pclass)} · pool={n_tiles}</p>"
            f"{pick_line}<p class='assess'>{assessment}</p>{img}</section>"
        )
    inp = payload.get("input_composition") or {}
    css = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
.wrap{padding:16px 24px}h1{font-size:1.25rem;border-bottom:1px solid #333;padding-bottom:8px}
.summary{background:#171a21;border:1px solid #333;border-radius:10px;padding:12px;margin:12px 0;font-size:14px;color:#9aa0a6}
.card{margin:16px 0;padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}
.card img{max-width:720px;width:100%;border-radius:6px}
.meta{color:#9aa0a6;font-size:13px;margin:4px 0}
.pick{color:#a5d6ff;font-size:15px;margin:8px 0}
.assess{color:#cbd5e1;font-size:14px;margin:8px 0 12px}
.ok{color:#81c995;font-weight:600}.na{color:#888}
.missing{color:#f28b82}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th,td{border:1px solid #333;padding:6px 8px;text-align:left}
th{background:#1a1d24}
"""
    n_vlm = sum(1 for r in results if r.get("vlm_result"))
    n_auto = sum(1 for r in results if r.get("vlm_skipped") == "single_candidate")
    table = (
        "<table><thead><tr><th>idx</th><th>cue</th><th>method</th>"
        "<th>pick</th><th>plausible_tile</th><th>xyz</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )
    montage_path = grid_dir / "expB_pick_vs_nonpick_montage.png"
    n = len(results)
    montage_block = ""
    if montage_path.is_file():
        href = _rel_href(out_path, montage_path)
        montage_block = (
            f"<section class='card'><h2>Pick vs non-pick montage</h2>"
            f"<p class='meta'>Row 1 = VLM pick · Row 2 = most different non-pick · multi-tile pools only · {n} cues</p>"
            f'<img src="{html.escape(href)}" loading="lazy" alt="pick vs non-pick montage"/>'
            f"</section>"
        )
    body = (
        f"<div class='summary'><p><b>Model:</b> {html.escape(str(payload.get('model', '')))} · "
        f"<b>VLM picks:</b> {n_vlm} · <b>auto (1 tile):</b> {n_auto} · "
        f"<b>total:</b> {payload.get('n_cues', 0)}</p>"
        f"<p><b>Filter:</b> {html.escape(inp.get('filter_pipeline', ''))}</p></div>"
        + montage_block
        + table
        + "".join(cards)
    )
    doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Exp B — representative pose pick</title><style>{css}</style></head>"
        f"<body><div class='wrap'><h1>Task B — representative pose selection</h1>{body}</div></body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"[html] wrote {out_path}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config_json)
    score_path = Path(args.score_json) if args.score_json else None
    template = Path(args.prompt).read_text(encoding="utf-8")
    grid_dir = Path(args.grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_correct_cues(config_path, score_path)
    if args.cues:
        want = {c.strip() for c in args.cues.split(",") if c.strip()}
        rows = [r for r in rows if r.get("cue") in want]
    if args.limit:
        rows = rows[: args.limit]

    client = None if args.dry_run else _gemini_client()
    out_json = Path(args.out_json)
    resume_map = _load_resume(out_json) if args.resume else {}
    env = _make_env()
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            idx = int(row.get("idx", -1))
            cue = str(row.get("cue", "?"))
            if args.resume and cue in resume_map and (
                resume_map[cue].get("vlm_result")
                or resume_map[cue].get("vlm_skipped")
                or resume_map[cue].get("error")
            ):
                results.append(resume_map[cue])
                print(f"[B] g{idx} {cue}: resume", flush=True)
                continue
            pose = first_pose_from_row(row)
            safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
            grid_path = grid_dir / f"mm19_g{idx:02d}_{safe}_pool.png"
            ranked = render_plausible_pool_tile(
                env,
                pose,
                grid_path,
                uniform_tiles=True,
                no_header=True,
                plausible_only=True,
                relax_elbow_if_empty=True,
            )
            entries = ranked.get("entries") or []
            n_tiles = len(entries)
            rec: dict[str, Any] = {
                "idx": idx,
                "cue": cue,
                "description": row.get("description"),
                "pose_class": _pose_class_key(pose),
                "nominal_pose": {
                    k: pose.get(k)
                    for k in (
                        "arm_position",
                        "gripper_orientation",
                        "elbow_bended",
                        "x",
                        "y",
                        "z",
                    )
                },
                "grid_image": str(grid_path.relative_to(_REPO)),
                "n_tiles": n_tiles,
                "pool_meta": {
                    k: ranked.get(k)
                    for k in (
                        "bank_dir",
                        "bank_orient",
                        "target_xyz",
                        "n_pool",
                        "n_hand_ok",
                        "source",
                        "elbow_bended_relaxed",
                    )
                },
            }
            if n_tiles == 0:
                rec["error"] = "empty pool after dir/orient/bended filter"
                results.append(rec)
                print(f"[B] g{idx} {cue}: empty pool", flush=True)
                continue

            if n_tiles == 1:
                rec["vlm_skipped"] = "single_candidate"
                rec["vlm_pick_index"] = 1
                rec["picked_entry"] = entries[0]
                results.append(rec)
                print(f"[B] g{idx} {cue}: skip VLM (1 tile only)", flush=True)
                continue

            if args.dry_run:
                rec["dry_run"] = True
                results.append(rec)
                print(f"[B] g{idx} {cue}: grid only ({n_tiles} tiles, class={rec['pose_class']})", flush=True)
                continue

            from google.genai import types  # noqa: F401

            prompt = _fill_prompt(template, row, pose, n_tiles)
            text, model_used = _generate_with_fallback(
                client,
                model=args.model,
                img_bytes=grid_path.read_bytes(),
                prompt=prompt,
            )
            parsed = _extract_json(text)
            pick = int(parsed.get("best_tile_index", 1))
            pick = max(1, min(pick, n_tiles))
            picked_ent = entries[pick - 1] if pick - 1 < len(entries) else entries[0]
            rec.update(
                {
                    "vlm_result": parsed,
                    "vlm_pick_index": pick,
                    "picked_entry": picked_ent,
                    "model": model_used,
                }
            )
            # Review image: highlight VLM pick (input grid stays uniform)
            review_path = grid_dir / f"mm19_g{idx:02d}_{safe}_pool_review.png"
            render_plausible_pool_tile(
                env,
                pose,
                review_path,
                uniform_tiles=True,
                no_header=True,
                plausible_only=True,
                relax_elbow_if_empty=True,
                gemini_rank=pick,
            )
            rec["review_image"] = str(review_path.relative_to(_REPO))
            results.append(rec)
            print(f"[B] g{idx} {cue}: pick #{pick}/{n_tiles}", flush=True)
            save_json(out_json, {"time": datetime.now().isoformat(timespec="seconds"), "results": results})
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    # group stats by pose class
    by_class: dict[str, list[str]] = {}
    for r in results:
        by_class.setdefault(str(r.get("pose_class", "?")), []).append(str(r.get("cue", "")))

    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "B",
        "title": "Representative pose pick (exp1 GT-correct cues)",
        "model": args.model,
        "prompt": str(Path(args.prompt).relative_to(_REPO)),
        "config_json": str(config_path.relative_to(_REPO)),
        "score_json": str(score_path.relative_to(_REPO)) if score_path else None,
        "groundtruth": str(GT_PATH.relative_to(_REPO)),
        "input_composition": {
            "filter_pipeline": "exp1 generation → pose matches GT → pool filter: arm_position + gripper_orientation + elbow_bended",
            "vlm_input": "numbered pool tile PNG + cue name + prompt_expB",
            "pool_source": "list_pose_bank_candidates (plausible metadata + bank fallback)",
        },
        "n_cues": len(results),
        "n_with_pick": sum(1 for r in results if r.get("vlm_pick_index") is not None),
        "by_pose_class": {k: len(v) for k, v in by_class.items()},
        "results": results,
    }
    out_json = Path(args.out_json)
    save_json(out_json, payload)

    html_path = HTML_EXP_DIR / f"expB_{args.model_tag}.html"
    montage_path = build_montage_from_payload(
        payload,
        config_path=config_path,
        out_path=grid_dir / "expB_pick_vs_nonpick_montage.png",
        cell_size=int(getattr(args, "cell_size", 256) or 256),
    )
    payload["montage_image"] = str(montage_path.relative_to(_REPO))
    save_json(out_json, payload)
    _write_review_html(payload, html_path, grid_dir=grid_dir)

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)
    return payload


def main() -> None:
    from pilot40_paths import model_to_tag  # noqa: WPS433

    p = argparse.ArgumentParser(description="Task B — representative pose from bended-filtered pool")
    p.add_argument("--config-json", type=Path, default=result_config_path("1", "gemini-2.5-pro"))
    p.add_argument("--score-json", type=Path, default=None)
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    p.add_argument("--model-tag", default=None)
    p.add_argument("--prompt", type=Path, default=prompt_exp_path("B"))
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    p.add_argument("--cues", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--open", action="store_true")
    p.add_argument(
        "--montage-only",
        action="store_true",
        help="Build n×2 pick vs non-pick montage PNG (+ HTML) from existing --out-json",
    )
    p.add_argument("--cell-size", type=int, default=256, help="Montage cell size in px (default 256)")
    args = p.parse_args()
    args.model_tag = args.model_tag or model_to_tag(args.model)
    if args.montage_only:
        payload = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        montage_path = build_montage_from_payload(
            payload,
            config_path=Path(args.config_json),
            out_path=Path(args.grid_dir) / "expB_pick_vs_nonpick_montage.png",
            cell_size=args.cell_size,
        )
        payload["montage_image"] = str(montage_path.relative_to(_REPO))
        # persist computed marks / acc / counts
        save_json(Path(args.out_json), payload)
        html_path = HTML_EXP_DIR / f"expB_{args.model_tag}.html"
        _write_review_html(payload, html_path, grid_dir=Path(args.grid_dir))
        if args.open and sys.platform == "darwin":
            subprocess.run(["open", str(html_path)], check=False)
        return
    run(args)


if __name__ == "__main__":
    main()
