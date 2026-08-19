#!/usr/bin/env python3
"""Gemini picks best pose from top-K bank grid per cue (natural + cue-fit)."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pose_bank_topk_tiles import first_pose_from_row, render_topk_tile  # noqa: E402
from legacy.render_mobile_config import _make_env  # noqa: E402

PROMPT_PATH = _REPO / "data/seed/prompt/tiago/prompt_pick_pose_topk_grid.txt"
DEFAULT_SHOTS = _REPO / "data/seed/shots/tiago/shot_configs_pilot40_mobile.json"
DEFAULT_OUT = _REPO / "data/results/verify/tiago/pose_topk_gemini_pick.json"
DEFAULT_GRID_DIR = _REPO / "data/results/visualize/tiago/pose_topk_gemini"


def _extract_json(text: str) -> dict[str, Any]:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def _fill_prompt(template: str, row: dict[str, Any], n_tiles: int) -> str:
    return (
        template.replace("{{CUE}}", str(row.get("cue", "")))
        .replace("{{CUE_TEXT}}", str(row.get("cue_text", row.get("description", ""))))
        .replace("{{DESCRIPTION}}", str(row.get("description", "")))
        .replace("{{N_TILES}}", str(n_tiles))
    )


def _gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
    return genai.Client(api_key=api_key)


def run(args: argparse.Namespace) -> None:
    shots = json.loads(Path(args.shots_json).read_text(encoding="utf-8"))
    if args.limit:
        shots = shots[: args.limit]
    if args.cues:
        want = {c.strip() for c in args.cues.split(",") if c.strip()}
        shots = [r for r in shots if r.get("cue") in want]

    template = Path(args.prompt).read_text(encoding="utf-8")
    model = args.model
    grid_dir = Path(args.grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else _gemini_client()

    env = _make_env()
    results: list[dict[str, Any]] = []
    try:
        for row in shots:
            idx = int(row.get("idx", -1))
            cue = str(row.get("cue", "?"))
            safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
            pose = first_pose_from_row(row)
            grid_path = grid_dir / f"mm19_g{idx:02d}_{safe}_top{args.top_k}_gemini.png"
            title = f"cue: {cue} | pick natural + cue-fit"
            ranked = render_topk_tile(env, pose, grid_path, top_k=args.top_k, title=title)
            entries = ranked.get("entries") or []
            n_tiles = len(entries)
            rec: dict[str, Any] = {
                "idx": idx,
                "cue": cue,
                "grid_image": str(grid_path.relative_to(_REPO)),
                "n_tiles": n_tiles,
                "xyz_rank1": entries[0] if entries else None,
                "bank_meta": {
                    k: ranked.get(k)
                    for k in ("bank_dir", "bank_orient", "target_xyz", "n_pool", "n_hand_ok")
                },
            }
            if n_tiles == 0:
                rec["error"] = "no tiles (still arm or empty bank)"
                results.append(rec)
                continue

            if args.dry_run:
                rec["dry_run"] = True
                results.append(rec)
                print(f"[{idx}] {cue}: grid only ({n_tiles} tiles)", flush=True)
                continue

            prompt = _fill_prompt(template, row, n_tiles)
            img_bytes = grid_path.read_bytes()
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            parsed = _extract_json(resp.text or "")
            pick = int(parsed.get("best_tile_index", 1))
            pick = max(1, min(pick, n_tiles))
            gem_entry = entries[pick - 1]
            rec["gemini"] = parsed
            rec["gemini_pick"] = gem_entry
            rec["same_as_xyz_rank1"] = pick == 1
            render_topk_tile(
                env,
                pose,
                grid_path,
                top_k=args.top_k,
                title=f"cue: {cue} | Gemini pick #{pick}",
                gemini_rank=pick,
            )
            print(
                f"[{idx}] {cue}: gemini=#{pick} xyz=#1 "
                f"({'SAME' if pick == 1 else 'DIFF'}) — {parsed.get('assessment', '')[:80]}",
                flush=True,
            )
            results.append(rec)
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "top_k": args.top_k,
        "prompt": str(Path(args.prompt).relative_to(_REPO)),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)

    if args.html:
        html_path = Path(args.html)
        _write_review_html(payload, html_path)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(html_path)])
        except Exception:
            pass


def _write_review_html(payload: dict[str, Any], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for r in payload.get("results") or []:
        cue = r.get("cue", "?")
        grid = r.get("grid_image", "")
        g = r.get("gemini") or {}
        pick = r.get("gemini_pick") or {}
        xyz = r.get("xyz_rank1") or {}
        same = r.get("same_as_xyz_rank1")
        pick_idx = g.get("best_tile_index")
        runner = g.get("runner_up_tile_index")
        conf = g.get("confidence")
        same_badge = "<span class='same-badge'>same</span>" if same else "<span class='diff-badge'>different</span>"
        badge = (
            f"<span class='vlm-badge'>◆ VLM #{pick_idx}</span>"
            f"<span class='xyz-badge'>★ xyz #1</span>"
            f"{same_badge}"
        )
        meta = f"<p class='badges'>{badge}</p>"
        if runner:
            meta += f"<p class='meta'>runner-up: #{runner} · confidence: {conf}</p>"
        cards.append(
            f"<article><h3>c{r.get('idx')} {cue}</h3>{meta}"
            f"<p class='assessment'>{g.get('assessment', '')}</p>"
            f"<img src='{os.path.relpath(_REPO / grid, html_path.parent)}' alt='{cue} grid'>"
            f"<details><summary>joint values</summary>"
            f"<pre>gemini_pick: {json.dumps(pick, ensure_ascii=False, indent=2)}\n"
            f"xyz_rank1: {json.dumps(xyz, ensure_ascii=False, indent=2)}</pre></details></article>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>pose topk gemini pick</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;padding:20px;max-width:1200px;margin:auto}"
        "article{border:1px solid #ddd;border-radius:8px;margin:16px 0;padding:16px;background:#fafafa}"
        "h3{margin:0 0 8px}"
        ".badges{display:flex;flex-wrap:wrap;gap:8px;align-items:center}"
        ".vlm-badge{background:#fce4f6;color:#a02080;padding:4px 10px;border-radius:6px;font-weight:700;border:2px solid #ff50dc}"
        ".xyz-badge{background:#e8f8e8;color:#1a6b1a;padding:4px 10px;border-radius:6px;font-weight:600;border:2px solid #50dc50}"
        ".same-badge,.diff-badge{padding:3px 8px;border-radius:4px;font-size:12px}"
        ".same-badge{background:#dff0df;color:#2d6b2d}"
        ".diff-badge{background:#fff0df;color:#8b5a00}"
        ".meta{color:#666;font-size:13px;margin:4px 0 8px}"
        ".assessment{margin:8px 0 12px;line-height:1.45}"
        "img{max-width:100%;border:1px solid #ccc;border-radius:4px}"
        "details{margin-top:10px;font-size:13px}"
        "</style></head><body>"
        f"<h1>Gemini pose pick (top-{payload.get('top_k')})</h1>"
        f"{''.join(cards)}</body></html>"
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"HTML {html_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shots-json", default=str(DEFAULT_SHOTS))
    p.add_argument("--prompt", default=str(PROMPT_PATH))
    p.add_argument("--out-json", default=str(DEFAULT_OUT))
    p.add_argument("--grid-dir", default=str(DEFAULT_GRID_DIR))
    p.add_argument("--html", default=str(_REPO / "data/results/html/tiago/pose_topk_gemini_pick.html"))
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--limit", type=int, default=5, help="First N shots (default 5)")
    p.add_argument("--cues", default=None)
    p.add_argument("--dry-run", action="store_true", help="Build grids only, skip Gemini API")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
