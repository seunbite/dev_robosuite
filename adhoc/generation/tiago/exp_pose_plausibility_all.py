#!/usr/bin/env python3
"""Task A — per-tile Gemini plausibility for all 12 dir×orient pose groups."""
from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_plausible_pose_set import USER_SELECTION  # noqa: E402
from generate_pose_group_tiles import ORDER_DIR, ORDER_GRIP  # noqa: E402
from pilot40_paths import HTML_EXP_DIR, VERIFY_EXP_DIR, prompt_exp_path, save_json  # noqa: E402
from verify_pose_group_plausibility_gemini import (  # noqa: E402
    _collect_group_entries,
    _draw_plausible_box,
    _fill_prompt,
    _finalize_outputs,
    _gemini_client,
    _generate_with_fallback,
    _load_resume,
    _parse_group_key,
    _render_tile,
    _save_payload,
    _extract_json,
)
from legacy.render_mobile_config import _make_env  # noqa: E402

DEFAULT_VIS = _REPO / "data/results/visualize/tiago/expA_plausibility"
DEFAULT_OUT = VERIFY_EXP_DIR / "result_expA_gemini-2.5-pro.json"


def _all_group_keys() -> list[str]:
    return [f"{d}_{g}" for d in ORDER_DIR for g in ORDER_GRIP]


def _merged_plausible_indices(
    group: str,
    results: list[dict[str, Any]],
) -> list[int]:
    manual = set(USER_SELECTION.get(group, []))
    gemini_ok = {int(r["tile_idx"]) for r in results if r.get("is_plausible") is True}
    return sorted(manual | gemini_ok)


def _write_suite_html(payload: dict[str, Any], out_path: Path, *, vis_root: Path) -> None:
    groups = payload.get("groups") or []
    cards: list[str] = []
    for g in groups:
        group = str(g.get("group", ""))
        merged = g.get("merged_plausible_indices") or []
        manual = USER_SELECTION.get(group, [])
        n_ok = int(g.get("n_plausible", 0))
        n_tiles = int(g.get("n_tiles", 0))
        review = vis_root / group / f"review_{group}.html"
        review_href = os.path.relpath(review.resolve(), out_path.parent.resolve()) if review.is_file() else ""
        link = f"<a href='{html.escape(review_href)}'>per-group review</a>" if review_href else ""
        cards.append(
            f"<section class='card'><h2>{html.escape(group)}</h2>"
            f"<p class='meta'>{n_ok}/{n_tiles} gemini-plausible · "
            f"manual={len(manual)} · merged={len(merged)}</p>"
            f"<p class='meta'>merged indices: {html.escape(str(merged))}</p>"
            f"<p>{link}</p></section>"
        )
    summary = payload.get("summary") or {}
    body = (
        f"<div class='summary'><p><b>Input:</b> 12 groups × ~25 tiles from pose bank "
        f"(dir/orient filter + head-hand gap + 5×5 xyz grid).</p>"
        f"<p><b>Model:</b> one PNG tile + prompt_expA per tile.</p>"
        f"<p><b>Output set:</b> USER_SELECTION (manual) ∪ {{tile | is_plausible=true}} "
        f"→ {summary.get('n_merged_total', '?')} poses total.</p></div>"
        + "".join(cards)
    )
    css = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}
.wrap{padding:16px 20px}h1{font-size:1.3rem}
.summary{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px;margin:12px 0}
.card{background:#111827;border:1px solid #334155;border-radius:10px;padding:12px;margin:10px 0}
.meta{color:#94a3b8;font-size:13px}a{color:#93c5fd}
"""
    doc = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Exp A — pose plausibility (all groups)</title><style>{css}</style></head>"
        f"<body><div class='wrap'><h1>Task A — per-tile plausibility (12 groups)</h1>{body}</div></body></html>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"[html] wrote {out_path}", flush=True)


def _run_one_group(
    args: argparse.Namespace,
    *,
    group: str,
    client: Any,
    template: str,
    env,
) -> dict[str, Any]:
    disp_dir, orient = _parse_group_key(group)
    vis_dir = Path(args.vis_dir) / group
    tiles_dir = vis_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    out_json = vis_dir / "result.json"
    resume = _load_resume(out_json) if args.resume else {}

    entries = _collect_group_entries(
        env,
        disp_dir,
        orient,
        grid_side=args.grid_cols,
        head_hand_gap_min=args.head_hand_gap_min,
    )
    print(f"[A:{group}] {len(entries)} tiles", flush=True)

    results: list[dict[str, Any]] = []
    boxed_tiles: list[Any] = []
    model_used = args.model

    for i, entry in enumerate(entries, start=1):
        tile_path = tiles_dir / f"tile_{i:02d}.png"
        if args.resume and i in resume and resume[i].get("is_plausible") is not None:
            rec = dict(resume[i])
            im = (
                __import__("PIL.Image", fromlist=["Image"]).Image.open(tile_path).convert("RGB")
                if tile_path.is_file()
                else _render_tile(env, entry, tile_idx=i)
            )
            plausible = bool(rec.get("is_plausible"))
            boxed = _draw_plausible_box(im, plausible=plausible)
            results.append(rec)
            boxed_tiles.append(boxed)
            model_used = str(rec.get("model") or model_used)
            continue

        im = _render_tile(env, entry, tile_idx=i)
        im.save(tile_path)
        manual = i in USER_SELECTION.get(group, [])
        rec: dict[str, Any] = {
            "tile_idx": i,
            "direction_group": disp_dir,
            "gripper_orientation": orient,
            "x": entry.get("x"),
            "y": entry.get("y"),
            "z": entry.get("z"),
            "joint_deg": entry.get("joint_deg"),
            "r_arm_rad": entry.get("r_arm_rad"),
            "manual_selected": manual,
            "tile_image": str(tile_path.relative_to(_REPO)),
        }
        if args.dry_run:
            rec["dry_run"] = True
            rec["is_plausible"] = None
            boxed = _draw_plausible_box(im, plausible=manual)
        else:
            prompt = _fill_prompt(template, entry, disp_dir=disp_dir, orient=orient)
            text, model_used = _generate_with_fallback(
                client,
                model=args.model,
                img_bytes=tile_path.read_bytes(),
                prompt=prompt,
            )
            parsed = _extract_json(text)
            plausible = bool(parsed.get("is_plausible"))
            rec.update(
                {
                    "is_plausible": plausible,
                    "confidence": parsed.get("confidence"),
                    "reason": parsed.get("reason"),
                    "model": model_used,
                    "raw": parsed,
                }
            )
            boxed = _draw_plausible_box(im, plausible=plausible or manual)
            print(f"  [{group} {i:02d}] {'OK' if plausible else 'NO'}{' *manual*' if manual else ''}", flush=True)

        boxed_path = tiles_dir / f"tile_{i:02d}_boxed.png"
        boxed.save(boxed_path)
        rec["boxed_image"] = str(boxed_path.relative_to(vis_dir))
        boxed_tiles.append(boxed)
        results.append(rec)

    group_args = argparse.Namespace(
        group=group,
        prompt=args.prompt,
        out_json=out_json,
        vis_dir=vis_dir,
        open=False,
    )
    payload = _finalize_outputs(
        args=group_args,
        vis_dir=vis_dir,
        tiles_dir=tiles_dir,
        disp_dir=disp_dir,
        orient=orient,
        model_used=model_used,
        results=results,
        boxed_tiles=boxed_tiles,
        grid_side=args.grid_cols,
    )
    merged = _merged_plausible_indices(group, results)
    return {
        "group": group,
        "direction_group": disp_dir,
        "gripper_orientation": orient,
        "n_tiles": len(results),
        "n_plausible": sum(1 for r in results if r.get("is_plausible") is True),
        "n_manual": len(USER_SELECTION.get(group, [])),
        "merged_plausible_indices": merged,
        "manual_indices": USER_SELECTION.get(group, []),
        "review_html": str((vis_dir / f"review_{group}.html").relative_to(DEFAULT_VIS.parent)),
        "result_json": str(out_json.relative_to(_REPO)),
        "results": results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    template = Path(args.prompt).read_text(encoding="utf-8")
    client = None if args.dry_run else _gemini_client()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()] if args.groups else _all_group_keys()

    env = _make_env()
    group_payloads: list[dict[str, Any]] = []
    try:
        for group in groups:
            group_payloads.append(_run_one_group(args, group=group, client=client, template=template, env=env))
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    n_merged = sum(len(g["merged_plausible_indices"]) for g in group_payloads)
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": "A",
        "title": "Per-tile pose plausibility (dir×orient groups)",
        "model": args.model,
        "prompt": str(Path(args.prompt).relative_to(_REPO)),
        "input_composition": {
            "pose_bank_filter": "dir + orient + head-hand x-gap",
            "grid": f"{args.grid_cols}x{args.grid_cols} xyz-balanced (~25 tiles/group)",
            "vlm_input": "single rendered PNG per tile + prompt_expA",
            "output_rule": "merged = USER_SELECTION ∪ {tile_idx | is_plausible=true}",
        },
        "user_selection": USER_SELECTION,
        "summary": {
            "n_groups": len(group_payloads),
            "n_merged_total": n_merged,
            "n_manual_total": sum(len(v) for v in USER_SELECTION.values()),
        },
        "groups": [
            {k: v for k, v in g.items() if k != "results"}
            for g in group_payloads
        ],
        "group_results": {g["group"]: g["results"] for g in group_payloads},
    }
    out_json = Path(args.out_json)
    save_json(out_json, payload)

    html_path = HTML_EXP_DIR / f"expA_{args.model_tag}.html"
    _write_suite_html(payload, html_path, vis_root=Path(args.vis_dir))

    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(html_path)], check=False)
    return payload


def main() -> None:
    from pilot40_paths import model_to_tag  # noqa: WPS433

    p = argparse.ArgumentParser(description="Task A — Gemini plausibility for all pose groups")
    p.add_argument("--model", default=os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    p.add_argument("--model-tag", default=None)
    p.add_argument("--prompt", type=Path, default=prompt_exp_path("A"))
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS)
    p.add_argument("--groups", default="", help="comma list; default all 12")
    p.add_argument("--grid-cols", type=int, default=5)
    p.add_argument("--head-hand-gap-min", type=float, default=-0.1)
    p.add_argument("--dry-run", action="store_true", help="render tiles + HTML only")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--open", action="store_true")
    args = p.parse_args()
    args.model_tag = args.model_tag or model_to_tag(args.model)
    run(args)


if __name__ == "__main__":
    main()
