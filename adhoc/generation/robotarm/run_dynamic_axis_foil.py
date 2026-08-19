#!/usr/bin/env python3
"""Dynamic-axis foil: flip one path/move axis per human-authored Motion Code.

For each GT cell that has a ``path(...)`` or ``move(...)`` with an axis / plane,
flip exactly one such axis (deterministic), convert, render true+foil, and write
a side-by-side HTML gallery for human preference eval.

Example::

    python adhoc/generation/robotarm/run_dynamic_axis_foil.py
    python adhoc/generation/robotarm/run_dynamic_axis_foil.py --platforms humanoid,tiago --skip-render
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import convert_code  # noqa: E402

GT_MASTER = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
DEFAULT_OUT = _REPO / "data/results/evaluation/embodiment_dynamic_axis_foil"
RENDER_TRUE = _REPO / "data/results/render/embodiment_motion_code_gt"

# Cycle axes so one flip is always a different letter.
_AXIS_NEXT = {"x": "y", "y": "z", "z": "x"}
_PLANE_NEXT = {
    "xy": "yz",
    "yx": "zy",
    "xz": "xy",
    "zx": "yx",
    "yz": "xz",
    "zy": "zx",
}

# path(...) or move(...); capture full call for replacement.
_CALL_RE = re.compile(r"\b(path|move)\s*\(([^)]*)\)", re.IGNORECASE)
_G1_COMPARE_MODEL: Any = None


def _flip_plane_token(plane: str) -> str | None:
    key = plane.lower()
    return _PLANE_NEXT.get(key)


def _flip_args_variants(kind: str, args: str) -> list[tuple[str, str]]:
    """Return ordered foil mutations for one path/move call."""
    variants: list[tuple[str, str]] = []
    is_arc = kind == "path" and bool(re.search(r"(^|,)\s*arc\s*(,|$)", args, re.I))
    if is_arc:
        # Preserve the authored arc plane. Reverse traversal direction instead.
        m = re.search(r"\b(deg|degrees|sweep)\s*=\s*([+-]?)(\d+(?:\.\d+)?)", args, re.I)
        if not m or float(m.group(3)) == 0.0:
            return variants
        old = f"{m.group(2)}{m.group(3)}"
        new = m.group(3) if m.group(2) == "-" else f"-{m.group(3)}"
        changed = args[: m.start(2)] + new + args[m.end(3) :]
        variants.append((changed, f"arc {m.group(1)} {old}→{new}"))
        return variants

    # plane=xz / plane = xy
    m = re.search(r"\bplane\s*=\s*([xyz]{2})\b", args, re.I)
    if m:
        nxt = _flip_plane_token(m.group(1))
        if nxt:
            new = args[: m.start(1)] + nxt + args[m.end(1) :]
            variants.append((new, f"plane {m.group(1)}→{nxt}"))
        return variants

    # axis=x / axis = y
    m = re.search(r"\baxis\s*=\s*([xyz])\b", args, re.I)
    if m:
        cur = m.group(1).lower()
        for nxt in (_AXIS_NEXT[cur], _AXIS_NEXT[_AXIS_NEXT[cur]]):
            new = args[: m.start(1)] + nxt + args[m.end(1) :]
            variants.append((new, f"axis={cur}→{nxt}"))
        return variants

    # positional axis after joint/part: move(shoulder, z, ...) or path(ee, line, z, ...)
    # Prefer a bare x|y|z token that is not inside a number / ± / list of numbers only.
    # Skip known non-axis keywords.
    skip = {
        "ee",
        "line",
        "arc",
        "circle",
        "spline",
        "hold",
        "rep",
        "spd",
        "speed",
        "dist",
        "r",
        "shape",
        "mode",
        "dir",
        "grip",
        "gripper",
        "torso",
        "elbow",
        "head",
        "base",
        "shoulder",
        "wrist",
        "forelegs",
        "right_front",
        "left_front",
        "right_hind",
        "left_hind",
        "pan",
        "tilt",
        "roll",
        "yaw",
        "pitch",
    }
    # tokenize loosely on commas
    parts = [p.strip() for p in args.split(",")]
    for i, part in enumerate(parts):
        # pure axis letter
        if re.fullmatch(r"[xyz]", part, re.I):
            cur = part.lower()
            for nxt in (_AXIS_NEXT[cur], _AXIS_NEXT[_AXIS_NEXT[cur]]):
                changed = list(parts)
                changed[i] = nxt
                variants.append((", ".join(changed), f"pos-axis {cur}→{nxt} ({kind})"))
            return variants
        # keyword=value already handled; skip keyword=...
        if "=" in part:
            continue
        tok = part.lower()
        if tok in skip:
            continue
    return variants


def _flip_args(kind: str, args: str) -> tuple[str, str] | None:
    """Backward-compatible first syntactic mutation for tests/callers."""
    variants = _flip_args_variants(kind, args)
    return variants[0] if variants else None


def _movement_differs(
    platform: str,
    original: dict[str, Any],
    foil: dict[str, Any],
) -> bool:
    """Reject foils that compile to the same effective motion."""
    original_moves = original.get("movements") or []
    foil_moves = foil.get("movements") or []
    if original_moves == foil_moves:
        return False
    if platform != "humanoid":
        return True

    # Humanoid x/y/z axes are resolved to physical DOFs at the active pose.
    # Compare the renderer's qpos sequence so two axes resolving to the same
    # joint/trajectory cannot enter the foil set.
    global _G1_COMPARE_MODEL
    import mujoco
    import numpy as np
    from embodiment_renderers import _g1_motion_qpos, _make_g1_scene_xml

    if _G1_COMPARE_MODEL is None:
        scene = _make_g1_scene_xml()
        try:
            _G1_COMPARE_MODEL = mujoco.MjModel.from_xml_path(str(scene))
        finally:
            scene.unlink(missing_ok=True)
    a = _g1_motion_qpos({"movements": original_moves}, _G1_COMPARE_MODEL, fps=20)
    b = _g1_motion_qpos({"movements": foil_moves}, _G1_COMPARE_MODEL, fps=20)
    if len(a) != len(b):
        return True
    if not a and not b:
        return False
    return not np.allclose(np.asarray(a), np.asarray(b), atol=1e-7, rtol=0.0)


def flip_one_axis(
    code: str,
    *,
    platform: str | None = None,
    original: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the first syntactic mutation that produces a different motion."""
    for m in _CALL_RE.finditer(code):
        kind, args = m.group(1), m.group(2)
        for new_args, note in _flip_args_variants(kind.lower(), args):
            new_call = f"{m.group(1)}({new_args})"
            new_code = code[: m.start()] + new_call + code[m.end() :]
            result = {
                "old_code": code,
                "new_code": new_code,
                "note": note,
                "call_kind": kind.lower(),
                "old_call": m.group(0),
                "new_call": new_call,
            }
            if platform is None:
                return result
            base = original or convert_code(code, platform=platform)
            converted = convert_code(new_code, platform=platform)
            if converted.get("parse_ok") and _movement_differs(platform, base, converted):
                result["converted"] = converted
                result["actual_motion_diff"] = True
                return result
    return None


def _gif_rel(render_root: Path, plat: str, idx: int, cue: str) -> Path:
    # GT layout: humanoid/gif, tiago/gif, manipulator/gif (single_arm), …
    plat_dir = {
        "single_arm": "manipulator",
        "humanoid": "humanoid",
        "tiago": "tiago",
        "stretch": "stretch",
        "roomba": "roomba",
        "quadruped": "quadruped",
    }.get(plat, plat)
    gif_dir = render_root / plat_dir / "gif"
    if plat == "single_arm":
        gif_dir = gif_dir / "IIWA"
    return gif_dir / f"emb_g{idx:02d}_{cue}.gif"


def build_gallery(
    rows: list[dict[str, Any]],
    *,
    out_html: Path,
    true_root: Path,
    foil_root: Path,
) -> Path:
    platform_order = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
    platform_labels = {
        "humanoid": "Humanoid",
        "tiago": "Tiago",
        "single_arm": "Single arm",
        "stretch": "Stretch",
        "roomba": "Roomba",
    }
    by_cue: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_cue.setdefault((int(row["idx"]), row["cue"]), {})[row["platform"]] = row

    sections: list[str] = []
    for (idx, cue), platform_rows in sorted(by_cue.items()):

        def media(p: Path, label: str) -> str:
            if p.is_file():
                rel = os.path.relpath(p, out_html.parent)
                bust = int(p.stat().st_mtime)
                return (
                    f"<h3>{html_mod.escape(label)}</h3>"
                    f'<img src="{html_mod.escape(rel)}?v={bust}" loading="lazy">'
                )
            return f"<h3>{html_mod.escape(label)}</h3><p class='miss'>no gif</p>"

        columns: list[str] = []
        for plat in platform_order:
            r = platform_rows.get(plat)
            if r is None:
                columns.append(
                    f'<div class="col"><h3 class="platform">{platform_labels[plat]}</h3>'
                    f'<p class="miss">no foil pair</p></div>'
                )
                continue
            true_gif = _gif_rel(true_root, plat, idx, cue)
            foil_gif = _gif_rel(foil_root, plat, idx, cue)
            columns.append(
                f'<div class="col"><h3 class="platform">{platform_labels[plat]}</h3>'
                f'{media(true_gif, "True")}'
                f'<pre class="mc">{html_mod.escape(r["old_code"])}</pre>'
                f'{media(foil_gif, "Foil")}'
                f'<pre class="mc">{html_mod.escape(r["new_code"])}</pre>'
                f'<div class="tag">{html_mod.escape(r.get("note") or "")}</div></div>'
            )
        sections.append(
            f'<section class="row"><h2>g{idx:02d} {html_mod.escape(cue)}</h2>'
            f'<div class="grid">{"".join(columns)}</div></section>'
        )

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Dynamic axis foil</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:10px 24px;color:#9aa0a6;font-size:0.9rem}}
.row{{margin:12px 16px;padding:12px;border:1px solid #333;border-radius:10px;background:#171a21}}
.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}}
.col{{background:#0f1115;border-radius:8px;padding:8px;min-width:0}}
.col h3{{margin:0 0 6px;font-size:0.85rem;color:#c4c7ce}}
.col h3.platform{{font-size:1rem;color:#fff;border-bottom:1px solid #333;padding-bottom:6px;margin-bottom:8px}}
img{{width:100%;height:auto;aspect-ratio:1;object-fit:cover;border-radius:6px;background:#000;display:block}}
.miss{{color:#f28b82}}
.tag{{font-size:0.75rem;color:#8ab4f8;font-weight:500}}
.mc{{font-size:11px;line-height:1.35;overflow:auto;max-height:160px;background:#0a0c10;
     padding:8px;border-radius:4px;white-space:pre-wrap;word-break:break-word;margin:8px 0 0;color:#d7dbe2}}
@media (max-width:1200px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style></head><body>
<h1>Dynamic axis foil — true vs flipped path/move axis</h1>
<p class="meta">{len(rows)} pairs · flip one path/move axis per human-authored script · which is more appropriate?</p>
{"".join(sections)}
</body></html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")
    return out_html


def render_foils(
    rows: list[dict[str, Any]],
    *,
    foil_json_dir: Path,
    foil_render_root: Path,
    true_render_root: Path,
    render_only: set[tuple[str, str]] | None = None,
) -> None:
    """Render paired True/Foil clips with the same current renderer and pose bank."""
    from motion_code_to_config import (  # noqa: WPS433
        render_manipulator,
        render_quadruped,
        render_tiago,
        write_json_files,
    )

    # Group foil cells into a GT-like master list keyed by cue idx.
    by_idx: dict[int, dict[str, Any]] = {}
    for r in rows:
        idx = int(r["idx"])
        if idx not in by_idx:
            by_idx[idx] = {
                "idx": idx,
                "cue": r["cue"],
                "category": r.get("category") or "",
                "platforms": {},
            }
        by_idx[idx]["platforms"][r["platform"]] = {
            "source_code": r["new_code"],
            "movements": r.get("movements") or [],
            "parse_ok": bool(r.get("parse_ok")),
            "platform": r["platform"],
            "cue": r["cue"],
            "idx": idx,
            "notes": r.get("convert_notes") or [],
            "foil_note": r.get("note"),
            "camera_reference_movements": r.get("original_movements") or [],
        }

    configs = [by_idx[i] for i in sorted(by_idx)]
    foil_json_dir.mkdir(parents=True, exist_ok=True)
    write_json_files(configs, foil_json_dir)

    # Flatten per-platform rows for renderers (same shape as sync_gt). Build a
    # paired authored set too, rather than comparing against potentially stale
    # global GT media rendered with an older pose bank / renderer revision.
    by_plat: dict[str, list[dict[str, Any]]] = {}
    true_by_plat: dict[str, list[dict[str, Any]]] = {}
    for e in configs:
        for plat, cell in (e.get("platforms") or {}).items():
            if render_only is not None and (plat, e["cue"]) not in render_only:
                continue
            if not cell.get("parse_ok") or not (cell.get("movements") or []):
                continue
            by_plat.setdefault(plat, []).append(cell)
            original = next(
                r for r in rows
                if int(r["idx"]) == int(e["idx"]) and r["platform"] == plat
            )
            true_by_plat.setdefault(plat, []).append(
                {
                    **cell,
                    "source_code": original["old_code"],
                    "movements": original.get("original_movements") or [],
                    "camera_reference_movements": original.get("original_movements") or [],
                }
            )

    os.environ.setdefault("MUJOCO_GL", "glfw")
    foil_render_root.mkdir(parents=True, exist_ok=True)

    def render_group(grouped: dict[str, list[dict[str, Any]]], root: Path) -> None:
        if grouped.get("tiago"):
            render_tiago(grouped["tiago"], out_root=root, force=True)
        if grouped.get("single_arm"):
            render_manipulator(grouped["single_arm"], out_root=root, force=True)
        if grouped.get("humanoid"):
            from embodiment_renderers import render_unitree_g1

            render_unitree_g1(grouped["humanoid"], root / "humanoid" / "gif", force=True)
        if grouped.get("stretch"):
            from embodiment_renderers import render_stretch

            render_stretch(grouped["stretch"], root / "stretch" / "gif", force=True)
        if grouped.get("roomba"):
            from embodiment_renderers import render_roomba

            render_roomba(grouped["roomba"], root / "roomba" / "gif", force=True)

    render_group(true_by_plat, true_render_root)
    render_group(by_plat, foil_render_root)
    if by_plat.get("quadruped"):
        rows_q = by_plat["quadruped"]
        tmp = foil_json_dir / "_foil_quad.json"
        tmp.write_text(json.dumps(rows_q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_quadruped(rows_q, config_path=tmp, out_root=foil_render_root, force=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--platforms", default="", help="comma filter")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument(
        "--render-only",
        default="",
        help="Comma list platform:cue; rebuild all pairs/gallery but render only these cells.",
    )
    ap.add_argument(
        "--render-changed",
        action="store_true",
        help="Render only cells whose generated foil differs from the previous foil_pairs.json.",
    )
    args = ap.parse_args()
    if args.render_only.strip() and args.render_changed:
        ap.error("use only one of --render-only and --render-changed")

    plats = {p.strip() for p in args.platforms.split(",") if p.strip()} or None
    master = json.loads(GT_MASTER.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    skipped = 0
    skipped_same_motion = 0
    excluded_quadruped = 0
    for e in master:
        for plat, cell in (e.get("platforms") or {}).items():
            if plat == "quadruped":
                excluded_quadruped += 1
                continue
            if plats is not None and plat not in plats:
                continue
            code = (cell.get("source_code") or "").strip()
            if not code:
                continue
            original = convert_code(code, platform=plat)
            flip = flip_one_axis(code, platform=plat, original=original)
            if not flip:
                # Includes no syntactic candidate and candidates that compile
                # to an identical effective movement.
                skipped += 1
                if re.search(r"\b(path|move)\s*\(", code, re.I):
                    skipped_same_motion += 1
                continue
            converted = flip.pop("converted")
            rows.append(
                {
                    "platform": plat,
                    "idx": int(e["idx"]),
                    "cue": e["cue"],
                    "category": e.get("category") or "",
                    "old_code": flip["old_code"],
                    "new_code": flip["new_code"],
                    "note": flip["note"],
                    "call_kind": flip["call_kind"],
                    "parse_ok": bool(converted.get("parse_ok")),
                    "movements": converted.get("movements") or [],
                    "original_movements": original.get("movements") or [],
                    "convert_notes": converted.get("notes") or [],
                    "actual_motion_diff": bool(flip.get("actual_motion_diff")),
                }
            )
            if args.max_items and len(rows) >= args.max_items:
                break
        if args.max_items and len(rows) >= args.max_items:
            break

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    previous_rows: dict[tuple[str, str], dict[str, Any]] = {}
    previous_path = out / "foil_pairs.json"
    if previous_path.is_file():
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_rows = {
            (r["platform"], r["cue"]): r
            for r in previous_payload.get("rows") or []
        }
    (out / "foil_pairs.json").write_text(
        json.dumps(
            {
                "n_pairs": len(rows),
                "n_skipped_no_axis": skipped,
                "n_skipped_same_or_unusable_motion": skipped_same_motion,
                "n_excluded_quadruped_cells": excluded_quadruped,
                "parse_ok": sum(1 for r in rows if r["parse_ok"]),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"foil pairs={len(rows)} skipped_no_axis_or_same={skipped} "
        f"same_or_unusable={skipped_same_motion} excluded_quadruped={excluded_quadruped} "
        f"parse_ok={sum(1 for r in rows if r['parse_ok'])}",
        flush=True,
    )

    foil_json = out / "motion_configs"
    foil_render = out / "render"
    if not args.skip_render:
        render_only = None
        if args.render_only.strip():
            render_only = set()
            for spec in args.render_only.split(","):
                if ":" not in spec:
                    ap.error(f"--render-only expects platform:cue, got {spec!r}")
                plat, cue = spec.split(":", 1)
                render_only.add((plat.strip(), cue.strip()))
            print(f"incremental foil render={sorted(render_only)}", flush=True)
        elif args.render_changed:
            render_only = {
                (r["platform"], r["cue"])
                for r in rows
                if (r["platform"], r["cue"]) not in previous_rows
                or previous_rows[(r["platform"], r["cue"])].get("new_code") != r.get("new_code")
            }
            print(f"changed foil render={sorted(render_only)}", flush=True)
        render_foils(
            rows,
            foil_json_dir=foil_json,
            foil_render_root=foil_render,
            true_render_root=out / "render_true",
            render_only=render_only,
        )

    html_out = out / "gallery_dynamic_axis_foil.html"
    paired_true = out / "render_true"
    build_gallery(
        rows,
        out_html=html_out,
        true_root=paired_true if paired_true.is_dir() else RENDER_TRUE,
        foil_root=foil_render,
    )
    print(f"gallery → {html_out}", flush=True)


if __name__ == "__main__":
    main()
