#!/usr/bin/env python3
"""Edit Human-GT Motion Code in a fill.md → convert JSON → re-render changed cues.

Workflow
--------
1. Edit Motion Codes in::

     data/seed/agent_fill/embodiment_short/codes_fill_gt.md

   Format (same as agent fill)::

     ## quadruped
     15 highfive_invite | pose(stance=sit) + move(right_front, x, +0.15, spd=1.0, hold=1.5)

2. Sync + render only what changed::

     python adhoc/generation/robotarm/sync_gt_motion_code.py

   Options::

     --skip-render
     --platforms quadruped,roomba
     --cues highfive_invite,salute
     --force-render          # re-render even if source_code unchanged
     --export                # rewrite fill.md from current GT JSON
     --copy-gemini plat:cue[,plat:cue...]
                             # copy Gemini source_code into fill, then sync+render

You only edit Motion Code strings. Renderable ``movements`` JSON is regenerated.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agent_fill_to_config import parse_fill  # noqa: E402
from motion_code_to_config import (  # noqa: E402
    build_html,
    convert_code,
    render_manipulator,
    render_quadruped,
    render_tiago,
    write_json_files,
)

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")
DEFAULT_FILL = _REPO / "data/seed/agent_fill/embodiment_short/codes_fill_gt.md"
DEFAULT_JSON = _REPO / "data/results/motion_configs/embodiment_motion_code_gt"
DEFAULT_RENDER = _REPO / "data/results/render/embodiment_motion_code_gt"
DEFAULT_HTML = _REPO / "data/results/html/embodiment_motion_code_gt.html"
GEMINI_JSON = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro"
    / "embodiment_motion_code_all.json"
)


def _norm(code: str) -> str:
    return " ".join(str(code or "").split())


def _load_all(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_fill(all_path: Path, fill_path: Path) -> None:
    configs = _load_all(all_path)
    lines = [
        "# Embodiment Short — Human GT Motion Code (EDIT THIS)",
        "#",
        "# Edit the Motion Code after `|`. Then:",
        "#   python adhoc/generation/robotarm/sync_gt_motion_code.py",
        "#",
        "# Copy one cell from Gemini into the fill + sync:",
        "#   python adhoc/generation/robotarm/sync_gt_motion_code.py --copy-gemini quadruped:highfive_invite",
        "#",
        "# Blank after `|` is ignored (keeps existing GT code).",
        "",
    ]
    for plat in PLATFORMS:
        lines.append(f"## {plat}")
        for e in sorted(configs, key=lambda x: int(x["idx"])):
            code = (e.get("platforms") or {}).get(plat, {}).get("source_code") or ""
            lines.append(f"{int(e['idx']):02d} {e['cue']} | {_norm(code)}")
        lines.append("")
    fill_path.parent.mkdir(parents=True, exist_ok=True)
    fill_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported fill → {fill_path}", flush=True)


def _set_fill_line(fill_text: str, platform: str, cue: str, code: str) -> str:
    """Replace or insert ``NN cue | code`` under ``## platform``."""
    lines = fill_text.splitlines()
    plat_re = re.compile(r"^##\s*([a-zA-Z0-9_]+)\s*$")
    line_re = re.compile(r"^\s*(\d+)\s+([A-Za-z0-9_]+)\s*\|\s*(.*?)\s*$")
    out: list[str] = []
    cur: str | None = None
    replaced = False
    for raw in lines:
        sec = plat_re.match(raw.strip())
        if sec:
            cur = sec.group(1).lower()
            if cur == "manipulator":
                cur = "single_arm"
            out.append(raw)
            continue
        m = line_re.match(raw)
        if m and cur == platform and m.group(2) == cue:
            out.append(f"{int(m.group(1)):02d} {cue} | {_norm(code)}")
            replaced = True
            continue
        out.append(raw)
    if not replaced:
        raise KeyError(f"fill has no line for {platform}:{cue}")
    return "\n".join(out) + ("\n" if fill_text.endswith("\n") else "")


def copy_gemini_into_fill(
    fill_path: Path,
    specs: list[str],
    gemini_path: Path,
) -> list[tuple[str, str]]:
    gem = {int(e["idx"]): e for e in _load_all(gemini_path)}
    # also by cue
    by_cue = {e["cue"]: e for e in gem.values()}
    text = fill_path.read_text(encoding="utf-8")
    copied: list[tuple[str, str]] = []
    for spec in specs:
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise ValueError(f"--copy-gemini expects plat:cue, got {spec!r}")
        plat, cue = spec.split(":", 1)
        plat, cue = plat.strip().lower(), cue.strip()
        if plat == "manipulator":
            plat = "single_arm"
        entry = by_cue.get(cue)
        if entry is None:
            raise KeyError(f"Gemini JSON missing cue {cue}")
        code = (entry.get("platforms") or {}).get(plat, {}).get("source_code") or ""
        if not code.strip():
            raise KeyError(f"Gemini has empty source_code for {plat}:{cue}")
        text = _set_fill_line(text, plat, cue, code)
        copied.append((plat, cue))
        print(f"copied Gemini → fill  {plat}:{cue}", flush=True)
        print(f"  {_norm(code)[:120]}", flush=True)
    fill_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return copied


def sync_from_fill(
    *,
    fill_path: Path,
    json_dir: Path,
    platforms: set[str] | None,
    cues: set[str] | None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    all_path = json_dir / "embodiment_motion_code_all.json"
    configs = _load_all(all_path)
    by_cue = {e["cue"]: e for e in configs}
    filled = parse_fill(fill_path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changed: list[tuple[str, str]] = []

    for (plat, cue), code in sorted(filled.items()):
        if platforms is not None and plat not in platforms:
            continue
        if cues is not None and cue not in cues:
            continue
        if not code.strip():
            continue
        entry = by_cue.get(cue)
        if entry is None:
            print(f"warn: unknown cue {cue}, skip", flush=True)
            continue
        row = (entry.get("platforms") or {}).get(plat)
        if row is None:
            print(f"warn: missing platform cell {plat}:{cue}, skip", flush=True)
            continue
        if _norm(row.get("source_code") or "") == _norm(code):
            continue
        converted = convert_code(code, platform=plat)
        row["source_code"] = converted["source_code"]
        row["movements"] = converted["movements"]
        row["parse_ok"] = converted["parse_ok"]
        row["notes"] = converted.get("notes") or []
        row["generation_valid"] = bool(converted["parse_ok"])
        row["time"] = now
        row["model"] = "human_gt_edit"
        changed.append((plat, cue))
        status = "ok" if converted["parse_ok"] else "PARSE_FAIL"
        print(f"updated {plat}:{cue} [{status}] notes={converted.get('notes')}", flush=True)

    write_json_files(configs, json_dir)
    print(f"JSON written → {all_path}  ({len(changed)} cells changed)", flush=True)
    return configs, changed


def render_changed(
    configs: list[dict[str, Any]],
    changed: list[tuple[str, str]],
    *,
    json_dir: Path,
    render_root: Path,
    force: bool,
) -> None:
    if not changed and not force:
        print("nothing to render", flush=True)
        return
    want_plats = {p for p, _ in changed} if changed else set(PLATFORMS)
    by_plat: dict[str, list[dict[str, Any]]] = {p: [] for p in PLATFORMS}
    changed_set = set(changed)
    for e in configs:
        for plat in PLATFORMS:
            if plat not in want_plats:
                continue
            # ``force`` only means overwrite existing GIFs in the renderers —
            # never expand the cell set beyond ``changed``.
            if changed and (plat, e["cue"]) not in changed_set:
                continue
            row = e["platforms"][plat]
            if not row.get("parse_ok") or not (row.get("movements") or []):
                print(f"skip render {plat}:{e['cue']} (parse_ok/movements)", flush=True)
                continue
            by_plat[plat].append(row)

    if "tiago" in want_plats and by_plat["tiago"]:
        render_tiago(by_plat["tiago"], out_root=render_root, force=True)
    if "single_arm" in want_plats and by_plat["single_arm"]:
        render_manipulator(by_plat["single_arm"], out_root=render_root, force=True)
    if "humanoid" in want_plats and by_plat["humanoid"]:
        from embodiment_renderers import render_unitree_g1

        render_unitree_g1(
            by_plat["humanoid"],
            render_root / "humanoid" / "gif",
            force=True,
        )
    if "stretch" in want_plats and by_plat["stretch"]:
        from embodiment_renderers import render_stretch

        render_stretch(
            by_plat["stretch"],
            render_root / "stretch" / "gif",
            force=True,
        )
    if "quadruped" in want_plats and by_plat["quadruped"]:
        # render_quadruped reads full result_quadruped.json; filter via temp
        rows = by_plat["quadruped"]
        tmp = json_dir / "_sync_rerender_quad.json"
        tmp.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        render_quadruped(rows, config_path=tmp, out_root=render_root, force=True)
    if "roomba" in want_plats and by_plat["roomba"]:
        from embodiment_renderers import render_roomba

        render_roomba(by_plat["roomba"], render_root / "roomba" / "gif", force=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fill", type=Path, default=DEFAULT_FILL)
    ap.add_argument("--json-dir", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--render-root", type=Path, default=DEFAULT_RENDER)
    ap.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--export", action="store_true", help="Rewrite fill.md from current GT JSON and exit")
    ap.add_argument(
        "--copy-gemini",
        default="",
        help="Comma list plat:cue to copy Gemini Motion Code into fill before sync",
    )
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--skip-html", action="store_true")
    ap.add_argument("--platforms", default="", help="Comma filter, e.g. quadruped")
    ap.add_argument("--cues", default="", help="Comma filter, e.g. highfive_invite,salute")
    args = ap.parse_args()

    all_path = args.json_dir / "embodiment_motion_code_all.json"
    if args.export:
        export_fill(all_path, args.fill)
        return

    if not args.fill.is_file():
        print(f"fill missing → exporting from JSON first: {args.fill}", flush=True)
        export_fill(all_path, args.fill)

    plats = {p.strip() for p in args.platforms.split(",") if p.strip()} or None
    cues = {c.strip() for c in args.cues.split(",") if c.strip()} or None

    if args.copy_gemini.strip():
        specs = [s for s in args.copy_gemini.split(",") if s.strip()]
        copy_gemini_into_fill(args.fill, specs, GEMINI_JSON)

    configs, changed = sync_from_fill(
        fill_path=args.fill,
        json_dir=args.json_dir,
        platforms=plats,
        cues=cues,
    )

    if args.force_render:
        # Always re-render the full filtered set (not only fill.md diffs).
        changed = []
        for e in configs:
            for plat in PLATFORMS:
                if plats is not None and plat not in plats:
                    continue
                if cues is not None and e["cue"] not in cues:
                    continue
                changed.append((plat, e["cue"]))

    if not args.skip_render:
        os.environ.setdefault("MUJOCO_GL", "glfw")
        render_changed(
            configs,
            changed,
            json_dir=args.json_dir,
            render_root=args.render_root,
            force=args.force_render,
        )

    if not args.skip_html:
        out = build_html(
            configs,
            args.render_root,
            args.html_out,
            title="Embodiment Motion Code — Human GT",
            meta=f"synced from {args.fill.name} · changed={len(changed)}",
        )
        print(f"HTML: {out}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
