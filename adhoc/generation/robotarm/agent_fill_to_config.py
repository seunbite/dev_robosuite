#!/usr/bin/env python3
"""Parse agent-filled Motion Code markdown/txt → JSON configs, Excel sheet, optional render.

Expected fill format (see data/seed/agent_fill/embodiment_short/codes_fill.md)::

    ## humanoid
    01 nod_yes | pose(...) + move(...)
    02 shake_head_no | ...

Also accepts TSV lines::

    idx\\tcue\\tplatform\\tcode

Contamination note: this script only reads the fill file + cue catalog. It does not
open Human GT motion-code sheets unless you pass --xlsx for writing a new sheet.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc/generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import (  # noqa: E402
    PLATFORM_COLS,
    PLATFORM_HEADERS,
    add_excel_sheet,
    build_html,
    convert_code,
    render_manipulator,
    render_quadruped,
    write_json_files,
)

PLATFORMS = list(PLATFORM_COLS.values())
_SECTION_RE = re.compile(r"^##\s*([a-zA-Z0-9_]+)\s*$")
_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z0-9_]+)\s*\|\s*(.*?)\s*$"
)
_TSV_RE = re.compile(
    r"^\s*(\d+)\t([A-Za-z0-9_]+)\t([A-Za-z0-9_]+)\t(.*)$"
)

DEFAULT_FILL = _REPO / "data/seed/agent_fill/embodiment_short/codes_fill.md"
DEFAULT_CUES = _REPO / "data/seed/agent_fill/embodiment_short/cues.json"
DEFAULT_OUT_JSON = _REPO / "data/results/motion_configs/embodiment_motion_code_agent"
DEFAULT_OUT_RENDER = _REPO / "data/results/render/embodiment_motion_code_agent"
DEFAULT_HTML = _REPO / "data/results/html/embodiment_motion_code_agent.html"
DEFAULT_XLSX = _REPO / "data/seed/gesture_embodiment_table.xlsx"


def load_cues(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data, key=lambda r: int(r["idx"]))


def parse_fill(text: str) -> dict[tuple[str, str], str]:
    """Return mapping (platform, cue) -> motion code."""
    out: dict[tuple[str, str], str] = {}
    platform: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        sec = _SECTION_RE.match(line)
        if sec:
            platform = sec.group(1).lower()
            if platform == "manipulator":
                platform = "single_arm"
            continue
        # single-# comments (not ## headings)
        if line.startswith("#"):
            mplat = re.match(r"^#\s*platform\s*:\s*([a-zA-Z0-9_]+)\s*$", line, re.I)
            if mplat:
                platform = mplat.group(1).lower()
            continue
        tsv = _TSV_RE.match(raw)
        if tsv:
            _idx, cue, plat, code = tsv.groups()
            plat = plat.lower()
            if plat == "manipulator":
                plat = "single_arm"
            code = code.strip()
            if code:
                out[(plat, cue)] = code
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        _idx, cue, code = m.groups()
        if platform is None:
            raise ValueError(f"code line before ## platform section: {line[:80]}")
        code = code.strip()
        if code:
            out[(platform, cue)] = code
    return out


def build_rows_from_fill(
    cues: list[dict[str, Any]],
    filled: dict[tuple[str, str], str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cue_row in cues:
        codes = {
            plat: filled.get((plat, cue_row["cue"]), "") for plat in PLATFORMS
        }
        rows.append(
            {
                "idx": int(cue_row["idx"]),
                "cue": cue_row["cue"],
                "category": cue_row.get("category", ""),
                "codes": codes,
            }
        )
    return rows


def build_all_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same shape as motion_code_to_config.build_all_configs."""
    out: list[dict[str, Any]] = []
    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        entry: dict[str, Any] = {
            "idx": row["idx"],
            "cue": row["cue"],
            "category": row["category"],
            "platforms": {},
        }
        for plat, code in row["codes"].items():
            converted = convert_code(str(code), platform=plat)
            domain = {
                "single_arm": "manipulator",
                "tiago": "tiago",
                "humanoid": "unitree_g1",
                "quadruped": "unitree_go2",
                "roomba": "roomba",
            }[plat]
            cfg = {
                "idx": row["idx"],
                "cue": row["cue"],
                "description": f"Agent-fill embodiment ({plat}) for {row['cue']}",
                "movements": converted["movements"],
                "state": "motion_code_agent",
                "model": "coding_agent",
                "time": now,
                "experiment": "embodiment_motion_code_agent",
                "generation_valid": bool(str(code).strip()),
                "platform": plat,
                "domain": domain,
                "parse_ok": converted["parse_ok"],
                "notes": converted["notes"],
                "source_code": converted["source_code"],
            }
            entry["platforms"][plat] = cfg
        out.append(entry)
    return out


def write_motion_code_sheet(
    xlsx: Path,
    rows: list[dict[str, Any]],
    *,
    sheet_name: str,
) -> None:
    """Write/replace a Motion-Code-style sheet (cue × platform cells)."""
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font

    if xlsx.is_file():
        wb = load_workbook(xlsx)
        if sheet_name in wb.sheetnames:
            del wb[sheet_name]
    else:
        wb = Workbook()
        # drop default empty sheet later if unused
        if "Sheet" in wb.sheetnames and len(wb.sheetnames) == 1:
            pass

    ws = wb.create_sheet(sheet_name, 0)
    headers = ["#", "Gesture", "Category"] + [PLATFORM_HEADERS[p] for p in PLATFORMS]
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(1, c, h)
        cell.font = Font(bold=True)
    ws.cell(
        2,
        1,
        "Key",
    )
    ws.cell(
        2,
        2,
        "pose(...) + move(...) + path(...) + hold(...)  |  agent-filled Short motion codes",
    )
    for i, row in enumerate(rows):
        r = i + 3
        ws.cell(r, 1, row["idx"])
        ws.cell(r, 2, row["cue"])
        ws.cell(r, 3, row.get("category", ""))
        for col, plat in PLATFORM_COLS.items():
            cell = ws.cell(r, col, row["codes"].get(plat, ""))
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["B"].width = 28
    for col in range(4, 9):
        ws.column_dimensions[chr(ord("A") + col - 1)].width = 42
    xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx)


def coverage_report(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> str:
    lines = ["coverage:"]
    for plat in PLATFORMS:
        filled = sum(1 for r in rows if str(r["codes"].get(plat, "")).strip())
        ok = sum(
            1
            for e in configs
            if e["platforms"][plat].get("generation_valid")
            and e["platforms"][plat].get("parse_ok")
        )
        bad = [
            e["cue"]
            for e in configs
            if str(e["platforms"][plat].get("source_code") or "").strip()
            and not e["platforms"][plat].get("parse_ok")
        ]
        lines.append(f"  {plat}: filled={filled}/{len(rows)} parse_ok={ok}")
        if bad:
            lines.append(f"    parse issues: {', '.join(bad[:8])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fill", type=Path, default=DEFAULT_FILL)
    ap.add_argument("--cues", type=Path, default=DEFAULT_CUES)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-render", type=Path, default=DEFAULT_OUT_RENDER)
    ap.add_argument("--html-out", type=Path, default=DEFAULT_HTML)
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    ap.add_argument("--sheet-name", default="Motion Code (Agent)")
    ap.add_argument("--renderable-sheet", default="Renderable JSON (Agent)")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument(
        "--platforms",
        default="humanoid,tiago,single_arm,quadruped,roomba",
        help="Comma list to render",
    )
    ap.add_argument("--max-cues", type=int, default=0)
    ap.add_argument(
        "--html-title",
        default="Embodiment — Motion Code (Agent)",
    )
    args = ap.parse_args()

    cues = load_cues(args.cues)
    if args.max_cues > 0:
        cues = cues[: args.max_cues]
    filled = parse_fill(args.fill.read_text(encoding="utf-8"))
    rows = build_rows_from_fill(cues, filled)
    configs = build_all_configs(rows)
    print(coverage_report(rows, configs), flush=True)

    master = write_json_files(configs, args.out_json)
    print(f"JSON: {master}", flush=True)

    write_motion_code_sheet(args.xlsx, rows, sheet_name=args.sheet_name)
    print(f"Excel sheet {args.sheet_name!r} written: {args.xlsx}", flush=True)
    add_excel_sheet(
        args.xlsx,
        configs,
        sheet_name=args.renderable_sheet,
        note=f"Agent-fill renderable JSON from {args.fill}",
    )
    print(f"Excel sheet {args.renderable_sheet!r} written: {args.xlsx}", flush=True)

    if args.skip_render:
        return

    want = {p.strip() for p in args.platforms.split(",") if p.strip()}
    if "tiago" in want:
        from motion_code_to_config import render_tiago

        render_tiago(
            [e["platforms"]["tiago"] for e in configs],
            out_root=args.out_render,
            force=args.force_render,
        )
    if "single_arm" in want:
        render_manipulator(
            [e["platforms"]["single_arm"] for e in configs],
            out_root=args.out_render,
            force=args.force_render,
        )
    from embodiment_renderers import render_roomba, render_unitree_g1

    if "humanoid" in want:
        render_unitree_g1(
            [e["platforms"]["humanoid"] for e in configs],
            args.out_render / "humanoid" / "gif",
            force=args.force_render,
        )
    if "quadruped" in want:
        render_quadruped(
            [e["platforms"]["quadruped"] for e in configs],
            config_path=args.out_json / "result_quadruped.json",
            out_root=args.out_render,
            force=args.force_render,
        )
    if "roomba" in want:
        render_roomba(
            [e["platforms"]["roomba"] for e in configs],
            args.out_render / "roomba" / "gif",
            force=args.force_render,
        )

    out = build_html(
        configs,
        args.out_render,
        args.html_out,
        title=args.html_title,
        meta=(
            f"{len(configs)} gestures · agent-fill · source {args.fill.name} · "
            f"render {args.out_render}"
        ),
    )
    print(f"HTML: {out}", flush=True)


if __name__ == "__main__":
    main()
