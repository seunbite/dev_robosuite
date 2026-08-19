#!/usr/bin/env python3
"""Export / apply editable few-shot xlsx → render → 5-col HTML gallery.

Workflow::

    # (already done) export current Motion Codes
    python adhoc/generation/robotarm/sync_fewshot_xlsx.py --export

    # edit data/seed/shots/embodiment_unified/fewshot_cues_yml_5.xlsx
    #   (edit motion_code cells; keep cue/platform/rank)

    # re-parse, re-extract traj, re-render changed rows, refresh HTML
    python adhoc/generation/robotarm/sync_fewshot_xlsx.py --apply

    # force re-render everything
    python adhoc/generation/robotarm/sync_fewshot_xlsx.py --apply --force-render
"""
from __future__ import annotations

import argparse
import html as H
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from build_embodiment_unified_shots import (  # noqa: E402
    PLATS,
    _add,
    enrich_and_render,
)

SEED = _REPO / "data/seed/shots/embodiment_unified"
OUT = _REPO / "data/results/shots/embodiment_unified"
XLSX = SEED / "fewshot_cues_yml_5.xlsx"
JSON_IN = OUT / "shot_cues_yml_5.json"
JSON_OUT = OUT / "shot_cues_yml_5.json"
YAML_OUT = SEED / "shot_cues_yml_5.yaml"
SEL_JSON = OUT / "gallery_5col_cues_yml_selection.json"
GALLERY = OUT / "gallery_5col_cues_yml.html"
PINS = SEED / "pins"

PLAT_LAB = {
    "humanoid": "Humanoid",
    "tiago": "Tiago",
    "stretch": "Stretch",
    "single_arm": "Single Arm",
    "roomba": "Roomba",
}

HEADERS = [
    "rank",
    "difficulty",
    "cue",
    "platform",
    "motion_code",
    "notes",
    "shot_id",
    "parse_ok",
    "render_ok",
    "render_gif",
    "source",
]


def _inject_pose_pins(bag: dict[str, list[dict[str, Any]]]) -> None:
    """Attach offline ``pinned_qpos`` when a pin file exists for cue×platform.

    Used for gestures the discrete G1 bank cannot express well (e.g. blow_kiss
    mouth-cover with ``up|horizontal|palmback`` then forward ``palmup``).
    """
    if not PINS.is_dir():
        return
    for plat, items in bag.items():
        for e in items:
            pin_path = PINS / f"{plat}_{e['cue']}.json"
            if not pin_path.is_file():
                continue
            payload = json.loads(pin_path.read_text())
            pins = payload.get("poses") or []
            pose_steps = [s for s in (e.get("movements") or []) if s.get("type") == "pose"]
            if len(pins) < len(pose_steps):
                print(
                    f"[pins] {pin_path.name}: {len(pins)} pins < {len(pose_steps)} poses; skip",
                    flush=True,
                )
                continue
            for step, pin in zip(pose_steps, pins):
                q = pin.get("qpos")
                if isinstance(q, list) and q:
                    step.setdefault("parameters", {}).setdefault("pose", {})["pinned_qpos"] = q
            print(f"[pins] injected {pin_path.name} → {e.get('shot_id')}", flush=True)


def _default_selection() -> list[dict[str, Any]]:
    if SEL_JSON.is_file():
        return json.loads(SEL_JSON.read_text())
    return [
        {"rank": 1, "difficulty": "Easy", "cue": "raising_hand_greeting", "why": ""},
        {"rank": 2, "difficulty": "Easy–Mid", "cue": "wave_hi", "why": ""},
        {"rank": 3, "difficulty": "Mid", "cue": "bow_greeting", "why": ""},
        {"rank": 4, "difficulty": "Mid–Hard", "cue": "blow_kiss", "why": ""},
        {"rank": 5, "difficulty": "Hard", "cue": "circle_temple_crazy", "why": ""},
    ]


def _load_shot_bag() -> dict[str, list[dict[str, Any]]]:
    if not JSON_IN.is_file():
        raise SystemExit(f"missing {JSON_IN}")
    return json.loads(JSON_IN.read_text())


def export_xlsx() -> Path:
    bag = _load_shot_bag()
    sel = _default_selection()
    by_cue = {s["cue"]: s for s in sel}

    wb = Workbook()
    # README
    ws0 = wb.active
    ws0.title = "README"
    lines = [
        "Few-shot Motion Codes for Fig1 baseline (5 cues × 5 robots).",
        "",
        "EDIT: sheet 'shots' → column motion_code (and optionally notes/difficulty).",
        "DO NOT rename cue / platform / rank unless you know what you are doing.",
        "",
        "After editing, run:",
        "  python adhoc/generation/robotarm/sync_fewshot_xlsx.py --apply",
        "",
        "That will: parse motion_code → xyzrpy/joint_angles → render GIF → refresh gallery HTML.",
        f"Gallery: {GALLERY.relative_to(_REPO)}",
        "",
        "Cues (easy→hard) are outside the Fig1 30-cue eval set.",
        "Source seed: data/seed/shots/tiago/shot_configs_19_mobile.json (+ _cues.yml keys).",
    ]
    for i, line in enumerate(lines, 1):
        ws0.cell(i, 1, line)
    ws0.column_dimensions["A"].width = 100

    ws = wb.create_sheet("shots", 0)
    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    for col, h in enumerate(HEADERS, 1):
        cell = ws.cell(1, col, h)
        cell.fill = header_fill
        cell.font = header_font

    # order: rank then platform order
    rows_out: list[dict[str, Any]] = []
    for s in sorted(sel, key=lambda x: x["rank"]):
        cue = s["cue"]
        for plat in PLATS:
            hit = next((e for e in bag.get(plat, []) if e.get("cue") == cue), None)
            if not hit:
                hit = {
                    "cue": cue,
                    "platform": plat,
                    "motion_code": "",
                    "notes": s.get("why") or "",
                    "shot_id": f"{plat}__{cue}__cues_yml",
                    "parse_ok": False,
                    "render_ok": False,
                    "render_gif": "",
                    "source": "",
                }
            rows_out.append(
                {
                    "rank": s["rank"],
                    "difficulty": s.get("difficulty") or "",
                    "cue": cue,
                    "platform": plat,
                    "motion_code": hit.get("motion_code") or "",
                    "notes": hit.get("notes") or s.get("why") or "",
                    "shot_id": hit.get("shot_id") or f"{plat}__{cue}__cues_yml",
                    "parse_ok": bool(hit.get("parse_ok")),
                    "render_ok": bool(hit.get("render_ok")),
                    "render_gif": hit.get("render_gif") or "",
                    "source": hit.get("source") or "",
                }
            )

    wrap = Alignment(wrap_text=True, vertical="top")
    for r_i, row in enumerate(rows_out, 2):
        for c_i, h in enumerate(HEADERS, 1):
            val = row.get(h, "")
            if isinstance(val, bool):
                val = "TRUE" if val else "FALSE"
            cell = ws.cell(r_i, c_i, val)
            cell.alignment = wrap

    widths = {
        "A": 6,
        "B": 12,
        "C": 24,
        "D": 12,
        "E": 80,
        "F": 40,
        "G": 36,
        "H": 10,
        "I": 10,
        "J": 50,
        "K": 40,
    }
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{len(rows_out)+1}"

    # selection sheet (edit difficulty/why here too)
    ws2 = wb.create_sheet("selection")
    ws2.append(["rank", "difficulty", "cue", "why"])
    for s in sel:
        ws2.append([s["rank"], s.get("difficulty"), s["cue"], s.get("why") or by_cue.get(s["cue"], {}).get("why", "")])
    for col in range(1, 5):
        ws2.cell(1, col).fill = header_fill
        ws2.cell(1, col).font = header_font
    ws2.column_dimensions["A"].width = 6
    ws2.column_dimensions["B"].width = 12
    ws2.column_dimensions["C"].width = 24
    ws2.column_dimensions["D"].width = 60

    SEED.mkdir(parents=True, exist_ok=True)
    wb.save(XLSX)
    print(f"wrote {XLSX} ({len(rows_out)} rows)", flush=True)
    return XLSX


def _read_xlsx() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not XLSX.is_file():
        raise SystemExit(f"missing {XLSX} — run --export first")
    wb = load_workbook(XLSX, data_only=True)
    ws = wb["shots"]
    headers = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
    shots: list[dict[str, Any]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not any(row):
            continue
        d = {headers[i]: row[i] for i in range(len(headers))}
        if not d.get("cue") or not d.get("platform"):
            continue
        d["motion_code"] = str(d.get("motion_code") or "").strip()
        d["notes"] = str(d.get("notes") or "").strip()
        d["cue"] = str(d["cue"]).strip()
        d["platform"] = str(d["platform"]).strip()
        d["rank"] = int(d["rank"]) if d.get("rank") is not None else 0
        d["difficulty"] = str(d.get("difficulty") or "").strip()
        d["shot_id"] = str(d.get("shot_id") or f"{d['platform']}__{d['cue']}__cues_yml")
        shots.append(d)

    sel: list[dict[str, Any]] = []
    if "selection" in wb.sheetnames:
        ws2 = wb["selection"]
        h2 = [c.value for c in next(ws2.iter_rows(min_row=1, max_row=1))]
        for row in ws2.iter_rows(min_row=2, values_only=True):
            if not any(row):
                continue
            d = {h2[i]: row[i] for i in range(len(h2))}
            if not d.get("cue"):
                continue
            sel.append(
                {
                    "rank": int(d["rank"]) if d.get("rank") is not None else 0,
                    "difficulty": str(d.get("difficulty") or ""),
                    "cue": str(d["cue"]).strip(),
                    "why": str(d.get("why") or "").strip(),
                }
            )
    else:
        # derive from shots
        seen = {}
        for s in shots:
            seen.setdefault(
                s["cue"],
                {
                    "rank": s["rank"],
                    "difficulty": s["difficulty"],
                    "cue": s["cue"],
                    "why": s.get("notes") or "",
                },
            )
        sel = sorted(seen.values(), key=lambda x: x["rank"])
    return shots, sel


def _slim(e: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "cue",
        "shot_id",
        "role",
        "motion_code",
        "xyzrpy",
        "joint_angles",
        "render_gif",
        "source",
        "notes",
        "parse_ok",
        "extract_ok",
        "render_ok",
        "hz",
        "extract_error",
        "render_error",
        "rank",
        "difficulty",
    }
    return {k: e[k] for k in keep if k in e}


def write_gallery(bag: dict[str, list[dict[str, Any]]], sel: list[dict[str, Any]]) -> Path:
    idx = {p: {e["cue"]: e for e in bag[p]} for p in PLATS}
    rows_html = []
    for s in sorted(sel, key=lambda x: x["rank"]):
        cells = []
        for plat in PLATS:
            e = idx.get(plat, {}).get(s["cue"])
            if not e:
                cells.append("<td class='fail'><div class='miss'>missing</div></td>")
                continue
            gif = e.get("render_gif") or ""
            # The gallery and rendered assets both live below the repository.
            # Resolve the asset relative to the gallery directory so file://
            # viewing keeps working regardless of the repository location.
            gif_url = (
                Path(gif).relative_to(GALLERY.parent.relative_to(_REPO)).as_posix()
                if gif
                else ""
            )
            # cache-bust when file mtime changes
            if gif and (_REPO / gif).is_file():
                gif_url += f"?t={int((_REPO / gif).stat().st_mtime)}"
            mc = H.escape((e.get("motion_code") or "")[:220])
            ok = "ok" if e.get("render_ok") else "fail"
            img = (
                f'<img src="{H.escape(gif_url)}" alt="{H.escape(s["cue"])}"/>'
                if gif_url
                else "<div class='miss'>no gif</div>"
            )
            cells.append(
                f"<td class='{ok}'><div class='plat'>{PLAT_LAB[plat]}</div>{img}"
                f"<pre class='mc'>{mc}</pre></td>"
            )
        why = s.get("why") or ""
        rows_html.append(
            f"<tr><th><div class='rank'>#{s['rank']} · {H.escape(s.get('difficulty') or '')}</div>"
            f"<div class='cue'>{H.escape(s['cue'])}</div>"
            f"<div class='why'>{H.escape(why)}</div></th>{''.join(cells)}</tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/>
<title>Few-shot · editable xlsx → render</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:#0f1115;color:#e8eaed}}
h1{{font-size:1.25rem;margin:0 0 6px}}
.sub{{color:#9aa0a6;margin:0 0 18px;font-size:.9rem;max-width:95ch}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #2a2f3a;padding:8px;vertical-align:top;width:18%}}
th:first-child{{width:14%;text-align:left;background:#161a22}}
img{{width:100%;max-width:220px;background:#000;border-radius:4px;display:block}}
.plat{{font-size:.75rem;color:#9aa0a6;margin-bottom:4px}}
.cue{{font-family:ui-monospace,monospace;font-size:.95rem;margin:4px 0}}
.rank{{color:#8ab4f8;font-size:.8rem}}
.why{{color:#9aa0a6;font-size:.75rem;margin-top:6px}}
.mc{{font-size:.65rem;white-space:pre-wrap;color:#bdc1c6;margin:6px 0 0;max-height:5.5em;overflow:auto}}
.fail{{outline:1px solid #c5221f}}
.miss{{color:#c5221f;padding:40px 0;text-align:center}}
code{{background:#1e2430;padding:1px 4px;border-radius:3px}}
</style></head><body>
<h1>Few-shot gallery (from <code>{XLSX.relative_to(_REPO)}</code>)</h1>
<p class="sub">Edit the xlsx → <code>python adhoc/generation/robotarm/sync_fewshot_xlsx.py --apply</code> → this page refreshes.
Outside Fig1 30-cue set · easy→hard · 5 robots.</p>
<table>
<thead><tr><th>Cue</th>{''.join(f'<th>{PLAT_LAB[p]}</th>' for p in PLATS)}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</body></html>
"""
    GALLERY.write_text(html)
    return GALLERY


def apply_xlsx(*, force_render: bool = False) -> None:
    shots, sel = _read_xlsx()
    # build bag via _add (parse)
    bag: dict[str, list[dict[str, Any]]] = {p: [] for p in PLATS}
    old = _load_shot_bag() if JSON_IN.is_file() else {p: [] for p in PLATS}
    old_code = {
        (e["cue"], plat): (e.get("motion_code") or "")
        for plat in PLATS
        for e in old.get(plat, [])
    }

    for s in shots:
        plat = s["platform"]
        if plat not in bag:
            print(f"[skip] unknown platform {plat}", flush=True)
            continue
        _add(
            bag,
            platform=plat,
            cue=s["cue"],
            motion_code=s["motion_code"],
            source=str(s.get("source") or XLSX.relative_to(_REPO)),
            role=f"cues_yml_e2h_r{s['rank']}",
            shot_id=s["shot_id"],
            notes=s.get("notes") or "",
        )
        # stamp meta
        bag[plat][-1]["rank"] = s["rank"]
        bag[plat][-1]["difficulty"] = s["difficulty"]

    _inject_pose_pins(bag)

    # only force-render rows whose motion_code changed (unless --force-render)
    if force_render:
        bag = enrich_and_render(bag, force=True)
    else:
        # split: changed vs unchanged
        changed = {p: [] for p in PLATS}
        unchanged_meta = {}
        for plat in PLATS:
            for e in bag[plat]:
                prev = old_code.get((e["cue"], plat), None)
                if prev is None or prev.strip() != (e["motion_code"] or "").strip():
                    changed[plat].append(e)
                else:
                    # reuse old traj/gif if present
                    old_e = next((x for x in old.get(plat, []) if x.get("cue") == e["cue"]), None)
                    if old_e and old_e.get("render_gif"):
                        e["xyzrpy"] = old_e.get("xyzrpy") or []
                        e["joint_angles"] = old_e.get("joint_angles") or {
                            "joint_names": [],
                            "joints_rad": [],
                        }
                        e["render_gif"] = old_e.get("render_gif")
                        e["render_ok"] = old_e.get("render_ok", True)
                        e["extract_ok"] = old_e.get("extract_ok", True)
                        e["hz"] = old_e.get("hz")
                        unchanged_meta[(plat, e["cue"])] = True
                    else:
                        changed[plat].append(e)
        n_chg = sum(len(v) for v in changed.values())
        print(f"[apply] changed rows to re-render: {n_chg}", flush=True)
        if n_chg:
            enrich_and_render(changed, force=True)
            # merge back
            ch_idx = {(plat, e["cue"]): e for plat in PLATS for e in changed[plat]}
            for plat in PLATS:
                for i, e in enumerate(bag[plat]):
                    key = (plat, e["cue"])
                    if key in ch_idx:
                        bag[plat][i] = ch_idx[key]

    # write json/yaml
    focus = {p: [_slim(e) for e in bag[p]] for p in PLATS}
    OUT.mkdir(parents=True, exist_ok=True)
    SEED.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(focus, indent=2, ensure_ascii=False) + "\n")
    YAML_OUT.write_text(yaml.safe_dump(focus, sort_keys=False, allow_unicode=True))
    SEL_JSON.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")

    # refresh xlsx status columns from new bag
    export_from_bag(focus, sel)

    g = write_gallery(bag, sel)
    n_ok = sum(1 for p in PLATS for e in bag[p] if e.get("render_ok"))
    n = sum(len(bag[p]) for p in PLATS)
    print(f"render_ok {n_ok}/{n}", flush=True)
    print(f"wrote {JSON_OUT}", flush=True)
    print(f"wrote {g}", flush=True)


def export_from_bag(bag: dict[str, list[dict[str, Any]]], sel: list[dict[str, Any]]) -> None:
    """Rewrite xlsx preserving motion_code from bag (after apply)."""
    # temporarily point JSON_IN logic: write bag then export
    JSON_OUT.write_text(json.dumps(bag, indent=2, ensure_ascii=False) + "\n")
    SEL_JSON.write_text(json.dumps(sel, indent=2, ensure_ascii=False) + "\n")
    export_xlsx()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", action="store_true", help="JSON → xlsx")
    ap.add_argument("--apply", action="store_true", help="xlsx → render → HTML")
    ap.add_argument("--force-render", action="store_true")
    ap.add_argument("--gallery-only", action="store_true", help="rebuild HTML from current JSON")
    args = ap.parse_args()

    if args.gallery_only:
        bag = _load_shot_bag()
        sel = _default_selection()
        g = write_gallery(bag, sel)
        print(f"wrote {g}", flush=True)
        return

    if args.export or (not args.apply):
        # default: export if no flag, or explicit --export
        if args.export or not args.apply:
            export_xlsx()
        if not args.apply:
            return

    if args.apply:
        apply_xlsx(force_render=args.force_render)


if __name__ == "__main__":
    main()
