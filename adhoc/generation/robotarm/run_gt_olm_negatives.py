#!/usr/bin/env python3
"""Build location / orientation / movement negatives from humanlike-90 GT.

Source: embodiment_motion_code_gt_humanlike90 (G1) + _gt_tiago_humanlike90 (Tiago).
Pilot: 10 cues × 2 bodies = 20 rows × 4 clips. Full run would be 90×2=180 rows (720 clips).

    python adhoc/generation/robotarm/run_gt_olm_negatives.py
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_PAPER = _REPO / "papers/icra2027_motion_abstraction"
for p in (_REPO, _HERE, _PAPER):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from motion_code_to_config import (  # noqa: E402
    _parse_call,
    _split_steps_with_ops,
    convert_code,
)
from run_embodiment_exp0_gemini import gif_path_for, render_pending_once  # noqa: E402
from run_olm_state_regen_ablation import format_sim_olm  # noqa: E402

GT_G1 = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gt_humanlike90"
    / "embodiment_motion_code_all.json"
)
GT_TIAGO = (
    _REPO
    / "data/results/motion_configs"
    / "embodiment_motion_code_gt_tiago_humanlike90"
    / "embodiment_motion_code_all.json"
)
RENDER_GT_PAPER30 = _REPO / "data/results/render" / "embodiment_motion_code_gt"
DEFAULT_OUT = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "results_gt_olm_negatives_pilot20"
)
HTML_COPY = _PAPER / "olm_gt_negatives_pilot" / "gallery.html"

PLATFORMS = ("humanoid", "tiago")
KINDS = ("loc", "ori", "move")
KIND_LABEL = {
    "gt": "GT (positive)",
    "loc": "NEG location",
    "ori": "NEG orientation",
    "move": "NEG movement",
}

# Mix of paper-30 emblems and later humanlike-90 hard cues.
PILOT_CUES = (
    "salute",
    "highfive_invite",
    "stop_palm_out",
    "cover_yawn_sleepy",
    "pat_self_back",
    "expand_space",
    "clarification_request_palm_tilt",
    "cover_ears_too_loud",
    "cheers_toast",
    "shh_gesture",
)

DIR_ORTHO = {
    "front": "up",
    "up": "front",
    "in": "out",
    "out": "in",
    "back": "front",
    "down": "up",
    "still": "front",
}
PALM_FLIP = {
    "palmin": "palmout",
    "palmout": "palmin",
    "palmfront": "palmback",
    "palmback": "palmfront",
    "palmup": "palmdown",
    "palmdown": "palmup",
}
AXIS_FLIP = {
    "x": "z",
    "z": "x",
    "y": "x",
    "pan": "tilt",
    "tilt": "pan",
    "roll": "pan",
    "xz": "yz",
    "yz": "xz",
    "xy": "yz",
}


def _fmt_val(v: Any) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, list):
        return "[" + ",".join(_fmt_val(x) for x in v) + "]"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _rebuild(name: str, pos: list[Any], kw: dict[str, Any]) -> str:
    bits = [_fmt_val(p) for p in pos]
    bits.extend(f"{k}={_fmt_val(v)}" for k, v in kw.items())
    return f"{name}({', '.join(bits)})"


def _join(parts: list[tuple[str, str]]) -> str:
    out = []
    for i, (op, step) in enumerate(parts):
        if i == 0:
            out.append(step)
        else:
            sep = f" {op} " if op else " + "
            out.append(sep + step)
    return "".join(out)


def _flip_pct(v: float) -> int:
    nv = 100.0 - v if abs(v - 50.0) >= 12 else (v + 40.0 if v < 50 else v - 40.0)
    return int(max(0, min(100, round(nv))))


def _negate(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return -v
    if isinstance(v, list):
        return [_negate(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("+"):
            return "-" + s[1:]
        if s.startswith("-"):
            return "+" + s[1:]
    return v


def _flip_grip(grip: Any) -> str:
    raw = str(grip)
    parts = [p.strip() for p in raw.replace(" ", "").split("|") if p.strip()]
    out: list[str] = []
    flipped_palm = False
    for p in parts:
        key = p.lower()
        if key in PALM_FLIP:
            out.append(PALM_FLIP[key])
            flipped_palm = True
        elif key in ("vert", "vertical"):
            out.append("horiz")
        elif key in ("horiz", "horizontal"):
            out.append("vert")
        else:
            out.append(p)
    if not flipped_palm and not any(p.lower() in PALM_FLIP for p in parts):
        # no palm token: add an opposing palm
        out.append("palmback")
    return "|".join(out)


def _mutate_loc_parts(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    done = False
    new: list[tuple[str, str]] = []
    for op, step in parts:
        parsed = _parse_call(step)
        if not done and parsed and parsed[0] == "pose":
            name, pos, kw = parsed
            axes = [a for a in ("x", "y", "z") if isinstance(kw.get(a), (int, float))]
            if axes:
                for a in axes:
                    kw[a] = _flip_pct(float(kw[a]))
            else:
                kw["x"] = 25
                kw["y"] = 85
                kw["z"] = 30
            new.append((op, _rebuild(name, pos, kw)))
            done = True
        else:
            new.append((op, step))
    return new


def _mutate_ori_parts(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    done = False
    new: list[tuple[str, str]] = []
    for op, step in parts:
        parsed = _parse_call(step)
        if not done and parsed and parsed[0] == "pose":
            name, pos, kw = parsed
            d = str(kw.get("dir") or "still").lower()
            kw["dir"] = DIR_ORTHO.get(d, "up")
            if "grip" in kw:
                kw["grip"] = _flip_grip(kw["grip"])
            else:
                kw["grip"] = "horiz|palmback"
            new.append((op, _rebuild(name, pos, kw)))
            done = True
        else:
            new.append((op, step))
    return new


def _mutate_move_parts(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    done = False
    new: list[tuple[str, str]] = []
    for op, step in parts:
        parsed = _parse_call(step)
        if done or not parsed:
            new.append((op, step))
            continue
        name, pos, kw = parsed
        if name in ("move", "movement"):
            if len(pos) >= 2:
                axis = str(pos[1]).lower()
                pos[1] = AXIS_FLIP.get(axis, "z")
            elif "axis" in kw:
                kw["axis"] = AXIS_FLIP.get(str(kw["axis"]).lower(), "z")
            if len(pos) >= 3:
                pos[2] = _negate(pos[2])
            elif "degrees" in kw:
                kw["degrees"] = _negate(kw["degrees"])
            elif "deg" in kw:
                kw["deg"] = _negate(kw["deg"])
            new.append((op, _rebuild(name, pos, kw)))
            done = True
        elif name == "path":
            if "axis" in kw:
                kw["axis"] = AXIS_FLIP.get(str(kw["axis"]).lower(), "z")
            if "plane" in kw:
                kw["plane"] = AXIS_FLIP.get(str(kw["plane"]).lower(), "yz")
            if "dist" in kw:
                kw["dist"] = _negate(kw["dist"])
            elif "distance" in kw:
                kw["distance"] = _negate(kw["distance"])
            new.append((op, _rebuild(name, pos, kw)))
            done = True
        else:
            new.append((op, step))
    if not done:
        new.append(("+", "path(ee, line, axis=z, dist=0.20, spd=1.5)"))
    return new


def load_gt90() -> dict[str, dict[str, Any]]:
    """Merge G1 90-cue GT with Tiago 90-cue GT (later cues live in separate masters)."""
    g1 = json.loads(GT_G1.read_text(encoding="utf-8"))
    tg = json.loads(GT_TIAGO.read_text(encoding="utf-8"))
    by: dict[str, dict[str, Any]] = {}
    for e in g1:
        cue = str(e["cue"])
        by[cue] = {
            "idx": int(e["idx"]),
            "cue": cue,
            "category": e.get("category") or "",
            "platforms": dict(e.get("platforms") or {}),
        }
    for e in tg:
        cue = str(e["cue"])
        cur = by.setdefault(
            cue,
            {
                "idx": int(e["idx"]),
                "cue": cue,
                "category": e.get("category") or "",
                "platforms": {},
            },
        )
        tcell = (e.get("platforms") or {}).get("tiago")
        if tcell:
            cur["platforms"]["tiago"] = tcell
    return by


def mutate_code(code: str, kind: str) -> str:
    parts = _split_steps_with_ops(code)
    if kind == "loc":
        parts = _mutate_loc_parts(parts)
    elif kind == "ori":
        parts = _mutate_ori_parts(parts)
    elif kind == "move":
        parts = _mutate_move_parts(parts)
    else:
        raise ValueError(kind)
    return _join(parts)


def _cell_cfg(
    *,
    idx: int,
    cue: str,
    platform: str,
    code: str,
    converted: dict[str, Any],
    experiment: str,
) -> dict[str, Any]:
    domain = {"humanoid": "unitree_g1", "tiago": "tiago"}.get(platform, platform)
    return {
        "idx": idx,
        "cue": cue,
        "description": "",
        "movements": converted.get("movements") or [],
        "state": "gt_olm_negative",
        "model": "human_gt_mutated",
        "time": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment,
        "generation_valid": bool(converted.get("movements")),
        "platform": platform,
        "domain": domain,
        "parse_ok": bool(converted.get("parse_ok")),
        "notes": converted.get("notes") or [],
        "source_code": converted.get("source_code") or code,
    }


def media_tag(path: Path | None, html_parent: Path) -> str:
    if path is None or not path.is_file():
        return '<p class="miss">missing</p>'
    rel = os.path.relpath(path, html_parent)
    bust = int(path.stat().st_mtime)
    src = html_mod.escape(f"{rel}?v={bust}")
    if path.suffix.lower() == ".mp4":
        return (
            f'<video src="{src}" autoplay loop muted playsinline></video>'
        )
    return f'<img src="{src}" loading="lazy">'


def resolve_gif(root: Path, platform: str, idx: int, cue: str) -> Path | None:
    gif = gif_path_for(root, platform, idx, cue)
    if gif.is_file():
        return gif
    mp4 = gif.with_suffix(".mp4")
    return mp4 if mp4.is_file() else None


def write_gallery(rows: list[dict[str, Any]], *, out_html: Path, render_root: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    sections = []
    for row in rows:
        plat, idx, cue = row["platform"], int(row["idx"]), row["cue"]
        cols = []
        for kind, label in KIND_LABEL.items():
            if kind == "gt":
                gif = resolve_gif(render_root / "gt", plat, idx, cue) or resolve_gif(
                    RENDER_GT_PAPER30, plat, idx, cue
                )
                code = row["gt_code"]
                olm = row.get("gt_olm") or ""
            else:
                gif = resolve_gif(render_root / kind, plat, idx, f"{cue}__{kind}")
                code = row["neg"][kind]["code"]
                olm = row["neg"][kind].get("olm") or ""
            cols.append(
                f'<div class="col {kind}">'
                f"<h3>{html_mod.escape(label)}</h3>"
                f"{media_tag(gif, out_html.parent)}"
                f"<pre>{html_mod.escape(code)}</pre>"
                f'<pre class="olm">{html_mod.escape(olm)}</pre></div>'
            )
        sections.append(
            f'<section class="row"><h2>{plat} · g{idx:02d} {html_mod.escape(cue)}</h2>'
            f'<div class="grid">{"".join(cols)}</div></section>'
        )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>GT O/L/M negatives · pilot 20</title>
<style>
body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:0;background:#111;color:#eee}}
h1{{padding:16px 24px;margin:0;border-bottom:1px solid #333}}
.meta{{padding:8px 24px;color:#9aa0a6;max-width:1100px}}
.row{{margin:14px 16px;padding:12px;border:1px solid #333;border-radius:12px;background:#1a1d24}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}}
.col h3{{margin:0 0 6px;font-size:.8rem}}
.col.gt h3{{color:#81c995}}
.col.loc h3{{color:#fdd663}}
.col.ori h3{{color:#8ab4f8}}
.col.move h3{{color:#f28b82}}
img,video{{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;background:#000;display:block}}
pre{{font-size:11px;white-space:pre-wrap;word-break:break-word;background:#0a0c10;padding:8px;border-radius:6px;max-height:120px;overflow:auto}}
.olm{{color:#c5e1a5;max-height:90px}}
.miss{{color:#f28b82}}
@media (max-width:1100px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head>
<body>
<h1>GT negatives · location / orientation / movement</h1>
<p class="meta">Pilot n={len(rows)} rows from <b>humanlike-90 GT</b> (10 cues × G1+Tiago).
Each row is GT plus one handle-wise foil (location / orientation / movement).
Full inventory = 90 cues × 2 bodies = 180 rows (720 clips). VLM not run yet — input set only.</p>
{"".join(sections)}
</body></html>
"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cues", default=",".join(PILOT_CUES))
    ap.add_argument("--platforms", default=",".join(PLATFORMS))
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--force-render", action="store_true")
    args = ap.parse_args()

    cues = tuple(c.strip() for c in args.cues.split(",") if c.strip())
    plats = tuple(p.strip() for p in args.platforms.split(",") if p.strip())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    by_cue = load_gt90()
    missing = [c for c in cues if c not in by_cue]
    if missing:
        raise SystemExit(f"missing GT cues: {missing}")

    rows: list[dict[str, Any]] = []
    render_cfgs: dict[str, list[dict[str, Any]]] = {k: [] for k in ("gt", *KINDS)}

    for cue in cues:
        entry = by_cue[cue]
        idx = int(entry["idx"])
        for plat in plats:
            cell = (entry.get("platforms") or {}).get(plat) or {}
            gt_code = str(cell.get("source_code") or "").strip()
            if not gt_code:
                print(f"[skip] no GT code {plat} {cue}", flush=True)
                continue
            gt_conv = convert_code(gt_code, platform=plat)
            if gt_conv.get("parse_ok") and gt_conv.get("movements"):
                render_cfgs["gt"].append(
                    {
                        "idx": idx,
                        "cue": cue,
                        "category": "",
                        "description": cue,
                        "platforms": {
                            plat: _cell_cfg(
                                idx=idx,
                                cue=cue,
                                platform=plat,
                                code=gt_code,
                                converted=gt_conv,
                                experiment="gt_pos",
                            )
                        },
                    }
                )
            row: dict[str, Any] = {
                "idx": idx,
                "cue": cue,
                "platform": plat,
                "gt_code": gt_code,
                "gt_olm": format_sim_olm(plat, gt_conv),
                "neg": {},
            }
            for kind in KINDS:
                neg_code = mutate_code(gt_code, kind)
                conv = convert_code(neg_code, platform=plat)
                file_cue = f"{cue}__{kind}"
                row["neg"][kind] = {
                    "code": conv.get("source_code") or neg_code,
                    "parse_ok": bool(conv.get("parse_ok")),
                    "notes": conv.get("notes") or [],
                    "olm": format_sim_olm(plat, conv),
                    "file_cue": file_cue,
                }
                if conv.get("parse_ok") and conv.get("movements"):
                    render_cfgs[kind].append(
                        {
                            "idx": idx,
                            "cue": file_cue,
                            "category": "",
                            "description": f"{cue} {kind} negative",
                            "platforms": {
                                plat: _cell_cfg(
                                    idx=idx,
                                    cue=file_cue,
                                    platform=plat,
                                    code=neg_code,
                                    converted=conv,
                                    experiment=f"gt_neg_{kind}",
                                )
                            },
                        }
                    )
            rows.append(row)
            print(f"--- {plat} {cue}", flush=True)
            print(f"  GT   {gt_code}", flush=True)
            for kind in KINDS:
                print(f"  {kind:4} {row['neg'][kind]['code']}", flush=True)

    (out_dir / "negatives.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.skip_render:
        for kind in ("gt", *KINDS):
            configs = _merge_platform_cfgs(render_cfgs[kind])
            if not configs:
                continue
            render_root = out_dir / "render" / kind
            master_out = out_dir / f"neg_{kind}.json"
            master_out.write_text(
                json.dumps(configs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"[render] {kind} cues={len(configs)}", flush=True)
            render_pending_once._skip = set()  # type: ignore[attr-defined]
            for _ in range(30):
                n = render_pending_once(
                    configs,
                    render_root=render_root,
                    out_json=master_out,
                    platforms=list(plats),
                    force=args.force_render,
                    max_items=6,
                )
                if n == 0:
                    break
                print(f"[render] {kind} attempted {n}", flush=True)

    html_out = out_dir / "gallery.html"
    write_gallery(rows, out_html=html_out, render_root=out_dir / "render")
    write_gallery(rows, out_html=HTML_COPY, render_root=out_dir / "render")
    print(f"html: {html_out}\nhtml copy: {HTML_COPY}", flush=True)


def _merge_platform_cfgs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in items:
        key = (int(entry["idx"]), str(entry["cue"]))
        cur = by.setdefault(key, {**entry, "platforms": {}})
        cur["platforms"].update(entry.get("platforms") or {})
    return [by[k] for k in sorted(by)]


if __name__ == "__main__":
    main()
