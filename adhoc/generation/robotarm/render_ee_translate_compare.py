#!/usr/bin/env python3
"""Re-render Gemini-made humanoid/tiago with EE residual translate; compare to nearest.

Original (nearest bank hit, residual xyz kept):
  data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro/{plat}/gif/

New (nearest bank + EE parallel-translate snap, no anim of the snap):
  data/results/render/embodiment_motion_code_gemini_gemini-2.5-pro_ee_translate/{plat}/gif/

HTML:
  data/results/html/gemini25pro_nearest_vs_ee_translate_hu_ti.html
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from embodiment_renderers import render_unitree_g1  # noqa: E402
from motion_code_to_config import render_tiago  # noqa: E402

TAG = "embodiment_motion_code_gemini_gemini-2.5-pro"
CFG_ROOT = _REPO / "data/results/motion_configs" / TAG
NEAREST_ROOT = _REPO / "data/results/render" / TAG
TRANSLATE_ROOT = _REPO / "data/results/render" / f"{TAG}_ee_translate"
HTML_OUT = _REPO / "data/results/html/gemini25pro_nearest_vs_ee_translate_hu_ti.html"
PLATS = ("humanoid", "tiago")


def _rows_for(plat: str) -> list[dict[str, Any]]:
    path = CFG_ROOT / f"result_{plat}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and "results" in raw:
        rows = list(raw["results"])
    else:
        # master all.json shape fallback
        master = json.loads((CFG_ROOT / "embodiment_motion_code_all.json").read_text(encoding="utf-8"))
        rows = []
        for entry in master:
            cell = (entry.get("platforms") or {}).get(plat) or {}
            if not cell.get("parse_ok"):
                continue
            rows.append(
                {
                    "idx": int(entry["idx"]),
                    "cue": str(entry["cue"]),
                    "description": entry.get("description") or "",
                    "movements": cell.get("movements") or [],
                    "parse_ok": True,
                    "source_code": cell.get("source_code") or "",
                    "platform": plat,
                }
            )
        return sorted(rows, key=lambda r: int(r["idx"]))
    out = []
    for r in rows:
        if not r.get("parse_ok"):
            continue
        out.append(
            {
                "idx": int(r["idx"]),
                "cue": str(r["cue"]),
                "description": r.get("description") or "",
                "movements": r.get("movements") or [],
                "parse_ok": True,
                "source_code": r.get("source_code") or "",
                "platform": plat,
            }
        )
    return sorted(out, key=lambda r: int(r["idx"]))


def _gif(root: Path, plat: str, idx: int, cue: str) -> Path:
    return root / plat / "gif" / f"emb_g{idx:02d}_{cue}.gif"


def cmd_render(args: argparse.Namespace) -> None:
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or list(PLATS)
    TRANSLATE_ROOT.mkdir(parents=True, exist_ok=True)
    for plat in plats:
        rows = _rows_for(plat)
        if args.only_idx:
            allow = {int(x) for x in args.only_idx.split(",") if x.strip()}
            rows = [r for r in rows if int(r["idx"]) in allow]
        print(f"[ee-translate] {plat} n={len(rows)} → {TRANSLATE_ROOT / plat}", flush=True)
        if plat == "humanoid":
            os.environ["G1_POSE_XYZ_MODE"] = "translate"
            render_unitree_g1(rows, TRANSLATE_ROOT / "humanoid" / "gif", force=args.force)
        elif plat == "tiago":
            os.environ["MOBILE_POSE_XYZ_MODE"] = "trim"
            # clear setdefault lock from prior nearest runs in-process
            render_tiago(rows, out_root=TRANSLATE_ROOT, force=args.force)
        else:
            raise SystemExit(f"unsupported platform {plat}")
    print(f"[ee-translate] done → {TRANSLATE_ROOT}", flush=True)


HUMAN_YESNO_JSON = (
    _REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "cueid_human_binary_acc_by_platform_gemini_made.json"
)


def _load_human_yesno() -> dict[tuple[str, int], dict[str, Any]]:
    """Match-only gemini-made yes/no: wrong_details → human NO; else YES."""
    data = json.loads(HUMAN_YESNO_JSON.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for plat, block in (data.get("yesno") or {}).get("by_platform", {}).items():
        wrong = {
            int(w["idx"]): str(w.get("note") or "")
            for w in (block.get("wrong_details") or [])
        }
        n = int(block.get("n") or 30)
        for idx in range(1, n + 1):
            if idx in wrong:
                out[(plat, idx)] = {"label": "NO", "note": wrong[idx]}
            else:
                out[(plat, idx)] = {"label": "YES", "note": ""}
    return out


def cmd_html(args: argparse.Namespace) -> None:
    plats = [p.strip() for p in args.platforms.split(",") if p.strip()] or list(PLATS)
    human = _load_human_yesno()
    cards: list[str] = []
    n_ok = n_miss = 0
    n_yes = n_no = 0
    for plat in plats:
        rows = _rows_for(plat)
        for r in rows:
            idx, cue = int(r["idx"]), str(r["cue"])
            a = _gif(NEAREST_ROOT, plat, idx, cue)
            b = _gif(TRANSLATE_ROOT, plat, idx, cue)
            ra = os.path.relpath(a, HTML_OUT.parent) if a.is_file() else ""
            rb = os.path.relpath(b, HTML_OUT.parent) if b.is_file() else ""
            if a.is_file() and b.is_file():
                n_ok += 1
            else:
                n_miss += 1
            hv = human.get((plat, idx))
            if hv and hv["label"] == "YES":
                n_yes += 1
                badge = '<span class="badge yes" title="human yes/no: YES">YES</span>'
            elif hv and hv["label"] == "NO":
                n_no += 1
                note = html_mod.escape(hv.get("note") or "")
                title = f' title="human yes/no: NO — {note}"' if note else ' title="human yes/no: NO"'
                badge = f'<span class="badge no"{title}>NO</span>'
            else:
                badge = '<span class="badge na" title="no human yes/no label">—</span>'
            media_a = (
                f'<div class="frame">{badge}'
                f'<img src="{html_mod.escape(ra)}" alt="nearest"></div>'
                if ra
                else f'<div class="frame">{badge}<div class="miss">missing nearest</div></div>'
            )
            media_b = (
                f'<img src="{html_mod.escape(rb)}" alt="ee-translate">'
                if rb
                else '<div class="miss">missing ee-translate</div>'
            )
            # first pose xyz hint from movements
            xyz_bits = []
            for step in r.get("movements") or []:
                if step.get("type") != "pose":
                    continue
                pose = (step.get("parameters") or {}).get("pose") or {}
                for k in ("x", "y", "z"):
                    if isinstance(pose.get(k), (int, float)):
                        xyz_bits.append(f"{k}={pose[k]}")
                break
            xyz = ", ".join(xyz_bits) if xyz_bits else "(no xyz on first pose)"
            note_line = ""
            if hv and hv["label"] == "NO" and hv.get("note"):
                note_line = f'<p class="note">human NO note: {html_mod.escape(hv["note"])}</p>'
            cards.append(
                f'<section class="card">'
                f"<h2>{html_mod.escape(plat)} · g{idx:02d} {html_mod.escape(cue)}</h2>"
                f'<p class="meta">first-pose EE targets: {html_mod.escape(xyz)}</p>'
                f"{note_line}"
                f'<div class="cols">'
                f'<div class="col"><div class="lab">nearest (original) · human yes/no</div>{media_a}</div>'
                f'<div class="col"><div class="lab">ee-translate (new)</div>{media_b}</div>'
                f"</div></section>"
            )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Gemini-2.5-Pro · nearest vs EE-translate (humanoid + tiago)</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}}
h1{{padding:16px 24px;margin:0;font-size:1.2rem;border-bottom:1px solid #333}}
.banner{{padding:12px 24px;color:#9aa0a6;font-size:.9rem;border-bottom:1px solid #222}}
.card{{margin:18px 24px;padding:14px;border:1px solid #333;border-radius:10px;background:#171a21}}
.card h2{{margin:0 0 6px;font-size:1rem}}
.meta{{margin:0 0 6px;color:#9aa0a6;font-size:.82rem}}
.note{{margin:0 0 10px;color:#f28b82;font-size:.8rem}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:900px){{.cols{{grid-template-columns:1fr}}}}
.lab{{font-size:.72rem;color:#9aa0a6;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}}
.frame{{position:relative}}
.frame img{{width:100%;height:auto;border-radius:6px;background:#000;display:block}}
img{{width:100%;height:auto;border-radius:6px;background:#000;display:block}}
.badge{{position:absolute;top:8px;right:8px;z-index:2;padding:3px 9px;border-radius:999px;
  font-size:.72rem;font-weight:700;letter-spacing:.04em;box-shadow:0 1px 4px rgba(0,0,0,.45)}}
.badge.yes{{background:#137333;color:#e6f4ea}}
.badge.no{{background:#c5221f;color:#fce8e6}}
.badge.na{{background:#3c4043;color:#dadce0}}
.miss{{color:#f28b82;padding:40px 12px;text-align:center;background:#000;border-radius:6px}}
</style></head><body>
<h1>Gemini-2.5-Pro made · nearest bank vs EE residual translate</h1>
<p class="banner">
humanoid + tiago · 30 cues each · pair both present: {n_ok} · missing side: {n_miss}
 · human yes/no on original: <span style="color:#81c995">YES {n_yes}</span> /
 <span style="color:#f28b82">NO {n_no}</span><br>
Left = original feasible-set nearest (badge = your human yes/no on that clip).
Right = same bank hit, then EE parallel-translated to Motion Code x/y/z (snap; not animated).
</p>
{''.join(cards)}
</body></html>
"""
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(doc, encoding="utf-8")
    print(
        f"gallery → {HTML_OUT} (pairs_ok={n_ok} miss={n_miss} yes={n_yes} no={n_no})",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_r = sub.add_parser("render", help="render ee-translate GIFs")
    p_r.add_argument("--platforms", default="humanoid,tiago")
    p_r.add_argument("--only-idx", default="")
    p_r.add_argument("--force", action="store_true")
    p_r.set_defaults(func=cmd_render)
    p_h = sub.add_parser("html", help="build comparison HTML")
    p_h.add_argument("--platforms", default="humanoid,tiago")
    p_h.set_defaults(func=cmd_html)
    p_a = sub.add_parser("all", help="render then html")
    p_a.add_argument("--platforms", default="humanoid,tiago")
    p_a.add_argument("--only-idx", default="")
    p_a.add_argument("--force", action="store_true")

    def _all(a: argparse.Namespace) -> None:
        cmd_render(a)
        cmd_html(a)

    p_a.set_defaults(func=_all)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
