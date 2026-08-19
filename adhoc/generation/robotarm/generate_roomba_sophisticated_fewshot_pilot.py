#!/usr/bin/env python3
"""Generate five difficult Roomba Motion Codes from sophisticated few-shots."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from google import genai

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from embodiment_renderers import render_roomba  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402

FEWSHOTS = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_cues_yml_5.json"
MASTER = (
    REPO
    / "data/results/motion_configs/embodiment_motion_code_gt"
    / "embodiment_motion_code_all.json"
)
OUT = REPO / "data/results/roomba_sophisticated_fewshot_pilot"
TARGET_IDXS = (12, 13, 18, 24, 25)


def parse_indices(raw: str) -> tuple[int, ...]:
    selected: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = (int(value) for value in token.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return tuple(sorted(selected))


def _target_rows(indices: tuple[int, ...]) -> list[dict[str, Any]]:
    targets = {idx: {} for idx in indices}
    for row in json.loads(MASTER.read_text(encoding="utf-8")):
        idx = int(row["idx"])
        if idx not in targets:
            continue
        cell = row["platforms"]["roomba"]
        targets[idx] = {
            "idx": idx,
            "cue": row["cue"],
            "description": cell.get("description") or row.get("description") or "",
        }
    return [targets[idx] for idx in indices]


def _clean_code(text: str) -> str:
    fenced = re.search(r"```(?:\w+)?\s*(.*?)```", text, re.S)
    code = (fenced.group(1) if fenced else text).strip()
    code = re.sub(r"^(?:motion[_ ]?code\s*:?)\s*", "", code, flags=re.I)
    return " ".join(line.strip() for line in code.splitlines() if line.strip())


def _prompt(shots: list[dict[str, Any]], target: dict[str, Any]) -> str:
    examples = "\n".join(
        f"- {shot['cue']}: {shot['motion_code']}" for shot in shots
    )
    return f"""You design expressive Motion Codes for a Roomba-style planar base.

Few-shot examples:
{examples}

Target cue: {target['cue']}
Description: {target['description']}

Roomba constraints:
- Use ONLY pose(dir=...), path(base, line, ...), path(base, arc, ...), and hold(t=...).
- +x is forward, +y is left. Use in-place yaw arcs (r=0) as hand/head gesture proxies.
- Use a clear temporal story: preparation → expressive proxy → release or recovery.
- Combine diverse motions to create a recognizable gesture that transcends the embodiment's limitations.
- Make it recognizable even though there are no arms. Avoid a generic single spin.
- Return exactly ONE Motion Code expression, no prose or markdown."""


def _write_gallery(rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        code = html.escape(row.get("motion_code") or "")
        if row.get("render_ok") and Path(row.get("render_gif") or "").is_file():
            image = (
                f"<img src='render/{html.escape(Path(row['render_gif']).name)}"
                f"?v={Path(row['render_gif']).stat().st_mtime_ns}' alt='{html.escape(row['cue'])}'>"
            )
        else:
            image = f"<p class='fail'>{html.escape(row.get('error') or 'render failed')}</p>"
        cards.append(
            f"<article><h2>g{row['idx']:02d} · {html.escape(row['cue'])}</h2>"
            f"<p>{html.escape(row['description'])}</p>{image}<pre>{code}</pre></article>"
        )
    (OUT / "gallery.html").write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Roomba Sophisticated Few-shot Pilot</title>
<style>
body{margin:24px;background:#10151d;color:#e8edf5;font-family:ui-sans-serif,system-ui}
h1{margin:0 0 5px}.sub{color:#9aa7b5}.grid{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:14px}
article{background:#171f2b;border:1px solid #334155;padding:12px;border-radius:8px}h2{font-size:15px;margin:0 0 5px}
p{font-size:12px;color:#b7c2d0}img{width:100%;background:#000;border-radius:5px}pre{white-space:pre-wrap;font-size:11px;line-height:1.45;color:#cbd5e1}.fail{color:#fca5a5}
</style></head><body><h1>Roomba sophisticated few-shot · difficult cue pilot</h1>
<p class='sub'>Generated from the five edited Roomba few-shots, then parsed and rendered as Motion Codes.</p>
<div class='grid'>"""
        + "".join(cards)
        + "</div></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    global OUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--only-idx",
        default=",".join(str(index) for index in TARGET_IDXS),
        help="Cue indices or inclusive ranges (for example 1-30).",
    )
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render previously generated pilot rows that do not yet have GIFs.",
    )
    args = parser.parse_args()
    OUT = args.out_dir.expanduser().resolve()
    if args.render_only:
        rows = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
        render_dir = OUT / "render"
        for row in rows:
            expected_gif = render_dir / f"emb_g{row['idx']:02d}_{row['cue']}.gif"
            if not row.get("parse_ok") or (
                row.get("render_ok") and expected_gif.is_file()
            ):
                continue
            converted = convert_code(row["motion_code"], platform="roomba")
            render_row = {
                "idx": row["idx"],
                "cue": row["cue"],
                "description": row["description"],
                "movements": converted["movements"],
                "source_code": row["motion_code"],
                "parse_ok": True,
                "platform": "roomba",
            }
            try:
                render_roomba([render_row], render_dir, force=True, fps=8)
                gif = render_dir / f"emb_g{row['idx']:02d}_{row['cue']}.gif"
                row["render_ok"] = gif.is_file()
                row["render_gif"] = str(gif)
                row.pop("error", None)
            except Exception as exc:  # noqa: BLE001
                row["render_ok"] = False
                row["error"] = str(exc)
            print(f"{row['cue']}: render={row.get('render_ok', False)}")
        (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        _write_gallery(rows)
        print(OUT / "gallery.html")
        return

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    shots = json.loads(FEWSHOTS.read_text(encoding="utf-8"))
    client = genai.Client(api_key=key)
    OUT.mkdir(parents=True, exist_ok=True)
    render_dir = OUT / "render"
    rows: list[dict[str, Any]] = []
    for target in _target_rows(parse_indices(args.only_idx)):
        response = client.models.generate_content(
            model=args.model,
            contents=_prompt(shots, target),
        )
        raw = (response.text or "").strip()
        code = _clean_code(raw)
        converted = convert_code(code, platform="roomba")
        row = {
            **target,
            "motion_code": code,
            "raw_output": raw,
            "parse_ok": bool(converted.get("parse_ok")),
        }
        if not row["parse_ok"]:
            row["error"] = str(converted.get("error") or "parse failed")
            rows.append(row)
            continue
        render_row = {
            "idx": target["idx"],
            "cue": target["cue"],
            "description": target["description"],
            "movements": converted["movements"],
            "source_code": code,
            "parse_ok": True,
            "platform": "roomba",
        }
        try:
            render_roomba([render_row], render_dir, force=True, fps=8)
            gif = render_dir / f"emb_g{target['idx']:02d}_{target['cue']}.gif"
            row["render_ok"] = gif.is_file()
            row["render_gif"] = str(gif)
        except Exception as exc:  # noqa: BLE001
            row["render_ok"] = False
            row["error"] = str(exc)
        rows.append(row)
        print(f"{target['cue']}: parse={row['parse_ok']} render={row.get('render_ok', False)}")
    (OUT / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    _write_gallery(rows)
    print(OUT / "gallery.html")


if __name__ == "__main__":
    main()
