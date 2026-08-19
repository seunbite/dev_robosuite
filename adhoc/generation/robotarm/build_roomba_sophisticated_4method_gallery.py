#!/usr/bin/env python3
"""Build a four-method gallery for sophisticated Roomba experiments."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TARGETS = (12, 13, 18, 24, 25)
MOTION = REPO / "data/results/roomba_sophisticated_fewshot_pilot/results.json"
VECTOR = REPO / "data/results/baseline_traj/roomba_sophisticated_pilot"
TRAJECTORY = REPO / "data/results/drag_draw/roomba_sophisticated_pilot/predictions_rendered.json"
OUT = REPO / "data/results/html/roomba_sophisticated_4method_pilot.html"


def _by_idx(rows: list[dict]) -> dict[int, dict]:
    return {int(row["idx"]): row for row in rows if row.get("parse_ok")}


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


def _img(path: str, gallery: Path) -> str:
    absolute = Path(path)
    if not absolute.is_absolute():
        absolute = REPO / absolute
    if not absolute.is_file():
        return "<div class='missing'>missing render</div>"
    # ``relpath`` works across the separate pilot directories.
    return (
        f"<img src='{html.escape(os.path.relpath(absolute, gallery.parent))}"
        f"?v={absolute.stat().st_mtime_ns}'>"
    )


def _vector_rows(vector: Path, mode: str) -> dict[int, dict]:
    path = vector / f"predictions_{mode}.jsonl"
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("parse_ok"):
            latest[int(row["idx"])] = row
    return latest


def _vector_gif(vector: Path, mode: str, row: dict) -> Path:
    return (
        vector
        / f"render_{mode}"
        / "roomba"
        / f"emb_g{int(row['idx']):02d}_{row['cue']}_{row.get('model', 'gemini-2.5-pro')}.gif"
    )


def _cell(title: str, image: str, code: str = "") -> str:
    extra = f"<pre>{html.escape(code)}</pre>" if code else ""
    return f"<article><h3>{html.escape(title)}</h3>{image}{extra}</article>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, default=MOTION)
    parser.add_argument("--vector", type=Path, default=VECTOR)
    parser.add_argument("--trajectory", type=Path, default=TRAJECTORY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--only-idx", default=",".join(str(index) for index in TARGETS))
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    motion = _by_idx(json.loads(args.motion.read_text(encoding="utf-8")))
    joint = _vector_rows(args.vector, "joint")
    cartesian = _vector_rows(args.vector, "xyzrpy")
    trajectory = _by_idx(json.loads(args.trajectory.read_text(encoding="utf-8")))

    rows = []
    for idx in parse_indices(args.only_idx):
        source = motion[idx]
        method_cells = [
            _cell("Joint-Space", _img(str(_vector_gif(args.vector, "joint", joint[idx])), out)),
            _cell("Cartesian-Space", _img(str(_vector_gif(args.vector, "xyzrpy", cartesian[idx])), out)),
            _cell(
                "Motion Code",
                _img(source["render_gif"], out),
                source["motion_code"],
            ),
            _cell(
                "Trajectory Sketching",
                _img(trajectory[idx]["render_gif"], out),
            ),
        ]
        rows.append(
            f"<section><h2>c{idx:02d} · {html.escape(source['cue'])}</h2>"
            f"<p>{html.escape(source['description'])}</p><div class='grid'>{''.join(method_cells)}</div></section>"
        )

    out.write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Roomba sophisticated few-shot · four methods</title>
<style>
body{margin:24px;background:#0d1117;color:#e6edf3;font-family:ui-sans-serif,system-ui}
h1{margin:0 0 5px}.sub,p{color:#9da7b3;font-size:13px}.grid{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:10px}
section{border-top:1px solid #30363d;margin-top:20px;padding-top:14px}h2{font-size:16px;margin:0}h3{font-size:13px;margin:0 0 6px}
article{background:#161b22;border:1px solid #30363d;padding:9px;min-width:0}img{display:block;width:100%;background:#000;border-radius:5px}
pre{white-space:pre-wrap;font-size:10px;line-height:1.4;color:#c9d1d9;max-height:8em;overflow:auto}.missing{height:160px;display:grid;place-items:center;color:#fca5a5}
</style></head><body><h1>Roomba sophisticated five-shot · four-method comparison</h1>
<p class='sub'>Same five sophisticated Roomba demonstrations. Motion Code, Joint-Space, Cartesian-Space, and Trajectory Sketching are independently generated.</p>"""
        + "".join(rows)
        + "</body></html>",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
