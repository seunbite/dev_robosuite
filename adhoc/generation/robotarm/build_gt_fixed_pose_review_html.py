#!/usr/bin/env python3
"""Build multi-robot GIF review HTML with config step tables."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_CFG = (
    _REPO
    / "data/results/motion_configs/manipulator"
    / "motion_configs_prompt_v19_gt_fixed_pose_pilot40.json"
)

STEP_COLS = ("#", "type", "joint", "axis", "angle", "rep", "speed", "hold (s)")


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "-")


def _step_rows(cfg: dict) -> list[tuple[str, ...]]:
    rows: list[tuple[str, ...]] = []
    for i, step in enumerate(cfg.get("movements", []), 1):
        t = step.get("type", "?")
        pr = step.get("parameters", {})

        if t == "pose":
            p = pr.get("pose", {})
            label = "pose (GT)" if i == 1 else "pose"
            rows.append(
                (
                    str(i),
                    label,
                    "-",
                    "-",
                    f"{p.get('dir')}/{p.get('gripper_orientation')} ({p.get('x')},{p.get('y')},{p.get('z')}) pid={p.get('pose_id', '-')}",
                    "-",
                    _esc(pr.get("speed")),
                    _esc(pr.get("hold_time", 0)),
                )
            )
            continue

        if t == "path":
            shape = pr.get("shape", "-")
            if shape == "line":
                detail = f"line axis={pr.get('axis')} dist={pr.get('distance')}m"
                axis_col = _esc(pr.get("axis"))
            else:
                detail = (
                    f"arc plane={pr.get('plane')} r={pr.get('radius')}m "
                    f"sweep={pr.get('sweep')}° {pr.get('direction', '')}"
                )
                axis_col = _esc(pr.get("plane"))
            rows.append(
                (
                    str(i),
                    "path (EE IK)",
                    "-",
                    axis_col,
                    detail,
                    "-",
                    _esc(pr.get("speed")),
                    _esc(pr.get("hold_time", 0)),
                )
            )
            continue

        if t == "movement":
            joint = pr.get("joint", "-")
            rep = pr.get("repetition", 1)
            dirs = pr.get("directions", [])
            if not dirs:
                rows.append((str(i), "movement", _esc(joint), "-", "-", str(rep), "-", "-"))
                continue
            for bi, d in enumerate(dirs):
                degs = d.get("degrees") or {}
                step_lbl = str(i) if bi == 0 else ""
                type_lbl = "movement" if bi == 0 else ""
                joint_lbl = _esc(joint) if bi == 0 else ""
                rep_lbl = str(rep) if bi == 0 else ""
                rows.append(
                    (
                        step_lbl,
                        type_lbl,
                        joint_lbl,
                        _esc(", ".join(sorted(degs.keys())) or "-"),
                        _esc(", ".join(f"{k}{v}°" for k, v in sorted(degs.items())) or "-"),
                        rep_lbl,
                        _esc(d.get("speed")),
                        _esc(d.get("hold_time", 0)),
                    )
                )
            continue

        rows.append((str(i), _esc(t), "-", "-", "-", "-", "-", "-"))
    return rows


def config_table_html(cfg: dict) -> str:
    meta = (
        "<table class='meta'><tbody>"
        f"<tr><th>groundtruth</th><td>{_esc(cfg.get('groundtruth', '-'))}</td></tr>"
        f"<tr><th>description</th><td>{_esc(cfg.get('description', ''))}</td></tr>"
        "</tbody></table>"
    )
    head = "".join(f"<th>{_esc(c)}</th>" for c in STEP_COLS)
    body_rows = []
    for row in _step_rows(cfg):
        body_rows.append("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>")
    steps = (
        f"<table class='steps'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
    return f"<div class='config'>{meta}{steps}</div>"


def build_html(
    *,
    config_json: Path,
    gif_root: Path,
    html_path: Path,
    title: str,
    subtitle: str = "",
) -> Path:
    configs = json.loads(config_json.read_text(encoding="utf-8"))
    robots = sorted(d.name for d in gif_root.iterdir() if d.is_dir())
    html_path.parent.mkdir(parents=True, exist_ok=True)

    cards: list[str] = []
    for cfg in sorted(configs, key=lambda c: c.get("idx", 0)):
        cue = cfg["cue"]
        idx = cfg["idx"]
        tbl = config_table_html(cfg)
        row = [
            "<section class='cue'>",
            f"<h2>c{idx} {cue}</h2>",
            tbl,
            "<div class='grid'>",
        ]
        for rob in robots:
            d = gif_root / rob
            gifs = sorted(d.glob(f"*_{cue}_*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
            rel = os.path.relpath(gifs[0], html_path.parent) if gifs else ""
            img = f'<img src="{rel}" loading="lazy">' if rel else "<p>missing</p>"
            row.append(f"<figure><figcaption>{rob}</figcaption>{img}</figure>")
        row.append("</div></section>")
        cards.append("".join(row))

    css = """
body{font-family:system-ui,sans-serif;margin:20px;max-width:1500px}
header{background:#f0f4ff;padding:12px 16px;border-radius:8px;margin-bottom:24px}
section.cue{margin-bottom:36px;border-bottom:1px solid #eee;padding-bottom:20px}
h2{margin:0 0 10px;font-size:1.15rem}
.config{margin:0 0 14px}
table{border-collapse:collapse;font-size:12px;width:100%;max-width:900px}
table.meta{margin-bottom:8px;max-width:100%}
table.meta th{text-align:left;width:110px;background:#fafafa;font-weight:600}
table.steps th{background:#e8eef8;font-weight:600}
th,td{border:1px solid #ddd;padding:5px 8px;vertical-align:top}
table.steps tbody tr:nth-child(even){background:#fafafa}
table.steps td:first-child{font-weight:600;text-align:center;width:2em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
figure{margin:0;border:1px solid #ddd;padding:6px;border-radius:4px}
img{width:100%;height:auto}
figcaption{font-size:12px;font-weight:600}
"""
    sub = f"<p>{_esc(subtitle)}</p>" if subtitle else ""
    html_doc = (
        f"<!DOCTYPE html><html><head><meta charset=utf-8><title>{_esc(title)}</title>"
        f"<style>{css}</style></head><body>"
        f"<header><h1>{_esc(title)}</h1>{sub}</header>"
        f"{''.join(cards)}</body></html>"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def main() -> None:
    root = _REPO
    cfg = DEFAULT_CFG
    pairs = [
        (
            root / "data/results/visualize/gt_fixed_pose_pilot20_hz10",
            root / "data/results/html/manipulator/gt_fixed_pose_pilot20_hz10_all_robots.html",
            "GT-fixed pilot40 — 7 robots @ hz=10",
            "GIF dir: gt_fixed_pose_pilot20_hz10/ (40 cues)",
        ),
        (
            root / "data/results/visualize/gt_fixed_pose_pilot20",
            root / "data/results/html/manipulator/gt_fixed_pose_pilot20_all_robots.html",
            "GT-fixed pilot40 — 7 robots @ hz=4 (partial GIFs)",
            "GIF dir: gt_fixed_pose_pilot20/ (first 20 cues only unless re-rendered)",
        ),
    ]
    for gif_root, html_path, title, sub in pairs:
        if not gif_root.is_dir():
            continue
        out = build_html(
            config_json=cfg,
            gif_root=gif_root,
            html_path=html_path,
            title=title,
            subtitle=sub,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
