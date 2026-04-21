import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire


WORKSPACE_ROOT = Path("/Users/sb/Downloads/workspace")
REPO_ROOT = WORKSPACE_ROOT / "dev_robosuite"
MOTION_SCRIPT = REPO_ROOT / "adhoc" / "robotarm" / "motion_generation.py"
POSE_DB = REPO_ROOT / "data" / "seed" / "closest_poses_results.jsonl"


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _open_preview(path: Path) -> None:
    try:
        subprocess.run(["open", str(path)], check=False)
    except Exception:
        pass


def _load_configs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _find_render(output_dir: Path, cue_idx: int, cue: str) -> str | None:
    candidates = []
    for pattern in (f"*_c{cue_idx}_*_tiled.gif", f"*_c{cue_idx}_*.gif"):
        candidates.extend(sorted(output_dir.rglob(pattern)))
    for path in candidates:
        if path.name.endswith("_preview.gif"):
            continue
        return str(path)
    cue_token = _norm_token(cue)
    if cue_token:
        for path in sorted(output_dir.rglob("*.gif")):
            if path.name.endswith("_preview.gif"):
                continue
            if cue_token in _norm_token(path.stem):
                return str(path)
    return None


def _render_one(
    config_path: Path,
    cue_idx: int,
    output_dir: Path,
    robot: str,
    hz: int,
    top_k: int,
    preview_speed_scale: float,
    preview_hold_scale: float,
) -> None:
    cmd = [
        sys.executable,
        str(MOTION_SCRIPT),
        "--robot",
        robot,
        "--config_path",
        str(config_path),
        "--jsonl_path",
        str(POSE_DB),
        "--cue_idx",
        str(cue_idx),
        "--output_dir",
        str(output_dir),
        "--hz",
        str(hz),
        "--top_k",
        str(top_k),
        "--preview_speed_scale",
        str(preview_speed_scale),
        "--preview_hold_scale",
        str(preview_hold_scale),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _write_html(
    sections: list[tuple[str, list[dict]]],
    html_path: Path,
    robot: str,
    hz: int,
    top_k: int,
    preview_speed_scale: float,
    preview_hold_scale: float,
) -> None:
    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Shot Config Render Preview</title>
<style>
:root {
  --bg: #f5f7fb; --surface: #fff; --surface2: #eff3f8; --border: #d7dee8; --text: #16202a; --muted: #647281; --accent: #0969da;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.wrap { max-width: 1800px; margin: 0 auto; padding: 24px; }
.hero { margin-bottom: 20px; }
.hero h1 { margin: 0 0 8px; font-size: 28px; }
.hero p { margin: 0; color: var(--muted); }
.hero .meta { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip { display: inline-block; padding: 4px 10px; border-radius: 999px; background: var(--surface2); border: 1px solid var(--border); color: var(--muted); font-size: 13px; }
.section { margin-bottom: 28px; }
.section h2 { margin: 0 0 12px; font-size: 22px; }
.grid { display: grid; grid-template-columns: repeat(2, minmax(340px, 1fr)); gap: 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; }
.hdr { padding: 12px 16px; background: var(--surface2); border-bottom: 1px solid var(--border); }
.title { font-size: 17px; font-weight: 700; }
.meta-line { margin-top: 4px; color: var(--muted); font-size: 13px; }
.media { background: #edf2f7; padding: 12px; }
.media img { max-width: 100%; display: block; border-radius: 10px; border: 1px solid var(--border); }
.body { padding: 14px 16px 16px; }
.label { margin: 10px 0 4px; color: var(--muted); font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
.mono { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; background: #fafcff; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.na { color: var(--muted); font-style: italic; }
@media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
"""]
    parts.append(
        f'<section class="hero"><h1>Shot Config Render Preview</h1><p>Rendered directly from shot config JSONs.</p><div class="meta">'
        f'<span class="chip">robot={_esc(robot)}</span>'
        f'<span class="chip">hz={hz}</span>'
        f'<span class="chip">top_k={top_k}</span>'
        f'<span class="chip">preview_speed_scale={preview_speed_scale}</span>'
        f'<span class="chip">preview_hold_scale={preview_hold_scale}</span>'
        f"</div></section>"
    )
    for section_title, cards in sections:
        parts.append(f'<section class="section"><h2>{_esc(section_title)}</h2><div class="grid">')
        for card in cards:
            parts.append('<article class="card">')
            parts.append(
                f'<div class="hdr"><div class="title">c{card["idx"]}: {_esc(card["cue"])}</div>'
                f'<div class="meta-line">state={_esc(card.get("state", ""))} | source={_esc(card["source_name"])}</div></div>'
            )
            parts.append('<div class="media">')
            if card.get("gif_path"):
                rel = os.path.relpath(card["gif_path"], html_path.parent)
                parts.append(f'<img src="{_esc(rel)}" alt="{_esc(card["cue"])}">')
            else:
                parts.append('<div class="na">No render found</div>')
            parts.append('</div><div class="body">')
            parts.append('<div class="label">Description</div>')
            parts.append(f'<div>{_esc(card.get("description", ""))}</div>')
            if card.get("planning_shot"):
                parts.append('<div class="label">Planning Shot</div>')
                parts.append(f'<div class="mono">{_esc(card.get("planning_shot", ""))}</div>')
            if card.get("reasoning"):
                parts.append('<div class="label">Reasoning</div>')
                parts.append(f'<div class="mono">{_esc(card.get("reasoning", ""))}</div>')
            parts.append('<div class="label">Movements</div>')
            parts.append(f'<div class="mono">{_esc(json.dumps(card.get("movements", []), indent=2, ensure_ascii=False))}</div>')
            parts.append('</div></article>')
        parts.append('</div></section>')
    parts.append("</div></body></html>")
    html_path.write_text("".join(parts), encoding="utf-8")


def main(
    v17_config: str = "data/seed/shot_configs_v17.json",
    current_config: str = "data/seed/shot_configs.json",
    output_root: str = "adhoc/test/results/shot_config_compare",
    robot: str = "IIWA",
    hz: int = 8,
    top_k: int = 1,
    preview_speed_scale: float = 1.0,
    preview_hold_scale: float = 1.0,
    open_html: bool = True,
):
    v17_path = (REPO_ROOT / v17_config).resolve() if not str(v17_config).startswith("/") else Path(v17_config)
    current_path = (REPO_ROOT / current_config).resolve() if not str(current_config).startswith("/") else Path(current_config)
    out_root = (REPO_ROOT / output_root).resolve() if not str(output_root).startswith("/") else Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    v17_out = out_root / f"v17_{timestamp}"
    current_out = out_root / f"current_{timestamp}"
    v17_out.mkdir(parents=True, exist_ok=True)
    current_out.mkdir(parents=True, exist_ok=True)

    sections = []
    for label, cfg_path, render_dir in [
        (v17_path.name, v17_path, v17_out),
        (current_path.name, current_path, current_out),
    ]:
        configs = _load_configs(cfg_path)
        cards = []
        print(f"\nRendering {label}: {len(configs)} cues")
        for item in sorted(configs, key=lambda x: x.get("idx", 0)):
            cue_idx = int(item["idx"])
            cue = item.get("cue", f"c{cue_idx}")
            print(f"  c{cue_idx}: {cue}")
            _render_one(
                config_path=cfg_path,
                cue_idx=cue_idx,
                output_dir=render_dir,
                robot=robot,
                hz=hz,
                top_k=top_k,
                preview_speed_scale=preview_speed_scale,
                preview_hold_scale=preview_hold_scale,
            )
            cards.append(
                {
                    **item,
                    "gif_path": _find_render(render_dir, cue_idx, cue),
                    "source_name": label,
                }
            )
        sections.append((f"{label} ({len(cards)} cues)", cards))

    html_path = out_root / f"shot_config_compare_{timestamp}.html"
    _write_html(
        sections=sections,
        html_path=html_path,
        robot=robot,
        hz=hz,
        top_k=top_k,
        preview_speed_scale=preview_speed_scale,
        preview_hold_scale=preview_hold_scale,
    )
    print(f"\nHTML: {html_path}")
    print(f"HTML URL: file://{html_path}")
    if open_html:
        _open_preview(html_path)


if __name__ == "__main__":
    fire.Fire(main)
