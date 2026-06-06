#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import fire
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[misc, assignment]

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.embodiment_sources import repo_rel_to_path, resolve_embodiment_paths  # noqa: E402
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from legacy.render_mobile_config import _make_env, render_config  # noqa: E402


def _palette_cues_ordered(path: Path) -> list[str]:
    if yaml is None:
        raise RuntimeError("palette_yml requires PyYAML")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ordered: list[str] = []
    seen: set[str] = set()
    if not isinstance(data, dict):
        return ordered
    for blob in data.values():
        if isinstance(blob, str):
            tokens = blob.split()
        elif isinstance(blob, list):
            tokens = [str(x) for x in blob]
        else:
            continue
        for cue in tokens:
            if cue not in seen:
                seen.add(cue)
                ordered.append(cue)
    return ordered


def _open_file(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        print(f"Could not open {path}: {e}")


def _write_html(
    config_path: Path,
    items: list[tuple[dict, Path]],
    *,
    html_subdir: str = "google_robot",
    html_stem: str = "render_google_robot",
) -> Path:
    html_path = dated_results_html_path(html_subdir, html_stem)
    out_dir = html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for cfg, gif_path in items:
        cue = cfg.get("cue", "?")
        idx = cfg.get("idx", -1)
        rel_gif = os.path.relpath(gif_path, out_dir) if gif_path.exists() else ""
        gif_html = f'<img src="{rel_gif}" loading="lazy">' if rel_gif else "<div>No GIF</div>"
        cards.append(
            f"<article><h3>c{idx} {cue}</h3>{gif_html}<pre>{json.dumps(cfg, indent=2, ensure_ascii=False)}</pre></article>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>google_robot render</title>"
        "<style>body{font-family:sans-serif;padding:20px}article{border:1px solid #ddd;margin:12px 0;padding:12px}"
        "img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f7f7f7;padding:8px}</style></head><body>"
        f"<h1>google_robot render ({config_path.name})</h1>{''.join(cards)}</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def run(
    config_json: str | None = None,
    output_dir: str | None = None,
    auto_generate_if_missing: bool = True,
    do_html: bool = False,
    sources_yml: str | None = None,
    palette_yml: str | None = None,
    overlay_progress_bar: bool = True,
    progress_bar_style: str = "typed",
    cues: str | None = None,
) -> None:
    src_opt = Path(sources_yml) if sources_yml else None
    _, _, yaml_cfg, blk = resolve_embodiment_paths("google_robot", src_opt)
    cfg = Path(config_json) if config_json else yaml_cfg
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False, sources_yml=sources_yml)
    gif_dir = Path(output_dir) if output_dir else repo_rel_to_path(blk["render_dir"])
    gif_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if palette_yml:
        pal = set(_palette_cues_ordered(Path(palette_yml)))
        configs = [r for r in configs if isinstance(r.get("cue"), str) and r["cue"] in pal]
    announce_output(gif_dir, len(configs), kind="gif")

    want_cues = {c.strip() for c in cues.split(",") if c.strip()} if cues else None
    rendered: list[tuple[dict, Path]] = []
    env = _make_env()
    try:
        for row in tqdm(configs, desc="render[google_robot]"):
            idx = int(row.get("idx", -1))
            cue = row.get("cue", "?")
            safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
            gif_path = gif_dir / f"mm19_g{idx:02d}_{safe}.gif"
            if want_cues is not None and cue not in want_cues:
                rendered.append((row, gif_path if gif_path.is_file() else Path("")))
                continue
            try:
                frames = render_config(
                    row,
                    env=env,
                    overlay_progress_bar=overlay_progress_bar,
                    progress_bar_style=progress_bar_style,
                )
                if frames:
                    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=50, loop=0)
                rendered.append((row, gif_path if gif_path.exists() else Path("")))
            except Exception as e:
                tqdm.write(f"skip g{idx}: {e}")
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    if do_html:
        html = _write_html(
            cfg,
            rendered,
            html_subdir=str(blk["render_html_subdir"]),
            html_stem=str(blk["render_html_stem"]),
        )
        print(f"HTML saved: {html}")
        _open_file(html)


if __name__ == "__main__":
    fire.Fire(run)
