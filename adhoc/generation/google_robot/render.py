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
_GOOGLE = Path(__file__).resolve().parent
for p in (_REPO, _GOOGLE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from adhoc.generation.embodiment_sources import repo_rel_to_path, resolve_embodiment_paths  # noqa: E402
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from legacy.render_mobile_config import _make_env, render_config  # noqa: E402
from pose_bank_topk_tiles import first_pose_from_row, render_topk_tile  # noqa: E402

DEFAULT_TOPK_TILE_DIR = _REPO / "data/results/visualize/google_robot/pose_topk"
POSE_BANK_TOP_K = 30


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
    items: list[tuple[dict, Path, Path | None]],
    *,
    html_subdir: str = "google_robot",
    html_stem: str = "render_google_robot",
) -> Path:
    html_path = dated_results_html_path(html_subdir, html_stem)
    out_dir = html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for cfg, gif_path, tile_path in items:
        cue = cfg.get("cue", "?")
        idx = cfg.get("idx", -1)
        rel_gif = os.path.relpath(gif_path, out_dir) if gif_path.exists() else ""
        gif_html = f'<img class="gif" src="{rel_gif}" loading="lazy">' if rel_gif else "<div>No GIF</div>"
        tile_html = ""
        if tile_path is not None and tile_path.is_file():
            rel_tile = os.path.relpath(tile_path, out_dir)
            tile_html = (
                f'<h4>Pose bank top-{POSE_BANK_TOP_K} (dir+orient → xyz distance)</h4>'
                f'<img class="tiles" src="{rel_tile}" loading="lazy">'
            )
        cards.append(
            f"<article><h3>c{idx} {cue}</h3>{gif_html}{tile_html}"
            f"<pre>{json.dumps(cfg, indent=2, ensure_ascii=False)}</pre></article>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>google_robot render</title>"
        "<style>body{font-family:sans-serif;padding:20px}article{border:1px solid #ddd;margin:12px 0;padding:12px}"
        "img.gif{max-width:min(512px,100%);height:auto;display:block;margin-bottom:12px}"
        "img.tiles{max-width:100%;height:auto;display:block;margin:8px 0 12px;border:1px solid #e0e0e8}"
        "h4{margin:16px 0 6px;font-size:14px;color:#333}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:8px}</style></head><body>"
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
    html_only: bool = False,
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
    topk_dir = DEFAULT_TOPK_TILE_DIR
    topk_dir.mkdir(parents=True, exist_ok=True)

    def _paths_for_row(row: dict) -> tuple[Path, Path]:
        idx = int(row.get("idx", -1))
        cue = str(row.get("cue", "?"))
        safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return (
            gif_dir / f"mm19_g{idx:02d}_{safe}.gif",
            topk_dir / f"mm19_g{idx:02d}_{safe}_top{POSE_BANK_TOP_K}.png",
        )

    env = _make_env()
    try:
        if not html_only:
            for row in tqdm(configs, desc="render[google_robot]"):
                idx = int(row.get("idx", -1))
                cue = row.get("cue", "?")
                if want_cues is not None and cue not in want_cues:
                    continue
                gif_path, _ = _paths_for_row(row)
                try:
                    frames = render_config(
                        row,
                        env=env,
                        overlay_progress_bar=overlay_progress_bar,
                        progress_bar_style=progress_bar_style,
                    )
                    if frames:
                        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=50, loop=0)
                except Exception as e:
                    tqdm.write(f"skip g{idx}: {e}")

        if do_html or html_only:
            for row in tqdm(configs, desc="pose_topk[google_robot]"):
                _, tile_path = _paths_for_row(row)
                pose = first_pose_from_row(row)
                if pose:
                    try:
                        render_topk_tile(env, pose, tile_path, top_k=POSE_BANK_TOP_K)
                    except Exception as e:
                        tqdm.write(f"skip topk g{row.get('idx')}: {e}")
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    if do_html or html_only:
        rendered: list[tuple[dict, Path, Path | None]] = []
        for row in configs:
            gif_path, tile_path = _paths_for_row(row)
            rendered.append(
                (
                    row,
                    gif_path if gif_path.is_file() else Path(""),
                    tile_path if tile_path.is_file() else None,
                )
            )
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
