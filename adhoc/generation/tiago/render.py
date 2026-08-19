#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import fire
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[misc, assignment]

_REPO = Path(__file__).resolve().parents[3]
_GOOGLE = Path(__file__).resolve().parent
_ROBOTARM = _REPO / "adhoc/generation/robotarm"
for p in (_REPO, _GOOGLE, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from render_paths import render_gif_basename  # noqa: E402

from adhoc.generation.embodiment_sources import repo_rel_to_path, resolve_embodiment_paths  # noqa: E402
from adhoc.generation.tiago.legacy import _make_env, render_config  # noqa: E402
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from pose_bank_topk_tiles import first_pose_from_row, render_plausible_pool_tile  # noqa: E402

DEFAULT_PLAUSIBLE_POOL_DIR = _REPO / "data/results/visualize/tiago/pose_plausible_pool"


def _save_gif_frames(frames: list, gif_path: Path, *, duration_ms: int = 50) -> None:
    gen = _REPO / "adhoc/generation"
    if str(gen) not in sys.path:
        sys.path.insert(0, str(gen))
    from gif_encode import save_gif_frames

    save_gif_frames(frames, gif_path, duration_ms=duration_ms)


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
    items: list[tuple[dict, Path, Path | None, dict[str, Any] | None]],
    *,
    html_subdir: str = "tiago",
    html_stem: str = "render_tiago",
) -> Path:
    html_path = dated_results_html_path(html_subdir, html_stem)
    out_dir = html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for cfg, gif_path, tile_path, tile_meta in items:
        cue = cfg.get("cue", "?")
        idx = cfg.get("idx", -1)
        rel_gif = os.path.relpath(gif_path.resolve(), out_dir) if gif_path.exists() else ""
        gif_html = f'<img class="gif" src="{rel_gif}" loading="lazy">' if rel_gif else "<div>No GIF</div>"
        tile_html = ""
        if tile_path is not None and tile_path.is_file():
            rel_tile = os.path.relpath(tile_path, out_dir)
            meta = tile_meta or {}
            n_show = len(meta.get("entries") or [])
            n_pool = int(meta.get("n_pool") or n_show)
            src = meta.get("source") or "pool"
            tile_html = (
                f'<h4>Plausible pose pool ({n_show}/{n_pool} — '
                f'dir + orient + elbow_bended, {src})</h4>'
                f'<img class="tiles" src="{rel_tile}" loading="lazy">'
            )
        cards.append(
            f"<article><h3>c{idx} {cue}</h3>{gif_html}{tile_html}"
            f"<pre>{json.dumps(cfg, indent=2, ensure_ascii=False)}</pre></article>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>tiago render</title>"
        "<style>body{font-family:sans-serif;padding:20px}article{border:1px solid #ddd;margin:12px 0;padding:12px}"
        "img.gif{max-width:min(512px,100%);height:auto;display:block;margin-bottom:12px}"
        "img.tiles{max-width:100%;height:auto;display:block;margin:8px 0 12px;border:1px solid #e0e0e8}"
        "h4{margin:16px 0 6px;font-size:14px;color:#333}"
        "pre{white-space:pre-wrap;background:#f7f7f7;padding:8px}</style></head><body>"
        f"<h1>tiago render ({config_path.name})</h1>{''.join(cards)}</body></html>"
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
    ignore_xyz: bool = False,
    topk: int | None = None,
    pose_tiles_only: bool = False,
) -> None:
    src_opt = Path(sources_yml) if sources_yml else None
    _, _, yaml_cfg, blk = resolve_embodiment_paths("tiago", src_opt)
    cfg = Path(config_json) if config_json else yaml_cfg
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False, sources_yml=sources_yml)
    gif_dir = Path(output_dir) if output_dir else repo_rel_to_path(blk["render_dir"])
    gif_dir = gif_dir.resolve()
    gif_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if palette_yml:
        pal = set(_palette_cues_ordered(Path(palette_yml)))
        configs = [r for r in configs if isinstance(r.get("cue"), str) and r["cue"] in pal]
    announce_output(gif_dir, len(configs), kind="gif")

    if cues:
        if isinstance(cues, (list, tuple)):
            want_cues = {str(c).strip() for c in cues if str(c).strip()}
        else:
            want_cues = {c.strip() for c in str(cues).split(",") if c.strip()}
    else:
        want_cues = None
    pool_dir = DEFAULT_PLAUSIBLE_POOL_DIR
    if ignore_xyz:
        pool_dir = gif_dir / "pose_tiles"
    pool_dir.mkdir(parents=True, exist_ok=True)
    tile_meta_by_cue: dict[str, dict[str, Any]] = {}

    def _paths_for_row(row: dict) -> tuple[Path, Path]:
        idx = int(row.get("idx", -1))
        cue = str(row.get("cue", "?"))
        tile_suffix = "_pose_pool.gif" if ignore_xyz else "_plausible_pool.png"
        gif_name = render_gif_basename(idx, cue, "gen")
        return (
            gif_dir / gif_name,
            pool_dir / f"{Path(gif_name).stem}{tile_suffix}",
        )

    env = _make_env()
    try:
        if not html_only and not pose_tiles_only:
            for row in tqdm(configs, desc="render[tiago]"):
                idx = int(row.get("idx", -1))
                cue = row.get("cue", "?")
                if want_cues is not None and cue not in want_cues:
                    continue
                gif_path, tile_path = _paths_for_row(row)
                try:
                    frames = render_config(
                        row,
                        env=env,
                        overlay_progress_bar=overlay_progress_bar,
                        progress_bar_style=progress_bar_style,
                        ignore_xyz=ignore_xyz,
                    )
                    if frames:
                        _save_gif_frames(frames, gif_path)
                    if ignore_xyz:
                        pose = first_pose_from_row(row)
                        if pose:
                            tile_env = _make_env()
                            try:
                                meta = render_plausible_pool_tile(
                                    tile_env,
                                    pose,
                                    tile_path,
                                    ignore_xyz=True,
                                    topk=topk,
                                    row=row,
                                    overlay_progress_bar=False,
                                    main_gif_path=gif_path,
                                )
                                tile_meta_by_cue[str(cue)] = meta
                            finally:
                                closer = getattr(tile_env, "close", None)
                                if callable(closer):
                                    closer()
                except Exception as e:
                    tqdm.write(f"skip g{idx}: {e}")

        if pose_tiles_only and ignore_xyz:
            for row in tqdm(configs, desc="pose_tiles[tiago]"):
                cue = str(row.get("cue", "?"))
                if want_cues is not None and cue not in want_cues:
                    continue
                gif_path, tile_path = _paths_for_row(row)
                pose = first_pose_from_row(row)
                if pose:
                    tile_env = _make_env()
                    try:
                        meta = render_plausible_pool_tile(
                            tile_env,
                            pose,
                            tile_path,
                            ignore_xyz=True,
                            topk=topk,
                            row=row,
                            overlay_progress_bar=False,
                            main_gif_path=gif_path,
                        )
                        tile_meta_by_cue[cue] = meta
                    except Exception as e:
                        tqdm.write(f"skip pool g{row.get('idx')}: {e}")
                    finally:
                        closer = getattr(tile_env, "close", None)
                        if callable(closer):
                            closer()

        if do_html or html_only:
            for row in tqdm(configs, desc="pose_pool[tiago]"):
                cue = str(row.get("cue", "?"))
                gif_path, tile_path = _paths_for_row(row)
                pose = first_pose_from_row(row)
                if pose:
                    try:
                        meta = render_plausible_pool_tile(
                            env,
                            pose,
                            tile_path,
                            ignore_xyz=ignore_xyz,
                            topk=topk,
                            row=row if ignore_xyz else None,
                            overlay_progress_bar=False if ignore_xyz else overlay_progress_bar,
                            main_gif_path=gif_path,
                        )
                        tile_meta_by_cue[cue] = meta
                    except Exception as e:
                        tqdm.write(f"skip pool g{row.get('idx')}: {e}")
    finally:
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()

    if do_html or html_only:
        rendered: list[tuple[dict, Path, Path | None, dict[str, Any] | None]] = []
        for row in configs:
            cue = str(row.get("cue", "?"))
            gif_path, tile_path = _paths_for_row(row)
            rendered.append(
                (
                    row,
                    gif_path if gif_path.is_file() else Path(""),
                    tile_path if tile_path.is_file() else None,
                    tile_meta_by_cue.get(cue),
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
    fire.Fire({"run": run})
