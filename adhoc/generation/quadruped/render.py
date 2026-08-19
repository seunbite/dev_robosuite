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
from adhoc.generation.joint_motion_schema import quadruped_movements_for_legacy_path  # noqa: E402
from adhoc.generation.utils import announce_output, dated_results_html_path  # noqa: E402
from adhoc.generation.render_progress_overlay import overlay_simple_progress_bar_on_frames  # noqa: E402
from adhoc.utils.repo_paths import unitree_rl_mjlab_root  # noqa: E402


def _apply_simple_progress_overlay_to_gif(gpath: Path) -> None:
    """Post-process mjlab GIF: neutral top scrubber (cue JSON not visible inside mjlab)."""
    from PIL import Image

    with Image.open(gpath) as im:
        n = int(getattr(im, "n_frames", 1) or 1)
        duration = int(im.info.get("duration") or 50)
        seq: list[Image.Image] = []
        for i in range(n):
            im.seek(i)
            seq.append(im.copy().convert("RGB"))
    if not seq:
        return
    layered = overlay_simple_progress_bar_on_frames(seq)
    if len(layered) == 1:
        layered[0].save(str(gpath), duration=duration, loop=0, optimize=False)
    else:
        layered[0].save(
            str(gpath),
            save_all=True,
            append_images=layered[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )


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


def _quad_prefix_index(row: dict) -> tuple[str, int]:
    if row.get("group") in ("iconic", "contextual") and "group_idx" in row:
        return ("g" if row["group"] == "iconic" else "c", int(row["group_idx"]))
    return ("g", int(row.get("idx", -1)))


def _write_html(
    config_path: Path,
    items: list[tuple[dict, Path]],
    *,
    html_subdir: str = "quadruped",
    html_stem: str = "render_quadruped",
) -> Path:
    html_path = dated_results_html_path(html_subdir, html_stem)
    out_dir = html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for cfg, gif_path in items:
        cue = cfg.get("cue", "?")
        pfx, num = _quad_prefix_index(cfg)
        idx = int(cfg.get("idx", -1))
        rel_gif = os.path.relpath(gif_path, out_dir) if gif_path.exists() else ""
        gif_html = f'<img src="{rel_gif}" loading="lazy">' if rel_gif else "<div>No GIF</div>"
        cards.append(
            f"<article><h3>{pfx}{num:02d} gidx={idx} {cue}</h3>{gif_html}"
            f"<pre>{json.dumps(cfg, indent=2, ensure_ascii=False)}</pre></article>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>quadruped render</title>"
        "<style>body{font-family:sans-serif;padding:20px}article{border:1px solid #ddd;margin:12px 0;padding:12px}"
        "img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f7f7f7;padding:8px}</style></head><body>"
        f"<h1>quadruped render ({config_path.name})</h1>{''.join(cards)}</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def run(
    config_json: str | None = None,
    output_dir: str | None = None,
    python_bin: str | None = None,
    auto_generate_if_missing: bool = True,
    do_html: bool = False,
    max_items: int | None = None,
    camera: str = "front",
    per_item_timeout_sec: int = 900,
    skip_existing: bool = True,
    sources_yml: str | None = None,
    palette_yml: str | None = None,
    overlay_progress_bar: bool = True,
) -> None:
    """Record GIFs. Needs a Python with ``mjlab``, ``torch``, and ``imageio`` (e.g. unitree env). Set
    env ``MOTION_RENDER_PYTHON`` or pass ``python_bin``; otherwise the current ``sys.executable`` is used.
    """
    src_opt = Path(sources_yml) if sources_yml else None
    _, _, yaml_cfg, blk = resolve_embodiment_paths("quadruped", src_opt)
    cfg = Path(config_json) if config_json else yaml_cfg
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False, sources_yml=sources_yml)
    gif_dir = Path(output_dir) if output_dir else repo_rel_to_path(blk["render_dir"])
    cfg_dir = gif_dir / "_per_cue_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)
    py = python_bin or os.environ.get("MOTION_RENDER_PYTHON") or sys.executable
    mjlab = unitree_rl_mjlab_root() / "scripts" / "record_go2_locomotion_config_mjlab_gif.py"

    with open(cfg, "r", encoding="utf-8") as f:
        configs = json.load(f)
    if palette_yml:
        pal = set(_palette_cues_ordered(Path(palette_yml)))
        configs = [r for r in configs if isinstance(r.get("cue"), str) and r["cue"] in pal]
    if max_items is not None:
        configs = configs[: int(max_items)]
    announce_output(gif_dir, len(configs), kind="gif")
    mj_root = str(unitree_rl_mjlab_root())
    env = os.environ.copy()
    env["PYTHONPATH"] = mj_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    rendered: list[tuple[dict, Path]] = []
    for row in tqdm(configs, desc="render[quadruped]"):
        cue = row.get("cue", "?")
        safe = cue.replace("/", "_").replace("\\", "_").replace(" ", "_")
        pfx, num = _quad_prefix_index(row)
        cpath = cfg_dir / f"lc19_{pfx}{num:02d}_{safe}.json"
        gpath = gif_dir / f"lc19_{pfx}{num:02d}_{safe}_mjlab_earth.gif"
        if skip_existing and gpath.exists():
            rendered.append((row, gpath))
            continue
        with open(cpath, "w", encoding="utf-8") as wf:
            leg_mvs = quadruped_movements_for_legacy_path(list(row.get("movements") or []))
            json.dump({"cue": cue, "movements": leg_mvs}, wf, indent=2, ensure_ascii=False)
        cmd = [
            py,
            str(mjlab),
            "--config",
            str(cpath),
            "--out",
            str(gpath),
            "--earth-gravity",
            "--no-hold-base",
            "--pose-mode",
            "gravity",
            "--pd-scale",
            "2.0",
            "--camera",
            str(camera),
        ]
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(per_item_timeout_sec),
                cwd=str(unitree_rl_mjlab_root()),
                env=env,
            )
        except subprocess.TimeoutExpired:
            tqdm.write(f"skip {pfx}{num:02d}: timeout>{int(per_item_timeout_sec)}s")
            rendered.append((row, Path("")))
            continue
        if r.returncode != 0 or not gpath.exists():
            tqdm.write(f"skip {pfx}{num:02d}: rc={r.returncode} {(r.stderr or '')[-200:]}")
        else:
            if overlay_progress_bar:
                try:
                    _apply_simple_progress_overlay_to_gif(gpath)
                except Exception as ex:
                    tqdm.write(f"warn progress bar {gpath.name}: {ex}")
        rendered.append((row, gpath if gpath.exists() else Path("")))

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
