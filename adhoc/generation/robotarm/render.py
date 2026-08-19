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
from adhoc.utils.repo_paths import dev_robosuite_root  # noqa: E402
from legacy.motion_generation_core import generate  # noqa: E402


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


def _pose_index_from_row(row: dict) -> int | None:
    """Use GT tile-pick pose_id when present (gt_fixed_first_pose or first pose step)."""
    gfp = row.get("gt_fixed_first_pose")
    if isinstance(gfp, dict) and gfp.get("pose_id") is not None:
        return int(gfp["pose_id"])
    for step in row.get("movements") or []:
        if step.get("type") != "pose":
            continue
        pose = (step.get("parameters") or {}).get("pose")
        if isinstance(pose, dict) and pose.get("pose_id") is not None:
            return int(pose["pose_id"])
        break
    return None


def _robot_arm_tiled_gif(out_root: Path, sim_robot: str, idx: int) -> Path | None:
    d = out_root / sim_robot
    if not d.is_dir():
        return None
    cands = sorted(d.glob(f"*_c{idx}_tiled.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


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
    html_subdir: str = "manipulator",
    html_stem: str = "render_manipulator",
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
        "<html><head><meta charset='utf-8'><title>robotarm render</title>"
        "<style>body{font-family:sans-serif;padding:20px}article{border:1px solid #ddd;margin:12px 0;padding:12px}"
        "img{max-width:100%;height:auto}pre{white-space:pre-wrap;background:#f7f7f7;padding:8px}</style></head><body>"
        f"<h1>robotarm render ({config_path.name})</h1>{''.join(cards)}</body></html>"
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def run(
    config_json: str | None = None,
    output_dir: str | None = None,
    sim_robot: str = "IIWA",
    jsonl_path: str | None = None,
    hz: int = 4,
    top_k: int = 1,
    cue_indices: list[int] | None = None,
    auto_generate_if_missing: bool = True,
    do_html: bool = False,
    sources_yml: str | None = None,
    skip_existing: bool = False,
    palette_yml: str | None = None,
) -> None:
    src_opt = Path(sources_yml) if sources_yml else None
    _, _, yaml_cfg, blk = resolve_embodiment_paths("manipulator", src_opt)
    cfg = Path(config_json) if config_json else yaml_cfg
    if auto_generate_if_missing and not cfg.exists():
        from motion_generation import run as motion_run

        motion_run(config_json=str(cfg), run_render=False, sources_yml=sources_yml)
    root = dev_robosuite_root()
    out = Path(output_dir) if output_dir else repo_rel_to_path(blk["render_dir"], root)
    out.mkdir(parents=True, exist_ok=True)
    jpath = jsonl_path or str(dev_robosuite_root() / "data" / "seed" / "_remainder" / "closest_poses_results.jsonl")

    with open(cfg, "r", encoding="utf-8") as f:
        rows = json.load(f)
    palette: set[str] | None = None
    if palette_yml:
        cue_list = _palette_cues_ordered(Path(palette_yml))
        palette = set(cue_list)
    ci = cue_indices
    if ci is not None and isinstance(ci, int):
        ci = [ci]
    todo = rows if ci is None else [r for r in rows if int(r.get("idx", -1)) in set(ci)]
    if palette is not None:
        todo = [r for r in todo if isinstance(r.get("cue"), str) and r["cue"] in palette]
    if top_k is None or not isinstance(top_k, int):
        top_k = 1

    announce_output(out, len(todo), kind="gif")

    rendered: list[tuple[dict, Path]] = []
    robot_out = out / sim_robot
    robot_out.mkdir(parents=True, exist_ok=True)
    for row in tqdm(todo, desc="render[robotarm]"):
        idx = row.get("idx")
        cue = row.get("cue")
        if idx is None or not cue:
            continue
        if skip_existing:
            hit = _robot_arm_tiled_gif(out, sim_robot, int(idx))
            if hit is not None:
                rendered.append((row, hit))
                continue
        before = set(robot_out.glob("*.gif"))
        try:
            pose_index = _pose_index_from_row(row)
            generate(
                robot=sim_robot,
                cue=str(cue),
                cue_idx=int(idx),
                pose_index=pose_index,
                jsonl_path=jpath,
                config_path=str(cfg),
                output_dir=str(out),
                hz=hz,
                top_k=top_k,
            )
            after = set(robot_out.glob("*.gif"))
            new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
            rendered.append((row, new_files[0] if new_files else Path("")))
        except Exception as e:
            tqdm.write(f"skip c{idx} {cue}: {e}")

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
    fire.Fire({"run": run})
