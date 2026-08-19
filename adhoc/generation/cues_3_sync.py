#!/usr/bin/env python3
"""Sync ``cues_3.yml`` across embodiments: LLM-fill gaps, trim merged JSON to the palette, prune duplicate GIFs, render one pass.

Typical full run (repo root, ``PYTHONPATH=``.):

  .venv/bin/python adhoc/generation/cues_3_sync.py all --do-html

Requires ``GOOGLE_API_KEY`` for ``generate``. Rendering uses each robot's usual environment (mjlab subprocess for quadruped, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.embodiment_sources import (  # noqa: E402
    repo_rel_to_path,
    resolve_embodiment_paths,
    see_html_sources_path,
)
from adhoc.utils.repo_paths import dev_robosuite_root, seed_yml_dir  # noqa: E402

_SLUG_SUBDIR = {"manipulator": "robotarm", "tiago": "tiago", "quadruped": "quadruped"}

_SLUG_ORDER = ["manipulator", "tiago", "quadruped"]


def _palette_cues(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    ordered: list[str] = []
    seen: set[str] = set()
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML mapping root: {path}")
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


def _merged_cues(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {str(r["cue"]) for r in rows if isinstance(r, dict) and isinstance(r.get("cue"), str)}


def _quad_group_map(palette: list[str]) -> dict[str, tuple[str, int]]:
    cues_new = seed_yml_dir() / "cues_new.yml"
    with open(cues_new, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    m: dict[str, tuple[str, int]] = {}
    icon = data.get("iconic") or {}
    ctx = data.get("contextual") or {}
    if isinstance(icon, dict):
        for i, cue in enumerate(icon.keys(), start=1):
            m[str(cue)] = ("iconic", i)
    if isinstance(ctx, dict):
        for i, cue in enumerate(ctx.keys(), start=1):
            m[str(cue)] = ("contextual", i)
    next_ctx = max((gi for grp, gi in m.values() if grp == "contextual"), default=0) + 1
    for cue in palette:
        if cue not in m:
            m[cue] = ("contextual", next_ctx)
            next_ctx += 1
    return m


def trim_merged_json(slug: str, merged: Path, palette: list[str]) -> None:
    with open(merged, encoding="utf-8") as f:
        rows = json.load(f)
    by_cue: dict[str, dict] = {}
    for r in rows:
        c = r.get("cue") if isinstance(r, dict) else None
        if isinstance(c, str) and c not in by_cue:
            by_cue[c] = r
    quad_map = _quad_group_map(palette) if slug == "quadruped" else None
    out: list[dict] = []
    for i, cue in enumerate(palette):
        row = by_cue.get(cue)
        if row is None:
            raise SystemExit(f"[trim] slug={slug}: missing config for cue {cue!r} in {merged} (run generate first).")
        new_r = deepcopy(row)
        new_r["idx"] = i
        if slug == "quadruped" and quad_map is not None:
            gname, gidx = quad_map[cue]
            new_r["group"] = gname
            new_r["group_idx"] = int(gidx)
        out.append(new_r)
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[trim] slug={slug} wrote {len(out)} rows -> {merged.relative_to(_REPO)}")


_TS_PREFIX = re.compile(r"^\d{8}_\d{6}_")


def _strip_ts_key(name: str) -> str:
    return _TS_PREFIX.sub("", name)


def _prune_duplicate_keys(directory: Path, glob_pat: str = "*.gif") -> int:
    if not directory.is_dir():
        return 0
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in directory.glob(glob_pat):
        if p.is_file():
            groups[_strip_ts_key(p.name)].append(p)
    removed = 0
    for paths in groups.values():
        if len(paths) <= 1:
            continue
        paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for dead in paths[1:]:
            dead.unlink(missing_ok=True)
            removed += 1
    return removed


_MANIP_TILED = re.compile(r"^\d{8}_\d{6}_(.+)_c\d+_tiled\.gif$")


def _prune_manipulator_tiled_by_cue_stem(sim_dir: Path) -> int:
    if not sim_dir.is_dir():
        return 0
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sim_dir.glob("*_tiled.gif"):
        m = _MANIP_TILED.match(p.name)
        if not m:
            continue
        groups[m.group(1)].append(p)
    removed = 0
    for paths in groups.values():
        if len(paths) <= 1:
            continue
        paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for dead in paths[1:]:
            dead.unlink(missing_ok=True)
            removed += 1
    return removed


def prune_renders(slug: str, render_root: Path, sim_robot: str = "IIWA") -> None:
    n = 0
    if slug == "manipulator":
        sim = render_root / sim_robot
        n += _prune_manipulator_tiled_by_cue_stem(sim)
        n += _prune_duplicate_keys(sim)
    else:
        n += _prune_duplicate_keys(render_root)
    print(f"[prune] slug={slug} removed={n} under {render_root.relative_to(_REPO)}")


def parse_slugs(raw: list[str] | None) -> list[str]:
    if not raw:
        raw = ["all"]
    if "all" in raw:
        return list(_SLUG_ORDER)
    aliases = {
        "arm": "manipulator",
        "robotarm": "manipulator",
        "iiwa": "manipulator",
        "google": "tiago",
        "mobile": "tiago",
        "bimanual": "tiago",
        "quad": "quadruped",
    }
    out: list[str] = []
    for s in raw:
        s = s.strip().lower().replace("-", "_")
        s = aliases.get(s, s)
        if s not in _SLUG_ORDER:
            raise SystemExit(f"Unknown slug {s!r}; use manipulator, tiago, quadruped, or all.")
        if s not in out:
            out.append(s)
    return out


def cmd_generate(args: argparse.Namespace, palette: list[str]) -> None:
    if not args.dry_run and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY for generate.")
    src = Path(args.sources_yml).resolve() if args.sources_yml else None
    yaml_catalog = str((seed_yml_dir() / "cues_new.yml").resolve())
    py = sys.executable
    worker = _REPO / "adhoc" / "generation" / "cues_3_gen_worker.py"
    for slug in args.slugs:
        _, _, merged, _ = resolve_embodiment_paths(slug, src)
        have = _merged_cues(merged)
        missing = [c for c in palette if c not in have]
        print(f"[generate] {slug} merged={merged.relative_to(_REPO)} missing={len(missing)}")
        if not missing:
            continue
        if args.dry_run:
            for c in missing:
                print(f"  {c}")
            continue
        prompt, shots, _, _ = resolve_embodiment_paths(slug, src)
        manifest = {
            "cues": missing,
            "config_json": str(merged.resolve()),
            "prompt_file": str(prompt.resolve()),
            "shots_json": str(shots.resolve()),
            "yaml_path": "" if slug == "manipulator" else yaml_catalog,
            "delay": float(args.delay),
            "max_retries": int(args.max_retries),
            "model": str(args.model),
            "require_reasoning": bool(args.require_reasoning),
        }
        import tempfile

        fd, mf = tempfile.mkstemp(suffix=".json", text=True)
        os.close(fd)
        try:
            Path(mf).write_text(json.dumps(manifest), encoding="utf-8")
            r = subprocess.run(
                [py, str(worker), slug, "--manifest", mf],
                cwd=str(_REPO),
                env=os.environ.copy(),
            )
            if r.returncode != 0:
                raise SystemExit(f"generate worker failed for {slug} rc={r.returncode}")
        finally:
            Path(mf).unlink(missing_ok=True)


def cmd_trim(args: argparse.Namespace, palette: list[str]) -> None:
    src = Path(args.sources_yml).resolve() if args.sources_yml else None
    for slug in args.slugs:
        _, _, merged, _ = resolve_embodiment_paths(slug, src)
        trim_merged_json(slug, merged, palette)


def cmd_prune(args: argparse.Namespace, _palette: list[str]) -> None:
    root = dev_robosuite_root()
    src = Path(args.sources_yml).resolve() if args.sources_yml else None
    for slug in args.slugs:
        _, _, _, blk = resolve_embodiment_paths(slug, src)
        render_root = repo_rel_to_path(blk["render_dir"], root)
        prune_renders(slug, render_root, sim_robot=str(args.sim_robot))


def cmd_render(args: argparse.Namespace, palette_path: Path) -> None:
    py = sys.executable
    pal = str(palette_path.resolve())
    src_opt = Path(args.sources_yml).resolve() if args.sources_yml else None
    for slug in args.slugs:
        sub = _SLUG_SUBDIR[slug]
        script = _REPO / "adhoc" / "generation" / sub / "render.py"
        cmd: list[str] = [py, str(script), f"--palette_yml={pal}", f"--do_html={args.do_html}"]
        if src_opt is not None:
            cmd.append(f"--sources_yml={str(src_opt)}")
        if slug == "manipulator":
            cmd.append(f"--skip_existing={args.skip_existing}")
            cmd.append(f"--sim_robot={args.sim_robot}")
        elif slug == "quadruped":
            cmd.append(f"--skip_existing={args.skip_existing}")
        print("[render]", " ".join(cmd))
        subprocess.run(cmd, cwd=str(_REPO), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="cues_3 palette: generate, trim, prune, render.")
    ap.add_argument("command", choices=["generate", "trim", "prune", "render", "all"])
    ap.add_argument("--slugs", nargs="*", default=None, metavar="SLUG", help="manipulator tiago quadruped or omit for all.")
    ap.add_argument("--palette-yml", type=str, default=str(seed_yml_dir() / "cues_3.yml"))
    ap.add_argument("--sources-yml", type=str, default=str(see_html_sources_path()))
    ap.add_argument("--sim-robot", type=str, default="IIWA")
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--delay", type=float, default=2.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--require-reasoning", action="store_true")
    ap.add_argument("--do-html", action="store_true", help="With render/all: write dated render HTML.")
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="With render/all: skip manipulator/quadruped items that already have an on-disk GIF.",
    )
    args = ap.parse_args()
    args.slugs = parse_slugs(args.slugs)
    pale_path = Path(args.palette_yml).resolve()
    palette = _palette_cues(pale_path)
    print(f"[cues_3_sync] palette={pale_path.relative_to(_REPO)} n={len(palette)} slugs={args.slugs}")

    if args.command == "generate":
        cmd_generate(args, palette)
    elif args.command == "trim":
        cmd_trim(args, palette)
    elif args.command == "prune":
        cmd_prune(args, palette)
    elif args.command == "render":
        cmd_render(args, pale_path)
    elif args.command == "all":
        cmd_generate(args, palette)
        cmd_trim(args, palette)
        cmd_prune(args, palette)
        cmd_render(args, pale_path)
    else:
        raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
