#!/usr/bin/env python3
"""One-shot: rename platform identity tiago → tiago (code, seed, results).

Does NOT invent the new Everyday-Robots Tiago platform — that is added after.
Safe to re-run: skips missing paths; already-renamed tokens are left alone.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]

# Directory moves (src relative to repo → dst). Longer / more specific first.
DIR_MOVES: list[tuple[str, str]] = [
    "adhoc/generation/tiago|adhoc/generation/tiago",
    "data/seed/tiago|data/seed/tiago",
    "data/seed/prompt/tiago|data/seed/prompt/tiago",
    "data/seed/shots/tiago|data/seed/shots/tiago",
    "data/seed/yml/context_variation_tiago.yml|data/seed/yml/context_variation_tiago.yml",
    "data/seed/groundtruth/gt_tiago.json|data/seed/groundtruth/gt_tiago.json",
    "data/seed/groundtruth/tiago_pose_gt_abbrev.txt|data/seed/groundtruth/tiago_pose_gt_abbrev.txt",
]

# Filename renames under adhoc (scripts) — applied after package dir move.
FILE_MOVES: list[tuple[str, str]] = [
    "adhoc/generation/tiago/build_gt_tiago.py|adhoc/generation/tiago/build_gt_tiago.py",
    "adhoc/generation/tiago/tiago_experiment_suite.py|adhoc/generation/tiago/tiago_experiment_suite.py",
    "scripts/sbatch_tiago_exp0_qwen7b.sh|scripts/sbatch_tiago_exp0_qwen7b.sh",
    "scripts/sbatch_tiago_exp0_qwen32b.sh|scripts/sbatch_tiago_exp0_qwen32b.sh",
    "scripts/rerender_tiago_shots.sh|scripts/rerender_tiago_shots.sh",
]

SKIP_DIR_NAMES = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".html",
    ".txt",
    ".sh",
    ".csv",
    ".tsv",
    ".toml",
    ".cfg",
    ".ini",
    ".schema",
}


def _should_scan(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.suffix != "":
        # allow extensionless scripts rarely; skip binaries
        if path.suffix:
            return False
    # skip huge binary-ish
    if path.suffix.lower() in {".gif", ".mp4", ".png", ".jpg", ".jpeg", ".npz", ".zip"}:
        return False
    return True


def _rewrite_text(text: str) -> str:
    # Order matters: longer / more specific tokens first.
    reps = [
        ("_import_tiago_render", "_import_tiago_render"),
        ("tiago_render_mobile_config", "tiago_render_mobile_config"),
        ("tiago_pose_bank_topk_tiles", "tiago_pose_bank_topk_tiles"),
        ("tiago_experiment_suite", "tiago_experiment_suite"),
        ("build_gt_tiago", "build_gt_tiago"),
        ("gt_tiago", "gt_tiago"),
        ("see_tiago", "see_tiago"),
        ("context_variation_tiago", "context_variation_tiago"),
        ("ROBOSUITE_TIAGO_RENDER_WARMED", "ROBOSUITE_TIAGO_RENDER_WARMED"),
        ("pose_to_tiago", "pose_to_tiago"),
        ("convert_step_tiago", "convert_step_tiago"),
        ("def render_tiago", "def render_tiago"),
        ("render_tiago(", "render_tiago("),
        ("Tiago", "Tiago"),
        ("Tiago", "Tiago"),
        ("TIAGO", "TIAGO"),
        ("tiago", "tiago"),
        ("tiago", "tiago"),
    ]
    out = text
    for a, b in reps:
        out = out.replace(a, b)
    out = out.replace("tiago", "tiago")
    out = out.replace("Tiago", "Tiago")
    return out


def _move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"[skip-missing] {src}")
        return
    if dst.exists():
        print(f"[skip-exists] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[mv] {src.relative_to(_REPO)} -> {dst.relative_to(_REPO)}")
    shutil.move(str(src), str(dst))


def _rename_result_google_dirs() -> None:
    """Rename any .../tiago directory under data/results to tiago."""
    root = _REPO / "data" / "results"
    if not root.is_dir():
        return
    # Collect deepest-first so nested renames work
    dirs = sorted(
        [p for p in root.rglob("tiago") if p.is_dir()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for d in dirs:
        target = d.with_name("tiago")
        if target.exists():
            print(f"[merge-skip] {d} (target exists)")
            continue
        print(f"[mv-dir] {d.relative_to(_REPO)} -> tiago/")
        d.rename(target)
    # Files named *tiago*
    files = sorted(
        [p for p in root.rglob("*tiago*") if p.is_file()],
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for f in files:
        target = f.with_name(f.name.replace("tiago", "tiago"))
        if target.exists():
            print(f"[skip-file] {target}")
            continue
        print(f"[mv-file] {f.relative_to(_REPO)}")
        f.rename(target)


def _rewrite_tree(root: Path) -> int:
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        for name in filenames:
            path = Path(dirpath) / name
            if not _should_scan(path):
                continue
            # skip this migrator itself after first write? allow rewrite
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if b"\0" in raw[:2048]:
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = raw.decode("latin-1")
                except Exception:
                    continue
            if "tiago" not in text and "Tiago" not in text and "Tiago" not in text and "pose_to_tiago" not in text and "convert_step_tiago" not in text and "render_tiago(" not in text and "_import_tiago" not in text and "TIAGO" not in text:
                continue
            new = _rewrite_text(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                n += 1
                print(f"[edit] {path.relative_to(_REPO)}")
    return n


def main() -> None:
    # 1) Known path moves
    for spec in DIR_MOVES + FILE_MOVES:
        a, b = spec.split("|", 1)
        _move(_REPO / a, _REPO / b)

    # 2) Results tiago dirs/files
    _rename_result_google_dirs()

    # 3) Text rewrite whole repo (excl. .git)
    n = _rewrite_tree(_REPO)
    print(f"\nDone. Rewrote {n} files.")
    # Leave adhoc/generation/tiago free for the new Everyday-Robots platform.


if __name__ == "__main__":
    main()
