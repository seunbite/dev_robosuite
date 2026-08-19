#!/usr/bin/env python3
"""Rewrite legacy ``type: path`` steps as ``movement`` rows with ``joint: base``.

Targets aggregate motion JSON lists under ``data/results/motion_configs/`` by default.
Safe to run multiple times: already-migrated configs are unchanged."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.joint_motion_schema import migrate_config_row  # noqa: E402


def _migrate_obj(data: object) -> tuple[object, bool]:
    if isinstance(data, list):
        changed = False
        out: list = []
        for row in data:
            if isinstance(row, dict):
                nrow = migrate_config_row(row)
                old_s = json.dumps(row, sort_keys=True, ensure_ascii=False)
                new_s = json.dumps(nrow, sort_keys=True, ensure_ascii=False)
                changed |= old_s != new_s
                out.append(nrow)
            else:
                out.append(row)
        return out, changed
    if isinstance(data, dict) and isinstance(data.get("movements"), list):
        nrow = migrate_config_row(data)
        old_s = json.dumps(data, sort_keys=True, ensure_ascii=False)
        new_s = json.dumps(nrow, sort_keys=True, ensure_ascii=False)
        return nrow, old_s != new_s
    return data, False


def run(root: str = "data/results/motion_configs", dry_run: bool = False, glob: str = "*.json") -> None:
    base = (_REPO / root).resolve()
    if not base.is_dir():
        print(f"No directory {base}")
        return
    touched = 0
    for p in sorted(base.rglob(glob)):
        if "_per_cue_configs" in p.parts:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {p}: {e}")
            continue
        migrated, changed = _migrate_obj(data)
        if not changed:
            continue
        touched += 1
        print(f"Migrated: {p}")
        if not dry_run:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(migrated, f, indent=2, ensure_ascii=False)
    print(f"Done. Updated {touched} file(s)." + (" (dry run)" if dry_run else ""))


if __name__ == "__main__":
    fire.Fire(run)
