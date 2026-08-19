#!/usr/bin/env python3
"""Build one motion list from ``prompt19_locomotion_configs`` aligned to ``cues_new.yml``.

Each file in ``prompt19_locomotion_configs`` is a single-cue ``{ "cue", "movements" }`` snapshot
(``lc19_gNN_*`` and ``lc19_cNN_*`` prefixes are legacy batch indices; this merge re-orders
by YAML and assigns **group_idx** 1..N within iconic / contextual).

Output schema per row::

  idx          — global order 1..114 (iconic 1..59, contextual 60..114)
  group        — "iconic" | "contextual"
  group_idx    — 1..59 or 1..55 (used in render filenames: ``lc19_g%02d`` / ``lc19_c%02d``)
  cue, movements, … — from the chosen per-cue file (optional metadata may be present)
  source_file  — basename of the per-cue JSON that was used
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fire
import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.utils.repo_paths import motion_configs_results_dir, seed_yml_dir  # noqa: E402


def _load_per_cue_index(per_cue_dir: Path) -> dict[str, list[tuple[str, int, str]]]:
    by: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for f in per_cue_dir.glob("lc19_*.json"):
        m = re.match(r"^lc19_(g|c)(\d+)_(.+)\.json$", f.name)
        if not m:
            continue
        by[m.group(3)].append((m.group(1), int(m.group(2)), f.name))
    for k in by:
        by[k].sort(key=lambda x: (x[0], x[1]))
    return by


def _choose_file(
    group: str,
    cue: str,
    by_cue: dict[str, list[tuple[str, int, str]]],
) -> str | None:
    lst = by_cue.get(cue)
    if not lst:
        return None
    if len(lst) == 1:
        return lst[0][2]
    want = "g" if group == "iconic" else "c"
    for t, _n, fn in lst:
        if t == want:
            return fn
    return lst[0][2]


def run(
    yaml_path: str | None = None,
    per_cue_dir: str | None = None,
    out_json: str | None = None,
) -> str:
    ypath = Path(yaml_path) if yaml_path else seed_yml_dir() / "cues_new.yml"
    pdir = Path(per_cue_dir) if per_cue_dir else motion_configs_results_dir("quadruped") / "prompt19_locomotion_configs"
    out = Path(out_json) if out_json else motion_configs_results_dir("quadruped") / "motion_configs_prompt19_cues_new_merged.json"
    with open(ypath, "r", encoding="utf-8") as f:
        catalog = yaml.safe_load(f)
    if not isinstance(catalog, dict) or "iconic" not in catalog or "contextual" not in catalog:
        raise ValueError(f"Expected 'iconic' and 'contextual' in {ypath}")

    by_cue = _load_per_cue_index(pdir)
    merged: list[dict] = []
    gidx = 0
    for group, block in (("iconic", catalog["iconic"]), ("contextual", catalog["contextual"])):
        for local_i, cue in enumerate(block.keys(), start=1):
            gidx += 1
            fn = _choose_file(group, cue, by_cue)
            row: dict = {
                "idx": gidx,
                "group": group,
                "group_idx": local_i,
                "cue": cue,
            }
            if not fn:
                row["state"] = "missing_per_cue_file"
                row["movements"] = []
            else:
                fpath = pdir / fn
                with open(fpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                row["movements"] = raw.get("movements", [])
                row["source_file"] = fn
                for k in ("state", "model", "time", "reasoning"):
                    if k in raw:
                        row[k] = raw[k]
            merged.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(merged)} rows to {out}")
    return str(out)


if __name__ == "__main__":
    fire.Fire(run)
