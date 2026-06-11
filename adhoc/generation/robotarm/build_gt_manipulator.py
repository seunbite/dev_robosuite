#!/usr/bin/env python3
"""Build data/seed/groundtruth/gt_manipulator.json from pose + movement GT sources."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pilot90_paths import GT_PATH, MANIFEST_TSV  # noqa: E402

CONSOLIDATED = _REPO / "data/results/verify/pilot40_pose_eval_consolidated.json"
MOTION_GT = _REPO / "data/results/verify/pilot40_motion_component_gt.json"
CUES_MEANING = _REPO / "data/seed/yml/cues_meaning.yml"


def _descriptions() -> dict[str, str]:
    try:
        import yaml
    except ImportError:
        return {}
    if not CUES_MEANING.is_file():
        return {}
    data = yaml.safe_load(CUES_MEANING.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for group in data.values():
        if not isinstance(group, dict):
            continue
        for cue, desc in group.items():
            if isinstance(desc, str):
                out[str(cue)] = desc
    return out


def build() -> dict:
    pose_by = {
        str(r["cue"]): r
        for r in json.loads(CONSOLIDATED.read_text(encoding="utf-8")).get("rows", [])
        if r.get("cue")
    }
    motion_by = {
        str(a["cue"]): a
        for a in json.loads(MOTION_GT.read_text(encoding="utf-8")).get("annotations", [])
        if a.get("cue")
    }
    desc = _descriptions()
    manifest: list[tuple[int, str]] = []
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        cue = parts[3]
        idx_raw = parts[2].strip()
        if idx_raw.isdigit():
            cue_idx = int(idx_raw)
        else:
            cue_idx = int(pose_by.get(cue, {}).get("cue_idx", 0))
        manifest.append((cue_idx, cue))

    rows = []
    for cue_idx, cue in sorted(manifest, key=lambda x: x[0]):
        pe = pose_by.get(cue, {})
        me = motion_by.get(cue, {})
        rows.append(
            {
                "cue_idx": cue_idx,
                "cue": cue,
                "description": desc.get(cue) or pe.get("description") or "",
                "pose_gt": pe.get("groundtruth", ""),
                "groundtruth": pe.get("groundtruth", ""),
                "movement_gt": {
                    "annotation_raw": me.get("annotation_raw", ""),
                    "component": me.get("component"),
                    "always_correct": me.get("always_correct", False),
                },
            }
        )

    return {
        "robot": "manipulator",
        "n_cues": len(rows),
        "sources": {
            "pose": str(CONSOLIDATED.relative_to(_REPO)),
            "movement": str(MOTION_GT.relative_to(_REPO)),
            "manifest": str(MANIFEST_TSV.relative_to(_REPO)),
        },
        "rows": rows,
    }


def main() -> None:
    payload = build()
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload['rows'])} cues → {GT_PATH}")


if __name__ == "__main__":
    main()
