#!/usr/bin/env python3
"""Build data/seed/groundtruth/gt_tiago.json from mobile pose abbrev + movement GT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parse_pose_gt_mobile import parse_pose_gt_line, pose_gt_display  # noqa: E402
from pilot40_paths import GT_PATH, MANIFEST_TSV  # noqa: E402

POSE_ABBREV = _REPO / "data/seed/groundtruth/tiago_pose_gt_abbrev.txt"
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


def _manifest() -> list[tuple[int, str]]:
    """Manifest TSV file order (same as manifest90_cue_names)."""
    manifest: list[tuple[int, str]] = []
    for line in MANIFEST_TSV.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 4 or parts[1] == "pending_essence10":
            continue
        idx_raw = parts[2].strip()
        cue = parts[3].strip()
        if idx_raw.isdigit():
            cue_idx = int(idx_raw)
        else:
            cue_idx = 0
        manifest.append((cue_idx, cue))
    return manifest


def _pose_lines() -> list[str]:
    lines = [
        ln
        for ln in POSE_ABBREV.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return lines


def build() -> dict:
    motion_by = {
        str(a["cue"]): a
        for a in json.loads(MOTION_GT.read_text(encoding="utf-8")).get("annotations", [])
        if a.get("cue")
    }
    desc = _descriptions()
    manifest = _manifest()
    pose_lines = _pose_lines()
    if len(pose_lines) != len(manifest):
        raise ValueError(
            f"pose abbrev lines ({len(pose_lines)}) != manifest cues ({len(manifest)}); "
            f"check {POSE_ABBREV}"
        )

    rows = []
    for (cue_idx, cue), pose_line in zip(manifest, pose_lines):
        pose_any, pose_options, pose_raw = parse_pose_gt_line(pose_line)
        pose_gt = pose_gt_display(pose_any, pose_options, pose_raw)
        me = motion_by.get(cue, {})
        rows.append(
            {
                "cue_idx": cue_idx,
                "cue": cue,
                "description": desc.get(cue) or "",
                "pose_gt": pose_gt,
                "pose_any": pose_any,
                "pose_options": pose_options,
                "groundtruth": pose_gt,
                "movement_gt": {
                    "annotation_raw": me.get("annotation_raw", ""),
                    "component": me.get("component"),
                    "always_correct": me.get("always_correct", False),
                },
            }
        )

    return {
        "robot": "tiago",
        "n_cues": len(rows),
        "sources": {
            "pose": str(POSE_ABBREV.relative_to(_REPO)),
            "movement": str(MOTION_GT.relative_to(_REPO)),
            "manifest": str(MANIFEST_TSV.relative_to(_REPO)),
        },
        "rows": rows,
    }


def main() -> None:
    payload = build()
    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_any = sum(1 for r in payload["rows"] if r.get("pose_any"))
    print(f"Wrote {len(payload['rows'])} cues → {GT_PATH} (pose_any={n_any})")


if __name__ == "__main__":
    main()
