#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import fire


def _pose(front: str, back: str, direction: str = "neutral") -> dict:
    return {
        "dir": direction,
        "front_height": front,
        "back_height": back,
        "three_legs": False,
    }


def _mk_row(idx: int, cue: str, movements: list[dict]) -> dict:
    return {
        "idx": idx,
        "group": "probe",
        "group_idx": idx,
        "cue": cue,
        "movements": movements,
    }


def _movement_base(duration: float, path: dict) -> dict:
    return {
        "type": "movement",
        "duration": duration,
        "parameters": {"movement": {"joints": [{"joint": "base", "path": dict(path)}]}},
    }


def build(out_json: str | None = None) -> str:
    """
    Build a probe config with many pose_to_pose combinations:
    - static pose
    - pose_to_pose with/without velocity
    - pose_to_pose repetition
    - bow-like and nod-like variants
    """
    out = Path(out_json) if out_json else Path("data/results/motion_configs/quadruped/pose_combo_probes.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    i = 1

    levels = ["low", "mid", "high"]
    # Stronger velocity sweep so gait differs visibly on GIF (was too subtle at 0.03–0.06).
    velocities = [
        ("v0", {"vx": 0.0, "vy": 0.0, "wz": 0.0}),
        ("vx12", {"vx": 0.12, "vy": 0.0, "wz": 0.0}),
        ("vx28", {"vx": 0.28, "vy": 0.0, "wz": 0.0}),
        ("wz09", {"vx": 0.0, "vy": 0.0, "wz": 0.9}),
    ]

    # 1) 9 pure static poses.
    for fh in levels:
        for bh in levels:
            rows.append(
                _mk_row(
                    i,
                    f"static_f{fh}_b{bh}",
                    [{"type": "pose", "duration": 1.0, "parameters": {"pose": _pose(fh, bh)}}],
                )
            )
            i += 1

    # 2) pose_to_pose: mid->low/high with velocity variations.
    for tag, vel in velocities:
        rows.append(
            _mk_row(
                i,
                f"p2p_midmid_to_flow_bhigh_{tag}",
                [
                    {"type": "pose", "duration": 0.3, "parameters": {"pose": _pose("mid", "mid")}},
                    _movement_base(
                        1.6,
                        {
                            "shape": "pose_to_pose",
                            "start_pose": _pose("mid", "mid"),
                            "end_pose": _pose("low", "high"),
                            "velocity": vel,
                            "gait_blend_xy": 0.98,
                            "gait_blend_z": 0.65,
                        },
                    ),
                ],
            )
        )
        i += 1

    # 3) same pose_to_pose but repetition control.
    for rep in (1, 2, 3):
        rows.append(
            _mk_row(
                i,
                f"p2p_repeat_{rep}_midmid_flow_bhigh",
                [
                    {"type": "pose", "duration": 0.25, "parameters": {"pose": _pose("mid", "mid")}},
                    _movement_base(
                        1.8,
                        {
                            "shape": "pose_to_pose",
                            "start_pose": _pose("mid", "mid"),
                            "end_pose": _pose("low", "high"),
                            "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
                            "repetition": rep,
                        },
                    ),
                ],
            )
        )
        i += 1

    # 4) bow-like: neutral->bow->neutral, with/without velocity.
    for tag, vel in (("still", {"vx": 0.0, "vy": 0.0, "wz": 0.0}), ("walk", {"vx": 0.04, "vy": 0.0, "wz": 0.0})):
        rows.append(
            _mk_row(
                i,
                f"bow_cycle_{tag}",
                [
                    {"type": "pose", "duration": 0.25, "parameters": {"pose": _pose("mid", "mid")}},
                    _movement_base(
                        2.2,
                        {
                            "shape": "pose_to_pose",
                            "start_pose": _pose("mid", "mid"),
                            "end_pose": _pose("low", "high"),
                            "velocity": vel,
                            "repetition": 2,
                        },
                    ),
                ],
            )
        )
        i += 1

    # 5) nod-like small oscillation around mid.
    rows.append(
        _mk_row(
            i,
            "nod_like_small_repeat3",
            [
                {"type": "pose", "duration": 0.2, "parameters": {"pose": _pose("mid", "mid")}},
                _movement_base(
                    1.8,
                    {
                        "shape": "pose_to_pose",
                        "start_pose": _pose("mid", "mid"),
                        "end_pose": _pose("low", "mid"),
                        "velocity": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
                        "repetition": 3,
                    },
                ),
            ],
        )
    )

    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} probe rows to {out}")
    return str(out)


if __name__ == "__main__":
    fire.Fire(build)

