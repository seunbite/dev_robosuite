#!/usr/bin/env python3
"""Expand 30 Human-GT cores into Core + Long1–4 across five embodiments."""
from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adhoc.generation.robotarm.motion_code_to_config import convert_code  # noqa: E402

PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "quadruped", "roomba")
LEVELS = ("core", "long1", "long2", "long3", "long4")
GT_PATH = (
    _REPO
    / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
)
OUT_ROOT = _REPO / "data/results/motion_configs/embodiment_long_1_4"
PLAN_PATH = _REPO / "data/results/plans/long_1_4_variation_plan_30cues.json"

ROLE_LABELS = {
    "hold_short": "짧게 멈춰 결과 상태를 강조",
    "hold_long": "길게 멈춰 상태를 명확히 유지",
    "orient": "상대 방향으로 주의를 정렬",
    "scan": "좌우를 크게 살피며 대상을 탐색",
    "approach": "상대 또는 대상 쪽으로 접근",
    "retreat": "뒤로 물러나 거리와 반응을 표현",
    "sway_small": "작게 흔들려 망설임이나 미세 반응을 표현",
    "sway_large": "크고 느리게 흔들어 상태를 강하게 framing",
    "turn_away": "반대 방향으로 돌아 관계를 끊거나 전환",
    "depart": "앞으로 이동해 장면을 떠나는 후속 서사",
    "celebrate": "크고 빠른 상승 동작으로 감정을 강화",
    "settle": "낮고 안정된 자세로 진정·종료",
    "crouch": "낮게 웅크려 준비 상태를 형성",
    "rise": "낮은 상태에서 크게 일어나 에너지를 축적",
    "stretch": "위쪽으로 길게 뻗어 기지개·확장을 표현",
    "circle": "큰 원 또는 반원을 그려 공간 관계를 제시",
    "point_target": "대상 방향을 명시적으로 가리킴",
    "point_self": "자신의 위치 또는 몸쪽을 가리킴",
    "open": "바깥쪽으로 열어 환영·크기·개방을 표현",
    "close": "몸쪽으로 모아 방어·집중·작음을 표현",
    "bow": "아래로 숙여 감사·피로·종료를 표현",
    "initial_pose": "renderer 실행을 위한 중립 initial pose",
}

# Each level independently wraps the unchanged core with PRE and POST intent roles.
RECIPES: dict[int, dict[str, tuple[list[str], list[str]]]] = {
    1: {"long1": ([], ["hold_short"]), "long2": (["approach"], []), "long3": (["scan"], []), "long4": ([], ["turn_away", "depart"])},
    2: {"long1": ([], ["retreat"]), "long2": (["approach", "hold_short"], []), "long3": (["sway_large"], []), "long4": ([], ["turn_away", "depart"])},
    3: {"long1": ([], ["hold_short"]), "long2": (["orient"], []), "long3": (["circle"], []), "long4": ([], ["celebrate", "settle"])},
    4: {"long1": ([], ["retreat"]), "long2": (["approach"], []), "long3": (["scan", "hold_short"], []), "long4": ([], ["turn_away", "depart"])},
    5: {"long1": ([], ["celebrate"]), "long2": (["crouch", "rise"], []), "long3": (["hold_short", "rise"], []), "long4": ([], ["celebrate", "settle"])},
    6: {"long1": ([], ["approach"]), "long2": (["orient"], []), "long3": (["scan"], []), "long4": ([], ["circle", "hold_short"])},
    7: {"long1": ([], ["stretch"]), "long2": (["sway_small"], []), "long3": (["sway_large", "hold_short"], []), "long4": ([], ["turn_away", "depart", "settle"])},
    8: {"long1": ([], ["hold_long"]), "long2": ([], ["sway_small", "retreat"]), "long3": (["sway_large"], []), "long4": ([], ["hold_short", "turn_away", "depart"])},
    9: {"long1": (["orient"], []), "long2": (["point_self"], []), "long3": (["scan"], []), "long4": ([], ["circle", "hold_short"])},
    10: {"long1": (["point_target"], []), "long2": ([], ["point_self"]), "long3": (["approach", "retreat"], []), "long4": ([], ["orient", "approach"])},
    11: {"long1": (["orient"], []), "long2": ([], ["hold_long"]), "long3": (["scan"], []), "long4": ([], ["approach", "retreat"])},
    12: {"long1": ([], ["hold_long"]), "long2": (["approach", "hold_short"], []), "long3": (["sway_large"], []), "long4": ([], ["retreat", "turn_away"])},
    13: {"long1": (["orient"], []), "long2": ([], ["settle"]), "long3": (["hold_short", "orient"], []), "long4": ([], ["turn_away", "depart"])},
    14: {"long1": ([], ["hold_long"]), "long2": (["approach"], []), "long3": (["scan", "hold_short"], []), "long4": ([], ["settle", "depart"])},
    15: {"long1": ([], ["hold_long"]), "long2": (["approach"], []), "long3": (["celebrate"], []), "long4": ([], ["retreat", "celebrate"])},
    16: {"long1": ([], ["hold_short"]), "long2": (["scan"], []), "long3": (["sway_large"], []), "long4": ([], ["turn_away", "settle"])},
    17: {"long1": ([], ["hold_long"]), "long2": (["crouch", "rise"], []), "long3": (["celebrate"], []), "long4": ([], ["celebrate", "settle"])},
    18: {"long1": ([], ["hold_short"]), "long2": (["orient"], []), "long3": (["sway_small"], []), "long4": ([], ["open", "settle"])},
    19: {"long1": ([], ["hold_long"]), "long2": (["approach"], []), "long3": (["scan", "open"], []), "long4": ([], ["circle", "settle"])},
    20: {"long1": ([], ["retreat"]), "long2": (["approach", "hold_short"], []), "long3": (["scan"], []), "long4": ([], ["turn_away", "settle"])},
    21: {"long1": ([], ["settle"]), "long2": (["crouch"], []), "long3": (["sway_large"], []), "long4": ([], ["approach", "stretch"])},
    22: {"long1": ([], ["celebrate"]), "long2": (["crouch", "rise"], []), "long3": (["hold_short", "celebrate"], []), "long4": ([], ["circle", "celebrate"])},
    23: {"long1": ([], ["retreat"]), "long2": (["approach"], []), "long3": (["scan", "close"], []), "long4": ([], ["turn_away", "depart"])},
    24: {"long1": ([], ["bow"]), "long2": (["approach"], []), "long3": (["point_target", "hold_short"], []), "long4": ([], ["bow", "retreat"])},
    25: {"long1": (["sway_small"], []), "long2": ([], ["retreat"]), "long3": (["sway_large", "hold_short"], []), "long4": ([], ["turn_away", "depart"])},
    26: {"long1": ([], ["hold_long"]), "long2": (["crouch", "rise"], []), "long3": (["circle"], []), "long4": ([], ["open", "settle"])},
    27: {"long1": ([], ["hold_long"]), "long2": (["circle"], []), "long3": (["retreat", "scan"], []), "long4": ([], ["approach", "settle"])},
    28: {"long1": (["sway_small"], []), "long2": (["retreat"], []), "long3": (["circle", "crouch"], []), "long4": ([], ["sway_large", "approach"])},
    29: {"long1": ([], ["hold_long"]), "long2": (["approach"], []), "long3": (["scan", "close"], []), "long4": ([], ["retreat", "hold_long"])},
    30: {"long1": ([], ["rise"]), "long2": (["crouch"], []), "long3": (["sway_large"], []), "long4": ([], ["circle", "depart"])},
}

ROLE_CODES: dict[str, dict[str, str]] = {
    "humanoid": {
        "hold_short": "hold(t=0.6)", "hold_long": "hold(t=1.5)",
        "orient": "move(head, pan, +20, spd=0.6, hold=0.3)",
        "scan": "move(head, pan, [-35,35], rep=1, spd=0.5)",
        "approach": "move(shoulder, x, +12, spd=0.7)",
        "retreat": "move(shoulder, x, -12, spd=0.7)",
        "sway_small": "move(shoulder, y, [-10,10], rep=1, spd=0.5)",
        "sway_large": "move(shoulder, y, [-25,25], rep=1, spd=0.35)",
        "turn_away": "move(head, pan, +40, spd=0.6)",
        "depart": "pose(dir=back, grip=vert, x=35, z=55)",
        "celebrate": "move(shoulder, z, +25, rep=2, spd=1.8)",
        "settle": "move(torso, height, low, spd=0.5, hold=0.8)",
        "crouch": "pose(torso=low, dir=in, grip=vert, z=40)",
        "rise": "move(torso, height, high, spd=1.2)",
        "stretch": "pose(dir=up, grip=horiz, z=85, left_arm=mirror)",
        "circle": "path(ee, arc, plane=xy, r=0.15, sweep=180, spd=0.5)",
        "point_target": "pose(dir=front, grip=vert, x=80, z=55)",
        "point_self": "pose(dir=in, grip=vert, x=35, z=60)",
        "open": "pose(dir=front, grip=horiz, y=25, z=65, left_arm=mirror)",
        "close": "pose(dir=in, grip=vert, y=50, z=55, left_arm=mirror)",
        "bow": "move(torso, height, low, spd=0.5, hold=0.8)",
    },
    "tiago": {
        "hold_short": "hold(t=0.6)", "hold_long": "hold(t=1.5)",
        "orient": "move(head, pan, +20, spd=0.6, hold=0.3)",
        "scan": "move(head, pan, [-35,35], rep=1, spd=0.5)",
        "approach": "path(base, line, x=+0.10, spd=0.5)",
        "retreat": "path(base, line, x=-0.12, spd=0.5)",
        "sway_small": "path(base, arc, r=0, deg=[-15,15], rep=1, spd=0.5)",
        "sway_large": "path(base, arc, r=0, deg=[-45,45], rep=1, spd=0.35)",
        "turn_away": "path(base, arc, r=0, deg=180, spd=0.7)",
        "depart": "path(base, line, x=+0.30, spd=1.0)",
        "celebrate": "move(shoulder, z, +25, rep=2, spd=1.8)",
        "settle": "move(torso, height, low, spd=0.5, hold=0.8)",
        "crouch": "pose(torso=low, dir=in, grip=vert, z=40)",
        "rise": "move(torso, height, high, spd=1.2)",
        "stretch": "pose(dir=up, grip=horiz, z=85, torso=high)",
        "circle": "path(base, arc, r=0.30, deg=180, spd=0.5)",
        "point_target": "pose(dir=front, grip=vert, x=80, z=55)",
        "point_self": "pose(dir=back, grip=vert, x=35, z=60)",
        "open": "pose(dir=front, grip=horiz, y=25, z=65)",
        "close": "pose(dir=in, grip=vert, y=50, z=55)",
        "bow": "move(torso, height, low, spd=0.5, hold=0.8)",
    },
    "single_arm": {
        "initial_pose": "pose(dir=front, grip=horiz, x=55, y=50, z=55)",
        "hold_short": "hold(t=0.6)", "hold_long": "hold(t=1.5)",
        "orient": "path(ee, line, axis=y, dist=+0.08, spd=0.5)",
        "scan": "path(ee, line, axis=y, dist=+-0.18, rep=1, spd=0.5)",
        "approach": "path(ee, line, axis=x, dist=+0.10, spd=0.5)",
        "retreat": "path(ee, line, axis=x, dist=-0.12, spd=0.5)",
        "sway_small": "path(ee, line, axis=y, dist=+-0.08, rep=1, spd=0.5)",
        "sway_large": "path(ee, line, axis=y, dist=+-0.20, rep=1, spd=0.35)",
        "turn_away": "pose(dir=back, grip=vert, x=35, z=55)",
        "depart": "path(ee, line, axis=x, dist=-0.20, spd=0.7)",
        "celebrate": "path(ee, line, axis=z, dist=+0.20, rep=2, spd=1.5)",
        "settle": "path(ee, line, axis=z, dist=-0.15, spd=0.4, hold=0.8)",
        "crouch": "pose(dir=in, grip=vert, z=30)",
        "rise": "path(ee, line, axis=z, dist=+0.20, spd=1.0)",
        "stretch": "pose(dir=up, grip=horiz, z=90)",
        "circle": "path(ee, arc, plane=xy, r=0.15, deg=180, spd=0.5)",
        "point_target": "pose(dir=front, grip=vert, x=80, z=55)",
        "point_self": "pose(dir=back, grip=vert, x=35, z=55)",
        "open": "pose(dir=front, grip=horiz, y=25, z=65)",
        "close": "pose(dir=in, grip=vert, y=50, z=55)",
        "bow": "path(ee, line, axis=z, dist=-0.12, spd=0.4, hold=0.8)",
    },
    "quadruped": {
        "hold_short": "hold(t=0.6)", "hold_long": "hold(t=1.5)",
        "orient": "move(body, roll, +20, spd=0.6, hold=0.3)",
        "scan": "move(body, roll, [-35,35], rep=1, spd=0.5)",
        "approach": "path(base, line, x=+0.12, spd=0.6)",
        "retreat": "path(base, line, x=-0.15, spd=0.5)",
        "sway_small": "path(base, arc, r=0, deg=[-15,15], rep=1, spd=0.5)",
        "sway_large": "path(base, arc, r=0, deg=[-45,45], rep=1, spd=0.35)",
        "turn_away": "path(base, arc, r=0, deg=180, spd=0.7)",
        "depart": "path(base, line, x=+0.35, spd=1.2)",
        "celebrate": "move(right_front, z, +0.20, rep=2, spd=1.5)",
        "settle": "pose(stance=sit, three_legs=none) + hold(t=0.8)",
        "crouch": "pose(stance=crouch, three_legs=none)",
        "rise": "pose(stance=stand, three_legs=none)",
        "stretch": "move(front_legs, z, +0.25, spd=0.7)",
        "circle": "path(base, arc, r=0.35, deg=180, spd=0.5)",
        "point_target": "move(right_front, x, +0.20, spd=0.8, hold=0.6)",
        "point_self": "move(right_front, x, -0.12, spd=0.6, hold=0.6)",
        "open": "pose(stance=stand, three_legs=none) + move(right_front, y, +0.15, spd=0.6)",
        "close": "pose(stance=crouch, three_legs=none)",
        "bow": "pose(stance=play_bow, three_legs=none) + hold(t=0.8)",
    },
    "roomba": {
        "hold_short": "hold(t=0.6)", "hold_long": "hold(t=1.5)",
        "orient": "path(base, arc, r=0, deg=30, spd=0.6)",
        "scan": "path(base, arc, r=0, deg=[-45,45], rep=1, spd=0.5)",
        "approach": "path(base, line, x=+0.12, spd=0.5)",
        "retreat": "path(base, line, x=-0.15, spd=0.5)",
        "sway_small": "path(base, arc, r=0, deg=[-15,15], rep=1, spd=0.5)",
        "sway_large": "path(base, arc, r=0, deg=[-45,45], rep=1, spd=0.35)",
        "turn_away": "path(base, arc, r=0, deg=180, spd=0.7)",
        "depart": "path(base, line, x=+0.35, spd=1.3)",
        "celebrate": "path(base, arc, r=0, deg=360, spd=1.5)",
        "settle": "hold(t=1.2)",
        "crouch": "hold(t=0.5)",
        "rise": "path(base, line, x=+0.08, spd=0.8)",
        "stretch": "path(base, line, x=[-0.12,0.12], rep=1, spd=0.5)",
        "circle": "path(base, arc, r=0.30, deg=180, spd=0.5)",
        "point_target": "path(base, line, x=+0.12, spd=1.0) + hold(t=0.6)",
        "point_self": "path(base, arc, r=0, deg=180, spd=0.6) + hold(t=0.6)",
        "open": "path(base, arc, r=0.35, deg=180, spd=0.6)",
        "close": "path(base, arc, r=0, deg=[-15,15], rep=1, spd=0.5)",
        "bow": "path(base, line, x=+0.05, spd=0.25) + hold(t=0.8)",
    },
}


def _context_fragment(platform: str, role: str) -> dict[str, Any]:
    code = ROLE_CODES[platform][role]
    converted = convert_code(code, platform=platform)
    if not converted["parse_ok"]:
        raise ValueError(f"failed context parse: {platform}/{role}: {converted['notes']}")
    movements = converted["movements"]
    if platform == "single_arm" and "prepended default pose" in converted["notes"]:
        movements = movements[1:]
    return {"role": role, "code": code, "movements": movements}


def _description(pre: list[str], post: list[str]) -> str:
    parts = []
    if pre:
        parts.append("Core 앞: " + " → ".join(ROLE_LABELS[r] for r in pre))
    if post:
        parts.append("Core 뒤: " + " → ".join(ROLE_LABELS[r] for r in post))
    return ". ".join(parts) + ". Core Motion Code와 움직임은 그대로 보존한다."


def _write_level(level: str, entries: list[dict[str, Any]]) -> None:
    out = OUT_ROOT / level
    out.mkdir(parents=True, exist_ok=True)
    (out / "embodiment_motion_code_all.json").write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for platform in PLATFORMS:
        rows = [entry["platforms"][platform] for entry in entries]
        (out / f"result_{platform}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cue_dir = out / platform
        cue_dir.mkdir(exist_ok=True)
        for row in rows:
            (cue_dir / f"{row['cue']}.json").write_text(
                json.dumps([row], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )


def build() -> dict[str, list[dict[str, Any]]]:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    now = datetime.now().isoformat(timespec="seconds")
    all_levels: dict[str, list[dict[str, Any]]] = {level: [] for level in LEVELS}
    plan = []

    for entry in gt:
        idx, cue = int(entry["idx"]), str(entry["cue"])
        if idx not in RECIPES:
            raise ValueError(f"missing recipe for {idx} {cue}")
        plan_entry = {
            "idx": idx,
            "cue": cue,
            "category": entry.get("category", ""),
            "levels": {},
        }
        for level in LEVELS:
            out_entry = {
                "idx": idx,
                "cue": cue,
                "category": entry.get("category", ""),
                "level": level,
                "platforms": {},
            }
            pre_roles, post_roles = ([], []) if level == "core" else RECIPES[idx][level]
            plan_entry["levels"][level] = {
                "pre_roles": pre_roles,
                "post_roles": post_roles,
                "description": (
                    "Human-GT Core only."
                    if level == "core"
                    else _description(pre_roles, post_roles)
                ),
            }
            for platform in PLATFORMS:
                base = copy.deepcopy(entry["platforms"][platform])
                pre = [_context_fragment(platform, role) for role in pre_roles]
                post = [_context_fragment(platform, role) for role in post_roles]
                setup: list[dict[str, Any]] = []
                if (
                    platform == "single_arm"
                    and pre
                    and pre[0]["movements"]
                    and pre[0]["movements"][0].get("type") != "pose"
                ):
                    # The manipulator executor requires a pose before any path.
                    # Keep this renderer setup outside the unchanged core range.
                    setup = [
                        _context_fragment(
                            platform,
                            "initial_pose",
                        )
                    ]
                pre_movements = [m for frag in pre for m in frag["movements"]]
                post_movements = [m for frag in post for m in frag["movements"]]
                setup_movements = [m for frag in setup for m in frag["movements"]]
                core_movements = copy.deepcopy(base["movements"])
                movements = (
                    setup_movements
                    + pre_movements
                    + core_movements
                    + post_movements
                )
                start = len(setup_movements) + len(pre_movements) + 1
                end = start + len(core_movements) - 1
                source_parts = [frag["code"] for frag in setup + pre]
                source_parts.append(str(base.get("source_code", "")).strip())
                source_parts.extend(frag["code"] for frag in post)
                row = {
                    **base,
                    "description": (
                        f"{level} embodiment motion ({platform}) for {cue}: "
                        + plan_entry["levels"][level]["description"]
                    ),
                    "movements": movements,
                    "source_code": " + ".join(part for part in source_parts if part),
                    "state": "embodiment_long_1_4",
                    "model": "human_designed_context_plan",
                    "time": now,
                    "experiment": "embodiment_long_1_4",
                    "level": level,
                    "variant_id": f"{idx:02d}:{cue}:{platform}:{level}",
                    "base_core_ref": (
                        "data/results/motion_configs/embodiment_motion_code_gt/"
                        "embodiment_motion_code_all.json"
                    ),
                    "main_component_range": {
                        "start": start,
                        "end": end,
                        "indexing": "1-based-inclusive",
                    },
                    "added_submotions": [
                        {
                            "position": "renderer_setup",
                            "role": "initial_pose",
                            "description": "Single-arm path 실행 전 필요한 중립 initial pose",
                            "motion_code": frag["code"],
                        }
                        for frag in setup
                    ]
                    + [
                        {
                            "position": "before_core",
                            "role": frag["role"],
                            "description": ROLE_LABELS[frag["role"]],
                            "motion_code": frag["code"],
                        }
                        for frag in pre
                    ]
                    + [
                        {
                            "position": "after_core",
                            "role": frag["role"],
                            "description": ROLE_LABELS[frag["role"]],
                            "motion_code": frag["code"],
                        }
                        for frag in post
                    ],
                }
                if movements[start - 1 : end] != core_movements:
                    raise AssertionError(f"core changed: {idx}/{platform}/{level}")
                out_entry["platforms"][platform] = row
            all_levels[level].append(out_entry)
        plan.append(plan_entry)

    for level, entries in all_levels.items():
        _write_level(level, entries)
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    combined = [
        row
        for level in LEVELS
        for entry in all_levels[level]
        for row in entry["platforms"].values()
    ]
    if len(combined) != 750:
        raise AssertionError(f"expected 750 configs, got {len(combined)}")
    (OUT_ROOT / "embodiment_long_1_4_all_750.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("levels", {level: len(rows) * 5 for level, rows in all_levels.items()})
    print("combined", len(combined))
    print("plan", PLAN_PATH)
    print("configs", OUT_ROOT)
    return all_levels


if __name__ == "__main__":
    build()
