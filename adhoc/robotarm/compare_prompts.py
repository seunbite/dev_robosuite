"""
Compare motion config quality across prompt versions v5–v10.

Usage:
    python compare_prompts.py summary                          # metric comparison table
    python compare_prompts.py cues                              # per-cue breakdown
    python compare_prompts.py cues --filter pose_only           # filtered
    python compare_prompts.py diff 6 10                         # two-version diff
    python compare_prompts.py detail 10                         # one-version deep dive

    python compare_prompts.py grid --versions "[5,10]" --start_idx 0 --end_idx 9   # render + grid comparison
    python compare_prompts.py grid --versions "[5,6,10]" --start_idx 10 --end_idx 20 --robot Panda

    python compare_prompts.py render 10                         # render v10 tiled GIFs (IIWA, sample cues)
    python compare_prompts.py render 10 --robot Panda --cue_idxs "[0,1,2,3,4]"

    python compare_prompts.py generate --versions "[8,9,10]"    # generate missing configs
"""

import fire
import json
import os
import sys
import time
import subprocess
import glob as globmod
from collections import Counter
from datetime import datetime
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SEED_DIR = os.path.join(PROJECT_ROOT, "data", "seed")
PROMPT_DIR = os.path.join(SEED_DIR, "prompt")
EXPECTED_CUES = 58
MOTION_BASE = os.path.join(PROJECT_ROOT, "data", "motions")
DEFAULT_VERSIONS = list(range(1, 17))
PROMPT_EVOLUTION_SUMMARY = {
    1: ("Schema Baseline", "Bare schema-first prompt. Recognizability is requested, but structure is mostly left to few-shot examples."),
    2: ("Anti-Pattern Pass", "Adds explicit anti-patterns like no return-to-rest filler and stronger speed variation guidance."),
    3: ("Planning Required", "Introduces required pre-JSON motion planning comments so the model reasons about phases before emitting steps."),
    4: ("Body Mapping", "Adds the body spatial reference table so x/y/z choices map more directly to forehead, chest, mouth, and other regions."),
    5: ("Pattern Templates", "Introduces composition patterns such as oscillation, reach-hold, sweep, circular, tap, and expressive transition."),
    6: ("Pattern-First Cleanup", "Keeps the pattern-template approach but tightens anti-pattern rules and reinforces deliberate phase planning."),
    7: ("Mandatory Planning Fields", "Makes planning fields explicit and mandatory, including category and pattern skeleton, to reduce degenerate outputs."),
    8: ("Pattern Selection Guide", "Adds a cue-to-pattern guide and explicit multi-pattern composition for more complex multi-phase cues."),
    9: ("Degree/Speed Calibration", "Adds concrete degree and speed calibration tables so motions are tuned to more visible amplitudes and tempos."),
    10: ("Checklist Baseline", "Adds an explicit quality checklist and output contract; still optimized for short, readable, pattern-driven gestures."),
    11: ("3-Act Choreography", "Shifts from short gesture templates to setup, core action, and follow-through as a required micro-story structure."),
    12: ("Rhythm and Pauses", "Keeps the 3-act choreography but makes rhythm, pause placement, and speed curves the main design principle."),
    13: ("Joint Layering + Path", "Pushes choreography further with joint cascade rules and required path diversity, making motions denser and more layered."),
    14: ("Iconic Readability", "Refocuses from generic choreography to iconic silhouette, anchor pose, and recognizability before flourish."),
    15: ("One Unmistakable Read", "Simplifies around one dominant iconic pose and main action; path becomes optional and decorative motion is deprioritized."),
    16: ("Repetition + Beckon Rule", "Keeps the iconic-read focus and adds explicit repetition rules plus orientation-sensitive beckon axis mapping."),
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _config_path(version: int) -> str:
    return os.path.join(SEED_DIR, f"motion_configs_prompt_v{version}.json")

def _prompt_path(version: int) -> str:
    return os.path.join(PROMPT_DIR, f"prompt_v{version}.txt")

def _motion_dir(version: int) -> str:
    return os.path.join(MOTION_BASE, f"v{version}")

def _load_configs(version: int) -> list[dict]:
    path = _config_path(version)
    if not os.path.exists(path):
        return []
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, ValueError):
        return []


def _covered_indices(data: list[dict]) -> list[int]:
    idxs = sorted({
        int(c.get("idx"))
        for c in data
        if c.get("idx") is not None and str(c.get("idx")).isdigit()
        and 0 <= int(c.get("idx")) < EXPECTED_CUES
    })
    return idxs


def _is_complete_config_set(data: list[dict]) -> bool:
    if _covered_indices(data) != list(range(EXPECTED_CUES)):
        return False
    expected = _expected_cue_map()
    by_idx = {}
    for cfg in data:
        idx = cfg.get("idx")
        if isinstance(idx, int) and 0 <= idx < EXPECTED_CUES and idx not in by_idx:
            by_idx[idx] = cfg.get("cue")
    return all(by_idx.get(idx) == cue for idx, cue in expected.items())


def _expected_cue_map() -> dict[int, str]:
    cues_path = os.path.join(SEED_DIR, "cues.yml")
    with open(cues_path, "r", encoding="utf-8") as f:
        cue_dict = yaml.safe_load(f)
    iconic_items = list(cue_dict["iconic"].items())[:EXPECTED_CUES]
    return {idx: desc for idx, (_, desc) in enumerate(iconic_items)}


def _canonicalize_config_rows(data: list[dict]) -> tuple[list[dict], list[str], int]:
    expected = _expected_cue_map()
    expected_cues = set(expected.values())
    latest_by_cue = {}
    for cfg in data:
        cue = cfg.get("cue")
        if cue in expected_cues:
            latest_by_cue[cue] = cfg

    normalized = []
    missing = []
    for idx in range(EXPECTED_CUES):
        cue = expected[idx]
        cfg = latest_by_cue.get(cue)
        if cfg is None:
            missing.append(cue)
            continue
        out = dict(cfg)
        out["idx"] = idx
        out["cue"] = cue
        normalized.append(out)

    extra_rows = max(len(data) - len(normalized), 0)
    return normalized, missing, extra_rows


def _canonicalize_config_file(path: str, backup: bool = False) -> tuple[int, int]:
    if not os.path.exists(path):
        return 0, 0
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    normalized, missing, extra_rows = _canonicalize_config_rows(data)
    if backup:
        backup_path = path.replace(".json", "_pre_canonicalize_backup.json")
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2, ensure_ascii=False)
    return len(normalized), len(missing)

def _cue_pattern(cfg: dict) -> str:
    return " → ".join(m["type"] for m in cfg.get("movements", []))

def _cue_steps(cfg: dict) -> int:
    return len(cfg.get("movements", []))

def _is_pose_only(cfg: dict) -> bool:
    return all(m["type"] == "pose" for m in cfg.get("movements", []))

def _has_path(cfg: dict) -> bool:
    return any(m["type"] == "path" for m in cfg.get("movements", []))

def _max_degree(cfg: dict) -> float:
    best = 0
    for m in cfg.get("movements", []):
        if m.get("type") == "movement":
            for d in m.get("parameters", {}).get("directions", []):
                for _, v in d.get("degrees", {}).items():
                    best = max(best, abs(v))
    return best

def _speed_values(cfg: dict) -> list[float]:
    speeds = []
    for m in cfg.get("movements", []):
        p = m.get("parameters", {})
        if "speed" in p:
            speeds.append(p["speed"])
        for d in p.get("directions", []):
            if "speed" in d:
                speeds.append(d["speed"])
    return speeds

def _joints_used(cfg: dict) -> list[str]:
    joints = []
    for m in cfg.get("movements", []):
        j = m.get("parameters", {}).get("joint")
        if j:
            joints.append(j)
    return joints

def _score_config(cfg: dict) -> tuple[float, dict]:
    """Score a single cue config on structural quality. Returns (score, breakdown)."""
    movements = cfg.get("movements", [])
    types = [m.get("type") for m in movements]
    n_steps = len(movements)
    breakdown = {}

    if not movements:
        return -100, {"empty": -100}

    score = 0

    # Pose-only is disqualifying
    if all(t == "pose" for t in types):
        breakdown["pose_only"] = -50
        score -= 50

    # Step count: 3-5 is sweet spot
    if n_steps == 1:
        breakdown["steps"] = -15
    elif n_steps == 2:
        breakdown["steps"] = 0
    elif 3 <= n_steps <= 5:
        breakdown["steps"] = 10 + (n_steps - 2) * 3
    elif n_steps <= 7:
        breakdown["steps"] = 15
    else:
        breakdown["steps"] = 5
    score += breakdown["steps"]

    # Has movement steps
    has_move = "movement" in types
    breakdown["has_movement"] = 10 if has_move else -10
    score += breakdown["has_movement"]

    # Has path steps (bonus)
    has_p = "path" in types
    breakdown["has_path"] = 12 if has_p else 0
    score += breakdown["has_path"]

    # Type diversity (pose + movement + path = 3 types max)
    unique = len(set(types))
    breakdown["type_diversity"] = (unique - 1) * 5
    score += breakdown["type_diversity"]

    # Degree range: moderate is best, extreme is penalized
    md = _max_degree(cfg)
    if md == 0:
        breakdown["degrees"] = 0
    elif md <= 35:
        breakdown["degrees"] = 5
    elif md <= 50:
        breakdown["degrees"] = 8
    elif md <= 60:
        breakdown["degrees"] = 2
    else:
        breakdown["degrees"] = -5
    score += breakdown["degrees"]

    # Speed variety
    speeds = _speed_values(cfg)
    if speeds:
        n_unique_speeds = len(set(speeds))
        if n_unique_speeds >= 3:
            breakdown["speed_variety"] = 8
        elif n_unique_speeds >= 2:
            breakdown["speed_variety"] = 4
        else:
            breakdown["speed_variety"] = -2 if len(speeds) > 2 else 0
    else:
        breakdown["speed_variety"] = 0
    score += breakdown["speed_variety"]

    # Joint diversity
    joints = set(_joints_used(cfg))
    breakdown["joint_diversity"] = min(len(joints), 3) * 3
    score += breakdown["joint_diversity"]

    # Pattern bonus: starts with pose (required structure)
    if types and types[0] == "pose":
        breakdown["starts_pose"] = 5
    else:
        breakdown["starts_pose"] = -10
    score += breakdown["starts_pose"]

    return score, breakdown


def _find_tiled_gif(
    version: int,
    robot: str,
    cue_idx: int,
    require_tiled: bool = False,
    allow_legacy: bool = True,
) -> str | None:
    """Find a GIF for a given version/robot/cue_idx.
    Prefer tiled GIFs, but fall back to single-variation GIFs when needed.
    Falls back to data/motions/{robot}/ for legacy v5 renders."""
    cue_name = None
    try:
        cue_name = next(
            (c.get("cue") for c in _load_configs(version) if c.get("idx") == cue_idx),
            None,
        )
    except Exception:
        cue_name = None
    search_dirs = [os.path.join(_motion_dir(version), robot)]
    if version == 5 and allow_legacy:
        search_dirs.append(os.path.join(MOTION_BASE, robot))
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        patterns = [
            os.path.join(search_dir, f"*_c{cue_idx}_tiled.gif"),
        ]
        if not require_tiled:
            patterns.append(os.path.join(search_dir, f"*_c{cue_idx}_*.gif"))
        for pattern in patterns:
            matches = globmod.glob(pattern)
            if matches:
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0]
        if cue_name and not require_tiled:
            for cue_pattern in (_motion_filename_cue_name(cue_name), _safe_cue_name(cue_name)):
                matches = globmod.glob(os.path.join(search_dir, f"*{cue_pattern}*.gif"))
                if matches:
                    matches.sort(key=os.path.getmtime, reverse=True)
                    return matches[0]
    return None


def _find_single_gif(version: int, robot: str, cue_idx: int) -> str | None:
    """Find a single-variation GIF if available; skip tiled and preview assets."""
    search_dir = os.path.join(_motion_dir(version), robot)
    if not os.path.isdir(search_dir):
        return None
    matches = []
    for path in globmod.glob(os.path.join(search_dir, f"*_c{cue_idx}_*.gif")):
        base = os.path.basename(path)
        if base.endswith("_tiled.gif") or base.endswith("_preview.gif"):
            continue
        matches.append(path)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def _primitive_checkpoint_count(cfg: dict) -> int:
    """Count visual checkpoints from movement primitives, ignoring repetition."""
    checkpoints = 0
    for m in cfg.get("movements", []):
        mtype = m.get("type")
        params = m.get("parameters", {})
        if mtype == "movement":
            checkpoints += max(len(params.get("directions", [])), 1)
        elif mtype in {"pose", "path"}:
            checkpoints += 1
        else:
            checkpoints += 1
    return max(checkpoints, 1)


def _build_top1_checkpoint_tile(
    src_gif: str,
    cfg: dict,
    out_path: str,
    tiled_columns: int = 5,
) -> str | None:
    """Build one PNG composed of primitive-aligned checkpoints from top_1.

    If the source is a tiled GIF (top_k variants laid out horizontally), crop the
    leftmost tile first so only top_1 is used.
    """
    from PIL import Image

    try:
        src_mtime = os.path.getmtime(src_gif)
        if os.path.exists(out_path) and os.path.getmtime(out_path) >= src_mtime:
            return out_path
        img = Image.open(src_gif)
        n_frames = max(getattr(img, "n_frames", 1), 1)

        img.seek(0)
        first = img.copy().convert("RGB")
        frame_w, frame_h = first.size
        crop_box = None
        if frame_w >= tiled_columns * 200:
            tile_w = frame_w // tiled_columns
            tile_h = min(frame_h, tile_w)  # strip bottom caption band from tiled renders
            crop_box = (0, 0, tile_w, tile_h)

        n_checkpoints = _primitive_checkpoint_count(cfg)
        if n_checkpoints == 1:
            sample_idxs = [max(0, min(n_frames - 1, round((n_frames - 1) * 0.5)))]
        else:
            sample_idxs = [
                min(n_frames - 1, round((n_frames - 1) * t / (n_checkpoints - 1)))
                for t in range(n_checkpoints)
            ]
        frames = []
        for frame_idx in sample_idxs:
            img.seek(frame_idx)
            frame = img.copy().convert("RGB")
            if crop_box is not None:
                frame = frame.crop(crop_box)
            frames.append(frame)

        cell_w = max(f.width for f in frames)
        cell_h = max(f.height for f in frames)
        gap = 8
        canvas = Image.new("RGB", (cell_w * len(frames) + gap * max(len(frames) - 1, 0), cell_h), (246, 248, 251))
        for i, frame in enumerate(frames):
            if frame.size != (cell_w, cell_h):
                frame = frame.resize((cell_w, cell_h))
            x = i * (cell_w + gap)
            canvas.paste(frame, (x, 0))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        canvas.save(out_path)
        img.close()
        return out_path
    except Exception:
        return None


def _normalize_versions(version: int | None = None, versions: list[int] | None = None) -> list[int]:
    if versions is not None:
        use_versions = versions
    elif version is not None:
        use_versions = [version]
    else:
        use_versions = DEFAULT_VERSIONS
    normalized = []
    for v in use_versions:
        try:
            vi = int(v)
        except Exception:
            continue
        if vi not in normalized:
            normalized.append(vi)
    return normalized

def _safe_cue_name(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")


def _motion_filename_cue_name(cue: str) -> str:
    return cue.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _movement_axes_label(params: dict) -> str:
    axis_order = ["x", "y", "z"]
    axes = []
    for d in params.get("directions", []):
        degrees = d.get("degrees", {})
        if isinstance(degrees, dict):
            for axis in axis_order:
                if axis in degrees and axis not in axes:
                    axes.append(axis)
    fallback_axis = params.get("axis")
    if fallback_axis in axis_order and fallback_axis not in axes:
        axes.append(fallback_axis)
    return "+".join(axes) if axes else "?"


def _step_pill_text(m: dict) -> str:
    mtype = m.get("type", "?")
    params = m.get("parameters", {})
    if mtype == "pose":
        pose = params.get("pose", {})
        return f"pose({pose.get('dir', '?')})"
    if mtype == "movement":
        joint = params.get("joint", "?")
        axes = _movement_axes_label(params)
        rep = params.get("repetition", 1)
        return f"move({joint}, {axes}, r={rep})"
    if mtype == "path":
        return f"path({params.get('shape', '?')}, {params.get('joint', '?')})"
    return str(mtype)


# ─── Analysis ────────────────────────────────────────────────────────────────

def _analyze_one(path: str) -> dict:
    data = json.load(open(path))
    name = os.path.basename(path).replace("motion_configs_", "").replace(".json", "")
    n = len(data)
    if n == 0:
        return {"name": name, "n": 0}

    pose_only = sum(1 for c in data if _is_pose_only(c))
    step_counts = Counter(_cue_steps(c) for c in data)
    type_counts = Counter()
    for c in data:
        for m in c.get("movements", []):
            type_counts[m["type"]] += 1

    extreme = sum(1 for c in data if _max_degree(c) > 50)
    max_deg = max((_max_degree(c) for c in data), default=0)

    uniform_speed = 0
    speed_variety_scores = []
    for c in data:
        speeds = _speed_values(c)
        if speeds:
            speed_variety_scores.append(len(set(speeds)))
        if len(speeds) > 2 and len(set(speeds)) == 1:
            uniform_speed += 1

    avg_speed_variety = sum(speed_variety_scores) / len(speed_variety_scores) if speed_variety_scores else 0

    return {
        "name": name, "n": n,
        "pose_only": pose_only,
        "two_step": step_counts.get(2, 0),
        "three_plus": sum(v for k, v in step_counts.items() if k >= 3),
        "avg_steps": sum(k * v for k, v in step_counts.items()) / n,
        "step_dist": dict(sorted(step_counts.items())),
        "types": dict(type_counts),
        "paths": type_counts.get("path", 0),
        "extreme": extreme,
        "max_deg": max_deg,
        "uniform_speed": uniform_speed,
        "avg_speed_variety": avg_speed_variety,
    }


def _best_markers(key: str, vals: list) -> list[bool]:
    higher_is_better = {"n", "three_plus", "avg_steps", "paths", "avg_speed_variety"}
    lower_is_better = {"pose_only", "two_step", "extreme", "max_deg", "uniform_speed"}
    if key in higher_is_better:
        best = max(vals)
    elif key in lower_is_better:
        best = min(vals)
    else:
        return [False] * len(vals)
    return [v == best for v in vals]


# ─── Commands ────────────────────────────────────────────────────────────────

def summary(versions: list[int] = None, include_v5: bool = True):
    """Show comparison table across all versions."""
    if versions is None:
        versions = [6, 7, 8, 9, 10]
    if include_v5 and 5 not in versions:
        versions = [5] + versions

    results = []
    for v in versions:
        path = _config_path(v)
        if os.path.exists(path):
            r = _analyze_one(path)
            if r["n"] > 0:
                results.append(r)

    if not results:
        print("No config files found. Run 'generate' first.")
        return

    for r in results:
        n = r["n"]
        print(f"\n{'─'*60}")
        print(f"  {r['name']}  ({n} configs)")
        print(f"{'─'*60}")
        print(f"  Pose-only:       {r['pose_only']}/{n} ({100*r['pose_only']/n:.0f}%)")
        print(f"  Avg steps:       {r['avg_steps']:.1f}   Step dist: {r['step_dist']}")
        print(f"  Types:           {r['types']}")
        print(f"  Path usage:      {r['paths']}")
        print(f"  Extreme (>50°):  {r['extreme']}  (max={r['max_deg']:.0f}°)")
        print(f"  Avg spd variety: {r['avg_speed_variety']:.1f}  (uniform: {r['uniform_speed']})")

    if len(results) < 2:
        return

    col_w = max(len(r["name"]) + 2 for r in results)
    col_w = max(col_w, 10)

    metrics = [
        ("Configs",        "n",                 "d"),
        ("Pose-only",      "pose_only",         "d"),
        ("Avg steps",      "avg_steps",         ".1f"),
        ("2-step",         "two_step",          "d"),
        ("3+ steps",       "three_plus",        "d"),
        ("Path count",     "paths",             "d"),
        ("Extreme (>50°)", "extreme",           "d"),
        ("Max degree",     "max_deg",           ".0f"),
        ("Uniform speed",  "uniform_speed",     "d"),
        ("Avg spd variety","avg_speed_variety",  ".1f"),
    ]

    print(f"\n{'═'*70}")
    print("  COMPARISON TABLE")
    print(f"{'═'*70}")
    header = f"{'Metric':<22}" + "".join(f"{r['name']:>{col_w}}" for r in results)
    print(header)
    print("─" * len(header))

    wins = Counter()
    for label, key, fmt in metrics:
        row = f"{label:<22}"
        vals = [r.get(key, 0) for r in results]
        for v in vals:
            row += f"{v:>{col_w}{fmt}}"
        markers = _best_markers(key, vals)
        if any(markers):
            for i, m in enumerate(markers):
                if m:
                    row += f"  ◀ best"
                    wins[results[i]["name"]] += 1
                    break
        print(row)

    print(f"\n{'─'*70}")
    print("  Wins per version:")
    for name, count in wins.most_common():
        bar = "█" * count
        print(f"    {name:<16} {bar} ({count})")
    print()


def cues(versions: list[int] = None, filter: str = None):
    """Per-cue comparison across versions.

    Args:
        versions: Versions to compare (default: [5, 6, 10])
        filter: 'pose_only', 'no_path', 'extreme', 'short', 'improved', 'regressed'
    """
    if versions is None:
        versions = [5, 6, 10]

    version_data = {}
    for v in versions:
        data = _load_configs(v)
        if data:
            by_cue = {}
            for cfg in data:
                cue = cfg.get("cue", "")
                if cue not in by_cue:
                    by_cue[cue] = cfg
            version_data[v] = by_cue

    if not version_data:
        print("No config files found.")
        return

    all_cues = []
    for v in versions:
        if v in version_data:
            for cue in version_data[v]:
                if cue not in all_cues:
                    all_cues.append(cue)

    col_w = 28
    header = f"{'Cue':<50}" + "".join(f"{'v'+str(v):>{col_w}}" for v in versions if v in version_data)
    print(f"\n{'═'*len(header)}")
    print("  PER-CUE COMPARISON")
    print(f"{'═'*len(header)}")
    print(header)
    print("─" * len(header))

    shown = 0
    for cue in all_cues:
        cells = []
        flags = set()
        for v in versions:
            if v not in version_data:
                continue
            cfg = version_data[v].get(cue)
            if cfg is None:
                cells.append("—")
                continue
            pattern = _cue_pattern(cfg)
            steps = _cue_steps(cfg)
            md = _max_degree(cfg)
            has_p = _has_path(cfg)
            po = _is_pose_only(cfg)
            markers = ""
            if po:
                markers += " ⚠PO"
                flags.add("pose_only")
            if md > 50:
                markers += f" ⚠{md:.0f}°"
                flags.add("extreme")
            if has_p:
                markers += " ◆path"
            if steps <= 2:
                flags.add("short")
            if not has_p:
                flags.add("no_path")
            cell = f"{steps}s {pattern}{markers}"
            cells.append(cell)

        if len(versions) >= 2:
            first_v = versions[0]
            last_v = versions[-1]
            if first_v in version_data and last_v in version_data:
                c_first = version_data[first_v].get(cue)
                c_last = version_data[last_v].get(cue)
                if c_first and c_last:
                    if _is_pose_only(c_first) and not _is_pose_only(c_last):
                        flags.add("improved")
                    if _cue_steps(c_last) > _cue_steps(c_first):
                        flags.add("improved")
                    if not _is_pose_only(c_first) and _is_pose_only(c_last):
                        flags.add("regressed")
                    if _max_degree(c_last) > _max_degree(c_first) and _max_degree(c_last) > 50:
                        flags.add("regressed")

        if filter and filter not in flags:
            continue

        cue_short = cue[:48] + ".." if len(cue) > 50 else cue
        row = f"{cue_short:<50}"
        for cell in cells:
            row += f"{cell:>{col_w}}"
        print(row)
        shown += 1

    print(f"\n  Shown: {shown}/{len(all_cues)} cues")
    if filter:
        print(f"  Filter: {filter}")
    print(f"  Filters: pose_only, no_path, extreme, short, improved, regressed\n")


def diff(v1: int, v2: int):
    """Side-by-side diff between two versions, highlighting changes."""
    data1 = _load_configs(v1)
    data2 = _load_configs(v2)
    if not data1 or not data2:
        print(f"Missing data: v{v1}={'found' if data1 else 'NOT FOUND'}, v{v2}={'found' if data2 else 'NOT FOUND'}")
        return

    by_cue1 = {c["cue"]: c for c in data1}
    by_cue2 = {c["cue"]: c for c in data2}
    all_cues = list(dict.fromkeys(list(by_cue1.keys()) + list(by_cue2.keys())))

    better = worse = same = 0

    print(f"\n{'═'*90}")
    print(f"  DIFF: v{v1} → v{v2}")
    print(f"{'═'*90}")
    print(f"{'Cue':<42} {'v'+str(v1):<22} {'v'+str(v2):<22} {'Change':<10}")
    print("─" * 90)

    for cue in all_cues:
        c1 = by_cue1.get(cue)
        c2 = by_cue2.get(cue)

        if c1 is None or c2 is None:
            label1 = "—" if c1 is None else f"{_cue_steps(c1)}s {_cue_pattern(c1)}"
            label2 = "—" if c2 is None else f"{_cue_steps(c2)}s {_cue_pattern(c2)}"
            cue_short = cue[:40] + ".." if len(cue) > 42 else cue
            print(f"{cue_short:<42} {label1:<22} {label2:<22} {'?':>6}")
            continue

        p1, p2 = _cue_pattern(c1), _cue_pattern(c2)
        s1, s2 = _cue_steps(c1), _cue_steps(c2)
        po1, po2 = _is_pose_only(c1), _is_pose_only(c2)
        d1, d2 = _max_degree(c1), _max_degree(c2)
        hp1, hp2 = _has_path(c1), _has_path(c2)

        score = 0
        if po1 and not po2: score += 2
        if not po1 and po2: score -= 2
        if s2 > s1: score += 1
        if s2 < s1: score -= 1
        if not hp1 and hp2: score += 1
        if hp1 and not hp2: score -= 1
        if d1 > 50 and d2 <= 50: score += 1
        if d1 <= 50 and d2 > 50: score -= 1

        if score > 0:
            change = f"  ✅ +{score}"
            better += 1
        elif score < 0:
            change = f"  ❌ {score}"
            worse += 1
        else:
            change = "  ↔" if p1 != p2 else "  ─"
            same += 1

        label1 = f"{s1}s {p1}"
        if po1: label1 += " ⚠PO"
        label2 = f"{s2}s {p2}"
        if hp2 and not hp1: label2 += " +path"

        cue_short = cue[:40] + ".." if len(cue) > 42 else cue
        if score != 0 or p1 != p2:
            print(f"{cue_short:<42} {label1:<22} {label2:<22} {change}")

    print(f"\n{'─'*90}")
    print(f"  Better: {better}  |  Worse: {worse}  |  Same/lateral: {same}")
    total = better + worse + same
    if total:
        print(f"  Net improvement: {better-worse:+d} / {total} cues ({100*(better-worse)/total:+.0f}%)")
    print()


def detail(version: int):
    """Deep dive into one version — per-cue stats, issues, pattern distribution."""
    data = _load_configs(version)
    if not data:
        print(f"v{version}: No config file found")
        return

    by_cue = {}
    for cfg in data:
        cue = cfg.get("cue", "unknown")
        by_cue[cue] = cfg

    n = len(by_cue)
    print(f"\n{'═'*70}")
    print(f"  DETAIL: v{version}  ({n} cues)")
    print(f"{'═'*70}")

    issues = []
    for cue, cfg in by_cue.items():
        cue_issues = []
        if _is_pose_only(cfg):
            cue_issues.append("pose-only")
        if _max_degree(cfg) > 50:
            cue_issues.append(f"extreme {_max_degree(cfg):.0f}°")
        if _cue_steps(cfg) < 2:
            cue_issues.append("too few steps")
        speeds = _speed_values(cfg)
        if len(speeds) > 2 and len(set(speeds)) == 1:
            cue_issues.append("uniform speed")
        if cue_issues:
            issues.append((cue, cue_issues))

    if issues:
        print(f"\n  ⚠ Issues ({len(issues)} cues):")
        for cue, iss in issues:
            cue_short = cue[:50] + ".." if len(cue) > 52 else cue
            print(f"    {cue_short:<54} {', '.join(iss)}")
    else:
        print(f"\n  ✅ No issues found!")

    patterns = Counter(_cue_pattern(cfg) for cfg in by_cue.values())
    print(f"\n  Pattern distribution:")
    for p, c in patterns.most_common():
        bar = "█" * c
        print(f"    {p:<40} {bar} ({c})")

    steps = Counter(_cue_steps(cfg) for cfg in by_cue.values())
    print(f"\n  Step distribution:")
    for s in sorted(steps):
        bar = "█" * steps[s]
        print(f"    {s} steps: {bar} ({steps[s]})")

    joint_counts = Counter()
    for cfg in by_cue.values():
        for j in _joints_used(cfg):
            joint_counts[j] += 1
    print(f"\n  Joint usage:")
    for j, c in joint_counts.most_common():
        bar = "█" * c
        print(f"    {j:<24} {bar} ({c})")

    all_speeds = []
    for cfg in by_cue.values():
        all_speeds.extend(_speed_values(cfg))
    if all_speeds:
        print(f"\n  Speed range: {min(all_speeds):.1f} – {max(all_speeds):.1f}  (median: {sorted(all_speeds)[len(all_speeds)//2]:.1f})")

    all_degs = [_max_degree(cfg) for cfg in by_cue.values() if _max_degree(cfg) > 0]
    if all_degs:
        print(f"  Max degree range: {min(all_degs):.0f}° – {max(all_degs):.0f}°  (median: {sorted(all_degs)[len(all_degs)//2]:.0f}°)")

    # Show render status
    rendered = 0
    for robot in ["IIWA", "Panda", "XArm7"]:
        rdir = os.path.join(_motion_dir(version), robot)
        if os.path.isdir(rdir):
            gifs = globmod.glob(os.path.join(rdir, "*_tiled.gif"))
            rendered += len(gifs)
    if rendered:
        print(f"\n  Rendered GIFs: {rendered} (in {_motion_dir(version)})")
    else:
        print(f"\n  No rendered GIFs yet. Run: compare_prompts.py render {version}")
    print()


# ─── Render ──────────────────────────────────────────────────────────────────

SAMPLE_CUE_IDXS = [0, 2, 5, 7, 14, 17, 21, 36, 38, 44]

def render(
    version: int | None = None,
    versions: list[int] | None = None,
    robot: str = "IIWA",
    cue_idxs: list[int] = None,
    all: bool = True,
    top_k: int = 5,
    hz: int = 4,
    path_hz: int = 12,
    timeout_s: int = 300,
    preview_speed_scale: float = 1.0,
    preview_hold_scale: float = 1.0,
    preview_max_hold_time: float = None,
    wipe: bool = False,
):
    """Render tiled GIFs for a prompt version via robosuite simulation.

    Args:
        version: Single prompt version
        versions: Multiple prompt versions (defaults to v1-v16 when omitted)
        robot: Robot name (IIWA, Panda, XArm7)
        cue_idxs: Specific cue indices to render
        all: Render all 58 cues by default
        top_k: Number of pose variations per cue
        hz: Frame rate for non-path motions
        path_hz: Frame rate for path motions
        timeout_s: Per-cue subprocess timeout in seconds
        preview_speed_scale: Multiplier for all speeds when rendering a fast preview
        preview_hold_scale: Multiplier for all hold times when rendering a fast preview
        preview_max_hold_time: Optional clamp for hold times when rendering a fast preview
        wipe: Delete existing GIFs in the target render directory before rendering
    """
    use_versions = _normalize_versions(version=version, versions=versions)
    if not use_versions:
        print("No versions requested.")
        return

    python_bin = sys.executable
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_generation.py")
    grand_succeeded = 0
    grand_failed = 0

    for cur_version in use_versions:
        config = _config_path(cur_version)
        if not os.path.exists(config):
            print(f"Config file not found: {config}")
            grand_failed += 1
            continue

        data = _load_configs(cur_version)
        if not data:
            print(f"No configs in {config}")
            grand_failed += 1
            continue

        output_dir = _motion_dir(cur_version)
        render_dir = os.path.join(output_dir, robot)
        os.makedirs(render_dir, exist_ok=True)

        if wipe:
            removed = 0
            wipe_targets = [render_dir]
            for gif_path in globmod.glob(os.path.join(render_dir, "*.gif")):
                os.remove(gif_path)
                removed += 1
            if cur_version == 5:
                legacy_dir = os.path.join(MOTION_BASE, robot)
                wipe_targets.append(legacy_dir)
                for gif_path in globmod.glob(os.path.join(legacy_dir, "*.gif")):
                    os.remove(gif_path)
                    removed += 1
            print(f"\n  Wiped {removed} existing GIFs from {', '.join(wipe_targets)}")

        if cue_idxs is not None:
            idxs = cue_idxs
        elif all:
            idxs = _covered_indices(data)
        else:
            idxs = [i for i in SAMPLE_CUE_IDXS if i < len(data)]

        require_tiled = top_k > 1

        todo = []
        for idx in idxs:
            existing = _find_tiled_gif(
                cur_version,
                robot,
                idx,
                require_tiled=require_tiled,
                allow_legacy=False,
            )
            if existing:
                cue_name = next((c["cue"] for c in data if c.get("idx") == idx), f"idx={idx}")
                print(f"  ✓ v{cur_version} c{idx} already rendered: {os.path.basename(existing)}")
            else:
                todo.append(idx)

        if not todo:
            print(f"\nAll {len(idxs)} cues already rendered for v{cur_version}/{robot}!")
            print(f"Output dir: {render_dir}")
            continue

        print(f"\n  Rendering {len(todo)} cues for v{cur_version}/{robot} (top_k={top_k})")
        print(f"  Output: {render_dir}")
        print(f"  Config: {config}\n")

        succeeded = 0
        failed = 0
        for i, idx in enumerate(todo):
            cue_name = next((c["cue"] for c in data if c.get("idx") == idx), f"idx={idx}")
            cue_short = cue_name[:45] + ".." if len(cue_name) > 47 else cue_name
            print(f"  [v{cur_version} {i+1}/{len(todo)}] c{idx}: {cue_short}", end=" ", flush=True)

            cmd = [
                python_bin, script_path,
                f"--robot={robot}",
                f"--cue_idx={idx}",
                f"--config_path={config}",
                f"--output_dir={output_dir}",
                f"--top_k={top_k}",
                f"--hz={hz}",
                f"--path_hz={path_hz}",
                f"--preview_speed_scale={preview_speed_scale}",
                f"--preview_hold_scale={preview_hold_scale}",
            ]
            if preview_max_hold_time is not None:
                cmd.append(f"--preview_max_hold_time={preview_max_hold_time}")

            start = time.time()
            try:
                result = subprocess.run(
                    cmd, text=True, capture_output=True, timeout=timeout_s
                )
                elapsed = time.time() - start

                if result.returncode == 0:
                    gif = _find_tiled_gif(
                        cur_version,
                        robot,
                        idx,
                        require_tiled=require_tiled,
                        allow_legacy=False,
                    )
                    if gif:
                        print(f"✅ ({elapsed:.0f}s)")
                        succeeded += 1
                    else:
                        print(f"⚠ done but no GIF ({elapsed:.0f}s)")
                        succeeded += 1
                else:
                    err = result.stderr.strip().splitlines()[-2:] if result.stderr else []
                    print(f"❌ ({elapsed:.0f}s)")
                    for line in err:
                        print(f"      {line}")
                    failed += 1
            except subprocess.TimeoutExpired:
                print(f"❌ timeout ({timeout_s}s)")
                failed += 1

        grand_succeeded += succeeded
        grand_failed += failed
        print(f"\n  v{cur_version} done: {succeeded} rendered, {failed} failed")
        print(f"  Output: {render_dir}\n")

    print(f"\n  Overall done: {grand_succeeded} rendered, {grand_failed} failed\n")


# ─── View (visual comparison) ───────────────────────────────────────────────

def view(
    versions: list[int] = None,
    robot: str = "IIWA",
    cue_idxs: list[int] = None,
    output: str = None,
    cols_per_version: int = 1,
):
    """Create side-by-side comparison image/GIF across prompt versions.

    For each cue, shows the tiled GIFs from different versions next to each other.
    Requires render to be run first for each version.

    Args:
        versions: Versions to compare (default: [5, 10])
        robot: Robot name
        cue_idxs: Cue indices to include (default: auto-detect from available GIFs)
        output: Output file path (default: auto-generated)
        cols_per_version: How many columns per version tile (1=compact)
    """
    from PIL import Image, ImageDraw, ImageFont

    if versions is None:
        versions = [5, 10]

    # Load cue names for labels
    all_cue_names = {}
    for v in versions:
        for cfg in _load_configs(v):
            idx = cfg.get("idx")
            if idx is not None:
                all_cue_names[idx] = cfg["cue"]

    # Find available GIFs
    available = {}  # {cue_idx: {version: gif_path}}
    for v in versions:
        for idx in (cue_idxs or sorted(all_cue_names.keys())):
            gif = _find_tiled_gif(v, robot, idx)
            if gif:
                if idx not in available:
                    available[idx] = {}
                available[idx][v] = gif

    if cue_idxs is None:
        use_idxs = sorted(idx for idx, vs in available.items() if len(vs) == len(versions))
    else:
        use_idxs = [idx for idx in cue_idxs if idx in available]

    if not use_idxs:
        print(f"No matching GIFs found for {robot} across versions {versions}.")
        print(f"Run 'render' first for each version. Example:")
        for v in versions:
            print(f"  python compare_prompts.py render {v} --robot {robot}")
        return

    print(f"\n  Comparing v{' vs v'.join(str(v) for v in versions)} for {robot}")
    print(f"  Cues: {use_idxs}")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Read first frame dimensions from first available GIF
    sample_gif_path = None
    for idx in use_idxs:
        for v in versions:
            p = available.get(idx, {}).get(v)
            if p:
                sample_gif_path = p
            break
        if sample_gif_path:
            break

    sample_img = Image.open(sample_gif_path)
    tile_w, tile_h = sample_img.size
    sample_img.close()

    label_col_w = 220
    version_label_h = 30
    cue_label_h = 24
    num_versions = len(versions)
    total_w = label_col_w + num_versions * tile_w
    total_h = version_label_h + len(use_idxs) * (tile_h + cue_label_h)

    # Determine max frame count for GIF animation
    max_frames = 1
    gif_cache = {}
    for idx in use_idxs:
        for v in versions:
            p = available.get(idx, {}).get(v)
            if p:
                img = Image.open(p)
                n_frames = getattr(img, 'n_frames', 1)
                max_frames = max(max_frames, n_frames)
                gif_cache[(idx, v)] = img

    print(f"  Grid: {len(use_idxs)} cues × {num_versions} versions, {max_frames} frames")
    print(f"  Canvas: {total_w} × {total_h}px")

    combined_frames = []
    for frame_i in range(max_frames):
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Version headers
        for vi, v in enumerate(versions):
            x = label_col_w + vi * tile_w
            label = f"v{v}"
            tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 30
            draw.text((x + (tile_w - tw) / 2, 5), label, fill="black", font=font)
            if vi > 0:
                draw.line([(x, 0), (x, total_h)], fill=(200, 200, 200), width=1)

        # Rows
        for row_i, idx in enumerate(use_idxs):
            y_base = version_label_h + row_i * (tile_h + cue_label_h)

            # Cue label
            cue_name = all_cue_names.get(idx, f"cue {idx}")
            cue_short = f"c{idx}: {cue_name[:28]}" if len(cue_name) > 28 else f"c{idx}: {cue_name}"
            draw.text((5, y_base + tile_h // 2 - 8), cue_short, fill="black", font=small_font)

            # Separator
            draw.line([(0, y_base), (total_w, y_base)], fill=(220, 220, 220), width=1)

            # Version tiles
            for vi, v in enumerate(versions):
                x = label_col_w + vi * tile_w
                gif_img = gif_cache.get((idx, v))
                if gif_img:
                    n_frames = getattr(gif_img, 'n_frames', 1)
                    try:
                        gif_img.seek(frame_i % n_frames)
                        frame = gif_img.copy().convert("RGB")
                        if frame.size != (tile_w, tile_h):
                            frame = frame.resize((tile_w, tile_h))
                        canvas.paste(frame, (x, y_base))
                    except EOFError:
                        pass
                else:
                    draw.rectangle([x, y_base, x + tile_w, y_base + tile_h], fill=(240, 240, 240))
                    draw.text((x + 10, y_base + tile_h // 2), "not rendered", fill=(180, 180, 180), font=small_font)

        combined_frames.append(canvas)

    # Close all GIF handles
    for img in gif_cache.values():
        try:
            img.close()
        except Exception:
            pass

    if not combined_frames:
        print("No frames generated.")
        return

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vs_label = "_vs_".join(f"v{v}" for v in versions)
        output = os.path.join(MOTION_BASE, f"compare_{vs_label}_{robot}_{ts}.gif")

    os.makedirs(os.path.dirname(output), exist_ok=True)

    if max_frames == 1:
        combined_frames[0].save(output.replace(".gif", ".png"))
        print(f"\n  Saved static comparison: {output.replace('.gif', '.png')}")
    else:
        import numpy as np
        sample_count = min(len(combined_frames), 50)
        sample_indices = np.unique(np.linspace(0, len(combined_frames) - 1, sample_count, dtype=int))
        palette_combined = Image.new("RGB", (total_w, total_h * len(sample_indices)))
        for idx_s, idx_f in enumerate(sample_indices):
            palette_combined.paste(combined_frames[idx_f].copy(), (0, idx_s * total_h))
        palette_img = palette_combined.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
        quantized = [f.copy().quantize(palette=palette_img, dither=Image.NONE) for f in combined_frames]
        del palette_combined, palette_img

        quantized[0].save(
            output,
            save_all=True,
            append_images=quantized[1:],
            duration=250,
            loop=0,
            disposal=1,
            optimize=False,
        )
        print(f"\n  Saved animated comparison ({max_frames} frames): {output}")

    for f in combined_frames:
        try:
            f.close()
        except Exception:
            pass

    print(f"  Open in browser/viewer to compare.\n")


# ─── Generate ────────────────────────────────────────────────────────────────

def generate(versions: list[int] = None, model: str = "gemini-2.5-flash", delay: float = 2.0):
    """Generate JSON configs for specified versions (skips already complete ones)."""
    if versions is None:
        versions = DEFAULT_VERSIONS

    print(f"\n── Generation (model: {model}) ──")
    for v in versions:
        prompt = _prompt_path(v)
        config = _config_path(v)

        if not os.path.exists(prompt):
            print(f"  ✗ v{v}: Prompt file not found ({prompt})")
            continue

        existing = _load_configs(v)
        if _is_complete_config_set(existing):
            print(f"  ✓ v{v}: Already complete ({len(_covered_indices(existing))} cues in c0-c{EXPECTED_CUES-1})")
            continue

        if existing:
            print(f"  ⚠ v{v}: Incomplete coverage ({len(_covered_indices(existing))}/{EXPECTED_CUES} cues in c0-c{EXPECTED_CUES-1}) — deleting and regenerating")
            os.remove(config)

        python_bin = sys.executable
        cmd = [
            python_bin, "adhoc/robotarm/config_gen_meta.py",
            f"--prompt_file={prompt}",
            f"--config_json={config}",
            f"--model={model}",
            f"--delay={delay}",
        ]

        print(f"  ▶ v{v}: Generating ...")
        start = time.time()
        result = subprocess.run(cmd, text=True, capture_output=True)
        elapsed = time.time() - start

        if result.returncode != 0:
            stderr_tail = result.stderr.strip().splitlines()[-3:] if result.stderr else []
            print(f"  ✗ v{v}: Failed ({elapsed:.0f}s)")
            for line in stderr_tail:
                print(f"    {line}")
        else:
            final = _load_configs(v)
            kept, missing = _canonicalize_config_file(config)
            final = _load_configs(v)
            suffix = f", canonicalized to {kept}" if kept else ""
            missing_note = f", missing {missing}" if missing else ""
            print(f"  ✓ v{v}: Done — {len(final)} configs{suffix}{missing_note} ({elapsed:.0f}s)")
    print()


def canonicalize(versions: list[int] = None, backup: bool = False):
    """Rewrite config files into canonical cue order c0-c57 using cues.yml."""
    if versions is None:
        versions = DEFAULT_VERSIONS

    print("\n── Canonicalize Configs ──")
    for v in versions:
        path = _config_path(v)
        if not os.path.exists(path):
            print(f"  ✗ v{v}: Missing ({path})")
            continue
        kept, missing = _canonicalize_config_file(path, backup=backup)
        if missing:
            print(f"  ⚠ v{v}: kept {kept}, missing {missing}")
        else:
            print(f"  ✓ v{v}: canonicalized to {kept} cues")
    print()


# ─── Status ──────────────────────────────────────────────────────────────────

def status(versions: list[int] = None):
    """Show what's available: configs generated, GIFs rendered per version/robot."""
    if versions is None:
        versions = DEFAULT_VERSIONS

    print(f"\n{'═'*70}")
    print("  STATUS OVERVIEW")
    print(f"{'═'*70}")

    col_w = 14
    robots = ["IIWA", "Panda", "XArm7"]
    header = f"{'Version':<12}{'Configs':>{col_w}}" + "".join(f"{r:>{col_w}}" for r in robots)
    print(header)
    print("─" * len(header))

    for v in versions:
        data = _load_configs(v)
        n = len(_covered_indices(data))
        row = f"  v{v:<9}{n:>{col_w}}"
        for robot in robots:
            dirs_to_check = [os.path.join(_motion_dir(v), robot)]
            if v == 5:
                dirs_to_check.append(os.path.join(MOTION_BASE, robot))
            gif_count = 0
            for rdir in dirs_to_check:
                if os.path.isdir(rdir):
                    gif_count += len(globmod.glob(os.path.join(rdir, "*_tiled.gif")))
            row += f"{gif_count:>{col_w}}" if gif_count else f"{'—':>{col_w}}"
        print(row)

    print(f"\n  Config files: {SEED_DIR}/motion_configs_prompt_v*.json")
    print(f"  GIF outputs:  {MOTION_BASE}/v*/{{robot}}/")
    print()


# ─── Grid: render + tile across versions ─────────────────────────────────────

def grid(
    versions: list[int] = None,
    robot: str = "IIWA",
    start_idx: int = 0,
    end_idx: int = 9,
    top_k: int = 1,
    hz: int = 4,
    path_hz: int = 12,
    output: str = None,
    proximal_degree_scale: float = 0.6,
):
    """Render cues across prompt versions and create a comparison grid GIF.

    Layout:
      columns = versions
      rows    = top_k poses × cue indices  (grouped by cue with label above)

    Args:
        versions: Prompt versions to compare (default: [5, 10])
        robot: Robot name (IIWA, Panda, XArm7)
        start_idx: First cue index
        end_idx: Last cue index (inclusive)
        top_k: Number of initial pose variations per cue (1=single, >1=multiple rows per cue)
        hz: Frame rate for non-path motions
        path_hz: Frame rate for path motions
        output: Output GIF path (auto-generated if None)
        proximal_degree_scale: Joint movement scale
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    if versions is None:
        versions = [5, 10]

    # Load configs for each version
    version_configs = {}
    for v in versions:
        data = _load_configs(v)
        if not data:
            print(f"  ✗ v{v}: No config file found")
            continue
        by_idx = {c.get("idx"): c for c in data if c.get("idx") is not None}
        version_configs[v] = by_idx

    if len(version_configs) < 2:
        print("Need at least 2 versions with configs.")
        return

    cue_idxs = list(range(start_idx, end_idx + 1))
    cue_names = {}
    for v in versions:
        if v in version_configs:
            for idx in cue_idxs:
                cfg = version_configs[v].get(idx)
                if cfg and idx not in cue_names:
                    cue_names[idx] = cfg.get("cue", f"cue_{idx}")

    cue_idxs = [i for i in cue_idxs if i in cue_names]
    if not cue_idxs:
        print("No valid cue indices in the given range.")
        return

    n_cues = len(cue_idxs)
    n_cols = len(version_configs)
    active_versions = [v for v in versions if v in version_configs]

    print(f"\n{'═'*60}")
    print(f"  GRID: {n_cues} cues × {n_cols} versions × top_k={top_k}  ({robot})")
    print(f"  Cues: c{start_idx}–c{end_idx}  Versions: {active_versions}")
    print(f"{'═'*60}\n")

    # Import MotionGenerator (heavy import, only when needed)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from motion_generation import MotionGenerator, _select_initial_poses, _config_has_path

    print(f"  Initializing {robot} simulator...", flush=True)
    generator = MotionGenerator(
        robot_name=robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        output_dir=os.path.join(MOTION_BASE, "grid_tmp"),
    )
    initial_joint_pos = generator.initial_joint_pos

    # grid_frames[(idx, v, k)] = list[Image]
    grid_frames = {}
    # actual_k[idx] = how many top_k rows this cue actually has
    actual_k = {}

    total_cells = n_cues * n_cols * top_k
    cell_num = 0

    for idx in cue_idxs:
        cue = cue_names[idx]

        if top_k == 1:
            actual_k[idx] = 1
            for v in active_versions:
                cell_num += 1
                cfg = version_configs[v].get(idx)
                if cfg is None:
                    print(f"  [{cell_num}/{total_cells}] c{idx} v{v}: SKIP")
                    continue

                first_pose_def = None
                for m in cfg.get("movements", []):
                    if m.get("type") == "pose":
                        first_pose_def = m["parameters"]["pose"]
                        break

                pose_id = None
                if first_pose_def is not None:
                    matching = generator._find_matching_poses(first_pose_def)
                    selected = _select_initial_poses(matching, first_pose_def, 1)
                    if selected:
                        pose_id = selected[0]["pose_id"]

                effective_hz = path_hz if _config_has_path(cfg) else hz
                cue_short = cue[:35] + ".." if len(cue) > 37 else cue
                print(f"  [{cell_num}/{total_cells}] c{idx} v{v}: {cue_short}", end=" ", flush=True)

                generator._set_joint_positions(initial_joint_pos)
                try:
                    frames, pid = generator.execute_cue(
                        cue=cue,
                        pose_index=pose_id,
                        config_path=_config_path(v),
                        proximal_degree_scale=proximal_degree_scale,
                        hz=effective_hz,
                        cue_idx=idx,
                        save_gif=False,
                    )
                    grid_frames[(idx, v, 0)] = frames
                    print(f"✅ p{pose_id} ({len(frames)}f)")
                except Exception as e:
                    print(f"❌ {str(e)[:60]}")
                generator._set_joint_positions(initial_joint_pos)
        else:
            # top_k > 1: select initial poses per (cue, version)
            # First, determine actual_k from the version with most matching poses
            max_k_for_cue = 0
            version_poses = {}

            for v in active_versions:
                cfg = version_configs[v].get(idx)
                if cfg is None:
                    continue
                first_pose_def = None
                for m in cfg.get("movements", []):
                    if m.get("type") == "pose":
                        first_pose_def = m["parameters"]["pose"]
                        break
                if first_pose_def is None:
                    continue
                matching = generator._find_matching_poses(first_pose_def)
                selected = _select_initial_poses(matching, first_pose_def, top_k)
                version_poses[v] = selected
                max_k_for_cue = max(max_k_for_cue, len(selected))

            actual_k[idx] = min(top_k, max(max_k_for_cue, 1))

            for k in range(actual_k[idx]):
                for v in active_versions:
                    cell_num += 1
                    cfg = version_configs[v].get(idx)
                    poses = version_poses.get(v, [])
                    if cfg is None or k >= len(poses):
                        continue

                    effective_hz = path_hz if _config_has_path(cfg) else hz
                    pose_id = poses[k]["pose_id"]
                    cue_short = cue[:30] + ".." if len(cue) > 32 else cue
                    print(f"  [{cell_num}/{total_cells}] c{idx} v{v} k{k}: {cue_short}", end=" ", flush=True)

                    generator._set_joint_positions(initial_joint_pos)
                    try:
                        frames, pid = generator.execute_cue(
                            cue=cue,
                            pose_index=pose_id,
                            config_path=_config_path(v),
                            proximal_degree_scale=proximal_degree_scale,
                            hz=effective_hz,
                            cue_idx=idx,
                            save_gif=False,
                        )
                        grid_frames[(idx, v, k)] = frames
                        print(f"✅ p{pose_id} ({len(frames)}f)")
                    except Exception as e:
                        print(f"❌ {str(e)[:60]}")
                    generator._set_joint_positions(initial_joint_pos)

    generator.close()

    # ── Assemble grid GIF ────────────────────────────────────────
    import gc

    print(f"\n  Assembling grid...", flush=True)

    tile_w = tile_h = 0
    for frames in grid_frames.values():
        if frames and len(frames) > 0:
            tile_w, tile_h = frames[0].size
            break

    if tile_w == 0:
        print("  No frames were rendered.")
        return

    MAX_GRID_FRAMES = 80
    raw_max = max((len(f) for f in grid_frames.values() if f), default=1)
    target_frames = min(raw_max, MAX_GRID_FRAMES)

    if raw_max > target_frames:
        for key, frames in grid_frames.items():
            if frames and len(frames) > target_frames:
                indices = np.unique(np.linspace(0, len(frames) - 1, target_frames, dtype=int))
                kept = [frames[i] for i in indices]
                for i, f in enumerate(frames):
                    if i not in indices:
                        try: f.close()
                        except: pass
                del frames
                grid_frames[key] = kept
        gc.collect()
        print(f"  Subsampled cell frames: {raw_max} → {target_frames}")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 56)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    version_header_h = 80
    cue_label_h = 64
    grid_w = n_cols * tile_w
    total_w = grid_w

    # Total height: version header + per-cue (label + top_k tile rows)
    total_h = version_header_h
    for idx in cue_idxs:
        k_count = actual_k.get(idx, 1)
        total_h += cue_label_h + k_count * tile_h

    max_frames = max((len(f) for f in grid_frames.values() if f), default=1)
    canvas_bytes = total_w * total_h * 3
    est_mb = (canvas_bytes * max_frames) / (1024 ** 2)
    print(f"  Grid: {total_w}×{total_h}px, {max_frames} frames, ~{est_mb:.0f} MB for canvases", flush=True)

    combined = []
    for fi in range(max_frames):
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        for ci, v in enumerate(active_versions):
            x = ci * tile_w
            label = f"v{v}"
            tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else 40
            draw.text((x + (tile_w - tw) / 2, 10), label, fill="black", font=font)
            if ci > 0:
                draw.line([(x, 0), (x, total_h)], fill=(180, 180, 180), width=2)

        draw.line([(0, version_header_h - 2), (total_w, version_header_h - 2)], fill=(120, 120, 120), width=3)

        cur_y = version_header_h
        for ci, idx in enumerate(cue_idxs):
            k_count = actual_k.get(idx, 1)

            if ci > 0:
                draw.line([(0, cur_y), (total_w, cur_y)], fill=(160, 160, 160), width=2)

            cue = cue_names.get(idx, "")
            cue_label = f"c{idx}: {cue}"
            tw = draw.textlength(cue_label, font=small_font) if hasattr(draw, "textlength") else 100
            if tw > total_w - 20:
                while tw > total_w - 20 and len(cue_label) > 10:
                    cue_label = cue_label[:-3] + ".."
                    tw = draw.textlength(cue_label, font=small_font) if hasattr(draw, "textlength") else 100
            draw.text(((total_w - tw) / 2, cur_y + 8), cue_label, fill="black", font=small_font)

            tile_start_y = cur_y + cue_label_h

            for k in range(k_count):
                ty = tile_start_y + k * tile_h
                if k > 0:
                    draw.line([(0, ty), (total_w, ty)], fill=(235, 235, 235), width=1)
                for col_i, v in enumerate(active_versions):
                    x = col_i * tile_w
                    frames = grid_frames.get((idx, v, k))
                    if frames and len(frames) > 0:
                        frame = frames[fi % len(frames)]
                        if frame.mode != "RGB":
                            frame = frame.convert("RGB")
                        if frame.size != (tile_w, tile_h):
                            frame = frame.resize((tile_w, tile_h))
                        canvas.paste(frame, (x, ty))
                    else:
                        draw.rectangle([x + 1, ty + 1, x + tile_w - 1, ty + tile_h - 1], fill=(240, 240, 240))
                        draw.text((x + tile_w // 2 - 10, ty + tile_h // 2 - 10), "—", fill=(180, 180, 180), font=font)

            cur_y = tile_start_y + k_count * tile_h

        combined.append(canvas)

    for frames in grid_frames.values():
        if frames:
            for f in frames:
                try: f.close()
                except: pass
    grid_frames.clear()
    gc.collect()

    if not combined:
        print("  No frames to save.")
        return

    # Save
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vs_label = "_".join(f"v{v}" for v in active_versions)
        tk_label = f"_k{top_k}" if top_k > 1 else ""
        output = os.path.join(
            MOTION_BASE,
            f"grid_{vs_label}_{robot}_c{start_idx}-{end_idx}{tk_label}_{ts}.gif",
        )

    os.makedirs(os.path.dirname(output), exist_ok=True)

    if max_frames == 1:
        png_path = output.replace(".gif", ".png")
        combined[0].save(png_path)
        print(f"  Saved static grid: {png_path}")
    else:
        PALETTE_SAMPLES = 8
        PALETTE_SCALE = 4
        sample_count = min(len(combined), PALETTE_SAMPLES)
        sample_indices = np.unique(np.linspace(0, len(combined) - 1, sample_count, dtype=int))
        small_w, small_h = max(total_w // PALETTE_SCALE, 1), max(total_h // PALETTE_SCALE, 1)
        palette_canvas = Image.new("RGB", (small_w, small_h * len(sample_indices)))
        for si, sf in enumerate(sample_indices):
            palette_canvas.paste(combined[sf].resize((small_w, small_h), Image.NEAREST), (0, si * small_h))
        palette_img = palette_canvas.quantize(colors=256, method=Image.FASTOCTREE, dither=Image.NONE)
        del palette_canvas
        gc.collect()

        QUANT_BATCH = 20
        quantized = []
        for batch_start in range(0, len(combined), QUANT_BATCH):
            batch = combined[batch_start:batch_start + QUANT_BATCH]
            for f in batch:
                quantized.append(f.quantize(palette=palette_img, dither=Image.NONE))
            for f in batch:
                try: f.close()
                except: pass
        del palette_img
        combined.clear()
        gc.collect()

        quantized[0].save(
            output,
            save_all=True,
            append_images=quantized[1:],
            duration=int(1000 / hz),
            loop=0,
            disposal=1,
            optimize=False,
        )
        for f in quantized:
            try: f.close()
            except: pass
        del quantized
        gc.collect()

        print(f"  Saved animated grid ({max_frames} frames): {output}")

    for f in combined:
        try: f.close()
        except: pass
    del combined
    gc.collect()

    total_rows = sum(actual_k.get(idx, 1) for idx in cue_idxs)
    print(f"  Grid: {n_cues} cues × {total_rows} tile-rows × {n_cols} versions, {total_w}×{total_h}px\n")


# ─── Best-of: per-cue optimal version selection ─────────────────────────────

def bestof(
    versions: list[int] = None,
    output: str = None,
    verbose: bool = False,
):
    """Auto-select the best prompt version for each cue and merge into one config.

    Scores every (cue × version) config on structural quality, picks the winner
    per cue, and writes a merged 'best-of' config file.

    Args:
        versions: Versions to evaluate (default: 5-10)
        output: Output JSON path (default: data/seed/motion_configs_bestof.json)
        verbose: Show scoring breakdown per cue
    """
    if versions is None:
        versions = [5, 6, 7, 8, 9, 10]

    if output is None:
        output = os.path.join(SEED_DIR, "motion_configs_bestof.json")

    # Load all versions
    version_data = {}
    for v in versions:
        data = _load_configs(v)
        if data:
            by_cue = {}
            for cfg in data:
                cue = cfg.get("cue", "")
                if cue and cue not in by_cue:
                    by_cue[cue] = cfg
            version_data[v] = by_cue
            print(f"  v{v}: {len(by_cue)} cues loaded")

    if not version_data:
        print("No config files found.")
        return

    # Collect all cues (ordered by first appearance)
    all_cues = []
    for v in versions:
        if v in version_data:
            for cue in version_data[v]:
                if cue not in all_cues:
                    all_cues.append(cue)

    # Score each (cue, version) and pick best
    best_configs = []
    version_wins = Counter()
    close_calls = []  # cues where top 2 scores are within 5 points

    col_w = 8
    header = f"{'Cue':<52}" + "".join(f"{'v'+str(v):>{col_w}}" for v in versions if v in version_data) + f"  {'Best':>6}  Margin"
    print(f"\n{'═'*len(header)}")
    print("  PER-CUE SCORING")
    print(f"{'═'*len(header)}")
    print(header)
    print("─" * len(header))

    for cue in all_cues:
        scores = {}
        breakdowns = {}
        for v in versions:
            if v not in version_data:
                continue
            cfg = version_data[v].get(cue)
            if cfg is None:
                continue
            sc, bd = _score_config(cfg)
            scores[v] = sc
            breakdowns[v] = bd

        if not scores:
            continue

        # Find best version for this cue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_v = ranked[0][0]
        best_score = ranked[0][1]
        margin = best_score - ranked[1][1] if len(ranked) > 1 else 999

        version_wins[best_v] += 1
        best_cfg = version_data[best_v][cue].copy()
        best_cfg["_selected_version"] = best_v
        best_cfg["_score"] = best_score
        best_cfg["_margin"] = margin
        best_configs.append(best_cfg)

        if margin <= 5 and len(ranked) > 1:
            close_calls.append((cue, ranked[:3]))

        # Print row
        cue_short = cue[:50] + ".." if len(cue) > 52 else cue
        row = f"{cue_short:<52}"
        for v in versions:
            if v not in version_data:
                continue
            sc = scores.get(v)
            if sc is not None:
                marker = " ◀" if v == best_v else "  "
                row += f"{sc:>{col_w-2}.0f}{marker}"
            else:
                row += f"{'—':>{col_w}}"
        row += f"  {'v'+str(best_v):>6}  {'+' + str(int(margin)) if margin < 999 else '—':>6}"
        print(row)

        if verbose:
            for v in [best_v]:
                bd = breakdowns.get(v, {})
                parts = [f"{k}={val:+d}" for k, val in bd.items() if val != 0]
                print(f"{'':>54}v{v}: {', '.join(parts)}")

    # Summary
    print(f"\n{'═'*70}")
    print("  VERSION DISTRIBUTION")
    print(f"{'═'*70}")
    for v, count in version_wins.most_common():
        pct = 100 * count / len(all_cues) if all_cues else 0
        bar = "█" * count
        print(f"    v{v:<4} {bar} {count} ({pct:.0f}%)")

    avg_score = sum(c.get("_score", 0) for c in best_configs) / len(best_configs) if best_configs else 0
    print(f"\n  Total cues: {len(best_configs)}")
    print(f"  Avg best score: {avg_score:.1f}")

    if close_calls:
        print(f"\n  ⚠ Close calls ({len(close_calls)} cues with margin ≤ 5):")
        for cue, ranked in close_calls[:10]:
            cue_short = cue[:40] + ".." if len(cue) > 42 else cue
            scores_str = ", ".join(f"v{v}={s:.0f}" for v, s in ranked)
            print(f"    {cue_short:<44} {scores_str}")
        if len(close_calls) > 10:
            print(f"    ... and {len(close_calls) - 10} more")

    # Quality breakdown of the best-of vs individual versions
    print(f"\n{'─'*70}")
    print("  BEST-OF vs INDIVIDUAL VERSIONS")
    print(f"{'─'*70}")

    bestof_metrics = {
        "pose_only": sum(1 for c in best_configs if _is_pose_only(c)),
        "avg_steps": sum(_cue_steps(c) for c in best_configs) / len(best_configs) if best_configs else 0,
        "paths": sum(1 for c in best_configs if _has_path(c)),
        "extreme": sum(1 for c in best_configs if _max_degree(c) > 50),
        "avg_score": avg_score,
    }

    metric_header = f"{'Metric':<20}{'best-of':>10}"
    for v in versions:
        if v in version_data:
            metric_header += f"{'v'+str(v):>10}"
    print(metric_header)
    print("─" * len(metric_header))

    for label, key in [("Pose-only", "pose_only"), ("Avg steps", "avg_steps"), ("Path count", "paths"), ("Extreme >50°", "extreme"), ("Avg score", "avg_score")]:
        fmt = ".1f" if "avg" in key else "d"
        row = f"{label:<20}{bestof_metrics[key]:>10{fmt}}"
        for v in versions:
            if v not in version_data:
                continue
            v_cfgs = list(version_data[v].values())
            if key == "pose_only":
                val = sum(1 for c in v_cfgs if _is_pose_only(c))
            elif key == "avg_steps":
                val = sum(_cue_steps(c) for c in v_cfgs) / len(v_cfgs) if v_cfgs else 0
            elif key == "paths":
                val = sum(1 for c in v_cfgs if _has_path(c))
            elif key == "extreme":
                val = sum(1 for c in v_cfgs if _max_degree(c) > 50)
            elif key == "avg_score":
                val = sum(_score_config(c)[0] for c in v_cfgs) / len(v_cfgs) if v_cfgs else 0
            row += f"{val:>10{fmt}}"
        print(row)

    # Write output
    output_configs = []
    for c in best_configs:
        out = {k: v for k, v in c.items() if not k.startswith("_")}
        out["_from_version"] = c["_selected_version"]
        output_configs.append(out)

    with open(output, "w") as f:
        json.dump(output_configs, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Saved best-of config: {output}")
    print(f"     {len(output_configs)} cues, cherry-picked from {len(version_data)} versions\n")


def top10_html(
    versions: list[int] | None = None,
    robot: str = "IIWA",
    count: int = 10,
    output: str | None = None,
):
    """Generate a focused HTML comparison page for the curated shared top-10 cues."""
    import html as html_mod

    if versions is None:
        versions = [10, 12, 13, 14, 15, 16]
    supported_versions = [10, 12, 13, 14, 15, 16]
    supported_prefixes = [supported_versions[:n] for n in range(3, len(supported_versions) + 1)]
    if versions not in supported_prefixes:
        raise ValueError(
            "top10_html currently supports prefix versions like "
            "[10, 12, 13], [10, 12, 13, 14], [10, 12, 13, 14, 15], "
            "or [10, 12, 13, 14, 15, 16]"
        )
    if robot != "IIWA":
        raise ValueError("top10_html currently supports robot='IIWA' only")

    cue_pool = list(range(21))  # c0-c20

    all_configs = {}
    for v in versions:
        cfgs = _load_configs(v)
        if not cfgs:
            raise ValueError(f"No config file found for v{v}")
        all_configs[v] = {c.get("idx"): c for c in cfgs if c.get("idx") is not None}

    def _find_exact_tiled_gif(version: int, cue_idx: int) -> str | None:
        return _find_tiled_gif(version, robot, cue_idx)

    candidates = []
    for idx in cue_pool:
        cfgs = {v: all_configs[v].get(idx) for v in versions}
        gifs = {v: _find_exact_tiled_gif(v, idx) for v in versions}
        if any(not cfgs[v] for v in versions):
            continue
        scores = {}
        for v in versions:
            scores[v], _ = _score_config(cfgs[v])
        candidates.append({
            "idx": idx,
            "cue": cfgs[versions[0]].get("cue", f"cue_{idx}"),
            "scores": scores,
            "combined_score": sum(scores.values()),
            "gifs": gifs,
            "cfgs": cfgs,
        })

    def _candidate_sort_key(candidate):
        ordered_scores = tuple(-candidate["scores"][v] for v in reversed(versions))
        return (-candidate["combined_score"],) + ordered_scores + (candidate["idx"],)

    candidates.sort(key=_candidate_sort_key)

    curated_order = [6, 7, 15, 16, 18, 19, 20, 9, 10, 17]
    candidate_by_idx = {c["idx"]: c for c in candidates}
    missing = [idx for idx in curated_order if idx not in candidate_by_idx]
    if missing:
        raise ValueError(
            f"Curated cue set is missing shared config/GIF assets: {missing}"
        )
    selected = [candidate_by_idx[idx] for idx in curated_order]
    if len(selected) != count:
        raise ValueError(f"Expected {count} curated cues, found {len(selected)}")

    def esc(s):
        return html_mod.escape(str(s))

    def _config_summary(cfg):
        mvs = cfg.get("movements", [])
        pattern = " → ".join(m.get("type", "?") for m in mvs)
        joints = []
        for j in _joints_used(cfg):
            if j not in joints:
                joints.append(j)
        speeds = _speed_values(cfg)
        speed_range = f"{min(speeds):.1f}–{max(speeds):.1f}" if speeds else "–"
        return f"{pattern} | joints: {', '.join(joints) or '–'} | speeds: {speed_range}"

    def _reasoning_html(cfg):
        reasoning = cfg.get("reasoning", "")
        if not reasoning:
            return '<span class="na">No CoT saved (re-generate to capture)</span>'
        parts = []
        for line in reasoning.strip().split("\n"):
            line_clean = line.lstrip("# ").strip()
            if ":" in line_clean:
                key, val = line_clean.split(":", 1)
                parts.append(f'<span class="cot-key">{esc(key.strip())}:</span> {esc(val.strip())}')
            else:
                parts.append(esc(line_clean))
        return "<br>".join(parts)

    def _config_json_html(cfg):
        display = {
            k: v for k, v in cfg.items()
            if k not in ("idx", "state", "model", "time", "reasoning", "validation_warnings")
        }
        return esc(json.dumps(display, indent=2, ensure_ascii=False))

    def _step_pills_html(cfg):
        pills = []
        for m in cfg.get("movements", []):
            mtype = m.get("type", "?")
            text = _step_pill_text(m)
            pills.append((mtype, text))
        out = ['<div class="step-viz">']
        for i, (mtype, text) in enumerate(pills):
            if i > 0:
                out.append('<span class="step-arrow">→</span>')
            out.append(f'<span class="step-pill {mtype}">{esc(text)}</span>')
        out.append('</div>')
        return "".join(out)

    def _version_stats(cfg):
        mvs = cfg.get("movements", [])
        path_steps = sum(1 for m in mvs if m.get("type") == "path")
        movement_steps = sum(1 for m in mvs if m.get("type") == "movement")
        unique_joints = []
        for j in _joints_used(cfg):
            if j not in unique_joints:
                unique_joints.append(j)
        speeds = _speed_values(cfg)
        holds = []
        for m in mvs:
            p = m.get("parameters", {})
            if "hold_time" in p and p["hold_time"] > 0:
                holds.append(p["hold_time"])
            for d in p.get("directions", []):
                if d.get("hold_time", 0) > 0:
                    holds.append(d["hold_time"])
        return {
            "steps": _cue_steps(cfg),
            "path_steps": path_steps,
            "movement_steps": movement_steps,
            "unique_joints": unique_joints,
            "speed_count": len(set(speeds)),
            "hold_count": len(holds),
        }

    def _difference_callout(cfg_by_version):
        stats = {v: _version_stats(cfg_by_version[v]) for v in versions}
        max_steps = max(v["steps"] for v in stats.values())
        min_steps = min(v["steps"] for v in stats.values())
        max_paths = max(v["path_steps"] for v in stats.values())
        min_paths = min(v["path_steps"] for v in stats.values())
        max_holds = max(v["hold_count"] for v in stats.values())
        max_joints = max(len(v["unique_joints"]) for v in stats.values())

        def describe(version: int) -> str:
            st = stats[version]
            bits = []
            if st["steps"] == min_steps:
                bits.append("most concise")
            elif st["steps"] == max_steps:
                bits.append("most choreographed")
            else:
                bits.append("mid-density")
            bits.append("more step detail" if st["steps"] > min_steps else "fewer steps")
            bits.append("more path" if st["path_steps"] == max_paths and max_paths > min_paths else "less path")
            bits.append("more pauses" if st["hold_count"] == max_holds and max_holds > 0 else "fewer pauses")
            bits.append("richer joints" if len(st["unique_joints"]) == max_joints and max_joints > 1 else "simpler joints")
            return ", ".join(bits)

        return (
            f'<div class="diff-callout">'
            + "".join(
                f'<div><strong>v{v}:</strong> {esc(describe(v))}</div>'
                for v in versions
            )
            + '</div>'
        )

    def _version_card(version: int, cfg: dict, gif_path: str | None):
        model = cfg.get("model", "–")
        desc = cfg.get("description", "")
        score, _ = _score_config(cfg)
        step_count = _cue_steps(cfg)
        is_pose_only = _is_pose_only(cfg)
        badge_class = "badge-bad" if is_pose_only or score < 3 else ("badge-warn" if score < 6 else "badge-good")
        parts = []
        parts.append(f'<div class="version-card">')
        parts.append(f'<div class="card-header"><span class="vtag">v{version}</span><span class="model">{esc(model)}</span></div>')
        parts.append('<div class="card-body">')
        if gif_path:
            rel_gif = os.path.relpath(gif_path, os.path.dirname(output))
            parts.append(f'<div class="gif-container"><img src="{esc(rel_gif)}" loading="lazy" alt="v{version} animation"></div>')
        else:
            parts.append('<div class="gif-container na" style="padding: 24px 0; border: 1px dashed var(--border); border-radius: 12px;">No tiled GIF yet</div>')
        if desc:
            parts.append(f'<div class="description">{esc(desc)}</div>')
        parts.append(_step_pills_html(cfg))
        parts.append(f'<div style="margin: 4px 0;"><span class="badge {badge_class}">score: {score:.1f}</span> <span class="badge badge-good">{step_count} steps</span>')
        if _has_path(cfg):
            parts.append(' <span class="badge badge-good">has path</span>')
        parts.append('</div>')
        parts.append('<div class="section-label">Summary</div>')
        parts.append(f'<div class="summary-line">{esc(_config_summary(cfg))}</div>')
        parts.append('<div class="section-label">Chain of Thought</div>')
        parts.append(f'<div class="cot-block">{_reasoning_html(cfg)}</div>')
        parts.append('<div class="section-label">Config JSON</div>')
        parts.append(f'<div class="config-json collapsed" onclick="this.classList.toggle(\'collapsed\')">{_config_json_html(cfg)}</div>')
        parts.append('</div></div>')
        return "".join(parts)

    if output is None:
        version_label = "_vs_".join(f"v{v}" for v in versions)
        output = os.path.join(MOTION_BASE, f"top10_{version_label}_IIWA.html")

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top 10 Prompt Comparison</title>
<style>
:root {
  --bg: #f6f8fb; --surface: #ffffff; --surface2: #eef2f7; --panel: #f3f6fb;
  --border: #d0d7de; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --warn: #9a6700; --red: #cf222e; --purple: #8250df;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: linear-gradient(180deg, #ffffff, #f6f8fb 160px); color: var(--text); line-height: 1.5; }
.page { max-width: 1520px; margin: 0 auto; padding: 18px 16px 40px; }
.hero { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; padding: 18px 20px; margin-bottom: 20px; box-shadow: 0 10px 28px rgba(31,35,40,0.08); }
.hero h1 { font-size: 24px; margin-bottom: 8px; }
.hero p { color: var(--text2); max-width: 980px; }
.hero .meta { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
.chip { display: inline-block; padding: 6px 10px; border-radius: 999px; font-size: 12px; background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }
.selected-list { margin-top: 14px; font-size: 13px; color: var(--text2); }
.selected-list code { color: var(--text); }
.cue-row { margin-bottom: 20px; }
.cue-header { font-size: 18px; font-weight: 650; margin-bottom: 10px; padding: 10px 14px; background: var(--surface); border-radius: 12px; border-left: 4px solid var(--accent); }
.cue-header .idx { color: var(--accent); margin-right: 8px; }
.cue-grid { display: grid; grid-template-columns: repeat(VAR_COLS, minmax(0, 1fr)); gap: 16px; }
.version-card { background: var(--surface); border: 1px solid var(--border); border-radius: 18px; overflow: hidden; box-shadow: 0 10px 24px rgba(31,35,40,0.08); }
.card-header { padding: 14px 18px; background: var(--surface2); font-weight: 700; font-size: 15px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
.card-header .vtag { color: var(--accent); font-size: 18px; }
.card-header .model { color: var(--text2); font-weight: 500; font-size: 13px; }
.card-body { padding: 16px 18px 18px; }
.gif-container { margin-bottom: 12px; }
.gif-container img { width: 100%; border-radius: 12px; border: 1px solid var(--border); display: block; background: var(--surface2); }
.description { font-size: 13px; color: var(--text2); margin: 8px 0 12px; font-style: italic; min-height: 56px; }
.step-viz { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 6px 0 8px; }
.step-pill { padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.step-pill.pose { background: #1f6feb33; color: var(--accent); }
.step-pill.movement { background: #2ea04333; color: var(--accent2); }
.step-pill.path { background: #bc8cff33; color: var(--purple); }
.step-arrow { color: var(--text2); font-size: 11px; }
.badge { display: inline-block; padding: 4px 8px; border-radius: 8px; font-size: 12px; font-weight: 600; }
.badge-good { background: #2ea04326; color: var(--accent2); }
.badge-warn { background: #d2992226; color: var(--warn); }
.badge-bad { background: #f8514926; color: var(--red); }
.section-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text2); margin: 14px 0 6px; }
.summary-line { font-size: 12px; color: var(--text2); padding: 10px 12px; background: var(--panel); border-radius: 10px; font-family: 'SF Mono', 'Fira Code', monospace; word-break: break-word; }
.cot-block { font-size: 12px; padding: 12px; background: var(--panel); border-radius: 10px; border-left: 4px solid var(--purple); height: 110px; min-height: 78px; max-height: 320px; overflow: auto; resize: vertical; }
.cot-key { color: var(--purple); font-weight: 700; }
.config-json { font-size: 11px; font-family: 'SF Mono', 'Fira Code', monospace; background: var(--panel); padding: 12px; border-radius: 10px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; cursor: pointer; }
.config-json.collapsed { max-height: 96px; position: relative; overflow: hidden; }
.config-json.collapsed::after { content: '▼ click to expand'; position: absolute; bottom: 0; left: 0; right: 0; text-align: center; padding: 6px; background: linear-gradient(transparent, var(--panel)); color: var(--text2); }
.diff-callout { margin: 10px 0 14px; padding: 12px 14px; background: linear-gradient(180deg, rgba(88,166,255,0.08), rgba(63,185,80,0.06)); border: 1px solid var(--border); border-radius: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px; color: var(--text2); }
.diff-callout strong { color: var(--text); }
.na { color: var(--text2); font-style: italic; }
@media (max-width: 1500px) {
  .cue-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 980px) {
  .cue-grid { grid-template-columns: 1fr; }
  .diff-callout { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="page">
""".replace("VAR_COLS", str(len(versions))))

    html_parts.append('<section class="hero">')
    version_title = " vs ".join(f"v{v}" for v in versions)
    html_parts.append(f'<h1>Top 10 Shared-Cue Comparison: {esc(version_title)}</h1>')
    html_parts.append(f'<p>This page compares the same 10 IIWA cues across {esc(version_title)}. Selection is deterministic: shared rendered cues only (`c0–c20`), validated against config and GIF availability for all requested versions, then displayed in a curated order chosen to make the version differences easy to read side-by-side.</p>')
    html_parts.append('<div class="meta">')
    html_parts.append(f'<span class="chip">versions: {esc(version_title)}</span>')
    html_parts.append('<span class="chip">robot: IIWA</span>')
    html_parts.append('<span class="chip">candidate pool: shared rendered c0–c20</span>')
    html_parts.append('<span class="chip">selection: shared assets + curated top-10 set</span>')
    html_parts.append('</div>')
    html_parts.append('<div class="selected-list"><strong>Selected cues:</strong> ')
    html_parts.append(", ".join(f"<code>c{c['idx']}</code> {esc(c['cue'])}" for c in selected))
    html_parts.append('</div>')
    html_parts.append('</section>')

    for item in selected:
        idx = item["idx"]
        cue = item["cue"]
        html_parts.append(f'<section class="cue-row" id="cue-{idx}">')
        html_parts.append(f'<div class="cue-header"><span class="idx">c{idx}</span>{esc(cue)}</div>')
        html_parts.append(_difference_callout(item["cfgs"]))
        html_parts.append('<div class="cue-grid">')
        for v in versions:
            html_parts.append(_version_card(v, item["cfgs"][v], item["gifs"][v]))
        html_parts.append('</div></section>')

    html_parts.append("""
</div>
</body>
</html>
""")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print("\n  ✅ Top-10 comparison page saved:")
    print(f"     {output}")
    print(f"     cues: {[c['idx'] for c in selected]}\n")


def dashboard(
    versions: list[int] | None = None,
    start_idx: int = 0,
    end_idx: int = 57,
    cue_idxs: list[int] | None = None,
    robot: str = "IIWA",
    output: str | None = None,
    serve: bool = False,
    include_legacy_base: bool = False,
):
    """Generate an interactive HTML dashboard showing prompt, CoT, config, and GIF side-by-side.

    Args:
        versions: Prompt versions to include (default: all available)
        start_idx: First cue index
        end_idx: Last cue index (inclusive)
        cue_idxs: Optional explicit cue index list (overrides start/end range)
        robot: Robot name for finding GIFs
        output: Output HTML path (auto-generated if None)
        serve: If True, start a local HTTP server to view the dashboard
        include_legacy_base: If True, also show legacy base renders from data/motions/{robot}
    """
    import html as html_mod
    import base64
    from pathlib import Path

    if versions is None:
        versions = DEFAULT_VERSIONS[:]

    if not versions:
        print("No config versions found.")
        return

    # Load data
    all_configs = {}
    all_prompts = {}
    for v in versions:
        cfgs = _load_configs(v)
        if cfgs:
            all_configs[v] = {c.get("idx"): c for c in cfgs if c.get("idx") is not None}
        ppath = _prompt_path(v)
        if os.path.exists(ppath):
            with open(ppath, "r", encoding="utf-8") as f:
                all_prompts[v] = f.read()

    if cue_idxs is None:
        cue_idxs = list(range(start_idx, end_idx + 1))
    cue_names = {}
    for v in versions:
        for idx in cue_idxs:
            cfg = all_configs.get(v, {}).get(idx)
            if cfg and idx not in cue_names:
                cue_names[idx] = cfg.get("cue", f"cue_{idx}")
    cue_idxs = [i for i in cue_idxs if i in cue_names]

    # Find per-version per-cue GIFs
    # gif_map_version[(idx, v)] = path   — version-specific renders
    # gif_map_base[idx] = path           — legacy base prompt renders
    gif_map_version = {}
    gif_map_base = {}
    for v in versions:
        for idx in cue_idxs:
            gif = _find_tiled_gif(v, robot, idx)
            if gif:
                gif_map_version[(idx, v)] = gif

    if include_legacy_base:
        legacy_dir = os.path.join(MOTION_BASE, robot)
        if os.path.isdir(legacy_dir):
            for gf in os.listdir(legacy_dir):
                if not gf.endswith(".gif"):
                    continue
                for idx in cue_idxs:
                    if f"_c{idx}_" in gf:
                        gif_map_base[idx] = os.path.join(legacy_dir, gf)
                        break

    # Find grid comparison GIFs
    grid_gifs = []
    for gf in sorted(os.listdir(MOTION_BASE)):
        if gf.startswith("grid_") and gf.endswith(".gif"):
            grid_gifs.append(gf)

    active_versions = [v for v in versions if v in all_configs]
    n_versions = len(active_versions)

    print(f"Dashboard: {len(cue_idxs)} cues × {n_versions} versions")
    print(f"  Versions: {active_versions}")
    print(f"  Version-specific GIFs: {len(gif_map_version)}")
    print(f"  Legacy base GIFs: {len(gif_map_base)}")
    print(f"  Grid comparison GIFs: {len(grid_gifs)}")

    def esc(s):
        return html_mod.escape(str(s))

    def _config_summary(cfg):
        if not cfg:
            return "N/A"
        mvs = cfg.get("movements", [])
        types = [m.get("type", "?") for m in mvs]
        pattern = " → ".join(types)
        joints = set()
        for m in mvs:
            j = m.get("parameters", {}).get("joint", "")
            if j:
                joints.add(j)
        speeds = _speed_values(cfg)
        speed_range = f"{min(speeds):.1f}–{max(speeds):.1f}" if speeds else "–"
        return f"{pattern} | joints: {', '.join(joints) or '–'} | speeds: {speed_range}"

    def _reasoning_html(cfg):
        if not cfg:
            return '<span class="na">N/A</span>'
        reasoning = cfg.get("reasoning", "")
        if not reasoning:
            return '<span class="na">No CoT saved (re-generate to capture)</span>'
        lines = reasoning.strip().split("\n")
        parts = []
        for line in lines:
            line_clean = line.lstrip("# ").strip()
            if ":" in line_clean:
                key, val = line_clean.split(":", 1)
                parts.append(f'<span class="cot-key">{esc(key.strip())}:</span> {esc(val.strip())}')
            else:
                parts.append(esc(line_clean))
        return "<br>".join(parts)

    def _config_json_html(cfg):
        if not cfg:
            return '<span class="na">N/A</span>'
        return esc(json.dumps(cfg.get("movements", []), indent=2, ensure_ascii=False))

    # Build HTML
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Motion Config Dashboard — {robot}</title>
<style>
:root {{
  --bg: #f6f8fb; --surface: #ffffff; --surface2: #eef2f7;
  --border: #d0d7de; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --warn: #9a6700;
  --red: #cf222e; --purple: #8250df;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); font-size: 17px; line-height: 1.5; }}
.header {{ position: sticky; top: 0; z-index: 100; background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 20px; }}
.header h1 {{ font-size: 21px; font-weight: 600; white-space: nowrap; }}
.header .version-toggles {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.header label {{ display: flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; background: var(--surface2); cursor: pointer; font-size: 13px; border: 1px solid var(--border); transition: all 0.15s; }}
.header label:hover {{ border-color: var(--accent); }}
.header label.active {{ background: #dbeafe; border-color: var(--accent); }}
.header input[type="checkbox"] {{ accent-color: var(--accent); }}
.filter-bar {{ display: flex; gap: 12px; align-items: center; margin-left: auto; }}
.filter-bar input {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 13px; width: 200px; }}
.sidebar {{ position: fixed; left: 0; top: 53px; bottom: 0; width: 240px; background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto; padding: 8px 0; }}
.sidebar a {{ display: block; padding: 6px 16px; color: var(--text2); text-decoration: none; font-size: 13px; border-left: 3px solid transparent; }}
.sidebar a:hover {{ background: var(--surface2); color: var(--text); }}
.sidebar a.active {{ border-left-color: var(--accent); color: var(--accent); background: #dbeafe; }}
.main {{ margin-left: 240px; padding: 20px 24px; }}
.cue-section {{ margin-bottom: 32px; scroll-margin-top: 60px; }}
.cue-header {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; padding: 8px 12px; background: var(--surface); border-radius: 8px; border-left: 4px solid var(--accent); }}
.cue-header .idx {{ color: var(--accent); margin-right: 8px; }}
.version-grid {{ display: grid; grid-template-columns: repeat({n_versions}, 1fr); gap: 12px; }}
.version-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.version-card .card-header {{ padding: 8px 12px; background: var(--surface2); font-weight: 600; font-size: 13px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
.version-card .card-header .vtag {{ color: var(--accent); }}
.version-card .card-header .model {{ color: var(--text2); font-weight: 400; font-size: 12px; }}
.card-body {{ padding: 10px 12px; }}
.section-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); margin: 8px 0 4px; }}
.section-label:first-child {{ margin-top: 0; }}
.summary-line {{ font-size: 12px; color: var(--text2); padding: 4px 8px; background: var(--surface2); border-radius: 4px; font-family: 'SF Mono', 'Fira Code', monospace; word-break: break-all; }}
.cot-block {{ font-size: 12px; padding: 8px; background: var(--surface2); border-radius: 4px; border-left: 3px solid var(--purple); height: 96px; min-height: 72px; max-height: 280px; overflow: auto; resize: vertical; }}
.cot-key {{ color: var(--purple); font-weight: 600; }}
.config-json {{ font-size: 11px; font-family: 'SF Mono', 'Fira Code', monospace; background: var(--surface2); padding: 8px; border-radius: 4px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; cursor: pointer; }}
.config-json.collapsed {{ max-height: 80px; position: relative; }}
.config-json.collapsed::after {{ content: '▼ click to expand'; position: absolute; bottom: 0; left: 0; right: 0; text-align: center; padding: 4px; background: linear-gradient(transparent, var(--bg)); color: var(--text2); font-size: 11px; }}
.na {{ color: var(--text2); font-style: italic; font-size: 12px; }}
.gif-container {{ margin-top: 6px; text-align: center; }}
.gif-container img {{ max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }}
.prompt-section {{ margin-bottom: 24px; }}
.prompt-toggle {{ cursor: pointer; padding: 10px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; font-size: 14px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; }}
.prompt-toggle:hover {{ border-color: var(--accent); }}
.prompt-content {{ display: none; background: var(--surface); border: 1px solid var(--border); border-top: none; border-radius: 0 0 8px 8px; padding: 16px; font-size: 12px; font-family: 'SF Mono', 'Fira Code', monospace; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }}
.prompt-content.open {{ display: block; }}
.diff-highlight {{ background: #2ea04326; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
.badge-good {{ background: #2ea04326; color: var(--accent2); }}
.badge-warn {{ background: #d2992226; color: var(--warn); }}
.badge-bad {{ background: #f8514926; color: var(--red); }}
.step-viz {{ display: flex; gap: 4px; align-items: center; flex-wrap: wrap; margin: 4px 0; }}
.step-pill {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.step-pill.pose {{ background: #1f6feb33; color: var(--accent); }}
.step-pill.movement {{ background: #2ea04333; color: var(--accent2); }}
.step-pill.path {{ background: #bc8cff33; color: var(--purple); }}
.step-arrow {{ color: var(--text2); font-size: 10px; }}
.description {{ font-size: 12px; color: var(--text2); margin: 4px 0 8px; font-style: italic; }}
.summary-intro {{ margin-bottom: 24px; }}
.summary-intro h2 {{ font-size: 18px; margin-bottom: 8px; }}
.summary-intro p {{ color: var(--text2); margin-bottom: 14px; max-width: 980px; }}
.prompt-summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
.prompt-summary-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
.prompt-summary-card .pver {{ color: var(--accent); font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.6px; }}
.prompt-summary-card .ptitle {{ font-size: 14px; font-weight: 600; margin-top: 4px; }}
.prompt-summary-card .pdesc {{ font-size: 12px; color: var(--text2); margin-top: 6px; line-height: 1.45; }}
</style>
</head>
<body>
<div class="header">
  <h1>Motion Config Dashboard</h1>
  <div class="version-toggles">
""")

    for v in active_versions:
        html_parts.append(f'    <label class="active"><input type="checkbox" data-version="{v}" checked onchange="toggleVersion({v}, this)"> v{v}</label>\n')

    html_parts.append(f"""  </div>
  <div class="filter-bar">
    <input type="text" id="searchInput" placeholder="Search cues..." oninput="filterCues(this.value)">
  </div>
</div>
<div class="sidebar" id="sidebar">
""")

    for idx in cue_idxs:
        name = cue_names[idx]
        short = name[:30] + ".." if len(name) > 32 else name
        html_parts.append(f'  <a href="#cue-{idx}" data-idx="{idx}">c{idx}: {esc(short)}</a>\n')

    html_parts.append('</div>\n<div class="main">\n')

    html_parts.append('<section class="summary-intro">\n')
    html_parts.append('  <h2>Prompt Evolution Summary</h2>\n')
    html_parts.append('  <p>Top-level summary of what changed from p1 to p16. The raw prompt texts remain below unchanged, and these summary cards follow the same version checkbox filters.</p>\n')
    html_parts.append('  <div class="prompt-summary-grid">\n')
    for v in active_versions:
        title, desc = PROMPT_EVOLUTION_SUMMARY.get(v, ("Prompt Variant", "No summary available."))
        html_parts.append(
            f'    <div class="prompt-summary-card" data-version="{v}">'
            f'<div class="pver">p{v}</div>'
            f'<div class="ptitle">{esc(title)}</div>'
            f'<div class="pdesc">{esc(desc)}</div>'
            f'</div>\n'
        )
    html_parts.append('  </div>\n')
    html_parts.append('</section>\n')

    # Prompt sections
    html_parts.append('<div id="prompt-sections">\n')
    for v in active_versions:
        prompt_text = all_prompts.get(v, "No prompt file found")
        # Only show sections unique to this version (skip few-shot placeholder area)
        display_text = prompt_text.split("{{FEW_SHOT_EXAMPLES}}")[0] if "{{FEW_SHOT_EXAMPLES}}" in prompt_text else prompt_text
        html_parts.append(f"""<div class="prompt-section" data-version="{v}">
  <div class="prompt-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.arrow').textContent = this.nextElementSibling.classList.contains('open') ? '▲' : '▼'">
    <span>📋 Prompt v{v} ({len(prompt_text)} chars)</span><span class="arrow">▼</span>
  </div>
  <div class="prompt-content">{esc(display_text)}</div>
</div>
""")
    html_parts.append('</div>\n')

    # Pipeline diagram
    html_parts.append("""
<div class="prompt-section">
  <div class="prompt-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.arrow').textContent = this.nextElementSibling.classList.contains('open') ? '▲' : '▼'" style="border-left: 4px solid var(--purple);">
    <span>Data Generation Pipeline</span><span class="arrow">▼</span>
  </div>
  <div class="prompt-content open" style="font-family: inherit; white-space: normal; padding: 0; max-height: none;">
    <svg viewBox="0 0 1100 520" xmlns="http://www.w3.org/2000/svg" style="width: 100%; background: var(--surface2); border-radius: 0 0 8px 8px;">
      <defs>
        <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#58a6ff"/></marker>
        <marker id="ah2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#3fb950"/></marker>
        <marker id="ah3" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#bc8cff"/></marker>
        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#1f6feb"/><stop offset="100%" stop-color="#58a6ff"/></linearGradient>
        <linearGradient id="g2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#238636"/><stop offset="100%" stop-color="#3fb950"/></linearGradient>
        <linearGradient id="g3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#8957e5"/><stop offset="100%" stop-color="#bc8cff"/></linearGradient>
        <linearGradient id="g4" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#9e6a03"/><stop offset="100%" stop-color="#d29922"/></linearGradient>
      </defs>

      <!-- Title -->
      <text x="550" y="30" text-anchor="middle" fill="#1f2328" font-size="16" font-weight="600" font-family="-apple-system, sans-serif">Config Generation &amp; Evaluation Pipeline</text>

      <!-- Row 1: Input Sources -->
      <text x="20" y="65" fill="#8b949e" font-size="11" font-family="-apple-system, sans-serif" font-weight="600">INPUT</text>

      <!-- Prompt Template -->
      <rect x="20" y="75" width="180" height="54" rx="8" fill="#161b22" stroke="#1f6feb" stroke-width="1.5"/>
      <text x="110" y="96" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">prompt_v{N}.txt</text>
      <text x="110" y="118" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Schema + Constraints + CoT</text>

      <!-- Cues -->
      <rect x="220" y="75" width="150" height="54" rx="8" fill="#161b22" stroke="#1f6feb" stroke-width="1.5"/>
      <text x="295" y="96" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">cues.yml</text>
      <text x="295" y="118" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">58 iconic + 58 contextual</text>

      <!-- Few-shot examples -->
      <rect x="390" y="75" width="170" height="54" rx="8" fill="#161b22" stroke="#1f6feb" stroke-width="1.5"/>
      <text x="475" y="96" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">shot_configs.json</text>
      <text x="475" y="118" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Handmade + Correction pairs</text>

      <!-- Arrows from inputs to orchestrator -->
      <line x1="110" y1="129" x2="110" y2="168" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="295" y1="129" x2="295" y2="168" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="475" y1="129" x2="475" y2="168" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#ah)"/>

      <!-- Row 2: Orchestration -->
      <text x="20" y="183" fill="#8b949e" font-size="11" font-family="-apple-system, sans-serif" font-weight="600">GENERATE</text>

      <!-- compare_prompts.py generate -->
      <rect x="20" y="190" width="230" height="50" rx="8" fill="#161b22" stroke="#3fb950" stroke-width="1.5"/>
      <text x="135" y="212" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">compare_prompts.py generate</text>
      <text x="135" y="230" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">--versions '[10,11,12]'</text>

      <!-- Arrow to config_gen_meta -->
      <line x1="250" y1="215" x2="280" y2="215" stroke="#3fb950" stroke-width="1.5" marker-end="url(#ah2)"/>

      <!-- config_gen_meta.py -->
      <rect x="282" y="190" width="200" height="50" rx="8" fill="#161b22" stroke="#3fb950" stroke-width="1.5"/>
      <text x="382" y="212" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">config_gen_meta.py</text>
      <text x="382" y="230" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">for each cue in cues.yml</text>

      <!-- Arrow to config_gen_single -->
      <line x1="482" y1="215" x2="512" y2="215" stroke="#3fb950" stroke-width="1.5" marker-end="url(#ah2)"/>

      <!-- config_gen_single.py -->
      <rect x="514" y="185" width="250" height="60" rx="8" fill="#21262d" stroke="#3fb950" stroke-width="2"/>
      <text x="639" y="207" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="700" font-family="-apple-system, sans-serif">config_gen_single.py</text>
      <text x="639" y="223" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Fill template → Gemini API → Parse</text>
      <text x="639" y="237" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Strip # CoT → Validate → Retry</text>

      <!-- Gemini API callout -->
      <rect x="800" y="190" width="140" height="50" rx="8" fill="#161b22" stroke="#d29922" stroke-width="1.5"/>
      <text x="870" y="212" text-anchor="middle" fill="#d29922" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">Gemini API</text>
      <text x="870" y="228" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">gemini-2.5-flash</text>
      <line x1="764" y1="210" x2="798" y2="210" stroke="#d29922" stroke-width="1.5" marker-end="url(#ah)"/>
      <line x1="798" y1="222" x2="764" y2="222" stroke="#d29922" stroke-width="1.5" stroke-dasharray="4" marker-end="url(#ah)"/>

      <!-- Row 3: Outputs -->
      <text x="20" y="290" fill="#8b949e" font-size="11" font-family="-apple-system, sans-serif" font-weight="600">OUTPUT</text>

      <!-- Arrow down from config_gen_single -->
      <line x1="639" y1="245" x2="639" y2="295" stroke="#3fb950" stroke-width="1.5" marker-end="url(#ah2)"/>

      <!-- Config JSON -->
      <rect x="510" y="297" width="260" height="54" rx="8" fill="#161b22" stroke="#3fb950" stroke-width="2"/>
      <text x="640" y="319" text-anchor="middle" fill="#3fb950" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">motion_configs_prompt_v{N}.json</text>
      <text x="640" y="339" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">58 configs: cue + movements + reasoning</text>

      <!-- Row 4: Rendering -->
      <text x="20" y="395" fill="#8b949e" font-size="11" font-family="-apple-system, sans-serif" font-weight="600">RENDER</text>

      <!-- Arrow down to render -->
      <line x1="640" y1="351" x2="640" y2="400" stroke="#bc8cff" stroke-width="1.5" marker-end="url(#ah3)"/>

      <!-- compare_prompts.py render -->
      <rect x="480" y="402" width="240" height="50" rx="8" fill="#161b22" stroke="#bc8cff" stroke-width="1.5"/>
      <text x="600" y="424" text-anchor="middle" fill="#bc8cff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">compare_prompts.py render</text>
      <text x="600" y="440" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">motion_generation.py → robosuite sim</text>

      <!-- Arrow to tiled GIFs -->
      <line x1="720" y1="427" x2="760" y2="427" stroke="#bc8cff" stroke-width="1.5" marker-end="url(#ah3)"/>

      <!-- Tiled GIFs output -->
      <rect x="762" y="402" width="200" height="50" rx="8" fill="#161b22" stroke="#bc8cff" stroke-width="1.5"/>
      <text x="862" y="424" text-anchor="middle" fill="#bc8cff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">v{N}/IIWA/*_tiled.gif</text>
      <text x="862" y="440" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Per-cue animated GIF</text>

      <!-- compare_prompts.py grid -->
      <rect x="200" y="402" width="240" height="50" rx="8" fill="#161b22" stroke="#bc8cff" stroke-width="1.5"/>
      <text x="320" y="424" text-anchor="middle" fill="#bc8cff" font-size="12" font-weight="600" font-family="-apple-system, sans-serif">compare_prompts.py grid</text>
      <text x="320" y="440" text-anchor="middle" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Multi-version grid comparison</text>

      <!-- Arrow from config to grid -->
      <path d="M510,330 L320,330 L320,400" fill="none" stroke="#bc8cff" stroke-width="1.5" marker-end="url(#ah3)"/>

      <!-- Arrow from grid to grid GIF -->
      <line x1="320" y1="452" x2="320" y2="480" stroke="#bc8cff" stroke-width="1.5" marker-end="url(#ah3)"/>
      <rect x="180" y="482" width="280" height="30" rx="6" fill="#161b22" stroke="#bc8cff" stroke-width="1"/>
      <text x="320" y="502" text-anchor="middle" fill="#bc8cff" font-size="11" font-family="-apple-system, sans-serif">grid_v10_v11_v12_v13_IIWA_c0-20.gif</text>

      <!-- Row 5: Dashboard -->
      <text x="20" y="485" fill="#8b949e" font-size="11" font-family="-apple-system, sans-serif" font-weight="600">VIEW</text>

      <!-- Arrow from tiled to dashboard -->
      <line x1="862" y1="452" x2="862" y2="480" stroke="#58a6ff" stroke-width="1.5" marker-end="url(#ah)"/>
      <rect x="740" y="482" width="240" height="30" rx="6" fill="#21262d" stroke="#58a6ff" stroke-width="2"/>
      <text x="860" y="502" text-anchor="middle" fill="#58a6ff" font-size="12" font-weight="700" font-family="-apple-system, sans-serif">dashboard (this page)</text>

      <!-- Dashed arrow from config to dashboard -->
      <path d="M770,330 L940,330 L940,480 L982,480" fill="none" stroke="#58a6ff" stroke-width="1" stroke-dasharray="4"/>

      <!-- Legend -->
      <rect x="20" y="310" width="12" height="12" rx="2" fill="#1f6feb"/>
      <text x="38" y="321" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Input files</text>
      <rect x="20" y="330" width="12" height="12" rx="2" fill="#3fb950"/>
      <text x="38" y="341" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Generation</text>
      <rect x="20" y="350" width="12" height="12" rx="2" fill="#bc8cff"/>
      <text x="38" y="361" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Rendering</text>
      <rect x="130" y="310" width="12" height="12" rx="2" fill="#d29922"/>
      <text x="148" y="321" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">External API</text>
      <rect x="130" y="330" width="12" height="12" rx="2" fill="#58a6ff"/>
      <text x="148" y="341" fill="#8b949e" font-size="10" font-family="-apple-system, sans-serif">Dashboard</text>
    </svg>
  </div>
</div>
""")

    # Grid GIF section
    if grid_gifs:
        html_parts.append('<div class="prompt-section">\n')
        html_parts.append('  <div class="prompt-toggle" onclick="this.nextElementSibling.classList.toggle(\'open\'); this.querySelector(\'.arrow\').textContent = this.nextElementSibling.classList.contains(\'open\') ? \'▲\' : \'▼\'">\n')
        html_parts.append(f'    <span>Grid Comparison GIFs ({len(grid_gifs)})</span><span class="arrow">▼</span>\n')
        html_parts.append('  </div>\n')
        html_parts.append('  <div class="prompt-content" style="font-family: inherit; white-space: normal;">\n')
        out_dir = os.path.dirname(output or os.path.join(MOTION_BASE, "dashboard.html"))
        for gf in grid_gifs:
            rel = os.path.relpath(os.path.join(MOTION_BASE, gf), out_dir)
            html_parts.append(f'    <a href="{esc(rel)}" style="color: var(--accent); display: block; margin: 4px 0;">{esc(gf)}</a>\n')
        html_parts.append('  </div>\n</div>\n')

    # Cue sections
    for idx in cue_idxs:
        name = cue_names[idx]
        html_parts.append(f'<div class="cue-section" id="cue-{idx}" data-cue-name="{esc(name.lower())}">\n')
        html_parts.append(f'  <div class="cue-header"><span class="idx">c{idx}</span>{esc(name)}</div>\n')

        # Legacy base-prompt GIF (from data/motions/{robot}/)
        base_gif = gif_map_base.get(idx)
        if base_gif:
            rel_gif = os.path.relpath(base_gif, os.path.dirname(output or os.path.join(MOTION_BASE, "dashboard.html")))
            html_parts.append(f'  <div class="gif-container" style="margin-bottom: 8px;">'
                              f'<div style="font-size:11px; color: var(--text2); margin-bottom:4px;">base (prompt.txt, gemini-3-flash-preview)</div>'
                              f'<img src="{esc(rel_gif)}" loading="lazy" alt="c{idx} base animation" style="max-height: 200px;"></div>\n')

        html_parts.append(f'  <div class="version-grid" style="grid-template-columns: repeat({n_versions}, 1fr);">\n')

        for v in active_versions:
            cfg = all_configs.get(v, {}).get(idx)
            model = cfg.get("model", "–") if cfg else "–"
            html_parts.append(f'  <div class="version-card" data-version="{v}">\n')
            html_parts.append(f'    <div class="card-header"><span class="vtag">v{v}</span><span class="model">{esc(model)}</span></div>\n')
            html_parts.append(f'    <div class="card-body">\n')

            # Per-version GIF
            v_gif = gif_map_version.get((idx, v))
            if v_gif:
                rel_gif = os.path.relpath(v_gif, os.path.dirname(output or os.path.join(MOTION_BASE, "dashboard.html")))
                html_parts.append(f'      <div class="gif-container"><img src="{esc(rel_gif)}" loading="lazy" alt="c{idx} v{v} animation"></div>\n')

            # Description
            desc = cfg.get("description", "") if cfg else ""
            if desc:
                html_parts.append(f'      <div class="description">{esc(desc)}</div>\n')

            # Step visualization
            if cfg:
                mvs = cfg.get("movements", [])
                html_parts.append('      <div class="step-viz">\n')
                for mi, m in enumerate(mvs):
                    mtype = m.get("type", "?")
                    pill_text = _step_pill_text(m)
                    if mi > 0:
                        html_parts.append('        <span class="step-arrow">→</span>\n')
                    html_parts.append(f'        <span class="step-pill {mtype}">{esc(pill_text)}</span>\n')
                html_parts.append('      </div>\n')

                # Quality badges
                score, details = _score_config(cfg)
                is_pose_only = _is_pose_only(cfg)
                step_count = _cue_steps(cfg)
                badge_class = "badge-bad" if is_pose_only or score < 3 else ("badge-warn" if score < 6 else "badge-good")
                html_parts.append(f'      <div style="margin: 4px 0;"><span class="badge {badge_class}">score: {score:.1f}</span> <span class="badge badge-good">{step_count} steps</span>')
                if is_pose_only:
                    html_parts.append(' <span class="badge badge-bad">pose-only</span>')
                if _has_path(cfg):
                    html_parts.append(' <span class="badge badge-good">has path</span>')
                html_parts.append('</div>\n')

            # Summary
            html_parts.append(f'      <div class="section-label">Summary</div>\n')
            html_parts.append(f'      <div class="summary-line">{esc(_config_summary(cfg))}</div>\n')

            # CoT reasoning
            html_parts.append(f'      <div class="section-label">Chain of Thought</div>\n')
            html_parts.append(f'      <div class="cot-block">{_reasoning_html(cfg)}</div>\n')

            # Config JSON
            html_parts.append(f'      <div class="section-label">Config JSON</div>\n')
            html_parts.append(f'      <div class="config-json collapsed" onclick="this.classList.toggle(\'collapsed\')">{_config_json_html(cfg)}</div>\n')

            html_parts.append('    </div>\n  </div>\n')

        html_parts.append('  </div>\n</div>\n')

    # JavaScript
    html_parts.append("""
<script>
function toggleVersion(v, el) {
  const label = el.closest('label');
  label.classList.toggle('active', el.checked);
  document.querySelectorAll(`[data-version="${v}"]`).forEach(card => {
    card.style.display = el.checked ? '' : 'none';
  });
  // Update grid columns
  const checked = document.querySelectorAll('.header input[type="checkbox"]:checked').length;
  document.querySelectorAll('.version-grid').forEach(grid => {
    grid.style.gridTemplateColumns = `repeat(${checked}, 1fr)`;
  });
}

function filterCues(query) {
  query = query.toLowerCase();
  document.querySelectorAll('.cue-section').forEach(section => {
    const name = section.dataset.cueName || '';
    const visible = !query || name.includes(query);
    section.style.display = visible ? '' : 'none';
  });
  document.querySelectorAll('.sidebar a').forEach(link => {
    const idx = link.dataset.idx;
    const section = document.getElementById(`cue-${idx}`);
    link.style.display = section && section.style.display !== 'none' ? '' : 'none';
  });
}

// Highlight active sidebar link on scroll
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
      const link = document.querySelector(`.sidebar a[href="#${entry.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { rootMargin: '-60px 0px -80% 0px' });

document.querySelectorAll('.cue-section').forEach(s => observer.observe(s));
</script>
</body>
</html>
""")

    # Write output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        vs_label = "_".join(f"v{v}" for v in active_versions)
        output = os.path.join(MOTION_BASE, f"dashboard_{vs_label}_{ts}.html")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\n  ✅ Dashboard saved: {output}")
    print(f"     {len(cue_idxs)} cues × {n_versions} versions\n")

    if serve:
        import http.server
        import functools
        port = 8765
        directory = os.path.dirname(os.path.abspath(output))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
        print(f"  Serving at http://localhost:{port}/{os.path.basename(output)}")
        with http.server.HTTPServer(("", port), handler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n  Server stopped.")


def robot_dashboard(
    version: int = 18,
    robots: list[str] | None = None,
    start_idx: int = 0,
    end_idx: int = 58,
    cue_idxs: list[int] | None = None,
    output: str | None = None,
):
    """Generate an interactive HTML dashboard for one prompt version on IIWA only."""
    import html as html_mod

    robots = ["IIWA"]
    robot = "IIWA"

    cfgs = _load_configs(version)
    if not cfgs:
        raise ValueError(f"No configs found for v{version}")
    configs_by_idx = {c.get("idx"): c for c in cfgs if c.get("idx") is not None}

    ppath = _prompt_path(version)
    prompt_text = ""
    if os.path.exists(ppath):
        with open(ppath, "r", encoding="utf-8") as f:
            prompt_text = f.read()

    requested_cue_idxs = list(range(start_idx, end_idx + 1)) if cue_idxs is None else list(cue_idxs)
    cue_idxs = [i for i in requested_cue_idxs if i in configs_by_idx]
    cue_names = {idx: configs_by_idx[idx].get("cue", f"cue_{idx}") for idx in cue_idxs}

    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(MOTION_BASE, f"dashboard_v{version}_{robot}_{ts}.html")
    tile_dir = os.path.join(os.path.dirname(output), "_dashboard_tiles", f"v{version}_{robot.lower()}")

    image_map = {}
    for idx in cue_idxs:
        cfg = configs_by_idx[idx]
        src_gif = _find_single_gif(version, robot, idx) or _find_tiled_gif(version, robot, idx)
        if not src_gif:
            continue
        image_path = _build_top1_checkpoint_tile(
            src_gif,
            cfg,
            os.path.join(tile_dir, f"c{idx:02d}_top1_checkpoints.png"),
        )
        if image_path:
            image_map[idx] = image_path

    def esc(s):
        return html_mod.escape(str(s))

    def _reasoning_html(cfg):
        if not cfg:
            return '<span class="na">N/A</span>'
        reasoning = cfg.get("reasoning", "")
        if not reasoning:
            return '<span class="na">No CoT saved</span>'
        lines = reasoning.strip().split("\n")
        parts = []
        for line in lines:
            line_clean = line.lstrip("# ").strip()
            if ":" in line_clean:
                key, val = line_clean.split(":", 1)
                parts.append(f'<span class="cot-key">{esc(key.strip())}:</span> {esc(val.strip())}')
            else:
                parts.append(esc(line_clean))
        return "<br>".join(parts)

    def _config_json_html(cfg):
        if not cfg:
            return '<span class="na">N/A</span>'
        return esc(json.dumps(cfg.get("movements", []), indent=2, ensure_ascii=False))

    primitive_stats = {
        "pose": 0,
        "movement": 0,
        "path": 0,
        "repetition": 0,
        "hold": 0,
        "multi-joint": 0,
    }
    for idx in cue_idxs:
        cfg = configs_by_idx[idx]
        movements = cfg.get("movements", [])
        joints = set()
        has_repetition = False
        has_hold = False
        for m in movements:
            mtype = m.get("type")
            if mtype in primitive_stats:
                primitive_stats[mtype] += 1
            params = m.get("parameters", {})
            joint = params.get("joint")
            if joint:
                joints.add(joint)
            if params.get("repetition", 1) not in (None, 1):
                has_repetition = True
            hold_time = params.get("hold_time")
            if hold_time not in (None, 0):
                has_hold = True
        if has_repetition:
            primitive_stats["repetition"] += 1
        if has_hold:
            primitive_stats["hold"] += 1
        if len(joints) >= 2:
            primitive_stats["multi-joint"] += 1

    prompt_display = prompt_text.split("{{FEW_SHOT_EXAMPLES}}")[0] if "{{FEW_SHOT_EXAMPLES}}" in prompt_text else prompt_text
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Robot Comparison Dashboard — v{version}</title>
<style>
:root {{
  --bg: #f6f8fb; --surface: #ffffff; --surface2: #eef2f7;
  --border: #d0d7de; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --warn: #9a6700;
  --red: #cf222e; --purple: #8250df;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); font-size: 18px; line-height: 1.5; }}
.header {{ position: sticky; top: 0; z-index: 100; background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 20px; }}
.header h1 {{ font-size: 20px; font-weight: 600; white-space: nowrap; }}
.filter-bar {{ display: flex; gap: 12px; align-items: center; margin-left: auto; }}
.controls {{ padding: 20px 24px 0; }}
.controls-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
.controls-bar {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
.controls-bar input {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 8px 10px; border-radius: 8px; font-size: 18px; width: 260px; }}
.controls-actions {{ display: flex; gap: 8px; }}
.controls-actions button {{ background: var(--surface2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 17px; cursor: pointer; }}
.controls-actions button:hover {{ border-color: var(--accent); color: var(--accent); }}
.cue-filter-list {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 8px; }}
.cue-filter-item {{ display: block; }}
.cue-filter-row {{ display: grid; grid-template-columns: 18px 1fr; gap: 8px; align-items: start; cursor: pointer; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface2); min-height: 100%; }}
.cue-filter-item input[type="checkbox"] {{ margin-top: 2px; accent-color: var(--accent); }}
.cue-filter-item.disabled {{ opacity: 0.45; }}
.cue-filter-item.selected .cue-filter-row {{ border-color: var(--accent); background: #dbeafe; }}
.cue-filter-item.in-view .cue-filter-row {{ box-shadow: inset 0 0 0 1px rgba(9,105,218,0.35); }}
.cue-filter-label {{ display: block; color: var(--text2); font-size: 18px; line-height: 1.35; }}
.main {{ padding: 20px 20px 32px; max-width: 1800px; margin: 0 auto; width: 100%; }}
.intro {{ margin-bottom: 20px; }}
.intro h2 {{ font-size: 23px; margin-bottom: 8px; }}
.intro p {{ color: var(--text2); max-width: 980px; margin-bottom: 12px; font-size: 19px; }}
.chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 17px; background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }}
.cue-stack {{ display: flex; flex-direction: row; align-items: flex-start; gap: 24px; overflow-x: auto; overflow-y: hidden; padding-bottom: 8px; }}
.cue-section {{ margin: 0; scroll-margin-top: 60px; width: min(340px, 72vw); min-width: min(340px, 72vw); display: block; flex: 0 0 auto; }}
.cue-header {{ font-size: 18px; font-weight: 600; margin-bottom: 12px; padding: 8px 12px; background: var(--surface); border-radius: 8px; border-left: 4px solid var(--accent); }}
.cue-header .idx {{ color: var(--accent); margin-right: 8px; }}
.robot-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.robot-card .card-header {{ padding: 8px 12px; background: var(--surface2); font-weight: 600; font-size: 18px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
.robot-card .card-header .rtag {{ color: var(--accent); }}
.card-body {{ padding: 10px 12px; }}
.section-label {{ font-size: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); margin: 8px 0 4px; }}
.section-label:first-child {{ margin-top: 0; }}
.cot-block {{ font-size: 15px; padding: 8px; background: var(--surface2); border-radius: 4px; border-left: 3px solid var(--purple); white-space: pre-wrap; overflow: visible; }}
.cot-key {{ color: var(--purple); font-weight: 600; }}
.config-json {{ font-size: 16px; font-family: 'SF Mono', 'Fira Code', monospace; background: var(--surface2); padding: 8px; border-radius: 4px; white-space: pre-wrap; overflow: visible; }}
.na {{ color: var(--text2); font-style: italic; font-size: 17px; }}
.gif-container {{ margin-top: 6px; text-align: center; }}
.gif-container img {{ max-width: 100%; border-radius: 4px; border: 1px solid var(--border); display: block; background: var(--surface2); }}
.step-viz {{ display: flex; gap: 4px; align-items: center; flex-wrap: wrap; margin: 4px 0; }}
.step-pill {{ padding: 2px 8px; border-radius: 12px; font-size: 19px; font-weight: 500; }}
.step-pill.pose {{ background: #1f6feb33; color: var(--accent); }}
.step-pill.movement {{ background: #2ea04333; color: var(--accent2); }}
.step-pill.path {{ background: #bc8cff33; color: var(--purple); }}
.step-arrow {{ color: var(--text2); font-size: 18px; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 14px; font-weight: 500; }}
.badge-good {{ background: #2ea04326; color: var(--accent2); }}
.badge-warn {{ background: #d2992226; color: var(--warn); }}
.badge-bad {{ background: #f8514926; color: var(--red); }}
.prompt-section {{ margin-bottom: 24px; }}
.prompt-toggle {{ cursor: pointer; padding: 10px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; font-size: 19px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; }}
.prompt-content {{ display: none; background: var(--surface); border: 1px solid var(--border); border-top: none; border-radius: 0 0 8px 8px; padding: 16px; font-size: 17px; font-family: 'SF Mono', 'Fira Code', monospace; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }}
.prompt-content.open {{ display: block; }}
@media (max-width: 900px) {{
  .header {{ padding: 12px 16px; flex-wrap: wrap; }}
  .controls {{ padding: 16px 16px 0; }}
  .main {{ padding: 16px; }}
  .controls-bar input {{ width: 100%; }}
  .cue-filter-list {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class="header">
  <h1>Robot Comparison Dashboard</h1>
  <div class="filter-bar">
    <span class="chip" id="selectedCueCount">selected: 0</span>
  </div>
</div>
<section class="controls">
  <div class="controls-card">
    <div class="controls-bar">
      <input type="text" id="searchInput" placeholder="Search cues..." oninput="applyCueFilters()">
      <div class="controls-actions">
      <button type="button" onclick="setAllCueFilters(true)">All</button>
      <button type="button" onclick="setAllCueFilters(false)">None</button>
      </div>
    </div>
    <div class="cue-filter-list">
""")
    for idx in requested_cue_idxs:
        has_cfg = idx in cue_names
        name = cue_names.get(idx, f"cue_{idx} (missing)")
        short = name[:30] + ".." if len(name) > 32 else name
        disabled = ' disabled' if not has_cfg else ''
        checked = ' checked' if has_cfg else ''
        item_class = "cue-filter-item" if has_cfg else "cue-filter-item disabled"
        html_parts.append(f'  <div class="{item_class}" data-filter-idx="{idx}" data-cue-label="{esc(name.lower())}">\n')
        html_parts.append(f'    <label class="cue-filter-row"><input type="checkbox" class="cue-filter-checkbox" data-idx="{idx}" onchange="applyCueFilters()"{checked}{disabled}><span class="cue-filter-label">c{idx}: {esc(short)}</span></label>\n')
        html_parts.append('  </div>\n')
    html_parts.append('    </div>\n  </div>\n</section>\n<div class="main">\n')
    html_parts.append('<section class="intro">\n')
    html_parts.append('  <h2>IIWA Cue Browser</h2>\n')
    html_parts.append('  <p>Select multiple cues above. Checked cues are rendered side-by-side in a horizontal strip below.</p>\n')
    html_parts.append('  <div class="chip-row">\n')
    html_parts.append(f'    <span class="chip">cues: {len(cue_idxs)}</span>\n')
    html_parts.append('  </div>\n')
    html_parts.append('  <div class="chip-row">\n')
    html_parts.append(f'    <span class="chip">pose: {primitive_stats["pose"]}</span>\n')
    html_parts.append(f'    <span class="chip">movement: {primitive_stats["movement"]}</span>\n')
    html_parts.append(f'    <span class="chip">path: {primitive_stats["path"]}</span>\n')
    html_parts.append(f'    <span class="chip">repetition cues: {primitive_stats["repetition"]}</span>\n')
    html_parts.append(f'    <span class="chip">hold cues: {primitive_stats["hold"]}</span>\n')
    html_parts.append(f'    <span class="chip">multi-joint cues: {primitive_stats["multi-joint"]}</span>\n')
    html_parts.append('  </div>\n')
    html_parts.append('</section>\n')
    html_parts.append(f"""<div class="prompt-section">
  <div class="prompt-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.arrow').textContent = this.nextElementSibling.classList.contains('open') ? '▲' : '▼'">
    <span>📋 Prompt ({len(prompt_text)} chars)</span><span class="arrow">▼</span>
  </div>
  <div class="prompt-content">{esc(prompt_display or 'No prompt file found')}</div>
</div>
""")
    html_parts.append('<div class="cue-stack">\n')

    for idx in cue_idxs:
        name = cue_names[idx]
        cfg = configs_by_idx[idx]
        html_parts.append(f'<div class="cue-section" id="cue-{idx}" data-idx="{idx}" data-cue-name="{esc(name.lower())}">\n')
        html_parts.append(f'  <div class="cue-header"><span class="idx">c{idx}</span>{esc(name)}</div>\n')
        html_parts.append(f'  <div class="robot-card" data-robot="{esc(robot)}">\n')
        html_parts.append('    <div class="card-header"></div>\n')
        html_parts.append('    <div class="card-body">\n')
        image_path = image_map.get(idx)
        if image_path:
            rel_image = os.path.relpath(image_path, os.path.dirname(output or os.path.join(MOTION_BASE, "dashboard.html")))
            html_parts.append(f'      <div class="gif-container"><img src="{esc(rel_image)}" loading="lazy" alt="c{idx} {esc(robot)} top1 5-frame tile"></div>\n')
        else:
            html_parts.append('      <div class="gif-container"><span class="na">No render found</span></div>\n')
        mvs = cfg.get("movements", [])
        html_parts.append('      <div class="step-viz">\n')
        for mi, m in enumerate(mvs):
            mtype = m.get("type", "?")
            pill_text = _step_pill_text(m)
            if mi > 0:
                html_parts.append('        <span class="step-arrow">→</span>\n')
            html_parts.append(f'        <span class="step-pill {mtype}">{esc(pill_text)}</span>\n')
        html_parts.append('      </div>\n')
        html_parts.append('      <div class="section-label">Chain of Thought</div>\n')
        html_parts.append(f'      <div class="cot-block">{_reasoning_html(cfg)}</div>\n')
        html_parts.append('      <div class="section-label">Config JSON</div>\n')
        html_parts.append(f'      <div class="config-json">{_config_json_html(cfg)}</div>\n')
        html_parts.append('    </div>\n  </div>\n</div>\n')

    html_parts.append('</div>\n')

    html_parts.append("""
<script>
function syncCueSelectionState() {
  document.querySelectorAll('.cue-filter-item').forEach(item => {
    const checkbox = item.querySelector('.cue-filter-checkbox');
    item.classList.toggle('selected', !!checkbox?.checked);
  });
  const selectedCount = document.querySelectorAll('.cue-filter-checkbox:checked').length;
  const chip = document.getElementById('selectedCueCount');
  if (chip) chip.textContent = `selected: ${selectedCount}`;
}

function applyCueFilters() {
  const query = (document.getElementById('searchInput')?.value || '').toLowerCase();
  const selected = new Set(
    Array.from(document.querySelectorAll('.cue-filter-checkbox:checked'))
      .map(el => Number(el.dataset.idx))
  );
  document.querySelectorAll('.cue-section').forEach(section => {
    const name = section.dataset.cueName || '';
    const idx = Number(section.dataset.idx);
    const visible = selected.has(idx) && (!query || name.includes(query) || `c${idx}`.includes(query));
    section.style.display = visible ? '' : 'none';
  });
  document.querySelectorAll('.cue-filter-item').forEach(item => {
    const idx = Number(item.dataset.filterIdx);
    const label = item.dataset.cueLabel || '';
    const visible = !query || label.includes(query) || `c${idx}`.includes(query);
    item.style.display = visible ? '' : 'none';
  });
  syncCueSelectionState();
}

function setAllCueFilters(checked) {
  document.querySelectorAll('.cue-filter-checkbox:not(:disabled)').forEach(el => {
    el.checked = checked;
  });
  applyCueFilters();
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.cue-filter-item').forEach(item => item.classList.remove('in-view'));
      const idx = entry.target.dataset.idx;
      const item = document.querySelector(`.cue-filter-item[data-filter-idx="${idx}"]`);
      if (item) item.classList.add('in-view');
    }
  });
}, { rootMargin: '-60px 0px -80% 0px' });

document.querySelectorAll('.cue-section').forEach(s => observer.observe(s));
applyCueFilters();
</script>
</body>
</html>
""")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\n  ✅ Robot dashboard saved: {output}")
    print(f"     {len(cue_idxs)} cues × {len(robots)} robots\n")


def contextual_dashboard(
    version: int = 18,
    robot: str = "IIWA",
    start_idx: int = 0,
    end_idx: int = 57,
    cue_idxs: list[int] | None = None,
    output: str | None = None,
):
    """Generate an interactive HTML dashboard for the contextual cue set."""
    import html as html_mod

    config_path = os.path.join(SEED_DIR, f"motion_configs_prompt_v{version}_contextual.json")
    prompt_path = _prompt_path(version)
    motion_dir = os.path.join(MOTION_BASE, f"v{version}_contextual", robot)

    if not os.path.exists(config_path):
        raise ValueError(f"Contextual config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)
    configs_by_idx = {c.get("idx"): c for c in cfgs if c.get("idx") is not None}

    prompt_text = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_text = f.read()

    if cue_idxs is None:
        cue_idxs = list(range(start_idx, end_idx + 1))
    cue_idxs = [i for i in cue_idxs if i in configs_by_idx]
    cue_names = {idx: configs_by_idx[idx].get("cue", f"cue_{idx}") for idx in cue_idxs}

    def esc(s):
        return html_mod.escape(str(s))

    def _config_summary(cfg):
        if not cfg:
            return "N/A"
        mvs = cfg.get("movements", [])
        types = [m.get("type", "?") for m in mvs]
        pattern = " -> ".join(types)
        joints = set()
        for m in mvs:
            j = m.get("parameters", {}).get("joint", "")
            if j:
                joints.add(j)
        speeds = _speed_values(cfg)
        speed_range = f"{min(speeds):.1f}-{max(speeds):.1f}" if speeds else "-"
        return f"{pattern} | joints: {', '.join(joints) or '-'} | speeds: {speed_range}"

    def _reasoning_html(cfg):
        reasoning = cfg.get("reasoning", "") if cfg else ""
        if not reasoning:
            return '<span class="na">No CoT saved</span>'
        lines = reasoning.strip().split("\n")
        parts = []
        for line in lines:
            line_clean = line.lstrip("# ").strip()
            if ":" in line_clean:
                key, val = line_clean.split(":", 1)
                parts.append(f'<span class="cot-key">{esc(key.strip())}:</span> {esc(val.strip())}')
            else:
                parts.append(esc(line_clean))
        return "<br>".join(parts)

    def _config_json_html(cfg):
        display = {k: v for k, v in cfg.items()
                   if k not in ("idx", "state", "model", "time", "reasoning", "validation_warnings")}
        return esc(json.dumps(display, indent=2, ensure_ascii=False))

    prompt_display = prompt_text.split("{{FEW_SHOT_EXAMPLES}}")[0] if "{{FEW_SHOT_EXAMPLES}}" in prompt_text else prompt_text
    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contextual Motion Dashboard — v{version} / {robot}</title>
<style>
:root {{
  --bg: #f6f8fb; --surface: #ffffff; --surface2: #eef2f7;
  --border: #d0d7de; --text: #1f2328; --text2: #59636e;
  --accent: #0969da; --accent2: #1a7f37; --warn: #9a6700;
  --red: #cf222e; --purple: #8250df;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, 'SF Pro Text', 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }}
.header {{ position: sticky; top: 0; z-index: 100; background: var(--surface); border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; align-items: center; gap: 20px; }}
.header h1 {{ font-size: 18px; font-weight: 600; white-space: nowrap; }}
.filter-bar {{ display: flex; gap: 12px; align-items: center; margin-left: auto; }}
.filter-bar input {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 5px 10px; border-radius: 6px; font-size: 13px; width: 240px; }}
.sidebar {{ position: fixed; left: 0; top: 53px; bottom: 0; width: 240px; background: var(--surface); border-right: 1px solid var(--border); overflow-y: auto; padding: 8px 0; }}
.sidebar a {{ display: block; padding: 6px 16px; color: var(--text2); text-decoration: none; font-size: 13px; border-left: 3px solid transparent; }}
.sidebar a:hover {{ background: var(--surface2); color: var(--text); }}
.sidebar a.active {{ border-left-color: var(--accent); color: var(--accent); background: #dbeafe; }}
.main {{ margin-left: 240px; padding: 20px 24px; }}
.intro {{ margin-bottom: 20px; }}
.intro h2 {{ font-size: 18px; margin-bottom: 8px; }}
.intro p {{ color: var(--text2); max-width: 980px; margin-bottom: 12px; }}
.chip-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; background: var(--surface2); color: var(--text2); border: 1px solid var(--border); }}
.cue-section {{ margin-bottom: 32px; scroll-margin-top: 60px; }}
.cue-header {{ font-size: 16px; font-weight: 600; margin-bottom: 12px; padding: 8px 12px; background: var(--surface); border-radius: 8px; border-left: 4px solid var(--accent); }}
.cue-header .idx {{ color: var(--accent); margin-right: 8px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }}
.card-header {{ padding: 8px 12px; background: var(--surface2); font-weight: 600; font-size: 13px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }}
.card-header .meta {{ color: var(--text2); font-weight: 400; font-size: 12px; }}
.card-body {{ padding: 10px 12px; }}
.gif-container {{ margin-top: 6px; text-align: center; }}
.gif-container img {{ max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }}
.description {{ font-size: 12px; color: var(--text2); margin: 4px 0 8px; font-style: italic; }}
.step-viz {{ display: flex; gap: 4px; align-items: center; flex-wrap: wrap; margin: 4px 0; }}
.step-pill {{ padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; }}
.step-pill.pose {{ background: #1f6feb33; color: var(--accent); }}
.step-pill.movement {{ background: #2ea04333; color: var(--accent2); }}
.step-pill.path {{ background: #bc8cff33; color: var(--purple); }}
.step-arrow {{ color: var(--text2); font-size: 10px; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
.badge-good {{ background: #2ea04326; color: var(--accent2); }}
.badge-warn {{ background: #d2992226; color: var(--warn); }}
.badge-bad {{ background: #f8514926; color: var(--red); }}
.section-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text2); margin: 8px 0 4px; }}
.summary-line {{ font-size: 12px; color: var(--text2); padding: 4px 8px; background: var(--surface2); border-radius: 4px; font-family: 'SF Mono', 'Fira Code', monospace; word-break: break-all; }}
.cot-block {{ font-size: 12px; padding: 8px; background: var(--surface2); border-radius: 4px; border-left: 3px solid var(--purple); height: 96px; min-height: 72px; max-height: 280px; overflow: auto; resize: vertical; }}
.cot-key {{ color: var(--purple); font-weight: 600; }}
.config-json {{ font-size: 11px; font-family: 'SF Mono', 'Fira Code', monospace; background: var(--surface2); padding: 8px; border-radius: 4px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; cursor: pointer; }}
.config-json.collapsed {{ max-height: 80px; position: relative; }}
.config-json.collapsed::after {{ content: '▼ click to expand'; position: absolute; bottom: 0; left: 0; right: 0; text-align: center; padding: 4px; background: linear-gradient(transparent, var(--bg)); color: var(--text2); font-size: 11px; }}
.prompt-section {{ margin-bottom: 24px; }}
.prompt-toggle {{ cursor: pointer; padding: 10px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; font-size: 14px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; }}
.prompt-content {{ display: none; background: var(--surface); border: 1px solid var(--border); border-top: none; border-radius: 0 0 8px 8px; padding: 16px; font-size: 12px; font-family: 'SF Mono', 'Fira Code', monospace; white-space: pre-wrap; max-height: 500px; overflow-y: auto; }}
.prompt-content.open {{ display: block; }}
</style>
</head>
<body>
<div class="header">
  <h1>Contextual Motion Dashboard</h1>
  <div class="filter-bar">
    <input type="text" id="searchInput" placeholder="Search cues..." oninput="filterCues(this.value)">
  </div>
</div>
<div class="sidebar">
""")
    for idx in cue_idxs:
        name = cue_names[idx]
        short = name[:30] + ".." if len(name) > 32 else name
        html_parts.append(f'  <a href="#cue-{idx}" data-idx="{idx}">c{idx}: {esc(short)}</a>\n')
    html_parts.append('</div>\n<div class="main">\n')
    html_parts.append('<section class="intro">\n')
    html_parts.append(f'  <h2>v{version} Contextual Cues</h2>\n')
    html_parts.append('  <p>This page shows the contextual cue set generated with prompt v18. Unlike the iconic set, these cues are more interactional and situational, so the descriptions are intentionally shorter and more compositional.</p>\n')
    html_parts.append('  <div class="chip-row">\n')
    html_parts.append(f'    <span class="chip">version: v{version}</span>\n')
    html_parts.append('    <span class="chip">cue group: contextual</span>\n')
    html_parts.append(f'    <span class="chip">robot: {esc(robot)}</span>\n')
    html_parts.append(f'    <span class="chip">cues: {len(cue_idxs)}</span>\n')
    html_parts.append('  </div>\n')
    html_parts.append('</section>\n')
    html_parts.append(f"""<div class="prompt-section">
  <div class="prompt-toggle" onclick="this.nextElementSibling.classList.toggle('open'); this.querySelector('.arrow').textContent = this.nextElementSibling.classList.contains('open') ? '▲' : '▼'">
    <span>📋 Prompt v{version} ({len(prompt_text)} chars)</span><span class="arrow">▼</span>
  </div>
  <div class="prompt-content">{esc(prompt_display or 'No prompt file found')}</div>
</div>
""")

    for idx in cue_idxs:
        cfg = configs_by_idx[idx]
        cue_name = cue_names[idx]
        model = cfg.get("model", "–")
        desc = cfg.get("description", "")
        matches = sorted(globmod.glob(os.path.join(motion_dir, f"*_c{idx}_tiled.gif")))
        gif = matches[-1] if matches else None
        html_parts.append(f'<div class="cue-section" id="cue-{idx}" data-cue-name="{esc(cue_name.lower())}">\n')
        html_parts.append(f'  <div class="cue-header"><span class="idx">c{idx}</span>{esc(cue_name)}</div>\n')
        html_parts.append('  <div class="card">\n')
        html_parts.append(f'    <div class="card-header"><span>{esc(robot)}</span><span class="meta">v{version} · {esc(model)}</span></div>\n')
        html_parts.append('    <div class="card-body">\n')
        if gif:
            rel_gif = os.path.relpath(gif, os.path.dirname(output or os.path.join(MOTION_BASE, "dashboard.html")))
            html_parts.append(f'      <div class="gif-container"><img src="{esc(rel_gif)}" loading="lazy" alt="c{idx} animation"></div>\n')
        if desc:
            html_parts.append(f'      <div class="description">{esc(desc)}</div>\n')
        mvs = cfg.get("movements", [])
        html_parts.append('      <div class="step-viz">\n')
        for mi, m in enumerate(mvs):
            mtype = m.get("type", "?")
            pill_text = _step_pill_text(m)
            if mi > 0:
                html_parts.append('        <span class="step-arrow">→</span>\n')
            html_parts.append(f'        <span class="step-pill {mtype}">{esc(pill_text)}</span>\n')
        html_parts.append('      </div>\n')
        score, _details = _score_config(cfg)
        is_pose_only = _is_pose_only(cfg)
        step_count = _cue_steps(cfg)
        badge_class = "badge-bad" if is_pose_only or score < 3 else ("badge-warn" if score < 6 else "badge-good")
        html_parts.append(f'      <div style="margin: 4px 0;"><span class="badge {badge_class}">score: {score:.1f}</span> <span class="badge badge-good">{step_count} steps</span>')
        if _has_path(cfg):
            html_parts.append(' <span class="badge badge-good">has path</span>')
        html_parts.append('</div>\n')
        html_parts.append('      <div class="section-label">Summary</div>\n')
        html_parts.append(f'      <div class="summary-line">{esc(_config_summary(cfg))}</div>\n')
        html_parts.append('      <div class="section-label">Chain of Thought</div>\n')
        html_parts.append(f'      <div class="cot-block">{_reasoning_html(cfg)}</div>\n')
        html_parts.append('      <div class="section-label">Config JSON</div>\n')
        html_parts.append(f'      <div class="config-json collapsed" onclick="this.classList.toggle(\'collapsed\')">{_config_json_html(cfg)}</div>\n')
        html_parts.append('    </div>\n  </div>\n</div>\n')

    html_parts.append("""
<script>
function filterCues(query) {
  query = query.toLowerCase();
  document.querySelectorAll('.cue-section').forEach(section => {
    const name = section.dataset.cueName || '';
    const visible = !query || name.includes(query);
    section.style.display = visible ? '' : 'none';
  });
  document.querySelectorAll('.sidebar a').forEach(link => {
    const idx = link.dataset.idx;
    const section = document.getElementById(`cue-${idx}`);
    link.style.display = section && section.style.display !== 'none' ? '' : 'none';
  });
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.sidebar a').forEach(a => a.classList.remove('active'));
      const link = document.querySelector(`.sidebar a[href="#${entry.target.id}"]`);
      if (link) link.classList.add('active');
    }
  });
}, { rootMargin: '-60px 0px -80% 0px' });

document.querySelectorAll('.cue-section').forEach(s => observer.observe(s));
</script>
</body>
</html>
""")

    if output is None:
        output = os.path.join(MOTION_BASE, f"dashboard_v{version}_contextual_{robot}.html")

    with open(output, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"\n  ✅ Contextual dashboard saved: {output}")
    print(f"     {len(cue_idxs)} cues × 1 robot\n")


if __name__ == "__main__":
    fire.Fire({
        "summary": summary,
        "cues": cues,
        "diff": diff,
        "detail": detail,
        "grid": grid,
        "render": render,
        "view": view,
        "generate": generate,
        "canonicalize": canonicalize,
        "status": status,
        "bestof": bestof,
        "dashboard": dashboard,
        "robot_dashboard": robot_dashboard,
        "contextual_dashboard": contextual_dashboard,
        "top10_html": top10_html,
    })
