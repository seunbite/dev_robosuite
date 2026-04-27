"""Analyze and compare motion config quality across prompt versions."""
import json
import fire
import glob
import os
from collections import Counter


def analyze(path: str):
    """Analyze a single config file."""
    data = json.load(open(path))
    name = os.path.basename(path).replace("motion_configs_", "").replace(".json", "")

    pose_only = [c for c in data if all(m["type"] == "pose" for m in c.get("movements", []))]
    step_counts = Counter(len(c.get("movements", [])) for c in data)
    type_counts = Counter()
    for c in data:
        for m in c.get("movements", []):
            type_counts[m["type"]] += 1

    patterns = Counter(
        tuple(m["type"] for m in c.get("movements", [])) for c in data
    )

    extreme = 0
    for c in data:
        for m in c.get("movements", []):
            if m["type"] == "movement":
                for d in m.get("parameters", {}).get("directions", []):
                    for _, v in d.get("degrees", {}).items():
                        if abs(v) > 50:
                            extreme += 1

    uniform_speed = 0
    for c in data:
        speeds = []
        for m in c.get("movements", []):
            p = m.get("parameters", {})
            if "speed" in p:
                speeds.append(p["speed"])
            for d in p.get("directions", []):
                if "speed" in d:
                    speeds.append(d["speed"])
        if len(speeds) > 2 and len(set(speeds)) == 1:
            uniform_speed += 1

    n = len(data)
    two_step_pct = 100 * step_counts.get(2, 0) / n if n else 0
    three_plus_pct = 100 * sum(v for k, v in step_counts.items() if k >= 3) / n if n else 0

    print(f"\n{'='*60}")
    print(f"  {name}  ({n} configs)")
    print(f"{'='*60}")
    print(f"  Pose-only:      {len(pose_only)}/{n} ({100*len(pose_only)/n:.0f}%)")
    print(f"  2-step:         {step_counts.get(2,0)}/{n} ({two_step_pct:.0f}%)")
    print(f"  3+ steps:       {sum(v for k,v in step_counts.items() if k>=3)}/{n} ({three_plus_pct:.0f}%)")
    print(f"  Step dist:      {dict(sorted(step_counts.items()))}")
    print(f"  Types:          {dict(type_counts)}")
    print(f"  Path usage:     {type_counts.get('path', 0)}")
    print(f"  Extreme (>50°): {extreme}")
    print(f"  Uniform speed:  {uniform_speed}")
    print(f"  Top patterns:")
    for p, c in patterns.most_common(5):
        print(f"    {' → '.join(p)}: {c}")

    return {
        "name": name, "n": n,
        "pose_only": len(pose_only),
        "two_step_pct": two_step_pct,
        "three_plus_pct": three_plus_pct,
        "paths": type_counts.get("path", 0),
        "extreme": extreme,
        "uniform_speed": uniform_speed,
    }


def compare(*paths: str, directory: str = None):
    """Compare multiple config files side by side.
    
    Usage:
      python analyze_configs.py compare path1.json path2.json
      python analyze_configs.py compare --directory data/seed/
    """
    if directory:
        paths = sorted(glob.glob(os.path.join(directory, "motion_configs_prompt_v*.json")))
    if not paths:
        print("No files to analyze.")
        return

    results = []
    for p in paths:
        if os.path.exists(p):
            results.append(analyze(p))

    if len(results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"{'Metric':<20}" + "".join(f"{r['name']:>14}" for r in results)
        print(header)
        print("-" * len(header))
        for metric, key in [
            ("Pose-only", "pose_only"),
            ("2-step %", "two_step_pct"),
            ("3+ step %", "three_plus_pct"),
            ("Path count", "paths"),
            ("Extreme deg", "extreme"),
            ("Uniform spd", "uniform_speed"),
        ]:
            row = f"{metric:<20}"
            for r in results:
                val = r[key]
                row += f"{val:>14.1f}" if isinstance(val, float) else f"{val:>14}"
            print(row)


if __name__ == "__main__":
    fire.Fire({"analyze": analyze, "compare": compare})
