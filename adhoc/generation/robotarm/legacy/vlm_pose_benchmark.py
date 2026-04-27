"""
VLM Pose Recognition Benchmark

Tests Gemini Vision's ability to understand robot arm poses from rendered frames.

Commands:
    main        — Original benchmark (static GIF tile)
    trajectory  — Trajectory-overlay benchmark (re-simulate + draw path on first frame)
    report      — Generate HTML report from results

Usage:
    python vlm_pose_benchmark.py main --n_cues 10
    python vlm_pose_benchmark.py trajectory --n_cues 10 --mode both
    python vlm_pose_benchmark.py report
"""

import fire
import json
import os
import sys
import time
import random
import glob as globmod
import base64
from io import BytesIO
from collections import Counter
from PIL import Image, ImageDraw
import numpy as np

MC_MANIP = "data/results/motion_configs/manipulator"
MOTION_DIR = "data/motions"


def _load_configs(version: int) -> list[dict]:
    path = os.path.join(MC_MANIP, f"motion_configs_prompt_v{version}.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path))


def _extract_tile(gif_path: str, tile_idx: int = 0, tile_w: int = 512) -> Image.Image:
    """Extract a single tile from a tiled GIF (first frame)."""
    gif = Image.open(gif_path)
    h = gif.size[1]
    # label_h is 24px at the bottom of each row
    tile_h = min(h, tile_w + 24)
    actual_tile_h = tile_w  # image part only (no label)
    left = tile_idx * tile_w
    tile = gif.crop((left, 0, left + tile_w, actual_tile_h)).convert("RGB")
    gif.close()
    return tile


def _find_gif(robot: str, cue_idx: int) -> str | None:
    patterns = [
        os.path.join(MOTION_DIR, robot, f"*_c{cue_idx}_tiled.gif"),
        os.path.join(MOTION_DIR, "v*", robot, f"*_c{cue_idx}_tiled.gif"),
    ]
    matches = []
    for pat in patterns:
        matches.extend(globmod.glob(pat))
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    return None


def _get_gt_info(cfg: dict) -> dict:
    """Extract ground truth from a config."""
    gt = {"cue": cfg.get("cue", ""), "idx": cfg.get("idx")}
    for m in cfg.get("movements", []):
        if m.get("type") == "pose":
            pose = m["parameters"]["pose"]
            if isinstance(pose, dict):
                gt["dir"] = pose.get("dir", "unknown")
                gt["gripper"] = pose.get("gripper_orientation", "unknown")
                gt["height"] = pose.get("height", "unknown")
                gt["x"] = pose.get("x")
                gt["y"] = pose.get("y")
                gt["z"] = pose.get("z")
            break
    return gt


# ── Tests ────────────────────────────────────────────────────────────────────

def _test_cue_recognition(client, model: str, image: Image.Image, gt_cue: str, all_cues: list[str]) -> dict:
    """Multiple choice: which gesture is the robot performing?"""
    distractors = [c for c in all_cues if c != gt_cue]
    random.shuffle(distractors)
    options = [gt_cue] + distractors[:4]
    random.shuffle(options)
    correct_letter = chr(65 + options.index(gt_cue))

    option_text = "\n".join(f"  {chr(65+i)}. {opt}" for i, opt in enumerate(options))
    prompt = (
        "You see a robot arm in a simulated environment. "
        "Which of the following gestures best matches the robot's current pose and movement?\n\n"
        f"{option_text}\n\n"
        "Reply with ONLY the letter (A, B, C, D, or E)."
    )

    try:
        response = client.models.generate_content(model=model, contents=[prompt, image])
        answer = response.text.strip().upper()
        # Extract just the letter
        for ch in answer:
            if ch in "ABCDE":
                answer = ch
                break
        correct = answer == correct_letter
    except Exception as e:
        answer = f"ERROR: {e}"
        correct = False

    return {
        "test": "cue_recognition",
        "gt_cue": gt_cue,
        "gt_letter": correct_letter,
        "answer": answer,
        "correct": correct,
        "options": {chr(65+i): opt for i, opt in enumerate(options)},
        "chosen_text": options[ord(answer) - 65] if len(answer) == 1 and answer in "ABCDE" else answer,
    }


def _test_arm_direction(client, model: str, image: Image.Image, gt_dir: str) -> dict:
    """Multiple choice: which direction is the arm pointing?"""
    directions = ["up", "down", "left", "right", "front"]
    if gt_dir not in directions:
        gt_dir_mapped = gt_dir
        if gt_dir in ("forward", "front"):
            gt_dir_mapped = "front"
        elif gt_dir in ("backward", "back"):
            gt_dir_mapped = "down"
        else:
            gt_dir_mapped = gt_dir
    else:
        gt_dir_mapped = gt_dir

    option_text = "\n".join(f"  {chr(65+i)}. {d}" for i, d in enumerate(directions))
    gt_letter = chr(65 + directions.index(gt_dir_mapped)) if gt_dir_mapped in directions else "?"

    prompt = (
        "Look at this robot arm in a simulated room. "
        "The camera is viewing from the front. "
        "Which direction is the robot's end-effector (gripper/hand) primarily pointing?\n\n"
        f"{option_text}\n\n"
        "Reply with ONLY the letter (A, B, C, D, or E)."
    )

    try:
        response = client.models.generate_content(model=model, contents=[prompt, image])
        answer = response.text.strip().upper()
        for ch in answer:
            if ch in "ABCDE":
                answer = ch
                break
        correct = answer == gt_letter
    except Exception as e:
        answer = f"ERROR: {e}"
        correct = False

    dir_options = {chr(65+i): d for i, d in enumerate(directions)}
    return {
        "test": "arm_direction",
        "gt_dir": gt_dir_mapped if gt_dir_mapped in directions else gt_dir,
        "gt_letter": gt_letter,
        "answer": answer,
        "correct": correct,
        "options": dir_options,
        "chosen_text": dir_options.get(answer, answer),
    }


def _test_gripper_orientation(client, model: str, image: Image.Image, gt_gripper: str) -> dict:
    """Binary choice: horizontal or vertical gripper?"""
    prompt = (
        "Look at this robot arm in a simulated room. "
        "Focus on the gripper (the end piece of the arm). "
        "Is the gripper oriented horizontally (parallel to the floor) or vertically (perpendicular to the floor)?\n\n"
        "  A. horizontal\n"
        "  B. vertical\n\n"
        "Reply with ONLY the letter (A or B)."
    )

    gt_letter = "A" if gt_gripper == "horizontal" else "B"

    try:
        response = client.models.generate_content(model=model, contents=[prompt, image])
        answer = response.text.strip().upper()
        for ch in answer:
            if ch in "AB":
                answer = ch
                break
        correct = answer == gt_letter
    except Exception as e:
        answer = f"ERROR: {e}"
        correct = False

    grip_options = {"A": "horizontal", "B": "vertical"}
    return {
        "test": "gripper_orientation",
        "gt_gripper": gt_gripper,
        "gt_letter": gt_letter,
        "answer": answer,
        "correct": correct,
        "options": grip_options,
        "chosen_text": grip_options.get(answer, answer),
    }


def _test_height_estimation(client, model: str, image: Image.Image, gt_z: int) -> dict:
    """Numeric estimation: how high is the end-effector? (0=low, 100=high)"""
    prompt = (
        "Look at this robot arm in a simulated room. "
        "On a scale from 0 to 100, how high is the robot's end-effector (gripper)? "
        "0 means the gripper is at the very bottom (near the base), "
        "100 means the gripper is at the very top (highest reachable point).\n\n"
        "Reply with ONLY a number between 0 and 100."
    )

    try:
        response = client.models.generate_content(model=model, contents=[prompt, image])
        raw = response.text.strip()
        # Extract first number
        num = ""
        for ch in raw:
            if ch.isdigit():
                num += ch
            elif num:
                break
        predicted = int(num) if num else -1
        error = abs(predicted - gt_z) if predicted >= 0 and gt_z is not None else -1
    except Exception as e:
        predicted = -1
        error = -1

    return {
        "test": "height_estimation",
        "gt_z": gt_z,
        "predicted": predicted,
        "error": error,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    robot: str = "IIWA",
    config_version: int = 10,
    model: str = "gemini-2.5-flash-lite",
    n_cues: int = 10,
    delay: float = 2.0,
    seed: int = 42,
):
    """Run VLM pose recognition benchmark.

    Args:
        robot: Robot name
        config_version: Which prompt version's configs to use for GT
        model: Gemini model name
        n_cues: Number of cues to test
        delay: Seconds between API calls
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    configs = _load_configs(config_version)
    if not configs:
        print(f"No configs found for v{config_version}")
        return

    all_cue_names = [c["cue"] for c in configs]

    # Find cues that have rendered GIFs
    test_cues = []
    for cfg in configs:
        idx = cfg.get("idx")
        if idx is None:
            continue
        gif_path = _find_gif(robot, idx)
        if gif_path:
            gt = _get_gt_info(cfg)
            test_cues.append({"cfg": cfg, "gif": gif_path, "gt": gt})

    if not test_cues:
        print(f"No rendered GIFs found for {robot}. Run: python adhoc/generation/rendering.py manipulator ...")
        return

    random.shuffle(test_cues)
    test_cues = test_cues[:n_cues]

    print(f"\n{'═'*70}")
    print(f"  VLM POSE RECOGNITION BENCHMARK")
    print(f"  Model: {model}  |  Robot: {robot}  |  Cues: {len(test_cues)}")
    print(f"{'═'*70}\n")

    results = {
        "cue_recognition": [],
        "arm_direction": [],
        "gripper_orientation": [],
        "height_estimation": [],
    }

    for i, tc in enumerate(test_cues):
        cfg = tc["cfg"]
        gt = tc["gt"]
        cue = gt["cue"]
        idx = gt["idx"]
        cue_short = cue[:40] + ".." if len(cue) > 42 else cue

        print(f"  [{i+1}/{len(test_cues)}] c{idx}: {cue_short}")

        # Extract one tile from the GIF
        image = _extract_tile(tc["gif"], tile_idx=0)

        meta = {"cue_idx": idx, "gif": tc["gif"]}

        # Test 1: Cue recognition
        r1 = _test_cue_recognition(client, model, image, cue, all_cue_names)
        r1.update(meta)
        results["cue_recognition"].append(r1)
        mark1 = "✅" if r1["correct"] else f"❌ (answered {r1['answer']}, expected {r1['gt_letter']})"
        print(f"    Cue recognition:     {mark1}")
        time.sleep(delay)

        # Test 2: Arm direction
        if gt.get("dir") and gt["dir"] != "unknown":
            r2 = _test_arm_direction(client, model, image, gt["dir"])
            r2.update(meta)
            results["arm_direction"].append(r2)
            mark2 = "✅" if r2["correct"] else f"❌ (answered {r2['answer']}, gt={r2['gt_letter']}:{gt['dir']})"
            print(f"    Arm direction:       {mark2}")
            time.sleep(delay)

        # Test 3: Gripper orientation
        if gt.get("gripper") and gt["gripper"] != "unknown":
            r3 = _test_gripper_orientation(client, model, image, gt["gripper"])
            r3.update(meta)
            results["gripper_orientation"].append(r3)
            mark3 = "✅" if r3["correct"] else f"❌ (answered {r3['answer']}, gt={r3['gt_letter']}:{gt['gripper']})"
            print(f"    Gripper orientation: {mark3}")
            time.sleep(delay)

        # Test 4: Height estimation
        if gt.get("z") is not None:
            r4 = _test_height_estimation(client, model, image, gt["z"])
            r4.update(meta)
            results["height_estimation"].append(r4)
            err_str = f"error={r4['error']}" if r4["error"] >= 0 else "FAIL"
            print(f"    Height (gt={gt['z']}):     predicted={r4['predicted']}  {err_str}")
            time.sleep(delay)

        image.close()
        print()

    # ── Report ───────────────────────────────────────────────────
    print(f"{'═'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*70}\n")

    for test_name, test_results in results.items():
        if not test_results:
            continue

        if test_name == "height_estimation":
            valid = [r for r in test_results if r["error"] >= 0]
            if valid:
                errors = [r["error"] for r in valid]
                mae = sum(errors) / len(errors)
                within_10 = sum(1 for e in errors if e <= 10)
                within_20 = sum(1 for e in errors if e <= 20)
                print(f"  {test_name}:")
                print(f"    N: {len(valid)}")
                print(f"    MAE: {mae:.1f}")
                print(f"    Within ±10: {within_10}/{len(valid)} ({100*within_10/len(valid):.0f}%)")
                print(f"    Within ±20: {within_20}/{len(valid)} ({100*within_20/len(valid):.0f}%)")
        else:
            n = len(test_results)
            correct = sum(1 for r in test_results if r.get("correct"))
            acc = 100 * correct / n if n else 0
            chance = {"cue_recognition": 20, "arm_direction": 20, "gripper_orientation": 50}
            chance_pct = chance.get(test_name, 0)
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"  {test_name}:")
            print(f"    Accuracy: {correct}/{n} ({acc:.0f}%)  {bar}")
            print(f"    Chance level: {chance_pct}%")
            if test_name == "cue_recognition":
                wrong = [r for r in test_results if not r["correct"]]
                if wrong:
                    print(f"    Misclassified:")
                    for r in wrong[:5]:
                        chosen = r["options"].get(r["answer"], "?")
                        print(f"      GT: {r['gt_cue'][:35]}")
                        print(f"      → Chose: {chosen[:35]}")
        print()

    # Save detailed results
    out_path = os.path.join(MOTION_DIR, f"vlm_benchmark_{model}_{robot}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Detailed results: {out_path}\n")


def report(
    models: list[str] = None,
    robot: str = "IIWA",
    output: str = None,
):
    """Generate an HTML report comparing VLM benchmark results across models.

    Args:
        models: List of model names (looks for vlm_benchmark_{model}_{robot}.json)
        robot: Robot name
        output: Output HTML path
    """
    import base64
    from io import BytesIO

    if models is None:
        found = globmod.glob(os.path.join(MOTION_DIR, f"vlm_benchmark_*_{robot}.json"))
        models = sorted(
            [os.path.basename(f).replace(f"vlm_benchmark_", "").replace(f"_{robot}.json", "") for f in found]
        )
    if not models:
        print("No benchmark results found.")
        return

    all_results = {}
    for m in models:
        path = os.path.join(MOTION_DIR, f"vlm_benchmark_{m}_{robot}.json")
        if os.path.exists(path):
            all_results[m] = json.load(open(path))

    configs = _load_configs(10)
    cue_map = {c["cue"]: c for c in configs}

    # Build per-cue merged view from cue_recognition (has most info)
    cue_items = {}
    for m, res in all_results.items():
        for r in res.get("cue_recognition", []):
            cue = r["gt_cue"]
            if cue not in cue_items:
                cue_items[cue] = {
                    "cue": cue,
                    "idx": r.get("cue_idx"),
                    "gif": r.get("gif"),
                    "models": {},
                }
            # Try to get idx/gif from this model's result if not set yet
            if cue_items[cue]["idx"] is None and r.get("cue_idx") is not None:
                cue_items[cue]["idx"] = r["cue_idx"]
            if cue_items[cue]["gif"] is None and r.get("gif"):
                cue_items[cue]["gif"] = r["gif"]
            cue_items[cue]["models"][m] = {"cue_rec": r}

    # Merge arm_direction, gripper_orientation, height_estimation
    for test_key in ("arm_direction", "gripper_orientation", "height_estimation"):
        for m, res in all_results.items():
            for r in res.get(test_key, []):
                idx = r.get("cue_idx")
                for cue_name, ci in cue_items.items():
                    if ci["idx"] == idx:
                        if m not in ci["models"]:
                            ci["models"][m] = {}
                        ci["models"][m][test_key] = r
                        break

    # For old results without cue_idx, try matching by position
    for m, res in all_results.items():
        for test_key in ("arm_direction", "gripper_orientation", "height_estimation"):
            test_list = res.get(test_key, [])
            cue_rec_list = res.get("cue_recognition", [])
            if test_list and not test_list[0].get("cue_idx"):
                # Fallback: match by index order within the same model
                rec_cues_in_order = [r["gt_cue"] for r in cue_rec_list]
                test_idx = 0
                for cue_name in rec_cues_in_order:
                    if cue_name in cue_items and test_idx < len(test_list):
                        ci = cue_items[cue_name]
                        if m not in ci["models"]:
                            ci["models"][m] = {}
                        if test_key not in ci["models"][m]:
                            ci["models"][m][test_key] = test_list[test_idx]
                            test_idx += 1

    # Fallback: fill missing idx/gif from configs
    for cue_name, ci in cue_items.items():
        if ci["idx"] is None and cue_name in cue_map:
            ci["idx"] = cue_map[cue_name].get("idx")
        if ci["gif"] is None and ci["idx"] is not None:
            ci["gif"] = _find_gif(robot, ci["idx"])

    sorted_items = sorted(cue_items.values(), key=lambda x: x.get("idx") or 999)

    def _img_b64(gif_path):
        if not gif_path or not os.path.exists(gif_path):
            return None
        try:
            tile = _extract_tile(gif_path, tile_idx=0)
            tile.thumbnail((256, 256))
            buf = BytesIO()
            tile.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None

    def _option_html(options, gt_letter, chosen_letter):
        if not options:
            return ""
        lines = []
        for letter, text in sorted(options.items()):
            cls = []
            if letter == gt_letter:
                cls.append("gt")
            if letter == chosen_letter:
                cls.append("chosen")
            cls_str = " ".join(cls)
            icon = ""
            if letter == gt_letter and letter == chosen_letter:
                icon = "✅ "
            elif letter == chosen_letter:
                icon = "❌ "
            elif letter == gt_letter:
                icon = "🎯 "
            lines.append(f'<div class="opt {cls_str}">{icon}<b>{letter}.</b> {text}</div>')
        return "\n".join(lines)

    def _height_html(r):
        if not r:
            return "<span class='na'>N/A</span>"
        gt = r.get("gt_z", "?")
        pred = r.get("predicted", "?")
        err = r.get("error", -1)
        cls = "correct" if err <= 10 else ("close" if err <= 20 else "wrong")
        return f'<span class="{cls}">GT={gt} → Pred={pred} (err={err})</span>'

    # Compute summary stats
    summary = {}
    for m in models:
        s = {"cue": [0, 0], "dir": [0, 0], "grip": [0, 0], "h_errors": []}
        res = all_results.get(m, {})
        for r in res.get("cue_recognition", []):
            s["cue"][1] += 1
            if r["correct"]: s["cue"][0] += 1
        for r in res.get("arm_direction", []):
            s["dir"][1] += 1
            if r["correct"]: s["dir"][0] += 1
        for r in res.get("gripper_orientation", []):
            s["grip"][1] += 1
            if r["correct"]: s["grip"][0] += 1
        for r in res.get("height_estimation", []):
            if r.get("error", -1) >= 0:
                s["h_errors"].append(r["error"])
        summary[m] = s

    html_parts = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLM Benchmark — {robot}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1 {{ color: #58a6ff; margin-bottom: 8px; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0 24px; }}
  .stat-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 16px 20px; min-width: 180px; }}
  .stat-card h3 {{ color: #8b949e; font-size: 12px; text-transform: uppercase; margin-bottom: 6px; }}
  .stat-card .val {{ font-size: 24px; font-weight: 700; }}
  .stat-card .val.good {{ color: #3fb950; }}
  .stat-card .val.mid {{ color: #d29922; }}
  .stat-card .val.bad {{ color: #f85149; }}
  .cue-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px;
               margin-bottom: 20px; overflow: hidden; }}
  .cue-header {{ background: #21262d; padding: 12px 16px; display: flex; align-items: center; gap: 12px; }}
  .cue-header .idx {{ background: #30363d; border-radius: 6px; padding: 4px 10px; font-weight: 700;
                      color: #58a6ff; font-size: 14px; }}
  .cue-header .name {{ font-size: 14px; font-weight: 600; }}
  .cue-body {{ display: flex; gap: 0; }}
  .cue-img {{ padding: 12px; flex-shrink: 0; background: #0d1117; }}
  .cue-img img {{ border-radius: 8px; display: block; }}
  .cue-tests {{ flex: 1; padding: 12px; display: flex; flex-direction: column; gap: 8px; overflow-x: auto; }}
  .test-section {{ border: 1px solid #30363d; border-radius: 8px; padding: 10px; }}
  .test-section h4 {{ font-size: 12px; color: #8b949e; margin-bottom: 6px; text-transform: uppercase; }}
  .model-row {{ display: flex; gap: 12px; margin-bottom: 8px; align-items: flex-start; }}
  .model-tag {{ background: #30363d; color: #58a6ff; border-radius: 4px; padding: 2px 8px;
                font-size: 11px; font-weight: 600; min-width: 100px; text-align: center; flex-shrink: 0; }}
  .opt {{ font-size: 13px; line-height: 1.6; padding: 2px 6px; border-radius: 4px; }}
  .opt.gt {{ background: rgba(63,185,80,0.15); border-left: 3px solid #3fb950; }}
  .opt.chosen {{ background: rgba(248,81,73,0.15); border-left: 3px solid #f85149; }}
  .opt.gt.chosen {{ background: rgba(63,185,80,0.25); border-left: 3px solid #3fb950; }}
  .correct {{ color: #3fb950; }}
  .close {{ color: #d29922; }}
  .wrong {{ color: #f85149; }}
  .na {{ color: #484f58; }}
  .model-results {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .model-col {{ flex: 1; min-width: 280px; }}
</style></head><body>
<h1>VLM Pose Recognition Benchmark — {robot}</h1>
<p style="color:#8b949e">Models: {', '.join(models)} | Cues tested: {len(sorted_items)}</p>
<div class="summary">"""]

    for m in models:
        s = summary[m]
        cue_acc = 100 * s["cue"][0] / s["cue"][1] if s["cue"][1] else 0
        dir_acc = 100 * s["dir"][0] / s["dir"][1] if s["dir"][1] else 0
        grip_acc = 100 * s["grip"][0] / s["grip"][1] if s["grip"][1] else 0
        mae = sum(s["h_errors"]) / len(s["h_errors"]) if s["h_errors"] else -1
        def _cls(v, thresh_good, thresh_mid):
            return "good" if v >= thresh_good else ("mid" if v >= thresh_mid else "bad")
        html_parts.append(f"""
  <div class="stat-card"><h3>{m}</h3>
    <div>Cue: <span class="val {_cls(cue_acc,30,20)}">{s['cue'][0]}/{s['cue'][1]} ({cue_acc:.0f}%)</span></div>
    <div>Dir: <span class="val {_cls(dir_acc,30,20)}">{s['dir'][0]}/{s['dir'][1]} ({dir_acc:.0f}%)</span></div>
    <div>Grip: <span class="val {_cls(grip_acc,60,50)}">{s['grip'][0]}/{s['grip'][1]} ({grip_acc:.0f}%)</span></div>
    <div>Height MAE: <span class="val {_cls(100-mae,85,70) if mae>=0 else 'na'}">{mae:.1f}</span></div>
  </div>""")

    html_parts.append("</div>")

    for ci in sorted_items:
        idx = ci.get("idx", "?")
        cue = ci["cue"]
        b64 = _img_b64(ci.get("gif"))
        img_tag = f'<img src="data:image/png;base64,{b64}" width="200">' if b64 else '<div style="width:200px;height:200px;background:#21262d;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#484f58">No image</div>'

        html_parts.append(f"""
<div class="cue-card">
  <div class="cue-header">
    <span class="idx">c{idx}</span>
    <span class="name">{cue}</span>
  </div>
  <div class="cue-body">
    <div class="cue-img">{img_tag}</div>
    <div class="cue-tests">""")

        # Cue Recognition test
        html_parts.append('<div class="test-section"><h4>Cue Recognition (5-choice)</h4><div class="model-results">')
        for m in models:
            md = ci["models"].get(m, {})
            r = md.get("cue_rec")
            if r:
                html_parts.append(f'<div class="model-col"><span class="model-tag">{m}</span>')
                html_parts.append(_option_html(r.get("options", {}), r.get("gt_letter", ""), r.get("answer", "")))
                html_parts.append('</div>')
        html_parts.append('</div></div>')

        # Arm Direction test
        html_parts.append('<div class="test-section"><h4>Arm Direction (5-choice)</h4><div class="model-results">')
        for m in models:
            md = ci["models"].get(m, {})
            r = md.get("arm_direction")
            if r:
                opts = r.get("options", {"A":"up","B":"down","C":"left","D":"right","E":"front"})
                html_parts.append(f'<div class="model-col"><span class="model-tag">{m}</span>')
                html_parts.append(_option_html(opts, r.get("gt_letter",""), r.get("answer","")))
                html_parts.append('</div>')
        html_parts.append('</div></div>')

        # Gripper Orientation test
        html_parts.append('<div class="test-section"><h4>Gripper Orientation (binary)</h4><div class="model-results">')
        for m in models:
            md = ci["models"].get(m, {})
            r = md.get("gripper_orientation")
            if r:
                opts = r.get("options", {"A":"horizontal","B":"vertical"})
                html_parts.append(f'<div class="model-col"><span class="model-tag">{m}</span>')
                html_parts.append(_option_html(opts, r.get("gt_letter",""), r.get("answer","")))
                html_parts.append('</div>')
            else:
                html_parts.append(f'<div class="model-col"><span class="model-tag">{m}</span><span class="na">N/A</span></div>')
        html_parts.append('</div></div>')

        # Height Estimation test
        html_parts.append('<div class="test-section"><h4>Height Estimation (0-100)</h4><div class="model-results">')
        for m in models:
            md = ci["models"].get(m, {})
            r = md.get("height_estimation")
            html_parts.append(f'<div class="model-col"><span class="model-tag">{m}</span>{_height_html(r)}</div>')
        html_parts.append('</div></div>')

        html_parts.append("</div></div></div>")

    html_parts.append("</body></html>")

    if output is None:
        output = os.path.join(MOTION_DIR, f"vlm_benchmark_report_{robot}.html")
    with open(output, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Report saved: {output}")


# ── Trajectory utilities ──────────────────────────────────────────────────────

def _get_gripper_tip_world(env, robot):
    """Get world-coordinate gripper finger-tip midpoint."""
    model = env.sim.model
    tips = []
    for i in range(model.nbody):
        bname = model.body_id2name(i)
        if "finger" in bname and "tip" in bname:
            tips.append(env.sim.data.body_xpos[i].copy())
    if len(tips) >= 2:
        return np.mean(tips, axis=0)
    for i in range(model.nsite):
        sname = model.site_id2name(i)
        if "grip_site" in sname and "cylinder" not in sname:
            return env.sim.data.site_xpos[i].copy()
    arm_key = list(robot._hand_pos.keys())[0]
    return robot._hand_pos[arm_key].copy()


def _project_3d(pt, cam_pos, cam_rot, fovy, img_size):
    """Project 3D world point → 2D pixel."""
    p = cam_rot.T @ (pt - cam_pos)
    d = -p[2]
    if d <= 0.01:
        return None
    f = 0.5 * img_size / np.tan(np.radians(fovy) / 2)
    return (int(f * p[0] / d + img_size / 2),
            int(img_size / 2 - f * p[1] / d))


def _draw_arrow_on(img, start, end, color="red", width=4, head=14):
    draw = ImageDraw.Draw(img)
    draw.line([start, end], fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    ln = max(1, (dx**2 + dy**2) ** 0.5)
    ux, uy = dx / ln, dy / ln
    px, py = -uy, ux
    p1 = (int(end[0] - head * ux + head * 0.5 * px),
          int(end[1] - head * uy + head * 0.5 * py))
    p2 = (int(end[0] - head * ux - head * 0.5 * px),
          int(end[1] - head * uy - head * 0.5 * py))
    draw.polygon([end, p1, p2], fill=color)


def _simulate_trajectory(gen, cue_name, cue_idx, config_path, hz=4,
                         pose_index=None):
    """Re-simulate a cue, capturing gripper trajectory at every frame.
    Returns (frames, trajectory, cam_pos, cam_rot, fovy).
    trajectory: list of dicts with 'pos' (3D) and 'rot' (3x3 orientation).
    """
    trajectory = []
    orig_capture = gen._capture_image

    def _capture_with_traj():
        tip = _get_gripper_tip_world(gen.env, gen.robot)
        arm_key = list(gen.robot._hand_orn.keys())[0]
        rot = gen.robot._hand_orn[arm_key].copy()
        trajectory.append({"pos": tip, "rot": rot})
        return orig_capture()

    gen._capture_image = _capture_with_traj
    try:
        gen._set_joint_positions(gen.initial_joint_pos)
        frames, pose_id = gen.execute_cue(
            cue=cue_name, cue_idx=cue_idx,
            pose_index=pose_index,
            config_path=config_path, hz=hz, save_gif=False,
        )
    finally:
        gen._capture_image = orig_capture

    cam_id = gen.env.sim.model.camera_name2id("frontview")
    cam_pos = gen.env.sim.data.cam_xpos[cam_id].copy()
    cam_rot = gen.env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy = gen.env.sim.model.cam_fovy[cam_id]

    return frames, trajectory, cam_pos, cam_rot, fovy


def _render_trajectory_image(base_frame, trajectory, cam_pos, cam_rot, fovy,
                             img_size=512, with_arrow=False, arrow_length=0.12):
    """Overlay gripper trajectory (and optionally direction arrow) on base frame."""
    img = base_frame.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    pts = []
    for t in trajectory:
        p = _project_3d(t["pos"], cam_pos, cam_rot, fovy, img_size)
        if p:
            pts.append(p)
    if not pts:
        return img

    # Trajectory line: green → yellow gradient
    for i in range(1, len(pts)):
        frac = i / max(1, len(pts) - 1)
        r = int(255 * frac)
        color = (r, 255, 0)
        draw.line([pts[i - 1], pts[i]], fill=color, width=3)

    # Green dot at start
    sx, sy = pts[0]
    draw.ellipse([sx - 8, sy - 8, sx + 8, sy + 8],
                 fill="lime", outline="white", width=2)

    # Yellow dot at end
    ex, ey = pts[-1]
    draw.ellipse([ex - 5, ey - 5, ex + 5, ey + 5],
                 fill="yellow", outline="white", width=1)

    if with_arrow and trajectory:
        last = trajectory[-1]
        pointing = last["rot"][:, 2]
        s3d = last["pos"]
        e3d = s3d + pointing * arrow_length
        s2d = _project_3d(s3d, cam_pos, cam_rot, fovy, img_size)
        e2d = _project_3d(e3d, cam_pos, cam_rot, fovy, img_size)
        if s2d and e2d:
            dx, dy = e2d[0] - s2d[0], e2d[1] - s2d[1]
            slen = (dx**2 + dy**2) ** 0.5
            if slen < 30 and slen > 1:
                sc = 30 / slen
                e2d = (int(s2d[0] + dx * sc), int(s2d[1] + dy * sc))
            _draw_arrow_on(img, s2d, e2d, color="red", width=4, head=14)

    return img


# ── Trajectory benchmark ─────────────────────────────────────────────────────

def trajectory(
    robot: str = "IIWA",
    config_version: int = 10,
    model: str = "gemini-2.5-flash-lite",
    n_cues: int = 10,
    delay: float = 2.0,
    seed: int = 42,
    mode: str = "both",
    hz: int = 4,
    camera_distance: float = 1.8,
    img_size: int = 512,
):
    """Run VLM cue recognition using trajectory-overlay images.

    Args:
        robot: Robot name
        config_version: Prompt version for configs
        model: Gemini model name
        n_cues: Number of cues to test
        mode: "traj" | "arrow" | "both"
        hz: Simulation frame rate
        camera_distance: FOV zoom factor
        img_size: Render resolution
    """
    random.seed(seed)
    from google import genai
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from motion_generation import MotionGenerator, _select_initial_poses

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY env var.")
    client = genai.Client(api_key=api_key)

    config_path = os.path.join(MC_MANIP, f"motion_configs_prompt_v{config_version}.json")
    configs = _load_configs(config_version)
    if not configs:
        print(f"No configs for v{config_version}")
        return

    all_cue_names = [c["cue"] for c in configs]

    # Filter cues that have rendered GIFs (proves they are valid)
    valid = []
    for cfg in configs:
        idx = cfg.get("idx")
        if idx is None:
            continue
        gif = _find_gif(robot, idx)
        if gif:
            valid.append(cfg)
    random.shuffle(valid)
    valid = valid[:n_cues]

    if not valid:
        print("No renderable cues found.")
        return

    modes = []
    if mode in ("traj", "both"):
        modes.append("traj")
    if mode in ("arrow", "both"):
        modes.append("arrow")

    print(f"\n{'═' * 70}")
    print(f"  VLM TRAJECTORY BENCHMARK")
    print(f"  Model: {model}  |  Robot: {robot}  |  Cues: {len(valid)}  |  Modes: {modes}")
    print(f"{'═' * 70}\n")

    gen = MotionGenerator(
        robot_name=robot,
        camera_distance=camera_distance,
        capture_image_width=img_size,
        capture_image_height=img_size,
        hz=hz,
    )

    out_dir = os.path.join(MOTION_DIR, "vlm_trajectory_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    results = {m: [] for m in modes}
    images_saved = {}

    for ci, cfg in enumerate(valid):
        cue = cfg["cue"]
        idx = cfg.get("idx")
        cue_short = cue[:45] + ".." if len(cue) > 47 else cue
        print(f"\n  [{ci + 1}/{len(valid)}] c{idx}: {cue_short}")

        first_pose_def = None
        for m in cfg.get("movements", []):
            if m.get("type") == "pose":
                first_pose_def = m["parameters"]["pose"]
                break
        pose_id = None
        if first_pose_def is not None:
            matching = gen._find_matching_poses(first_pose_def)
            selected = _select_initial_poses(matching, first_pose_def, 1)
            if selected:
                pose_id = selected[0]["pose_id"]

        try:
            gen._set_joint_positions(gen.initial_joint_pos)
            frames, traj, cam_pos, cam_rot, fovy = _simulate_trajectory(
                gen, cue, idx, config_path, hz=hz, pose_index=pose_id,
            )
        except Exception as e:
            print(f"    ⚠ Simulation failed: {e}")
            continue

        if not frames or not traj:
            print("    ⚠ No frames/trajectory captured")
            continue

        base = Image.fromarray(frames[0] if isinstance(frames[0], np.ndarray) else np.array(frames[0]))

        for m_key in modes:
            with_arrow = (m_key == "arrow")
            img = _render_trajectory_image(
                base, traj, cam_pos, cam_rot, fovy,
                img_size=img_size, with_arrow=with_arrow,
            )

            fname = f"{robot}_v{config_version}_c{idx}_{m_key}.png"
            img.save(os.path.join(out_dir, fname))
            images_saved[(idx, m_key)] = fname

            # VLM cue recognition
            distractors = [c for c in all_cue_names if c != cue]
            random.shuffle(distractors)
            options = [cue] + distractors[:4]
            random.shuffle(options)
            correct_letter = chr(65 + options.index(cue))
            option_text = "\n".join(f"  {chr(65 + i)}. {o}" for i, o in enumerate(options))

            hint = ""
            if m_key == "traj":
                hint = (
                    "The image has a colored trajectory line overlaid, showing "
                    "the path the gripper moved. A green dot marks the start and "
                    "a yellow dot marks the end.\n"
                )
            else:
                hint = (
                    "The image has a colored trajectory line (green→yellow) showing "
                    "the gripper's path, plus a RED ARROW at the end showing the "
                    "gripper's final pointing direction.\n"
                )

            prompt = (
                f"You see a robot arm in a simulated room. "
                f"{hint}"
                f"Which gesture is the robot performing?\n\n"
                f"{option_text}\n\n"
                f"Reply with ONLY the letter (A–E)."
            )

            try:
                resp = client.models.generate_content(model=model, contents=[prompt, img])
                raw = resp.text.strip().upper()
                answer = "?"
                for ch in raw:
                    if ch in "ABCDE":
                        answer = ch
                        break
            except Exception as e:
                answer = f"ERR"
                raw = str(e)

            correct = answer == correct_letter
            mark = "✅" if correct else f"❌ chose={options[ord(answer) - 65] if answer in 'ABCDE' else '?'}"
            suffix = " +arrow" if with_arrow else " traj"
            print(f"    {suffix}: {mark}")

            results[m_key].append({
                "cue_idx": idx,
                "gt_cue": cue,
                "gt_letter": correct_letter,
                "answer": answer,
                "correct": correct,
                "options": {chr(65 + i): o for i, o in enumerate(options)},
                "chosen_text": options[ord(answer) - 65] if answer in "ABCDE" else raw[:40],
                "image": fname,
                "mode": m_key,
            })
            time.sleep(delay)

    gen.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═' * 70}\n")

    for m_key in modes:
        res = results[m_key]
        n = len(res)
        if not n:
            continue
        corr = sum(1 for r in res if r["correct"])
        acc = 100 * corr / n
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        label = "Trajectory + Arrow" if m_key == "arrow" else "Trajectory Only"
        print(f"  {label}: {corr}/{n} ({acc:.0f}%)  {bar}  (chance=20%)")

        wrong = [r for r in res if not r["correct"]]
        if wrong:
            print(f"    Misclassified:")
            for r in wrong[:5]:
                chosen = r.get("chosen_text", "?")[:35]
                print(f"      GT: {r['gt_cue'][:35]}")
                print(f"      → Chose: {chosen}")
        print()

    # Save results
    save_data = {
        "model": model, "robot": robot, "n_cues": len(valid),
        "config_version": config_version, "seed": seed,
        "modes": modes,
    }
    for m_key in modes:
        save_data[f"results_{m_key}"] = results[m_key]

    out_json = os.path.join(out_dir, f"results_v{config_version}_{model}_{robot}.json")
    with open(out_json, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results: {out_json}")

    return save_data


def compare(
    versions: list[int] = None,
    robot: str = "IIWA",
    model: str = "gemini-2.5-flash-lite",
    n_cues: int = 10,
    delay: float = 2.0,
    seed: int = 42,
    mode: str = "both",
    hz: int = 4,
    camera_distance: float = 1.8,
    img_size: int = 512,
):
    """Run trajectory benchmark across multiple prompt versions, then generate comparison report.

    Args:
        versions: List of config versions to compare (default: [5,6,7,8,9,10])
    """
    if versions is None:
        versions = [5, 6, 7, 8, 9, 10]

    out_dir = os.path.join(MOTION_DIR, "vlm_trajectory_benchmark")
    os.makedirs(out_dir, exist_ok=True)

    for v in versions:
        result_path = os.path.join(out_dir, f"results_v{v}_{model}_{robot}.json")
        if os.path.exists(result_path):
            print(f"\n  ⏩ v{v} already has results, skipping. Delete {result_path} to re-run.")
            continue
        print(f"\n{'━' * 70}")
        print(f"  Running v{v}...")
        print(f"{'━' * 70}")
        try:
            trajectory(
                robot=robot, config_version=v, model=model,
                n_cues=n_cues, delay=delay, seed=seed, mode=mode,
                hz=hz, camera_distance=camera_distance, img_size=img_size,
            )
        except Exception as e:
            print(f"  ⚠ v{v} failed: {e}")

    _generate_comparison_report(out_dir, versions, model, robot)


def _generate_comparison_report(out_dir, versions, model, robot):
    """Generate cross-version comparison HTML report."""
    ver_data = {}
    for v in versions:
        path = os.path.join(out_dir, f"results_v{v}_{model}_{robot}.json")
        if os.path.exists(path):
            ver_data[v] = json.load(open(path))

    if not ver_data:
        print("No results to compare.")
        return

    modes = list(ver_data.values())[0].get("modes", ["traj", "arrow"])

    def _img_b64(fname):
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            return None
        img = Image.open(fpath)
        img.thumbnail((220, 220))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _opt_html(r):
        if not r:
            return '<span style="color:#484f58">N/A</span>'
        lines = []
        for letter, text in sorted(r.get("options", {}).items()):
            cls, icon = [], ""
            if letter == r["gt_letter"]:
                cls.append("gt")
            if letter == r["answer"]:
                cls.append("chosen")
            if letter == r["gt_letter"] and letter == r["answer"]:
                icon = "✅ "
            elif letter == r["answer"]:
                icon = "❌ "
            elif letter == r["gt_letter"]:
                icon = "🎯 "
            lines.append(f'<div class="opt {" ".join(cls)}">{icon}<b>{letter}.</b> {text}</div>')
        return "\n".join(lines)

    sorted_vers = sorted(ver_data.keys())
    n_ver = len(sorted_vers)

    # Compute summary stats
    summary = {}
    for v in sorted_vers:
        summary[v] = {}
        for mk in modes:
            res = ver_data[v].get(f"results_{mk}", [])
            n = len(res)
            corr = sum(1 for r in res if r["correct"])
            summary[v][mk] = {"n": n, "correct": corr, "acc": 100 * corr / n if n else 0}

    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Trajectory Benchmark — Version Comparison</title><style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0d1117; color:#c9d1d9; padding:20px; max-width:1800px; margin:0 auto; }}
  h1 {{ color:#58a6ff; margin-bottom:4px; }}
  .sub {{ color:#8b949e; margin-bottom:16px; }}
  table {{ border-collapse:collapse; margin:16px 0 24px; }}
  th, td {{ border:1px solid #30363d; padding:8px 14px; text-align:center; }}
  th {{ background:#21262d; color:#8b949e; font-size:12px; text-transform:uppercase; }}
  td.good {{ color:#3fb950; font-weight:700; }} td.mid {{ color:#d29922; font-weight:700; }}
  td.bad {{ color:#f85149; font-weight:700; }}
  .card {{ background:#161b22; border:1px solid #30363d; border-radius:10px;
           margin-bottom:20px; overflow:hidden; }}
  .card-hdr {{ background:#21262d; padding:10px 14px; font-weight:600; }}
  .card-hdr .tag {{ background:#30363d; border-radius:5px; padding:3px 8px;
                    font-weight:700; color:#58a6ff; font-size:13px; margin-right:8px; }}
  .ver-grid {{ display:grid; grid-template-columns:repeat({n_ver}, 1fr);
               gap:0; border-top:1px solid #30363d; }}
  .ver-cell {{ border-right:1px solid #30363d; padding:10px; }}
  .ver-cell:last-child {{ border-right:none; }}
  .ver-cell h4 {{ font-size:12px; color:#8b949e; margin-bottom:4px; text-transform:uppercase; }}
  .ver-cell img {{ border-radius:6px; display:block; margin-bottom:6px; width:100%; max-width:220px; }}
  .opt {{ font-size:11px; line-height:1.5; padding:1px 4px; border-radius:3px; }}
  .opt.gt {{ background:rgba(63,185,80,0.15); border-left:3px solid #3fb950; }}
  .opt.chosen {{ background:rgba(248,81,73,0.15); border-left:3px solid #f85149; }}
  .opt.gt.chosen {{ background:rgba(63,185,80,0.25); border-left:3px solid #3fb950; }}
  .mark {{ font-size:16px; margin-right:4px; }}
</style></head><body>
<h1>VLM Trajectory Benchmark — Version Comparison</h1>
<p class="sub">Model: {model} | Robot: {robot} | Versions: {', '.join(f'v{v}' for v in sorted_vers)}</p>
"""]

    # ── Summary table ────
    for mk in modes:
        label = "Trajectory + Arrow" if mk == "arrow" else "Trajectory Only"
        html.append(f'<h2 style="color:#8b949e;margin:12px 0 4px">{label}</h2>')
        html.append('<table><tr><th>Version</th>')
        for v in sorted_vers:
            html.append(f'<th>v{v}</th>')
        html.append('</tr><tr><th>Accuracy</th>')
        best_acc = max(summary[v][mk]["acc"] for v in sorted_vers if mk in summary[v])
        for v in sorted_vers:
            s = summary[v].get(mk, {"n": 0, "correct": 0, "acc": 0})
            cls = "good" if s["acc"] >= 30 else ("mid" if s["acc"] >= 20 else "bad")
            bold = " style='font-size:18px;text-decoration:underline'" if s["acc"] == best_acc and s["acc"] > 0 else ""
            html.append(f'<td class="{cls}"{bold}>{s["correct"]}/{s["n"]} ({s["acc"]:.0f}%)</td>')
        html.append('</tr></table>')

    # ── Per-cue comparison cards ────
    # Collect all unique cue_idxs across versions
    all_cue_idxs = {}
    for v in sorted_vers:
        for mk in modes:
            for r in ver_data[v].get(f"results_{mk}", []):
                idx = r["cue_idx"]
                if idx not in all_cue_idxs:
                    all_cue_idxs[idx] = r["gt_cue"]

    for mk in modes:
        mode_label = "Trajectory + Arrow" if mk == "arrow" else "Trajectory Only"
        html.append(f'<h2 style="color:#58a6ff;margin:20px 0 8px">{mode_label} — Per-Cue Detail</h2>')

        for idx in sorted(all_cue_idxs.keys()):
            cue_name = all_cue_idxs[idx]
            # Check if any version got it right / wrong
            marks = []
            for v in sorted_vers:
                res = [r for r in ver_data[v].get(f"results_{mk}", []) if r["cue_idx"] == idx]
                if res:
                    marks.append("✅" if res[0]["correct"] else "❌")
                else:
                    marks.append("—")
            mark_str = " ".join(f'v{v}:{m}' for v, m in zip(sorted_vers, marks))

            html.append(f'<div class="card"><div class="card-hdr">'
                        f'<span class="tag">c{idx}</span> {cue_name}'
                        f'<span style="float:right;font-size:12px;color:#8b949e">{mark_str}</span></div>')
            html.append(f'<div class="ver-grid">')

            for v in sorted_vers:
                res = [r for r in ver_data[v].get(f"results_{mk}", []) if r["cue_idx"] == idx]
                r = res[0] if res else None
                fname = r.get("image", "") if r else ""
                b64 = _img_b64(fname) if fname else None
                no_img = '<div style="height:160px;background:#21262d;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#484f58;font-size:11px">No img</div>'
                img_tag = f'<img src="data:image/png;base64,{b64}">' if b64 else no_img
                mark = '<span class="mark">✅</span>' if r and r["correct"] else ('<span class="mark">❌</span>' if r else '—')

                html.append(f'<div class="ver-cell"><h4>{mark} v{v}</h4>{img_tag}')
                html.append(_opt_html(r))
                html.append('</div>')

            html.append('</div></div>')

    html.append("</body></html>")
    out_path = os.path.join(out_dir, "compare_report.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html))
    print(f"\n  Comparison report: {out_path}")


def trajectory_report(versions: list[int] = None, model: str = "gemini-2.5-flash-lite", robot: str = "IIWA"):
    """Regenerate comparison report from existing results."""
    if versions is None:
        versions = [5, 6, 7, 8, 9, 10]
    out_dir = os.path.join(MOTION_DIR, "vlm_trajectory_benchmark")
    _generate_comparison_report(out_dir, versions, model, robot)


if __name__ == "__main__":
    fire.Fire({"main": main, "report": report, "trajectory": trajectory,
               "compare": compare, "trajectory_report": trajectory_report})
