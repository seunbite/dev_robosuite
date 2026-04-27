"""
VLM Direction / Orientation / Height Benchmark

Uses pre-computed poses from closest_poses_results.jsonl to test
whether VLM can identify arm direction, gripper orientation, and height
from a single rendered image.

10 diverse test cases are selected (covering 6 directions, 2 orientations,
varied xyz positions) and rendered across multiple robots.

Usage:
    python vlm_direction_benchmark.py run                         # run with defaults
    python vlm_direction_benchmark.py run --robots IIWA Panda     # specific robots
    python vlm_direction_benchmark.py run --model gemini-2.5-flash-lite
    python vlm_direction_benchmark.py report                      # view HTML report
"""

import fire
import json
import os
import sys
import time
import random
import base64
import glob as globmod
from io import BytesIO
from collections import Counter, defaultdict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED_DIR = "data/seed"
MOTION_DIR = "data/motions"
JSONL_PATH = os.path.join(SEED_DIR, "closest_poses_results.jsonl")


# ── Pose selection ────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def select_test_poses(
    jsonl_path: str = JSONL_PATH,
    n: int = 10,
    seed: int = 42,
) -> list[dict]:
    """Select n diverse test poses covering directions, orientations, and heights.

    Algorithm:
      1. Group poses by (dir, gripper_orientation) — 12 buckets
      2. Round-robin pick from each bucket to ensure coverage
      3. Within each bucket, prefer poses with diverse z_pct (low/med/high)
    """
    rng = random.Random(seed)
    entries = _load_jsonl(jsonl_path)

    # Use IIWA as reference robot for selection (all robots have same poses)
    ref_entries = [e for e in entries if e.get("robot") == "IIWA"]
    if not ref_entries:
        ref_entries = entries

    # Filter out poses where EE is likely out of camera frame
    ref_entries = [e for e in ref_entries if _is_in_frame(e)]

    # Group by (dir, gripper_orientation)
    buckets = defaultdict(list)
    for e in ref_entries:
        key = (e.get("dir", "?"), e.get("gripper_orientation", "?"))
        buckets[key].append(e)

    # Sort each bucket by z_pct to enable stratified sampling
    for key in buckets:
        buckets[key].sort(key=lambda x: x.get("z_pct", 50))

    # Round-robin: first ensure all 6 directions are covered
    directions = ["front", "up", "down", "left", "right", "back"]
    orientations = ["horizontal", "vertical"]
    all_keys = [(d, o) for d in directions for o in orientations]
    rng.shuffle(all_keys)

    selected = []
    used_pose_ids = set()

    def _pick_from(bucket, prefer_z=None):
        """Pick a pose from bucket, preferring specific z_pct range."""
        candidates = [p for p in bucket if p["pose_id"] not in used_pose_ids]
        if not candidates:
            return None
        if prefer_z == "low":
            candidates.sort(key=lambda x: x.get("z_pct", 50))
        elif prefer_z == "high":
            candidates.sort(key=lambda x: -x.get("z_pct", 50))
        elif prefer_z == "mid":
            candidates.sort(key=lambda x: abs(x.get("z_pct", 50) - 50))
        else:
            rng.shuffle(candidates)
        pick = candidates[0]
        used_pose_ids.add(pick["pose_id"])
        return pick

    z_prefs = ["low", "mid", "high"]
    z_idx = 0
    for key in all_keys:
        if len(selected) >= n:
            break
        if key not in buckets or not buckets[key]:
            continue
        pick = _pick_from(buckets[key], prefer_z=z_prefs[z_idx % 3])
        if pick:
            selected.append(pick)
            z_idx += 1

    # If still need more, fill from remaining buckets
    remaining_keys = list(buckets.keys())
    rng.shuffle(remaining_keys)
    for key in remaining_keys:
        if len(selected) >= n:
            break
        pick = _pick_from(buckets[key], prefer_z=z_prefs[z_idx % 3])
        if pick:
            selected.append(pick)
            z_idx += 1

    return selected[:n]


# ── Rendering ─────────────────────────────────────────────────────────────────

def _project_3d_to_2d(point_3d, cam_pos, cam_rot, fovy, img_size):
    """Project a 3D world point to 2D pixel (image y=0 at top after flip)."""
    p_cam = cam_rot.T @ (point_3d - cam_pos)
    depth = -p_cam[2]
    if depth <= 0.01:
        return None
    f = 0.5 * img_size / np.tan(np.radians(fovy) / 2)
    px = int(f * p_cam[0] / depth + img_size / 2)
    py = int(img_size / 2 - f * p_cam[1] / depth)
    return (px, py)


def _draw_arrow(img, start, end, color="green", width=5, head_size=18):
    """Draw an arrow with arrowhead on a PIL Image."""
    draw = ImageDraw.Draw(img)
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max(1, (dx**2 + dy**2) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (int(end[0] - head_size * ux + head_size * 0.5 * px),
          int(end[1] - head_size * uy + head_size * 0.5 * py))
    p2 = (int(end[0] - head_size * ux - head_size * 0.5 * px),
          int(end[1] - head_size * uy - head_size * 0.5 * py))
    draw.polygon([end, p1, p2], fill=color)


def _render_pose(
    robot_name: str, pose: dict, img_size: int = 512, camera_fov_scale: float = 2.0,
    arrow_color: str = None, arrow_length: float = 0.15,
) -> Image.Image:
    """Render a single pose. Optionally draw direction arrow from EE z-axis."""
    import robosuite as suite
    from robosuite.controllers.composite.composite_controller_factory import (
        refactor_composite_controller_config,
    )

    arm_ctrl = suite.load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_ctrl, robot_name, ["right", "left"])

    env = suite.make(
        env_name="EmptySpace",
        robots=robot_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        control_freq=20,
        controller_configs=ctrl_cfg,
        horizon=1000,
    )
    env.reset()
    robot = env.robots[0]

    cam_id = env.sim.model.camera_name2id("frontview")
    original_fov = env.sim.model.cam_fovy[cam_id]
    env.sim.model.cam_fovy[cam_id] = min(120.0, original_fov * camera_fov_scale)

    joint_pos = robot._joint_positions.copy()
    active = pose.get("active_joint_indices", [])
    angles = pose.get("joint_angles_rad", [])
    for i, idx in enumerate(active):
        if i < len(angles) and idx < len(joint_pos):
            joint_pos[idx] = angles[i]

    robot.set_robot_joint_positions(joint_pos)
    env.sim.forward()

    cam_pos = env.sim.data.cam_xpos[cam_id].copy()
    cam_rot = env.sim.data.cam_xmat[cam_id].reshape(3, 3).copy()
    fovy = env.sim.model.cam_fovy[cam_id]

    obs = env.sim.render(camera_name="frontview", width=img_size, height=img_size, depth=False)
    img = Image.fromarray(obs[::-1])

    if arrow_color:
        arm_key = list(robot._hand_pos.keys())[0]
        ee_rot = robot._hand_orn[arm_key].copy()
        pointing_dir = ee_rot[:, 2]  # z-axis = gripper pointing direction

        # Use world-coordinate finger tip positions for accurate placement
        tip_pos = None
        model = env.sim.model
        tip_bodies = []
        for i in range(model.nbody):
            bname = model.body_id2name(i)
            if "finger" in bname and "tip" in bname:
                tip_bodies.append(env.sim.data.body_xpos[i].copy())
        if len(tip_bodies) >= 2:
            tip_pos = np.mean(tip_bodies, axis=0)
        else:
            for i in range(model.nsite):
                sname = model.site_id2name(i)
                if "grip_site" in sname and "cylinder" not in sname:
                    tip_pos = env.sim.data.site_xpos[i].copy()
                    break
        if tip_pos is None:
            tip_pos = robot._hand_pos[arm_key].copy()

        s3d = tip_pos
        e3d = tip_pos + pointing_dir * arrow_length
        s2d = _project_3d_to_2d(s3d, cam_pos, cam_rot, fovy, img_size)
        e2d = _project_3d_to_2d(e3d, cam_pos, cam_rot, fovy, img_size)

        if s2d and e2d:
            dx, dy = e2d[0] - s2d[0], e2d[1] - s2d[1]
            screen_len = (dx**2 + dy**2) ** 0.5
            min_len = 40
            if screen_len < min_len and screen_len > 1:
                scale = min_len / screen_len
                e2d = (int(s2d[0] + dx * scale), int(s2d[1] + dy * scale))
            _draw_arrow(img, s2d, e2d, color=arrow_color, width=5, head_size=18)

    env.close()
    return img


def _is_in_frame(pose: dict) -> bool:
    """Check if the end-effector is likely within camera frame.
    Reject extreme positions where the gripper would be clipped."""
    ee = pose.get("ee_position", {})
    z = ee.get("z", 0.5)
    if z < 0:
        return False
    x = abs(ee.get("x", 0))
    y = abs(ee.get("y", 0))
    if x > 0.6 or y > 0.7:
        return False
    return True


def render_all(
    test_poses: list[dict],
    robots: list[str],
    img_size: int = 512,
    output_dir: str = None,
    arrow_color: str = None,
) -> dict:
    """Render test poses across all robots. Returns {(pose_idx, robot): PIL.Image}."""
    if output_dir is None:
        output_dir = os.path.join(MOTION_DIR, "vlm_dir_benchmark")
    os.makedirs(output_dir, exist_ok=True)

    images = {}
    for robot in robots:
        print(f"\n  Rendering {len(test_poses)} poses for {robot}...")
        for pi, pose in enumerate(test_poses):
            robot_pose = _find_robot_pose(pose, robot)
            if robot_pose is None:
                print(f"    [SKIP] No pose data for {robot} pose_id={pose.get('pose_id')}")
                continue

            img = _render_pose(robot, robot_pose, img_size=img_size, arrow_color=arrow_color)
            suffix = "_arrow" if arrow_color else ""
            fname = f"{robot}_pose{pi}_{pose.get('dir','?')}_{pose.get('gripper_orientation','?')}{suffix}.png"
            img.save(os.path.join(output_dir, fname))
            images[(pi, robot)] = img
            print(f"    [{pi+1}/{len(test_poses)}] {pose.get('dir')} / {pose.get('gripper_orientation')} — rendered")

    return images


def _find_robot_pose(ref_pose: dict, robot: str) -> dict | None:
    """Find matching pose for a specific robot from JSONL (same dir + orientation, closest pose_id)."""
    entries = _load_jsonl(JSONL_PATH)
    target_dir = ref_pose.get("dir")
    target_grip = ref_pose.get("gripper_orientation")

    candidates = [
        e for e in entries
        if e.get("robot") == robot
        and e.get("dir") == target_dir
        and e.get("gripper_orientation") == target_grip
    ]
    if not candidates:
        return None

    # Pick the one with the best (lowest) orientation_diff
    candidates.sort(key=lambda x: x.get("orientation_diff_deg", 999))
    return candidates[0]


# ── VLM Tests ─────────────────────────────────────────────────────────────────

def _ask_vlm(client, model: str, image: Image.Image, prompt: str) -> str:
    """Send image + prompt to VLM, return raw text response."""
    try:
        response = client.models.generate_content(model=model, contents=[prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"ERROR: {e}"


def _test_direction(client, model: str, image: Image.Image, gt_dir: str,
                    arrow_color: str = None) -> dict:
    directions = ["up", "down", "left", "right", "front", "back"]
    option_text = "\n".join(f"  {chr(65+i)}. {d}" for i, d in enumerate(directions))
    gt_letter = chr(65 + directions.index(gt_dir)) if gt_dir in directions else "?"

    arrow_hint = ""
    if arrow_color:
        arrow_hint = (
            f"\nHINT: A {arrow_color.upper()} ARROW is drawn on the image starting from the gripper tip, "
            "showing the exact direction the gripper is pointing. "
            "Follow the arrow to determine the direction.\n"
        )

    prompt = (
        "This image shows a robot arm in a simulated room. The camera faces the robot from the front.\n\n"
        "TASK: Determine which direction the gripper TIP is pointing.\n"
        "Focus ONLY on the very tip of the gripper (the two-finger end piece). "
        "Ignore the arm joints, links, and elbow.\n"
        f"{arrow_hint}\n"
        "Direction definitions (from the camera's perspective):\n"
        "  - up: gripper tip points toward the ceiling\n"
        "  - down: gripper tip points toward the floor\n"
        "  - left: gripper tip points to the LEFT side of the image\n"
        "  - right: gripper tip points to the RIGHT side of the image\n"
        "  - front: gripper tip points toward the camera (toward you). "
        "The arrow may appear as a short dot because it's pointing directly at you.\n"
        "  - back: gripper tip points away from the camera (into the scene). "
        "The arrow may appear as a short dot because it's pointing away.\n\n"
        f"{option_text}\n\n"
        "Reply with ONLY the letter (A-F)."
    )
    raw = _ask_vlm(client, model, image, prompt)
    answer = "?"
    for ch in raw.upper():
        if ch in "ABCDEF":
            answer = ch
            break

    options = {chr(65+i): d for i, d in enumerate(directions)}
    return {
        "test": "direction",
        "gt_dir": gt_dir,
        "gt_letter": gt_letter,
        "answer": answer,
        "correct": answer == gt_letter,
        "options": options,
        "chosen_text": options.get(answer, raw[:40]),
    }


def _test_orientation(client, model: str, image: Image.Image, gt_grip: str) -> dict:
    prompt = (
        "This image shows a robot arm with a two-finger gripper in a simulated room.\n\n"
        "TASK: Determine the gripper orientation — how the gripper's opening plane is aligned "
        "relative to the ground.\n\n"
        "Focus ONLY on the gap (opening) between the two finger pads at the very tip of the arm.\n\n"
        "DEFINITIONS:\n"
        "  VERTICAL — The gripper's opening plane is PERPENDICULAR to the ground. "
        "The two fingers are side by side horizontally (one left, one right, or one in front, one behind). "
        "The gap between them runs roughly left-right or front-back. "
        "Think of it like a handshake grip or a shushing gesture — "
        "the flat part of the gripper faces sideways.\n\n"
        "  HORIZONTAL — The gripper's opening plane is PARALLEL to the ground. "
        "The two fingers are stacked vertically (one on top, one on bottom). "
        "The gap between them runs roughly up-down. "
        "Think of it like picking something up from a table — "
        "the flat part of the gripper faces up or down.\n\n"
        "HINT: Look at the two finger pads. If one finger is ABOVE the other → HORIZONTAL. "
        "If the fingers are BESIDE each other (same height) → VERTICAL.\n\n"
        "  A. vertical\n"
        "  B. horizontal\n\n"
        "Reply with ONLY the letter (A or B)."
    )
    gt_letter = "A" if gt_grip == "vertical" else "B"
    raw = _ask_vlm(client, model, image, prompt)
    answer = "?"
    for ch in raw.upper():
        if ch in "AB":
            answer = ch
            break

    options = {"A": "vertical", "B": "horizontal"}
    return {
        "test": "orientation",
        "gt_gripper": gt_grip,
        "gt_letter": gt_letter,
        "answer": answer,
        "correct": answer == gt_letter,
        "options": options,
        "chosen_text": options.get(answer, raw[:40]),
    }


def _test_height(client, model: str, image: Image.Image, gt_z_pct: int) -> dict:
    prompt = (
        "Look at this robot arm.\n"
        "On a scale from 0 to 100, how high is the robot's end-effector (gripper)?\n"
        "0 = very low (near the base), 100 = very high (highest reachable point).\n\n"
        "Reply with ONLY a number between 0 and 100."
    )
    raw = _ask_vlm(client, model, image, prompt)
    num = ""
    for ch in raw:
        if ch.isdigit():
            num += ch
        elif num:
            break
    predicted = int(num) if num else -1
    error = abs(predicted - gt_z_pct) if predicted >= 0 else -1

    return {
        "test": "height",
        "gt_z": gt_z_pct,
        "predicted": predicted,
        "error": error,
    }


# ── Main run ──────────────────────────────────────────────────────────────────

def run(
    robots: list[str] = None,
    model: str = "gemini-2.5-flash-lite",
    n_poses: int = 10,
    delay: float = 2.0,
    seed: int = 42,
    img_size: int = 512,
    arrow_color: str = None,
):
    """Run VLM direction/orientation/height benchmark.

    Args:
        robots: List of robots to test (default: IIWA, Panda, XArm7)
        model: Gemini model name
        n_poses: Number of test poses
        delay: Seconds between API calls
        seed: Random seed
        img_size: Render image size
    """
    if robots is None:
        robots = ["IIWA", "Panda", "XArm7"]

    random.seed(seed)
    from google import genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)

    # 1. Select diverse test poses
    print(f"\n{'═'*70}")
    print(f"  VLM DIRECTION BENCHMARK")
    print(f"  Model: {model}  |  Robots: {robots}  |  Poses: {n_poses}")
    print(f"{'═'*70}\n")

    test_poses = select_test_poses(n=n_poses, seed=seed)
    print(f"  Selected {len(test_poses)} test poses:")
    for i, p in enumerate(test_poses):
        z = p.get("z_pct", "?")
        print(f"    [{i}] dir={p['dir']:6s}  grip={p['gripper_orientation']:10s}  z_pct={z}")
    print()

    # 2. Render
    arrow_label = f" (arrow={arrow_color})" if arrow_color else ""
    print(f"  Rendering poses...{arrow_label}")
    images = render_all(test_poses, robots, img_size=img_size, arrow_color=arrow_color)
    print(f"\n  Rendered {len(images)} images total.\n")

    # 3. VLM tests
    results = []
    total = len(test_poses) * len(robots)
    done = 0
    for pi, pose in enumerate(test_poses):
        gt_dir = pose.get("dir", "?")
        gt_grip = pose.get("gripper_orientation", "?")
        gt_z = pose.get("z_pct", 50)

        for robot in robots:
            img = images.get((pi, robot))
            if img is None:
                continue

            done += 1
            label = f"[{done}/{total}] {robot} pose{pi} dir={gt_dir} grip={gt_grip}"
            print(f"  {label}")

            r_dir = _test_direction(client, model, img, gt_dir, arrow_color=arrow_color)
            time.sleep(delay)
            r_ori = _test_orientation(client, model, img, gt_grip)
            time.sleep(delay)
            r_hgt = _test_height(client, model, img, gt_z)
            time.sleep(delay)

            mark_d = "✅" if r_dir["correct"] else f"❌ chose={r_dir['chosen_text']}"
            mark_o = "✅" if r_ori["correct"] else f"❌ chose={r_ori['chosen_text']}"
            mark_h = f"err={r_hgt['error']}" if r_hgt["error"] >= 0 else "FAIL"

            print(f"    dir: {mark_d}  |  grip: {mark_o}  |  height(gt={gt_z}): pred={r_hgt['predicted']} {mark_h}")

            results.append({
                "pose_idx": pi,
                "robot": robot,
                "gt_dir": gt_dir,
                "gt_grip": gt_grip,
                "gt_z": gt_z,
                "direction": r_dir,
                "orientation": r_ori,
                "height": r_hgt,
            })

    # 4. Summary
    print(f"\n{'═'*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*70}\n")

    dirs_list = ["up", "down", "left", "right", "front", "back"]
    grips_list = ["horizontal", "vertical"]

    for test_key, label, chance in [
        ("direction", "Direction (6-way)", "17%"),
        ("orientation", "Gripper Orientation", "50%"),
    ]:
        correct = sum(1 for r in results if r[test_key]["correct"])
        n = len(results)
        acc = 100 * correct / n if n else 0
        bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"  {label}:")
        print(f"    Overall: {correct}/{n} ({acc:.0f}%)  {bar}  (chance: {chance})")

        # Per-robot breakdown
        print(f"    By robot:")
        for robot in robots:
            rc = [r for r in results if r["robot"] == robot]
            c = sum(1 for r in rc if r[test_key]["correct"])
            a = 100 * c / len(rc) if rc else 0
            print(f"      {robot:10s}: {c}/{len(rc)} ({a:.0f}%)")

        # Per-direction breakdown
        print(f"    By direction:")
        for d in dirs_list:
            rc = [r for r in results if r["gt_dir"] == d]
            if rc:
                c = sum(1 for r in rc if r[test_key]["correct"])
                a = 100 * c / len(rc) if rc else 0
                wrong = [r[test_key].get("chosen_text", "?") for r in rc if not r[test_key]["correct"]]
                wrong_str = f"  wrong→{dict(Counter(wrong))}" if wrong else ""
                print(f"      {d:6s}: {c}/{len(rc)} ({a:.0f}%){wrong_str}")

        # Per-orientation breakdown
        print(f"    By grip orientation:")
        for g in grips_list:
            rc = [r for r in results if r["gt_grip"] == g]
            if rc:
                c = sum(1 for r in rc if r[test_key]["correct"])
                a = 100 * c / len(rc) if rc else 0
                print(f"      {g:12s}: {c}/{len(rc)} ({a:.0f}%)")
        print()

    # Height
    h_errors = [r["height"]["error"] for r in results if r["height"]["error"] >= 0]
    if h_errors:
        mae = sum(h_errors) / len(h_errors)
        w10 = sum(1 for e in h_errors if e <= 10)
        w20 = sum(1 for e in h_errors if e <= 20)
        print(f"  Height Estimation:")
        print(f"    Overall MAE: {mae:.1f}  |  ±10: {w10}/{len(h_errors)} ({100*w10/len(h_errors):.0f}%)  |  ±20: {w20}/{len(h_errors)} ({100*w20/len(h_errors):.0f}%)")
        print(f"    By robot:")
        for robot in robots:
            rc = [r for r in results if r["robot"] == robot and r["height"]["error"] >= 0]
            if rc:
                rm = sum(r["height"]["error"] for r in rc) / len(rc)
                print(f"      {robot:10s}: MAE={rm:.1f}")
        print(f"    By direction:")
        for d in dirs_list:
            rc = [r for r in results if r["gt_dir"] == d and r["height"]["error"] >= 0]
            if rc:
                rm = sum(r["height"]["error"] for r in rc) / len(rc)
                print(f"      {d:6s}: MAE={rm:.1f}")
    print()

    # 5. Save
    out_dir = os.path.join(MOTION_DIR, "vlm_dir_benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{model}.json")
    save_data = {
        "model": model,
        "robots": robots,
        "n_poses": n_poses,
        "seed": seed,
        "test_poses": [{k: v for k, v in p.items() if k != "_rot_mat"} for p in test_poses],
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Results saved: {out_path}")

    # 6. Auto-generate report
    _generate_report(out_dir, robots)


# ── HTML Report ───────────────────────────────────────────────────────────────

def _generate_report(out_dir: str, default_robots: list[str] = None):
    """Generate HTML report from all result JSONs in out_dir."""
    result_files = sorted(globmod.glob(os.path.join(out_dir, "results_*.json")))
    if not result_files:
        print("No result files found.")
        return

    all_data = {}
    for rf in result_files:
        d = json.load(open(rf))
        model_key = d["model"]
        stem = os.path.splitext(os.path.basename(rf))[0].replace("results_", "")
        if stem != model_key:
            suffix = stem[len(model_key):] if stem.startswith(model_key) else f" ({stem})"
            model_key = model_key + suffix
        if model_key in all_data:
            model_key = model_key + f" ({os.path.basename(rf)})"
        all_data[model_key] = d

    models = list(all_data.keys())
    first = all_data[models[0]]
    test_poses = first["test_poses"]
    robots = first.get("robots", default_robots or ["IIWA"])

    def _img_b64(robot, pose_idx, pose, suffix=""):
        d = pose.get("dir", "?")
        g = pose.get("gripper_orientation", "?")
        fname = f"{robot}_pose{pose_idx}_{d}_{g}{suffix}.png"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            return None
        img = Image.open(fpath)
        img.thumbnail((200, 200))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _opt_html(options, gt_letter, answer):
        lines = []
        for letter, text in sorted(options.items()):
            cls = []
            if letter == gt_letter:
                cls.append("gt")
            if letter == answer:
                cls.append("chosen")
            icon = ""
            if letter == gt_letter and letter == answer:
                icon = "✅ "
            elif letter == answer:
                icon = "❌ "
            elif letter == gt_letter:
                icon = "🎯 "
            lines.append(f'<div class="opt {" ".join(cls)}">{icon}<b>{letter}.</b> {text}</div>')
        return "\n".join(lines)

    html = [f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VLM Direction Benchmark</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background:#0d1117; color:#c9d1d9; padding:20px; }}
  h1 {{ color:#58a6ff; margin-bottom:4px; }}
  .sub {{ color:#8b949e; margin-bottom:16px; }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin:12px 0 20px; }}
  .sc {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 16px; min-width:160px; }}
  .sc h3 {{ color:#8b949e; font-size:11px; text-transform:uppercase; margin-bottom:4px; }}
  .sc .v {{ font-size:20px; font-weight:700; }}
  .sc .v.good {{ color:#3fb950; }} .sc .v.mid {{ color:#d29922; }} .sc .v.bad {{ color:#f85149; }}
  .pose-card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; margin-bottom:16px; overflow:hidden; }}
  .pose-hdr {{ background:#21262d; padding:10px 14px; display:flex; align-items:center; gap:10px; }}
  .pose-hdr .tag {{ background:#30363d; border-radius:5px; padding:3px 8px; font-weight:700; color:#58a6ff; font-size:13px; }}
  .pose-hdr .info {{ font-size:13px; }}
  .pose-body {{ display:flex; flex-wrap:wrap; gap:0; }}
  .robot-cell {{ border-right:1px solid #30363d; padding:10px; min-width:240px; }}
  .robot-cell:last-child {{ border-right:none; }}
  .robot-cell h4 {{ font-size:12px; color:#8b949e; margin-bottom:6px; }}
  .robot-cell img {{ border-radius:6px; display:block; margin-bottom:8px; }}
  .test-box {{ border:1px solid #30363d; border-radius:6px; padding:6px; margin-bottom:6px; }}
  .test-box h5 {{ font-size:11px; color:#8b949e; margin-bottom:4px; text-transform:uppercase; }}
  .opt {{ font-size:12px; line-height:1.5; padding:1px 4px; border-radius:3px; }}
  .opt.gt {{ background:rgba(63,185,80,0.15); border-left:3px solid #3fb950; }}
  .opt.chosen {{ background:rgba(248,81,73,0.15); border-left:3px solid #f85149; }}
  .opt.gt.chosen {{ background:rgba(63,185,80,0.25); border-left:3px solid #3fb950; }}
  .correct {{ color:#3fb950; }} .close {{ color:#d29922; }} .wrong {{ color:#f85149; }}
</style></head><body>
<h1>VLM Direction Benchmark</h1>
<p class="sub">Models: {', '.join(models)} | Robots: {', '.join(robots)} | Poses: {len(test_poses)}</p>
<div class="summary">"""]

    # Summary cards per model
    for m in models:
        res = all_data[m]["results"]
        n = len(res)
        d_corr = sum(1 for r in res if r["direction"]["correct"])
        o_corr = sum(1 for r in res if r["orientation"]["correct"])
        h_err = [r["height"]["error"] for r in res if r["height"]["error"] >= 0]
        mae = sum(h_err) / len(h_err) if h_err else -1
        dc = "good" if d_corr/n>.3 else ("mid" if d_corr/n>.17 else "bad")
        oc = "good" if o_corr/n>.6 else ("mid" if o_corr/n>.5 else "bad")
        hc = "good" if mae<15 else ("mid" if mae<25 else "bad")
        html.append(f"""<div class="sc"><h3>{m}</h3>
  <div>Dir: <span class="v {dc}">{d_corr}/{n} ({100*d_corr/n:.0f}%)</span></div>
  <div>Grip: <span class="v {oc}">{o_corr}/{n} ({100*o_corr/n:.0f}%)</span></div>
  <div>Height MAE: <span class="v {hc}">{mae:.1f}</span></div></div>""")
    html.append("</div>")

    # Per-pose cards
    for pi, pose in enumerate(test_poses):
        gt_d = pose.get("dir", "?")
        gt_g = pose.get("gripper_orientation", "?")
        gt_z = pose.get("z_pct", "?")
        html.append(f"""<div class="pose-card">
  <div class="pose-hdr">
    <span class="tag">Pose {pi}</span>
    <span class="info">dir=<b>{gt_d}</b> | grip=<b>{gt_g}</b> | z_pct=<b>{gt_z}</b></span>
  </div><div class="pose-body">""")

        for robot in robots:
            b64 = _img_b64(robot, pi, pose)
            b64_arrow = _img_b64(robot, pi, pose, suffix="_arrow")
            no_img = '<div style="width:180px;height:180px;background:#21262d;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#484f58">No img</div>'
            img_tag = f'<img src="data:image/png;base64,{b64}" width="180">' if b64 else no_img
            if b64_arrow:
                arrow_tag = f'<img src="data:image/png;base64,{b64_arrow}" width="180">'
                imgs_html = (f'<div style="display:flex;gap:6px;margin-bottom:6px">'
                             f'<div><div style="font-size:10px;color:#8b949e;margin-bottom:2px">Original</div>{img_tag}</div>'
                             f'<div><div style="font-size:10px;color:#8b949e;margin-bottom:2px">With Arrow</div>{arrow_tag}</div></div>')
            else:
                imgs_html = img_tag

            html.append(f'<div class="robot-cell"><h4>{robot}</h4>{imgs_html}')

            for m in models:
                res_list = [r for r in all_data[m]["results"]
                            if r["pose_idx"] == pi and r["robot"] == robot]
                if not res_list:
                    continue
                r = res_list[0]

                # Direction
                rd = r["direction"]
                html.append(f'<div class="test-box"><h5>Direction ({m})</h5>')
                html.append(_opt_html(rd.get("options", {}), rd.get("gt_letter", ""), rd.get("answer", "")))
                html.append("</div>")

                # Orientation
                ro = r["orientation"]
                html.append(f'<div class="test-box"><h5>Gripper ({m})</h5>')
                html.append(_opt_html(ro.get("options", {}), ro.get("gt_letter", ""), ro.get("answer", "")))
                html.append("</div>")

                # Height
                rh = r["height"]
                err = rh.get("error", -1)
                cls = "correct" if err <= 10 else ("close" if err <= 20 else "wrong")
                html.append(f'<div class="test-box"><h5>Height ({m})</h5>')
                html.append(f'<span class="{cls}">GT={rh.get("gt_z","?")} → Pred={rh.get("predicted","?")} (err={err})</span>')
                html.append("</div>")

            html.append("</div>")

        html.append("</div></div>")

    html.append("</body></html>")

    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html))
    print(f"  Report saved: {out_path}")


def report():
    """Generate/open HTML report from existing results."""
    out_dir = os.path.join(MOTION_DIR, "vlm_dir_benchmark")
    _generate_report(out_dir)


def preview(
    robots: list[str] = None,
    n_poses: int = 10,
    seed: int = 42,
    arrow_color: str = "green",
    img_size: int = 512,
    output: str = None,
):
    """Render poses with direction arrows and save a grid preview image.

    Args:
        robots: Robots to render
        n_poses: Number of poses
        seed: Random seed
        arrow_color: Arrow color (e.g. "green", "red", "#00FF00")
        img_size: Render size per cell
        output: Output path for grid PNG
    """
    if robots is None:
        robots = ["IIWA", "Panda", "XArm7"]

    test_poses = select_test_poses(n=n_poses, seed=seed)
    print(f"  Selected {len(test_poses)} poses, rendering with arrow_color={arrow_color}...")
    images = render_all(test_poses, robots, img_size=img_size, arrow_color=arrow_color)

    # Build grid: rows = poses, cols = robots
    n_rows = len(test_poses)
    n_cols = len(robots)
    cell = 256
    label_h = 32
    header_h = 40

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    grid_w = n_cols * cell
    grid_h = header_h + n_rows * (cell + label_h)
    grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 30))
    draw = ImageDraw.Draw(grid)

    # Header row with robot names
    for ci, robot in enumerate(robots):
        x = ci * cell + cell // 2
        draw.text((x, 8), robot, fill=(100, 180, 255), font=font, anchor="mt")

    for ri, pose in enumerate(test_poses):
        y_off = header_h + ri * (cell + label_h)
        d = pose.get("dir", "?")
        g = pose.get("gripper_orientation", "?")

        for ci, robot in enumerate(robots):
            x_off = ci * cell
            img = images.get((ri, robot))
            if img:
                thumb = img.copy()
                thumb.thumbnail((cell, cell))
                grid.paste(thumb, (x_off, y_off))

            label = f"p{ri} {d}/{g}"
            draw.text((x_off + cell // 2, y_off + cell + 2), label,
                      fill=(180, 180, 180), font=small_font, anchor="mt")

    if output is None:
        output = os.path.join(MOTION_DIR, "vlm_dir_benchmark",
                              f"preview_arrow_{arrow_color}.png")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    grid.save(output)
    print(f"  Preview saved: {output}")
    return output


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report, "select": select_test_poses, "preview": preview})
