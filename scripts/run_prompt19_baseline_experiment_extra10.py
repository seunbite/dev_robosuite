from __future__ import annotations

import copy
import html
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fire


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
BASE_OUT = SEED / "baseline_prompt19_direct_experiment_extra10"
MOTION_OUT = MOTIONS / "baseline_prompt19_direct_experiment_extra10"

ICONIC_SRC = SEED / "motion_configs_prompt_v19_sophisticated.json"
CONTEXTUAL_SRC = SEED / "motion_configs_prompt_v19_sophisticated_contextual.json"
SHOT_SRC = SEED / "q4_contrastive_experiment" / "shot_configs_v19_sophisticated_q4_contrastive.json"
BASE_PROMPT = SEED / "q4_contrastive_experiment" / "prompt_v19_sophisticated_q4_contrastive_full.txt"
POSE_DB = SEED / "closest_poses_results.jsonl"

NO_REASONING_PROMPT = BASE_OUT / "prompt_v19_sophisticated_no_reasoning_baseline.txt"
NO_REASONING_SHOTS = BASE_OUT / "shot_configs_v19_sophisticated_no_reasoning.json"
JOINT_PROMPT = BASE_OUT / "prompt_v19_sophisticated_direct_joint.txt"
XYZ_PROMPT = BASE_OUT / "prompt_v19_sophisticated_direct_xyz_theta.txt"
JOINT_SHOTS_JSON = BASE_OUT / "fewshot_joint_examples.json"
XYZ_SHOTS_JSON = BASE_OUT / "fewshot_xyz_theta_examples.json"

NO_REASONING_ICONIC_JSON = BASE_OUT / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json"
NO_REASONING_CONTEXTUAL_JSON = BASE_OUT / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json"
DIRECT_JOINT_JSON = BASE_OUT / "motion_configs_prompt_v19_sophisticated_direct_joint.json"
DIRECT_XYZ_JSON = BASE_OUT / "motion_configs_prompt_v19_sophisticated_direct_xyz_theta.json"
MANIFEST_JSON = BASE_OUT / "manifest.json"
HTML_OUT = BASE_OUT / "prompt19_baseline_experiment_extra10_compare_20260406_ko.html"
FALLBACK_GEMINI_KEY_FILE = Path("/Users/sb/Downloads/workspace/Motion2Mind/src/motion2mind/vlm/delete_gemini.py")


TARGETS = [
    {"dataset": "iconic", "idx": 1, "cue": "raising_hand_greeting"},
    {"dataset": "iconic", "idx": 13, "cue": "point_self"},
    {"dataset": "iconic", "idx": 20, "cue": "rub_eye_tired"},
    {"dataset": "iconic", "idx": 22, "cue": "circle_temple_crazy"},
    {"dataset": "iconic", "idx": 50, "cue": "firm_accept_forward_reach"},
    {"dataset": "contextual", "idx": 0, "cue": "nod_yes"},
    {"dataset": "contextual", "idx": 9, "cue": "deep_bow_apology"},
    {"dataset": "contextual", "idx": 14, "cue": "curl_fingers_give_me"},
    {"dataset": "contextual", "idx": 24, "cue": "disbelief_hold_then_drop"},
    {"dataset": "contextual", "idx": 42, "cue": "laughter_substitute"},
]


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text)).strip("_") or "item"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _strip_reasoning_from_shots() -> list[dict]:
    rows = copy.deepcopy(_load_json(SHOT_SRC))
    for row in rows:
        row.pop("reasoning", None)
        row.pop("planning_shot", None)
    return rows


def _build_no_reasoning_prompt() -> str:
    return """You are an expert in iconic robot motion design. Your goal is to create one highly readable motion cue for a single robot arm with a parallel-jaw gripper and output it as a structured JSON.

[pose Selection Rules]
1. Direction (dir):
- front: Handshake / offering.
- back: Indicating "me".
- up: Greeting / high-five.
- down: At attention / resting.
- left/right: Lateral actions (e.g., rubbing eyes, tapping temple).

2. Orientation:
- vertical: Perpendicular to ground (Handshake, shush).
- horizontal: Parallel to ground (Wave, presenting, Grabbing cup).

3. Axis Mapping for movement axis and gripper location:
- x: Depth (Forward/Backward extension).
- y: Lateral (Side-to-side negation/wave).
- z: Height (Up-and-down affirmation/nod).
- Coupled Motion: Use x+z for bowing or recoiling.

4. Body Reference:
Head: z85-100, Eye: z65-75, Chest: z45-55, Waist: z35-45, Extended: x70-90.

[Hard Constraints]
- Every motion MUST start with a pose.
- Total steps: 4-8.
- Speed: 0.5 (slow) - 4.0 (sharp). Default: 1.0-2.0.
- Path (line) distance: Default 5.0-120.0.
- Path (arc) radius: Default 5.0-25.0.
- If the cue resembles an acted animation beat, prefer a richer multi-phase sequence whose neighboring motions express the same context as the core action.
- Do not add a final static pose unless it is essential.
- No prose or extra explanation. No markdown code blocks.

Output only a single JSON object starting with {.

{{FEW_SHOT_EXAMPLES}}
---
Target Cue:
Cue Name: {{CUE_NAME}}
---
"""


def _compact_shot_block(row: dict) -> str:
    clean = {k: v for k, v in row.items() if k not in {"reasoning", "planning_shot"}}
    return json.dumps(clean, ensure_ascii=False, indent=2)


def _build_no_reasoning_examples(shots: list[dict]) -> str:
    return "\n\n".join(_compact_shot_block(row) for row in shots)


def _parse_json_object(raw_text: str) -> dict:
    raw = raw_text.strip()
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates = [raw] + [blk.strip() for blk in fenced if blk.strip()]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for pos, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[pos:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError(f"Could not parse JSON object from model output: {raw[:500]}")


def _ensure_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    if FALLBACK_GEMINI_KEY_FILE.exists():
        text = FALLBACK_GEMINI_KEY_FILE.read_text(encoding="utf-8")
        match = re.search(r'api_key="([^"]+)"', text)
        if match:
            api_key = match.group(1).strip()
            os.environ["GOOGLE_API_KEY"] = api_key
            return api_key
    raise ValueError("GOOGLE_API_KEY is not set and no local fallback key was found")


def _latest_gif(base: Path, cue: str, cue_idx: int | None = None) -> Path | None:
    safe = _safe_name(cue)
    if cue_idx is not None:
        tiled = sorted(base.rglob(f"*_{safe}_c{cue_idx}_tiled.gif"))
        if tiled:
            return tiled[-1]
    single = sorted(base.rglob(f"*_{safe}_p*.gif"))
    if single:
        return single[-1]
    any_match = sorted(base.rglob(f"*_{safe}_*.gif"))
    return any_match[-1] if any_match else None


def _latest_single_gif(base: Path, cue: str) -> Path | None:
    safe = _safe_name(cue)
    single = sorted(base.rglob(f"*_{safe}_p*.gif"))
    if single:
        return single[-1]
    any_match = sorted(base.rglob(f"*_{safe}_*.gif"))
    return any_match[-1] if any_match else None


def _target_rows() -> dict[str, dict[int, dict]]:
    iconic = {int(r["idx"]): r for r in _load_json(ICONIC_SRC)}
    contextual = {int(r["idx"]): r for r in _load_json(CONTEXTUAL_SRC)}
    return {"iconic": iconic, "contextual": contextual}


def _upsert_row(path: Path, row: dict) -> list[dict]:
    rows = _load_json(path) if path.exists() else []
    rows = [r for r in rows if not (int(r.get("idx", -1)) == int(row["idx"]) and r.get("cue") == row["cue"])]
    rows.append(row)
    rows.sort(key=lambda x: (int(x["idx"]), x["cue"]))
    _write_json(path, rows)
    return rows


@dataclass
class ShotTrace:
    idx: int
    cue: str
    state: str
    sampled_joint_keyframes: list[dict]
    sampled_xyz_keyframes: list[dict]
    gif_path: str


def extract_shot_sequences(
    robot: str = "IIWA",
    hz: int = 8,
    sample_count: int = 8,
) -> None:
    import numpy as np

    sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
    from motion_generation import MotionGenerator, _select_initial_poses  # noqa: E402
    import robosuite.utils.transform_utils as T  # noqa: E402

    BASE_OUT.mkdir(parents=True, exist_ok=True)
    shots = _load_json(SHOT_SRC)
    shot_render_dir = MOTION_OUT / "shot_examples" / robot
    shot_render_dir.mkdir(parents=True, exist_ok=True)

    traces: list[ShotTrace] = []
    gen = MotionGenerator(
        robot_name=robot,
        jsonl_path=str(POSE_DB),
        output_dir=str(MOTION_OUT / "shot_examples"),
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        for row in shots:
            states: list[dict] = []
            orig_capture = gen._capture_image

            def _capture_and_log():
                q = gen._get_joint_positions().copy()
                site_id = gen.env.sim.model.site_name2id(gen.jacobian_calculator.eef_site_name)
                pos = gen.env.sim.data.site_xpos[site_id].copy()
                rot = gen.env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
                euler = T.mat2euler(rot)
                states.append(
                    {
                        "q_deg": np.round(np.rad2deg(q), 2).tolist(),
                        "xyz": np.round(pos, 5).tolist(),
                        "theta_deg": round(float(np.rad2deg(euler[1])), 2),
                    }
                )
                return orig_capture()

            gen._capture_image = _capture_and_log  # type: ignore[assignment]
            pose_def = next(
                m["parameters"]["pose"]
                for m in row["movements"]
                if m.get("type") == "pose"
            )
            matching = gen._find_matching_poses(pose_def)
            selected = _select_initial_poses(matching, pose_def, top_k=1)
            gen.output_dir = str(shot_render_dir)
            gen._set_joint_positions(gen.initial_joint_pos)
            gen.execute_cue(
                cue=row["cue"],
                pose_index=selected[0]["pose_id"],
                config_path=str(SHOT_SRC),
                hz=int(hz),
                cue_idx=int(row["idx"]),
                save_gif=True,
            )
            gen._capture_image = orig_capture  # type: ignore[assignment]

            if not states:
                continue

            pick = sorted(set(int(round(i * (len(states) - 1) / max(1, sample_count - 1))) for i in range(sample_count)))
            joint_keys = [{"t": round(i / max(1, len(pick) - 1), 3), "q_deg": states[idx]["q_deg"]} for i, idx in enumerate(pick)]
            xyz_keys = [
                {
                    "t": round(i / max(1, len(pick) - 1), 3),
                    "x": states[idx]["xyz"][0],
                    "y": states[idx]["xyz"][1],
                    "z": states[idx]["xyz"][2],
                    "theta_deg": states[idx]["theta_deg"],
                }
                for i, idx in enumerate(pick)
            ]

            gif = _latest_gif(shot_render_dir, row["cue"], int(row["idx"]))
            traces.append(
                ShotTrace(
                    idx=int(row["idx"]),
                    cue=row["cue"],
                    state=row.get("state", ""),
                    sampled_joint_keyframes=joint_keys,
                    sampled_xyz_keyframes=xyz_keys,
                    gif_path=str(gif) if gif else "",
                )
            )
    finally:
        gen.close()

    _write_json(
        JOINT_SHOTS_JSON,
        [
            {
                "idx": t.idx,
                "cue": t.cue,
                "state": t.state,
                "joint_keyframes": t.sampled_joint_keyframes,
                "gif_path": t.gif_path,
            }
            for t in traces
        ],
    )
    _write_json(
        XYZ_SHOTS_JSON,
        [
            {
                "idx": t.idx,
                "cue": t.cue,
                "state": t.state,
                "cartesian_keyframes": t.sampled_xyz_keyframes,
                "gif_path": t.gif_path,
            }
            for t in traces
        ],
    )


def _fewshot_joint_text() -> str:
    rows = _load_json(JOINT_SHOTS_JSON)
    parts = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"# Example Cue: {row['cue']}",
                    json.dumps(
                        {
                            "cue": row["cue"],
                            "joint_keyframes": row["joint_keyframes"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
        )
    return "\n\n".join(parts)


def _fewshot_xyz_text() -> str:
    rows = _load_json(XYZ_SHOTS_JSON)
    parts = []
    for row in rows:
        parts.append(
            "\n".join(
                [
                    f"# Example Cue: {row['cue']}",
                    json.dumps(
                        {
                            "cue": row["cue"],
                            "cartesian_keyframes": row["cartesian_keyframes"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                ]
            )
        )
    return "\n\n".join(parts)


def write_baseline_prompts() -> None:
    shots = _strip_reasoning_from_shots()
    _write_json(NO_REASONING_SHOTS, shots)
    NO_REASONING_PROMPT.write_text(
        _build_no_reasoning_prompt().replace("{{FEW_SHOT_EXAMPLES}}", _build_no_reasoning_examples(shots)),
        encoding="utf-8",
    )

    joint_prompt = f"""You are designing one motion cue for a single robot arm.

Instead of using Pose / Movement / Path primitives, output the motion directly as absolute joint-angle keyframes.

Rules:
- Output only one JSON object starting with {{.
- Use keys: description, joint_keyframes.
- joint_keyframes must be a list of 5-9 items.
- Each keyframe must have:
  - t: normalized time from 0.0 to 1.0, strictly increasing
  - q_deg: absolute joint angles in degrees for the full robot arm
- Keep the motion readable and cue-specific.
- No reasoning, no markdown.

Few-shot examples:
{_fewshot_joint_text()}

Target Cue:
Cue Name: {{CUE_NAME}}
"""
    xyz_prompt = f"""You are designing one motion cue for a single robot arm.

Instead of using Pose / Movement / Path primitives, output the motion directly as end-effector cartesian keyframes.

Rules:
- Output only one JSON object starting with {{.
- Use keys: description, cartesian_keyframes.
- cartesian_keyframes must be a list of 5-9 items.
- Each keyframe must have:
  - t: normalized time from 0.0 to 1.0, strictly increasing
  - x, y, z: absolute end-effector position in meters
  - theta_deg: end-effector pitch angle in degrees
- Keep the motion readable and cue-specific.
- No reasoning, no markdown.

Few-shot examples:
{_fewshot_xyz_text()}

Target Cue:
Cue Name: {{CUE_NAME}}
"""
    JOINT_PROMPT.write_text(joint_prompt, encoding="utf-8")
    XYZ_PROMPT.write_text(xyz_prompt, encoding="utf-8")


def generate_no_reasoning(model_name: str = "gemini-2.5-pro") -> None:
    sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
    from config_gen_single import generate_motion_config  # noqa: E402

    _ensure_google_api_key()
    for spec in TARGETS:
        out_path = NO_REASONING_ICONIC_JSON if spec["dataset"] == "iconic" else NO_REASONING_CONTEXTUAL_JSON
        print(f"[no_reasoning] {spec['dataset']} c{spec['idx']} {spec['cue']}", flush=True)
        generate_motion_config(
            cue_name=spec["cue"],
            cue_idx=int(spec["idx"]),
            model_name=model_name,
            prompt_file=str(NO_REASONING_PROMPT),
            shots_json=str(NO_REASONING_SHOTS),
            config_json=str(out_path),
            max_handmade_examples=10,
            max_correction_examples=0,
            temperature=None,
            use_shots=False,
            require_reasoning=False,
        )


def _llm_client():
    from google import genai  # type: ignore

    api_key = _ensure_google_api_key()
    return genai.Client(api_key=api_key)


def _generate_direct_one(prompt_text: str, model_name: str) -> dict:
    client = _llm_client()
    response = client.models.generate_content(model=model_name, contents=prompt_text)
    return _parse_json_object(response.text.strip())


def generate_direct_sequences(model_name: str = "gemini-2.5-pro") -> None:
    joint_template = JOINT_PROMPT.read_text(encoding="utf-8")
    xyz_template = XYZ_PROMPT.read_text(encoding="utf-8")

    for spec in TARGETS:
        print(f"[direct_joint] {spec['dataset']} c{spec['idx']} {spec['cue']}", flush=True)
        prompt = joint_template.replace("{CUE_NAME}", spec["cue"]).replace("{{CUE_NAME}}", spec["cue"])
        obj = _generate_direct_one(prompt, model_name=model_name)
        row = {
            "idx": int(spec["idx"]),
            "cue": spec["cue"],
            "dataset": spec["dataset"],
            "description": obj.get("description", ""),
            "joint_keyframes": obj.get("joint_keyframes", []),
            "model": model_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _upsert_row(DIRECT_JOINT_JSON, row)

        print(f"[direct_xyz] {spec['dataset']} c{spec['idx']} {spec['cue']}", flush=True)
        prompt = xyz_template.replace("{CUE_NAME}", spec["cue"]).replace("{{CUE_NAME}}", spec["cue"])
        obj = _generate_direct_one(prompt, model_name=model_name)
        row = {
            "idx": int(spec["idx"]),
            "cue": spec["cue"],
            "dataset": spec["dataset"],
            "description": obj.get("description", ""),
            "cartesian_keyframes": obj.get("cartesian_keyframes", []),
            "model": model_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _upsert_row(DIRECT_XYZ_JSON, row)


def render_no_reasoning(robot: str = "IIWA", hz: int = 8) -> None:
    sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
    from motion_generation import MotionGenerator, _select_initial_poses  # noqa: E402

    rows_by_file = [
        ("iconic", NO_REASONING_ICONIC_JSON, MOTION_OUT / "no_reasoning_iconic" / robot),
        ("contextual", NO_REASONING_CONTEXTUAL_JSON, MOTION_OUT / "no_reasoning_contextual" / robot),
    ]

    gen = MotionGenerator(
        robot_name=robot,
        jsonl_path=str(POSE_DB),
        output_dir=str(MOTION_OUT / "no_reasoning"),
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        for _, config_path, out_dir in rows_by_file:
            out_dir.mkdir(parents=True, exist_ok=True)
            gen.output_dir = str(out_dir)
            rows = _load_json(config_path) if config_path.exists() else []
            for row in rows:
                pose_def = next(m["parameters"]["pose"] for m in row["movements"] if m.get("type") == "pose")
                matching = gen._find_matching_poses(pose_def)
                selected = _select_initial_poses(matching, pose_def, top_k=1)
                gen._set_joint_positions(gen.initial_joint_pos)
                gen.execute_cue(
                    cue=row["cue"],
                    pose_index=selected[0]["pose_id"],
                    config_path=str(config_path),
                    hz=int(hz),
                    cue_idx=int(row["idx"]),
                    save_gif=True,
                )
    finally:
        gen.close()


class DirectSequenceRenderer:
    def __init__(self, robot: str = "IIWA"):
        import numpy as np

        sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
        from alphabet_jacobian import JacobianCalculator  # noqa: E402
        import robosuite.utils.transform_utils as T  # noqa: E402

        self.np = np
        self.T = T
        self.calc = JacobianCalculator(
            robot_name=robot,
            jsonl_path=str(POSE_DB),
            has_renderer=False,
            has_offscreen_renderer=True,
            save_jacobian_gif=False,
        )
        self.env = self.calc.env
        self.robot = self.calc.robot
        self.site_id = self.env.sim.model.site_name2id(self.calc.eef_site_name)
        self.camera_name = "frontview"
        self.capture_width = 512
        self.capture_height = 512

        # The default frontview is too tight for direct-sequence baselines and
        # systematically crops overhead / extended motions. Widen it so the
        # comparison is about motion quality, not camera framing artifacts.
        cam_id = self.env.sim.model.camera_name2id(self.camera_name)
        self.env.sim.model.cam_fovy[cam_id] = 75.0

    def close(self) -> None:
        self.calc.close()

    def _capture(self):
        image = self.env.sim.render(
            camera_name=self.camera_name,
            width=self.capture_width,
            height=self.capture_height,
            depth=False,
        )
        return image[::-1]

    def _reset_robot(self) -> None:
        self.robot.set_robot_joint_positions(self.calc.initial_joint_pos.copy())
        self.env.sim.forward()
        for _ in range(10):
            self.env.sim.data.qvel[:] = 0
            self.env.sim.forward()

    def _save_gif(self, frames: list, out_path: Path, duration_ms: int = 100) -> None:
        from PIL import Image

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    def render_joint_keyframes(self, keyframes: list[dict], out_path: Path) -> None:
        if len(keyframes) < 2:
            raise ValueError("Need at least 2 joint keyframes")
        self._reset_robot()
        frames = []
        q_keys = [self.np.deg2rad(self.np.array(k["q_deg"], dtype=float)) for k in keyframes]
        t_keys = [float(k["t"]) for k in keyframes]
        self.robot.set_robot_joint_positions(q_keys[0])
        self.env.sim.forward()
        for i in range(len(q_keys) - 1):
            q0, q1 = q_keys[i], q_keys[i + 1]
            seg = max(2, int((t_keys[i + 1] - t_keys[i]) * 36))
            for step in range(seg):
                alpha = (step + 1) / seg
                q = q0 * (1 - alpha) + q1 * alpha
                self.robot.set_robot_joint_positions(q)
                self.env.sim.forward()
                frames.append(self._capture())
        self._save_gif(frames, out_path)

    def render_cartesian_keyframes(self, keyframes: list[dict], out_path: Path) -> None:
        if len(keyframes) < 2:
            raise ValueError("Need at least 2 cartesian keyframes")
        self._reset_robot()
        frames = []
        t_keys = [float(k["t"]) for k in keyframes]
        for i in range(len(keyframes) - 1):
            k0, k1 = keyframes[i], keyframes[i + 1]
            seg = max(2, int((t_keys[i + 1] - t_keys[i]) * 36))
            for step in range(seg):
                alpha = (step + 1) / seg
                pos = self.np.array(
                    [
                        float(k0["x"]) * (1 - alpha) + float(k1["x"]) * alpha,
                        float(k0["y"]) * (1 - alpha) + float(k1["y"]) * alpha,
                        float(k0["z"]) * (1 - alpha) + float(k1["z"]) * alpha,
                    ],
                    dtype=float,
                )
                theta_deg = float(k0["theta_deg"]) * (1 - alpha) + float(k1["theta_deg"]) * alpha
                mat = self.T.euler2mat(self.np.array([0.0, self.np.deg2rad(theta_deg), 0.0]))
                quat = self.T.mat2quat(mat)
                axis_angle = self.T.quat2axisangle(quat)
                action = self.np.concatenate([pos, axis_angle])

                for _ in range(4):
                    q_des = self.calc.ik_solver.solve(action)
                    current_joint_pos = self.robot._joint_positions.copy()
                    updated = current_joint_pos.copy()
                    updated[self.calc.ik_solver.dof_ids] = q_des
                    self.robot.set_robot_joint_positions(updated)
                    self.env.sim.forward()
                frames.append(self._capture())
        self._save_gif(frames, out_path)


def render_direct_sequences(robot: str = "IIWA") -> None:
    joint_rows = _load_json(DIRECT_JOINT_JSON) if DIRECT_JOINT_JSON.exists() else []
    xyz_rows = _load_json(DIRECT_XYZ_JSON) if DIRECT_XYZ_JSON.exists() else []
    renderer = DirectSequenceRenderer(robot=robot)
    try:
        for row in joint_rows:
            dataset = row["dataset"]
            out_dir = MOTION_OUT / "direct_joint" / dataset / robot
            out_path = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{robot}_{_safe_name(row['cue'])}_p{row['idx']}.gif"
            renderer.render_joint_keyframes(row["joint_keyframes"], out_path)
        for row in xyz_rows:
            dataset = row["dataset"]
            out_dir = MOTION_OUT / "direct_xyz_theta" / dataset / robot
            out_path = out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{robot}_{_safe_name(row['cue'])}_p{row['idx']}.gif"
            renderer.render_cartesian_keyframes(row["cartesian_keyframes"], out_path)
    finally:
        renderer.close()


def build_html(robot: str = "IIWA") -> None:
    refs = _target_rows()
    no_reason_iconic = {int(r["idx"]): r for r in (_load_json(NO_REASONING_ICONIC_JSON) if NO_REASONING_ICONIC_JSON.exists() else [])}
    no_reason_contextual = {int(r["idx"]): r for r in (_load_json(NO_REASONING_CONTEXTUAL_JSON) if NO_REASONING_CONTEXTUAL_JSON.exists() else [])}
    joint_rows = {(r["dataset"], int(r["idx"])): r for r in (_load_json(DIRECT_JOINT_JSON) if DIRECT_JOINT_JSON.exists() else [])}
    xyz_rows = {(r["dataset"], int(r["idx"])): r for r in (_load_json(DIRECT_XYZ_JSON) if DIRECT_XYZ_JSON.exists() else [])}

    cards = []
    for spec in TARGETS:
        dataset = spec["dataset"]
        idx = int(spec["idx"])
        cue = spec["cue"]
        ref_row = refs[dataset][idx]
        no_reason_row = no_reason_iconic.get(idx) if dataset == "iconic" else no_reason_contextual.get(idx)
        joint_row = joint_rows.get((dataset, idx))
        xyz_row = xyz_rows.get((dataset, idx))

        ref_dir = MOTIONS / ("v19_sophisticated" if dataset == "iconic" else "v19_sophisticated_contextual_q4filled") / robot
        no_reason_dir = MOTION_OUT / (f"no_reasoning_{dataset}") / robot
        joint_dir = MOTION_OUT / "direct_joint" / dataset / robot
        xyz_dir = MOTION_OUT / "direct_xyz_theta" / dataset / robot

        ref_gif = _latest_single_gif(ref_dir, cue) or _latest_gif(ref_dir, cue, idx)
        no_reason_gif = _latest_gif(no_reason_dir, cue, idx)
        joint_gif = _latest_gif(joint_dir, cue, idx)
        xyz_gif = _latest_gif(xyz_dir, cue, idx)

        def media(path: Path | None, label: str) -> str:
            if path is None:
                return f'<div class="missing">{label}<br>missing</div>'
            return f'<img src="{path.resolve().as_uri()}" alt="{html.escape(label)}" loading="lazy">'

        cards.append(
            f"""
            <article class="card">
              <div class="hdr">
                <div class="title">{dataset} · c{idx} · {html.escape(cue)}</div>
              </div>
              <div class="media-grid four">
                <section><div class="label">Reference</div>{media(ref_gif, 'reference')}</section>
                <section><div class="label">No Reasoning</div>{media(no_reason_gif, 'no reasoning')}</section>
                <section><div class="label">Direct q</div>{media(joint_gif, 'direct joint')}</section>
                <section><div class="label">Direct xyzθ</div>{media(xyz_gif, 'direct xyz')}</section>
              </div>
              <div class="body body-grid">
                <section class="cfg-block">
                  <div class="label">Reference Config</div>
                  <pre>{html.escape(json.dumps(ref_row, ensure_ascii=False, indent=2))}</pre>
                </section>
                <section class="cfg-block">
                  <div class="label">No Reasoning Config</div>
                  <pre>{html.escape(json.dumps(no_reason_row, ensure_ascii=False, indent=2) if no_reason_row else '{}')}</pre>
                </section>
                <section class="cfg-block">
                  <div class="label">Direct Joint Output</div>
                  <pre>{html.escape(json.dumps(joint_row, ensure_ascii=False, indent=2) if joint_row else '{}')}</pre>
                </section>
                <section class="cfg-block">
                  <div class="label">Direct xyz-theta Output</div>
                  <pre>{html.escape(json.dumps(xyz_row, ensure_ascii=False, indent=2) if xyz_row else '{}')}</pre>
                </section>
              </div>
            </article>
            """
        )

    text = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Prompt19 Baseline Experiment Extra 10</title>
  <style>
    :root {{ --bg:#fff; --surface:#fff; --line:#dde4ea; --muted:#64707b; --ink:#111; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
    .wrap {{ max-width:1840px; margin:0 auto; padding:24px; }}
    h1 {{ margin:0 0 8px; font-size:30px; }}
    .lead {{ margin:0 0 10px; color:var(--muted); }}
    .meta {{ margin:0 0 22px; color:var(--muted); font-size:13px; }}
    .grid {{ display:grid; gap:18px; }}
    .card {{ border:1px solid var(--line); background:var(--surface); }}
    .hdr {{ padding:14px 16px; border-bottom:1px solid var(--line); }}
    .title {{ font-size:18px; font-weight:700; }}
    .media-grid {{ display:grid; gap:12px; padding:14px 16px; border-bottom:1px solid var(--line); }}
    .media-grid.four {{ grid-template-columns:repeat(4, minmax(0, 1fr)); }}
    .media-grid img {{ width:100%; display:block; border:1px solid var(--line); background:#fff; }}
    .body {{ padding:14px 16px 18px; }}
    .body-grid {{ display:grid; gap:12px; grid-template-columns:repeat(2, minmax(0, 1fr)); align-items:start; }}
    .cfg-block {{ min-width:0; }}
    .label {{ margin:0 0 6px; font-size:12px; font-weight:700; text-transform:uppercase; color:var(--muted); }}
    pre {{ margin:0; max-height:320px; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#f8fafb; border:1px solid #edf1f4; padding:10px 12px; font-size:12px; line-height:1.45; }}
    .missing {{ min-height:220px; display:grid; place-items:center; text-align:center; background:#f8fafb; border:1px solid var(--line); color:var(--muted); }}
    @media (max-width: 1400px) {{ .media-grid.four {{ grid-template-columns:repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 900px) {{ .body-grid {{ grid-template-columns:1fr; }} }}
    @media (max-width: 760px) {{ .media-grid.four {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>Prompt 19 Baseline Experiment Extra 10</h1>
    <p class="lead">추가 10개 cue에 대해 baseline 3종을 비교합니다: no-reasoning structured, direct joint-angle, direct xyz-theta.</p>
    <p class="meta">targets={len(TARGETS)} | robot={html.escape(robot)} | output_root={html.escape(str(BASE_OUT))}</p>
    <div class="grid">{''.join(cards)}</div>
  </main>
</body>
</html>
"""
    HTML_OUT.write_text(text, encoding="utf-8")


def write_manifest() -> None:
    _write_json(
        MANIFEST_JSON,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "targets": TARGETS,
            "no_reasoning_prompt": str(NO_REASONING_PROMPT),
            "no_reasoning_shots": str(NO_REASONING_SHOTS),
            "joint_prompt": str(JOINT_PROMPT),
            "xyz_prompt": str(XYZ_PROMPT),
            "joint_shots_json": str(JOINT_SHOTS_JSON),
            "xyz_shots_json": str(XYZ_SHOTS_JSON),
            "no_reasoning_iconic_json": str(NO_REASONING_ICONIC_JSON),
            "no_reasoning_contextual_json": str(NO_REASONING_CONTEXTUAL_JSON),
            "direct_joint_json": str(DIRECT_JOINT_JSON),
            "direct_xyz_json": str(DIRECT_XYZ_JSON),
            "html": str(HTML_OUT),
        },
    )


def run_all(
    robot: str = "IIWA",
    model_name: str = "gemini-2.5-pro",
    hz: int = 8,
) -> None:
    random.seed(19)
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    extract_shot_sequences(robot=robot, hz=hz)
    write_baseline_prompts()
    generate_no_reasoning(model_name=model_name)
    generate_direct_sequences(model_name=model_name)
    render_no_reasoning(robot=robot, hz=hz)
    render_direct_sequences(robot=robot)
    build_html(robot=robot)
    write_manifest()
    print(f"HTML: {HTML_OUT}")


if __name__ == "__main__":
    fire.Fire(
        {
            "extract_shot_sequences": extract_shot_sequences,
            "write_baseline_prompts": write_baseline_prompts,
            "generate_no_reasoning": generate_no_reasoning,
            "generate_direct_sequences": generate_direct_sequences,
            "render_no_reasoning": render_no_reasoning,
            "render_direct_sequences": render_direct_sequences,
            "build_html": build_html,
            "write_manifest": write_manifest,
            "run_all": run_all,
        }
    )
