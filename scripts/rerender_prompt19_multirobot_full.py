from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import fire


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTIONS = ROOT / "data" / "motions"
POSE_DB = SEED / "closest_poses_results.jsonl"

sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
from motion_generation import MotionGenerator, _select_initial_poses  # noqa: E402


ROBOTS = ["IIWA", "Panda", "XArm7"]

JOBS = [
    {
        "name": "sophisticated_iconic",
        "config": SEED / "motion_configs_prompt_v19_sophisticated.json",
        "out_dir": MOTIONS / "v19_sophisticated",
    },
    {
        "name": "sophisticated_contextual",
        "config": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
        "out_dir": MOTIONS / "v19_sophisticated_contextual",
    },
    {
        "name": "no_reasoning_iconic",
        "config": SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json",
        "out_dir": MOTIONS / "baseline_prompt19_full_no_reasoning" / "no_reasoning_iconic",
    },
    {
        "name": "no_reasoning_contextual",
        "config": SEED / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json",
        "out_dir": MOTIONS / "baseline_prompt19_full_no_reasoning" / "no_reasoning_contextual",
    },
]


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_pose_def(row: dict) -> dict | None:
    for movement in row.get("movements", []):
        if movement.get("type") == "pose":
            return movement.get("parameters", {}).get("pose")
    return None


def _clear_dir_contents(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _latest_single_gif(base: Path, cue: str) -> Path | None:
    safe_cue = _safe_name(cue)
    matches = sorted(base.rglob(f"*_{safe_cue}_p*.gif"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def rerender_robot(robot: str, hz: int = 8, clear: bool = True) -> dict:
    if robot not in ROBOTS:
        raise ValueError(f"Unsupported robot: {robot}")

    summary: dict[str, object] = {
        "robot": robot,
        "hz": hz,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": [],
    }

    generator = MotionGenerator(
        robot_name=robot,
        jsonl_path=str(POSE_DB),
        output_dir=str(MOTIONS),
        has_renderer=False,
        has_offscreen_renderer=True,
    )
    try:
        for job in JOBS:
            rows = _load_rows(job["config"])
            out_dir = Path(job["out_dir"]) / robot
            if clear:
                _clear_dir_contents(out_dir)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
            generator.output_dir = str(out_dir)

            job_summary = {
                "name": job["name"],
                "config": str(job["config"]),
                "out_dir": str(out_dir),
                "total": len(rows),
                "rendered": 0,
                "failed": 0,
            }

            print(f"=== START {robot} {job['name']} total={len(rows)} hz={hz} ===", flush=True)
            for order, row in enumerate(rows, start=1):
                cue_idx = int(row["idx"])
                cue = row["cue"]
                pose_def = _first_pose_def(row)
                if pose_def is None:
                    job_summary["failed"] += 1
                    print(f"[{order}/{len(rows)}] fail c{cue_idx} {cue}: missing initial pose", flush=True)
                    continue

                try:
                    matching = generator._find_matching_poses(pose_def)
                    selected = _select_initial_poses(matching, pose_def, top_k=1)
                    if not selected:
                        raise ValueError("no matching initial pose")
                    generator._set_joint_positions(generator.initial_joint_pos)
                    generator.execute_cue(
                        cue=cue,
                        pose_index=selected[0]["pose_id"],
                        config_path=str(job["config"]),
                        hz=int(hz),
                        cue_idx=cue_idx,
                        save_gif=True,
                    )
                    latest = _latest_single_gif(out_dir, cue)
                    if latest is None:
                        raise FileNotFoundError("latest single gif not found")
                    job_summary["rendered"] += 1
                    print(f"[{order}/{len(rows)}] rendered c{cue_idx} {cue} -> {latest.name}", flush=True)
                except Exception as exc:
                    job_summary["failed"] += 1
                    print(f"[{order}/{len(rows)}] fail c{cue_idx} {cue}: {exc}", flush=True)

            summary["jobs"].append(job_summary)
            print(
                f"=== DONE {robot} {job['name']} rendered={job_summary['rendered']} failed={job_summary['failed']} ===",
                flush=True,
            )
    finally:
        generator.close()

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    out_summary = ROOT / "adhoc" / "test" / "results" / f"prompt19_multirobot_full_rerender_{robot.lower()}.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {out_summary}", flush=True)
    return summary


def rerender_all(hz: int = 8, clear: bool = True) -> None:
    summaries = []
    for robot in ROBOTS:
        summaries.append(rerender_robot(robot=robot, hz=hz, clear=clear))
    merged = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hz": hz,
        "clear": clear,
        "robots": summaries,
    }
    out_path = ROOT / "adhoc" / "test" / "results" / "prompt19_multirobot_full_rerender_all.json"
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged summary: {out_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(
        {
            "rerender_robot": rerender_robot,
            "rerender_all": rerender_all,
        }
    )
