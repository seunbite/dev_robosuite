import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import fire


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))

from motion_generation import MotionGenerator, _select_initial_poses  # noqa: E402


JOBS = [
    ("prompt19_sophisticated_iconic.json", "v19_sophisticated"),
    ("prompt19_sophisticated_contextual.json", "v19_sophisticated_contextual"),
]


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _load_rows(config_path: Path) -> list[dict]:
    return json.loads(config_path.read_text())


def _first_pose_def(row: dict) -> dict | None:
    for movement in row.get("movements", []):
        if movement.get("type") == "pose":
            return movement.get("parameters", {}).get("pose")
    return None


def _latest_new_gif(out_dir: Path, before_names: set[str], cue: str) -> Path | None:
    created = sorted(
        [p for p in out_dir.glob("*.gif") if p.name not in before_names],
        key=lambda p: p.stat().st_mtime,
    )
    if created:
        return created[-1]
    safe_cue = _safe_name(cue)
    matches = sorted(out_dir.glob(f"*_{safe_cue}_p*.gif"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def main(
    robot: str,
    hz: int = 8,
    force: bool = False,
):
    jsonl_path = ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    motions_root = ROOT / "data" / "motions"

    generator = MotionGenerator(
        robot_name=robot,
        jsonl_path=str(jsonl_path),
        output_dir=str(motions_root),
        has_renderer=False,
        has_offscreen_renderer=True,
    )

    summary = {
        "robot": robot,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "jobs": [],
    }

    try:
        for config_name, motion_subdir in JOBS:
            config_path = ROOT / "data" / "seed" / config_name
            out_dir = motions_root / motion_subdir / robot
            out_dir.mkdir(parents=True, exist_ok=True)
            generator.output_dir = str(out_dir)

            rows = _load_rows(config_path)
            job_summary = {
                "config": str(config_path),
                "out_dir": str(out_dir),
                "total": len(rows),
                "rendered": 0,
                "skipped": 0,
                "failed": 0,
            }
            print(f"=== START {robot} {config_name} total={len(rows)} ===", flush=True)

            for idx, row in enumerate(rows, start=1):
                cue_idx = int(row["idx"])
                cue = row["cue"]
                safe_cue = _safe_name(cue)
                tiled_matches = sorted(out_dir.glob(f"*_{safe_cue}_c{cue_idx}_tiled.gif"))
                if tiled_matches and not force:
                    job_summary["skipped"] += 1
                    print(f"[{idx}/{len(rows)}] skip c{cue_idx} {cue}", flush=True)
                    continue

                pose_def = _first_pose_def(row)
                if pose_def is None:
                    job_summary["failed"] += 1
                    print(f"[{idx}/{len(rows)}] fail c{cue_idx} {cue}: no pose movement", flush=True)
                    continue

                try:
                    matching = generator._find_matching_poses(pose_def)
                    selected = _select_initial_poses(matching, pose_def, top_k=1)
                    if not selected:
                        raise ValueError("no matching initial pose")

                    before_names = {p.name for p in out_dir.glob("*.gif")}
                    generator._set_joint_positions(generator.initial_joint_pos)
                    generator.execute_cue(
                        cue=cue,
                        pose_index=selected[0]["pose_id"],
                        config_path=str(config_path),
                        hz=int(hz),
                        cue_idx=cue_idx,
                        save_gif=True,
                    )
                    latest = _latest_new_gif(out_dir, before_names, cue)
                    if latest is None:
                        raise FileNotFoundError("rendered gif not found after execute_cue")

                    tiled_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{robot}_{safe_cue}_c{cue_idx}_tiled.gif"
                    tiled_path = out_dir / tiled_name
                    shutil.copy2(latest, tiled_path)
                    job_summary["rendered"] += 1
                    print(f"[{idx}/{len(rows)}] rendered c{cue_idx} {cue} -> {tiled_path.name}", flush=True)
                except Exception as exc:
                    job_summary["failed"] += 1
                    print(f"[{idx}/{len(rows)}] fail c{cue_idx} {cue}: {exc}", flush=True)

            summary["jobs"].append(job_summary)
            print(f"=== DONE {robot} {config_name} rendered={job_summary['rendered']} skipped={job_summary['skipped']} failed={job_summary['failed']} ===", flush=True)
    finally:
        generator.close()

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary_path = ROOT / "adhoc" / "test" / "results" / f"prompt19_sophisticated_render_{robot.lower()}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
