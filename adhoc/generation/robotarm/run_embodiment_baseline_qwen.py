#!/usr/bin/env python3
"""Generate the 5 robots × 30 cues × 2 trajectory baselines with local vLLM.

This is generation-only.  It writes append-only JSONL files and resumes by
job_id; rendering remains local because Babel does not carry the robot assets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_embodiment_baseline_traj import (  # noqa: E402
    DEFAULT_FEWSHOT_SHOTS,
    HZ,
    PART_JOINTS,
    PLATFORMS,
    _build_prompt,
    _extract_json,
    _fewshot_artifacts,
    _index_gt,
    _validate,
    flat_joint_names,
    link_order,
)
from vllm_local import get_vllm_engine  # noqa: E402


def _done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("parse_ok"):
            out.add(str(row["job_id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fewshot-shots", type=Path, default=DEFAULT_FEWSHOT_SHOTS)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = _index_gt()
    engine = get_vllm_engine(model=args.model)
    jobs: list[dict[str, Any]] = []
    for mode in ("joint", "xyzrpy"):
        pred_path = args.out_dir / f"predictions_{mode}.jsonl"
        done = _done(pred_path)
        for plat in PLATFORMS:
            shots = _fewshot_artifacts(args.out_dir, plat, gt, shots_path=args.fewshot_shots)
            if len(shots) != 5:
                raise RuntimeError(f"{plat}: expected exactly 5 shots, got {len(shots)}")
            for idx in range(1, 31):
                cell = gt[(idx, plat)]
                job_id = f"{mode}.{plat}.g{idx:02d}"
                if job_id in done:
                    continue
                jobs.append({
                    "job_id": job_id, "mode": mode, "platform": plat,
                    "idx": idx, "cell": cell, "shots": shots,
                    "prompt": _build_prompt(
                        mode, plat=plat,
                        target={"cue": cell["cue"], "description": cell.get("description") or ""},
                        shots=shots,
                    ),
                })
    print(f"pending={len(jobs)} total_expected=300 model={args.model}", flush=True)

    pending = jobs
    completed = 0
    for attempt in range(args.retries + 1):
        if not pending:
            break
        retry: list[dict[str, Any]] = []
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start:start + max(1, args.batch_size)]
            texts = engine.generate_batch([{"prompt": j["prompt"]} for j in batch])
            for job, raw in zip(batch, texts):
                plat, mode, cell, shots = job["platform"], job["mode"], job["cell"], job["shots"]
                try:
                    joint_names = flat_joint_names(plat) if plat in PART_JOINTS else shots[0]["joint_names"]
                    link_names = link_order(plat) if plat in PART_JOINTS else (shots[0].get("link_names") or ["end_effector"])
                    payload = _validate(
                        mode, _extract_json(raw), cue=cell["cue"],
                        joint_names=joint_names, link_names=link_names, plat=plat,
                    )
                    row = {
                        "job_id": job["job_id"], "mode": mode, "platform": plat,
                        "idx": job["idx"], "cue": cell["cue"],
                        "description": cell.get("description") or "", "payload": payload,
                        "raw_output": raw, "model": args.model, "parse_ok": True,
                    }
                    completed += 1
                except Exception as exc:  # noqa: BLE001
                    # Persist every inference immediately, including malformed
                    # raw responses that will be retried. JSONL stays append-only.
                    failed_row = {
                        "job_id": job["job_id"], "mode": mode, "platform": plat,
                        "idx": job["idx"], "cue": cell["cue"], "raw_output": raw,
                        "model": args.model, "parse_ok": False, "error": str(exc),
                        "attempt": attempt + 1,
                    }
                    path = args.out_dir / f"predictions_{mode}.jsonl"
                    with path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(failed_row, ensure_ascii=False) + "\n")
                    if attempt < args.retries:
                        retry.append(job)
                        continue
                    continue
                path = args.out_dir / f"predictions_{mode}.jsonl"
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if completed % 10 == 0:
                print(f"completed={completed} retry_pending={len(retry)}", flush=True)
        pending = retry
        print(f"attempt={attempt + 1} remaining={len(pending)}", flush=True)
    print(f"done successful_this_run={completed} terminal_failures={len(pending)}", flush=True)


if __name__ == "__main__":
    main()
