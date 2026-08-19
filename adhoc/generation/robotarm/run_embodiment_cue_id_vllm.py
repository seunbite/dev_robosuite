#!/usr/bin/env python3
"""Run embodiment cue-ID yes/no + pairwise eval with Qwen2.5-VL via in-process vLLM."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_video(repo: Path, question: dict[str, Any]) -> Path | None:
    for key in ("video", "mp4", "gif"):
        raw = question.get(key)
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = repo / path
        if path.is_file():
            return path
    return None


def _parse_yesno(text: str) -> str | None:
    cleaned = text.strip().upper()
    if cleaned.startswith("YES"):
        return "YES"
    if cleaned.startswith("NO"):
        return "NO"
    match = re.search(r"\b(YES|NO)\b", cleaned)
    return match.group(1) if match else None


def _parse_ab(text: str) -> str | None:
    cleaned = text.strip().upper()
    match = re.search(r"(?:^|\b)([AB])(?:\b|[.)])", cleaned)
    return match.group(1) if match else None


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("prediction") is not None]
    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "invalid": len(rows) - len(valid),
        "accuracy": correct / len(rows) if rows else 0.0,
        "valid_accuracy": (
            sum(bool(row.get("correct")) for row in valid) / len(valid) if valid else 0.0
        ),
        "prediction_counts": dict(Counter(row.get("prediction") or "invalid" for row in rows)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_platform: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_platform[row["platform"]].append(row)
        key = row.get("trial") or row.get("option_order") or row.get("task") or "all"
        by_trial[str(key)].append(row)
    return {
        "overall": _stats(rows),
        "by_platform": {k: _stats(v) for k, v in sorted(by_platform.items())},
        "by_split": {k: _stats(v) for k, v in sorted(by_trial.items())},
    }


def _run(
    *,
    questions: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    engine: Any,
    batch_size: int,
    repo: Path,
    task: str,
) -> dict[str, Any]:
    predictions_path = output_dir / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["question_id"]] = row

    pending = [q for q in questions if q["question_id"] not in done]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        requests: list[dict[str, Any]] = []
        usable: list[dict[str, Any]] = []
        for question in batch:
            video = _resolve_video(repo, question)
            if video is None:
                row = {
                    **question,
                    "model": model,
                    "raw_output": "",
                    "prediction": None,
                    "correct": False,
                    "error": "missing_video",
                }
                _append_jsonl(predictions_path, row)
                done[row["question_id"]] = row
                continue
            requests.append(
                {
                    "prompt": question["prompt"],
                    "videos": [str(video)],
                }
            )
            usable.append(question)

        if requests:
            texts = engine.generate_batch(requests)
            for question, raw in zip(usable, texts):
                pred = _parse_yesno(raw) if task == "yesno" else _parse_ab(raw)
                row = {
                    **question,
                    "model": model,
                    "raw_output": raw,
                    "prediction": pred,
                    "correct": pred == question["correct_answer"],
                }
                _append_jsonl(predictions_path, row)
                done[row["question_id"]] = row

        print(
            f"{output_dir.name}: {min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )

    ordered = [done[q["question_id"]] for q in questions if q["question_id"] in done]
    summary = {"model": model, "task": task, **_summarize(ordered)}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="*_yesno.json or *_pairwise.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=0)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    args = parser.parse_args()

    payload = _load(Path(args.input))
    questions = payload["questions"]
    task = payload.get("task") or questions[0].get("task")
    if task not in ("yesno", "pairwise"):
        raise SystemExit(f"Unknown task: {task}")

    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_USE_CUSTOM_ALLREDUCE", "0")
    if args.tensor_parallel_size:
        os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)
    os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_mem)
    os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    os.environ["VLLM_MAX_NUM_SEQS"] = str(args.max_num_seqs)
    os.environ["VLM_MODEL"] = args.model

    from vllm_local import VLLMLocalEngine

    # Video prompts need video modality enabled.
    engine = VLLMLocalEngine(model=args.model)
    # Patch limit if engine only allowed images (older default).
    try:
        engine.llm.llm_engine.model_config  # noqa: B018 — touch
    except Exception:
        pass

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _run(
        questions=questions,
        output_dir=out,
        model=args.model,
        engine=engine,
        batch_size=args.batch_size,
        repo=_REPO,
        task=task,
    )


if __name__ == "__main__":
    main()
