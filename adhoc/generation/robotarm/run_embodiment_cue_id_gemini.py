#!/usr/bin/env python3
"""Run embodiment cue-ID pairwise/yesno eval with Gemini (video bytes).

Thinking is minimized: budget=0 when the model allows it (Flash); for
gemini-2.5-pro the API minimum is 128, which we use as the floor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _client():
    from google import genai

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY (source APIKEY.sh)")
    return genai.Client(api_key=key)


def _build_config(model: str, thinking_budget: int):
    from google.genai import types

    # Pro rejects 0; Flash accepts 0.
    budget = thinking_budget
    if "pro" in model.lower() and budget == 0:
        budget = 128
    # Thinking tokens count against the output budget on 2.5-Pro; keep headroom
    # so the final A/B (or YES/NO) token is not truncated away.
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_output_tokens": max(256, budget + 64),
    }
    try:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
    except Exception:
        pass
    return types.GenerateContentConfig(**kwargs), budget


def _call_one(
    *,
    client: Any,
    model: str,
    config: Any,
    prompt: str,
    video: Path,
    mime: str,
) -> tuple[str, dict[str, Any]]:
    from google.genai import types

    part = types.Part.from_bytes(data=video.read_bytes(), mime_type=mime)
    resp = client.models.generate_content(
        model=model,
        contents=[part, prompt],
        config=config,
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        # Fall back when SDK hides parts / empty .text after thinking-only output.
        try:
            for cand in getattr(resp, "candidates", None) or []:
                content = getattr(cand, "content", None)
                for part in getattr(content, "parts", None) or []:
                    piece = getattr(part, "text", None)
                    if piece:
                        text += piece
        except Exception:
            pass
        text = text.strip()
    usage: dict[str, Any] = {}
    meta = getattr(resp, "usage_metadata", None)
    if meta is not None:
        usage = {
            "prompt_token_count": getattr(meta, "prompt_token_count", None),
            "candidates_token_count": getattr(meta, "candidates_token_count", None),
            "thoughts_token_count": getattr(meta, "thoughts_token_count", None),
            "total_token_count": getattr(meta, "total_token_count", None),
        }
    return text, usage


def _run_pack(
    *,
    questions: list[dict[str, Any]],
    output_dir: Path,
    model: str,
    task: str,
    repo: Path,
    workers: int,
    thinking_budget: int,
    retries: int,
) -> dict[str, Any]:
    predictions_path = output_dir / "predictions.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["question_id"]] = row

    pending = [q for q in questions if q["question_id"] not in done]
    print(
        f"{output_dir.name}: resume={len(done)} pending={len(pending)}/{len(questions)} "
        f"model={model} workers={workers}",
        flush=True,
    )
    if not pending:
        ordered = [done[q["question_id"]] for q in questions if q["question_id"] in done]
        summary = {"model": model, "task": task, **_summarize(ordered)}
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary["overall"], indent=2), flush=True)
        return summary

    client = _client()
    config, used_budget = _build_config(model, thinking_budget)
    print(f"thinking_budget={used_budget}", flush=True)

    def work(question: dict[str, Any]) -> dict[str, Any]:
        video = _resolve_video(repo, question)
        if video is None:
            return {
                **question,
                "model": model,
                "raw_output": "",
                "prediction": None,
                "correct": False,
                "error": "missing_video",
            }
        mime = "video/mp4" if video.suffix.lower() == ".mp4" else "image/gif"
        last_err = None
        for attempt in range(retries + 1):
            try:
                raw, usage = _call_one(
                    client=client,
                    model=model,
                    config=config,
                    prompt=question["prompt"],
                    video=video,
                    mime=mime,
                )
                pred = _parse_yesno(raw) if task == "yesno" else _parse_ab(raw)
                return {
                    **question,
                    "model": model,
                    "raw_output": raw,
                    "prediction": pred,
                    "correct": pred == question["correct_answer"],
                    "usage": usage,
                    "thinking_budget": used_budget,
                }
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                time.sleep(min(2**attempt, 20))
        return {
            **question,
            "model": model,
            "raw_output": "",
            "prediction": None,
            "correct": False,
            "error": last_err or "unknown_error",
            "thinking_budget": used_budget,
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(work, q): q for q in pending}
        for fut in as_completed(futures):
            row = fut.result()
            _append_jsonl(predictions_path, row)
            done[row["question_id"]] = row
            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(
                    f"{output_dir.name}: {len(done)}/{len(questions)} "
                    f"(+{completed}/{len(pending)})",
                    flush=True,
                )

    ordered = [done[q["question_id"]] for q in questions if q["question_id"] in done]
    summary = {
        "model": model,
        "task": task,
        "thinking_budget": used_budget,
        **_summarize(ordered),
    }
    # token rollup
    prompt_tok = cand_tok = thought_tok = 0
    n_usage = 0
    for row in ordered:
        u = row.get("usage") or {}
        if u.get("prompt_token_count") is None:
            continue
        n_usage += 1
        prompt_tok += int(u.get("prompt_token_count") or 0)
        cand_tok += int(u.get("candidates_token_count") or 0)
        thought_tok += int(u.get("thoughts_token_count") or 0)
    summary["usage_totals"] = {
        "n_with_usage": n_usage,
        "prompt_token_count": prompt_tok,
        "candidates_token_count": cand_tok,
        "thoughts_token_count": thought_tok,
        "output_like_token_count": cand_tok + thought_tok,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], indent=2), flush=True)
    print(json.dumps(summary["usage_totals"], indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="*_pairwise.json or *_yesno.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="0 = off when allowed; Pro is clamped to 128",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--repo", type=Path, default=_REPO)
    args = parser.parse_args()

    pack = _load(Path(args.input))
    task = pack.get("task") or (
        "pairwise" if "pairwise" in Path(args.input).name else "yesno"
    )
    questions = pack["questions"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _run_pack(
        questions=questions,
        output_dir=out,
        model=args.model,
        task=task,
        repo=Path(args.repo),
        workers=args.workers,
        thinking_budget=args.thinking_budget,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
