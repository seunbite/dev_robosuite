#!/usr/bin/env python3
"""Run the H4 binary MCQ set with cached Qwen2.5-VL models via vLLM."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def _load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prediction(text: str, question: dict) -> str | None:
    if question.get("response_mode") == "value":
        candidate = text.strip().strip("`").strip()
        final = re.search(r"(?is)FINAL_VALUE\s*:\s*(.+?)\s*$", candidate)
        if final:
            candidate = final.group(1).strip()
        candidates = [candidate]
        labeled = re.match(r"(?is)^[AB]\s*[\).:=-]\s*(.+?)\s*$", candidate)
        if labeled:
            candidates.insert(0, labeled.group(1).strip())
        for value_text in candidates:
            try:
                parsed = json.loads(value_text)
            except (json.JSONDecodeError, TypeError):
                parsed = value_text.strip().strip("\"'")
            for option in question["options"]:
                value = option["value"]
                if parsed == value or str(parsed).strip().lower() == str(value).strip().lower():
                    return option["id"]
        matched = []
        lower = candidate.lower()
        for option in question["options"]:
            forms = {
                str(option["value"]).strip().lower(),
                str(option.get("text", "")).strip().lower(),
                json.dumps(option["value"], ensure_ascii=False).strip().lower(),
            }
            if any(form and form in lower for form in forms):
                matched.append(option["id"])
        if len(set(matched)) == 1:
            return matched[0]
        letter = re.search(r"(?i)(?:^|\b)([AB])(?:\b|[\).:])", candidate)
        if letter:
            return letter.group(1).upper()
        return None
    match = re.search(r"(?i)(?:^|\b)([AB])(?:\b|[.)])", text.strip())
    if match:
        return match.group(1).upper()
    number = re.search(r"\b([12])\b", text)
    return {"1": "A", "2": "B"}.get(number.group(1)) if number else None


def _prompt(processor, question: dict) -> str:
    options = "\n".join(
        f"{option['id']}. {option['text']}" for option in question["options"]
    )
    messages = [
        {
            "role": "system",
            "content": question.get(
                "system_prompt",
                "You are an expert in robot gesture motion codes. "
                "Select the correct attribute value. Answer only A or B.",
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"{question['prompt']}\n\n{options}\n\n"
                        f"{question.get('answer_instruction', 'Answer only A or B.')}"
                    ),
                }
            ],
        },
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _summarize(rows: list[dict]) -> dict:
    def stats(group: list[dict]) -> dict:
        valid = [row for row in group if row["prediction"] in ("A", "B")]
        correct = sum(bool(row["correct"]) for row in group)
        return {
            "n": len(group),
            "correct": correct,
            "invalid": len(group) - len(valid),
            "accuracy": correct / len(group) if group else 0.0,
            "valid_accuracy": (
                sum(bool(row["correct"]) for row in valid) / len(valid) if valid else 0.0
            ),
        }

    by_platform: dict[str, list[dict]] = defaultdict(list)
    by_attribute: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_platform[row["platform"]].append(row)
        by_attribute[row["attribute"]].append(row)
    return {
        "overall": stats(rows),
        "by_platform": {key: stats(value) for key, value in sorted(by_platform.items())},
        "by_attribute": {key: stats(value) for key, value in sorted(by_attribute.items())},
        "prediction_counts": dict(Counter(row["prediction"] or "invalid" for row in rows)),
    }


def _run_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    model: str,
    llm,
    processor,
    sampling,
    batch_size: int,
) -> None:
    questions = _load(input_path)["questions"]
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    done: dict[str, dict] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["question_id"]] = row

    pending = [question for question in questions if question["question_id"] not in done]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [_prompt(processor, question) for question in batch]
        outputs = llm.generate(prompts, sampling_params=sampling)
        for question, output in zip(batch, outputs):
            raw = output.outputs[0].text.strip()
            pred = _prediction(raw, question)
            row = {
                **question,
                "model": model,
                "raw_output": raw,
                "prediction": pred,
                "correct": pred == question["correct_option"],
            }
            _append_jsonl(predictions_path, row)
            done[row["question_id"]] = row
        print(
            f"{input_path.name}: completed "
            f"{min(start + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )

    ordered = [done[question["question_id"]] for question in questions]
    summary = {
        "model": model,
        "source": str(input_path),
        **_summarize(ordered),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--second-input")
    parser.add_argument("--second-output-dir")
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=2,
        metavar=("INPUT", "OUTDIR"),
        help=(
            "Repeatable input/output pair. Prefer this to bundle many packs in one "
            "job (single model load). When set, --input/--output-dir are optional."
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Context length. Raise for schema-YAML prompts (e.g. 16384).",
    )
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    if bool(args.second_input) != bool(args.second_output_dir):
        parser.error("--second-input and --second-output-dir must be used together")

    datasets: list[tuple[Path, Path]] = []
    if args.dataset:
        datasets.extend((Path(inp), Path(out)) for inp, out in args.dataset)
    if args.input:
        if not args.output_dir:
            parser.error("--output-dir is required with --input")
        datasets.append((Path(args.input), Path(args.output_dir)))
        if args.second_input:
            datasets.append((Path(args.second_input), Path(args.second_output_dir)))
    if not datasets:
        parser.error("Provide --dataset INPUT OUTDIR (repeatable) and/or --input/--output-dir")

    os.environ.setdefault("VLLM_USE_CUSTOM_ALLREDUCE", "0")
    tensor_parallel = max(1, torch.cuda.device_count())
    kwargs = {
        "model": args.model,
        "tensor_parallel_size": tensor_parallel,
        "dtype": "half",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_mem,
        "trust_remote_code": True,
        "disable_custom_all_reduce": True,
    }
    if "32B" in args.model or "32b" in args.model:
        kwargs["enforce_eager"] = True
    llm = LLM(**kwargs)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    for input_path, output_dir in datasets:
        print(f"=== dataset {input_path.name} → {output_dir} ===", flush=True)
        _run_dataset(
            input_path=input_path,
            output_dir=output_dir,
            model=args.model,
            llm=llm,
            processor=processor,
            sampling=sampling,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
