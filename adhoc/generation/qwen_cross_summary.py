"""Cross-model Qwen summary tables (32B / 7B / 3B) for SUMMARY mode."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

QWEN_MODEL_COLUMNS: list[tuple[str, str]] = [
    ("Qwen-32B", "qwen32b"),
    ("Qwen-7B", "qwen7b"),
    ("Qwen-3B", "qwen3b"),
]


def _fmt_acc(metrics: dict[str, Any]) -> str:
    if metrics.get("status") == "missing":
        return "-"
    n = metrics.get("n")
    if n == 0:
        return "-"
    if metrics.get("accuracy") is not None:
        return f"{100 * float(metrics['accuracy']):.1f}%"
    if metrics.get("accuracy_pct") is not None:
        return f"{float(metrics['accuracy_pct']):.1f}%"
    headline = str(metrics.get("headline") or "-")
    return headline if len(headline) <= 32 else headline[:29] + "..."


def _fmt_mtime(path: Path) -> str:
    if not path.is_file():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _rel(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def print_qwen_cross_summary(
    *,
    suite_label: str,
    specs: list[dict[str, Any]],
    result_path_for: Callable[[dict[str, Any], str], Path],
    metrics_for: Callable[..., dict[str, Any]],
    repo: Path,
    metrics_kwargs: dict[str, Any] | None = None,
) -> None:
    """Print 표 1 (acc + file date) and 표 2 (file paths) for Qwen 32B/7B/3B."""
    metrics_kwargs = metrics_kwargs or {}
    task_w = max(4, max(len(str(s["id"])) for s in specs))

    # --- 표 1: accuracy + 생성일 ---
    print("\n" + "=" * 120)
    print(f"표 1) {suite_label} — Qwen 32B / 7B / 3B accuracy 및 결과 파일 생성일")
    print("=" * 120)
    header = f"{'Task':<{task_w}}  {'Title':<34}"
    for label, _ in QWEN_MODEL_COLUMNS:
        short = label.replace("Qwen-", "")
        header += f"  {short + ' acc':>10}  {short + ' date':>12}"
    print(header)
    print("-" * 120)

    path_grid: dict[str, dict[str, Path]] = {}
    for spec in specs:
        eid = str(spec["id"])
        title = str(spec.get("title", ""))[:34]
        row = f"{eid:<{task_w}}  {title:<34}"
        path_grid[eid] = {}
        for _label, tag in QWEN_MODEL_COLUMNS:
            path = result_path_for(spec, tag)
            path_grid[eid][tag] = path
            met = metrics_for(path, {**spec, "model_tag": tag}, model_tag=tag, **metrics_kwargs)
            row += f"  {_fmt_acc(met):>10}  {_fmt_mtime(path):>12}"
        print(row)
    print("=" * 120)

    # --- 표 2: 파일 위치 ---
    print("\n" + "=" * 120)
    print(f"표 2) {suite_label} — Qwen 32B / 7B / 3B 결과 JSON 경로")
    print("=" * 120)
    path_header = f"{'Task':<{task_w}}  {'Title':<28}"
    for label, _ in QWEN_MODEL_COLUMNS:
        path_header += f"  {label:<36}"
    print(path_header)
    print("-" * 120)
    for spec in specs:
        eid = str(spec["id"])
        title = str(spec.get("title", ""))[:28]
        row = f"{eid:<{task_w}}  {title:<28}"
        for _label, tag in QWEN_MODEL_COLUMNS:
            p = path_grid[eid][tag]
            row += f"  {_rel(p, repo):<36}"
        print(row)
    print("=" * 120 + "\n")
