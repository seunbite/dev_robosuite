"""Accuracy line plots for pilot-90 (pose tasks 1–3, motion tasks 7–9)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROBOTARM = _HERE.parent
_REPO = _ROBOTARM.parents[2]
for p in (_REPO, _ROBOTARM):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from paper_figures._style import (  # noqa: E402
    DEFAULT_SUITES,
    EXP_NAMES,
    MODEL_ORDER,
    MODEL_STYLE,
    MOTION_TASKS,
    POSE_TASKS,
    apply_paper_style,
    save_fig,
)
from pilot90_experiment_suite import experiment_specs_all, metrics_from_json  # noqa: E402

OUT_DIR = _REPO / "data/results/paper_figures"


def _acc_for_suite(suite_dir: Path, task_ids: tuple[str, ...]) -> list[float | None]:
    specs = {s["id"]: s for s in experiment_specs_all()}
    accs: list[float | None] = []
    for tid in task_ids:
        spec = specs[tid]
        path = suite_dir / spec["out_name"]
        m = metrics_from_json(path, spec)
        if m.get("status") != "ok" or m.get("accuracy") is None:
            accs.append(None)
        else:
            accs.append(float(m["accuracy"]))
    return accs


def _inline_generation_acc(exp_id: str) -> float | None:
    """Fallback: score generation JSON directly (shared across models for exp 1/7)."""
    from run_pilot90_qwen_suite import _score_motion_generation, _score_pose_generation  # noqa: WPS433

    specs = {s["id"]: s for s in experiment_specs_all()}
    tmp = OUT_DIR / "_tmp" / f"exp{exp_id}_inline.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if exp_id == "1":
        _score_pose_generation(tmp)
    elif exp_id == "7":
        _score_motion_generation(tmp)
    else:
        return None
    m = metrics_from_json(tmp, specs[exp_id])
    return m.get("accuracy")


def collect_accuracies(suites: list) -> dict[str, list[float | None]]:
    out: dict[str, list[float | None]] = {}
    for suite in suites:
        d = _REPO / suite.out_dir
        pose = _acc_for_suite(d, POSE_TASKS)
        motion = _acc_for_suite(d, MOTION_TASKS)
        # Generation steps share one config — fill missing from inline score
        if pose[0] is None:
            pose[0] = _inline_generation_acc("1")
        if motion[0] is None:
            motion[0] = _inline_generation_acc("7")
        out[suite.label] = {"pose": pose, "motion": motion}  # type: ignore[assignment]
    return out  # type: ignore[return-value]


def _plot_line(
    accs: dict[str, dict[str, list[float | None]]],
    *,
    task_ids: tuple[str, ...],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    caption: str,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    xlabels = [EXP_NAMES[t] for t in task_ids]
    xs = list(range(len(task_ids)))

    for label in MODEL_ORDER:
        ys = accs.get(label, {}).get(key, [])
        if not ys or all(y is None for y in ys):
            continue
        style = MODEL_STYLE[label]
        ax.plot(
            xs,
            [100 * (y or 0) for y in ys],
            label=label,
            color="black",
            marker=style["marker"],
            linestyle=style["linestyle"],
            markerfacecolor=style.get("mfc", "black"),
            markeredgecolor=style.get("mec", "black"),
            markeredgewidth=style.get("markeredgewidth", 1.0),
            fillstyle=style.get("fillstyle", "full"),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_title(title, pad=2)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3, borderpad=0.2)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_path, caption=caption)


def run(
    *,
    gemini_dir: str = "data/results/verify/pilot90_gemini",
    qwen32_dir: str = "data/results/verify/pilot90_qwen32b",
    qwen7_dir: str = "data/results/verify/pilot90_qwen7b",
    qwen3_dir: str = "data/results/verify/pilot90_qwen3b",
    out_dir: str | None = None,
) -> None:
    from dataclasses import replace

    suites = [
        replace(DEFAULT_SUITES[0], out_dir=gemini_dir),
        replace(DEFAULT_SUITES[1], out_dir=qwen32_dir),
        replace(DEFAULT_SUITES[2], out_dir=qwen7_dir),
        replace(DEFAULT_SUITES[3], out_dir=qwen3_dir),
    ]
    accs = collect_accuracies(suites)
    od = Path(out_dir) if out_dir else OUT_DIR
    od.mkdir(parents=True, exist_ok=True)

    summary_path = od / "acc_summary.json"
    summary_path.write_text(json.dumps(accs, indent=2), encoding="utf-8")

    _plot_line(
        accs,
        task_ids=POSE_TASKS,
        key="pose",
        ylabel="Accuracy (%)",
        title="Pose",
        out_path=od / "lineplot_pose.pdf",
        caption=(
            "Pose benchmark accuracy on 90 non-essence cues. "
            "Tasks: generation (Gen), tile VLM verify (VLM), text verify (Text). "
            "Models: Gm=Gemini, Qw32/7/3=Qwen2.5-VL."
        ),
    )
    _plot_line(
        accs,
        task_ids=MOTION_TASKS,
        key="motion",
        ylabel="Accuracy (%)",
        title="Movement",
        out_path=od / "lineplot_movement.pdf",
        caption=(
            "Movement benchmark accuracy on 90 non-essence cues. "
            "Tasks: generation (Gen), MP4 VLM verify (VLM), text verify (Text)."
        ),
    )
    print(f"Wrote {od / 'lineplot_pose.pdf'}")
    print(f"Wrote {od / 'lineplot_movement.pdf'}")
    print(f"Summary: {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Pilot-90 accuracy line plots")
    p.add_argument("--gemini-dir", default="data/results/verify/pilot90_gemini")
    p.add_argument("--qwen32-dir", default="data/results/verify/pilot90_qwen32b")
    p.add_argument("--qwen7-dir", default="data/results/verify/pilot90_qwen7b")
    p.add_argument("--qwen3-dir", default="data/results/verify/pilot90_qwen3b")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    run(
        gemini_dir=args.gemini_dir,
        qwen32_dir=args.qwen32_dir,
        qwen7_dir=args.qwen7_dir,
        qwen3_dir=args.qwen3_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
