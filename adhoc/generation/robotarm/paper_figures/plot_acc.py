"""Accuracy line plots for pilot-90 (pose tasks 1–3, motion tasks 7–9)."""
from __future__ import annotations

import argparse
import json
import math
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
MANUAL_TABLE = _HERE / "pilot90_acc_table.json"
ALL_TASKS = tuple(str(i) for i in range(1, 11))

TASK_SHORT = {
    "1": "P Gen",
    "2": "P VLM",
    "3": "P Txt",
    "4": "P Pair",
    "5": "P M6",
    "6": "P M12",
    "7": "M Gen",
    "8": "M VLM",
    "9": "M Txt",
    "10": "M Pair",
}

BAR_PLOT_TASKS = ("4", "5", "6", "10")
BAR_TASK_LABELS = {
    "4": "Pose (2)",
    "5": "Pose (6)",
    "6": "Pose (12)",
    "10": "Movement (2)",
}
BAR_GROUP_GAP = 0.7
WARM_BG = "#FFE8D6"
COOL_BG = "#D6E8FF"
_WARM_BAR = {"0.15": "#7A4A35", "0.55": "#C9A088", "white": "#FFF8F2"}
_COOL_BAR = {"0.15": "#354A6E", "0.55": "#88A0C9", "white": "#F2F8FF"}


def _bar_x_for_task(tid: str) -> float:
    i = BAR_PLOT_TASKS.index(tid)
    return float(i) if i < 3 else 3.0 + BAR_GROUP_GAP


def _bar_group_for_task(tid: str) -> str:
    return "warm" if tid in BAR_PLOT_TASKS[:3] else "cool"


def _group_bar_color(style: dict, group: str) -> str:
    key = style.get("bar_color", "0.25")
    if style.get("bar_hatch"):
        return _WARM_BAR["white"] if group == "warm" else _COOL_BAR["white"]
    palette = _WARM_BAR if group == "warm" else _COOL_BAR
    return palette.get(key, palette["0.55"])


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
    from pilot90_experiment_suite import score_exp1, score_exp7  # noqa: WPS433
    from pilot90_paths import POSE_CFG_LEGACY, config_for_experiment  # noqa: WPS433

    specs = {s["id"]: s for s in experiment_specs_all()}
    tmp = OUT_DIR / "_tmp" / f"exp{exp_id}_inline.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if exp_id == "1":
        cfg = config_for_experiment("1", "qwen32b")
        if not cfg.is_file():
            cfg = POSE_CFG_LEGACY
        score_exp1(cfg, tmp)
    elif exp_id == "7":
        cfg = config_for_experiment("7", "qwen32b")
        if not cfg.is_file():
            return None
        score_exp7(cfg, tmp)
    else:
        return None
    m = metrics_from_json(tmp, specs[exp_id])
    return m.get("accuracy")


def load_manual_table(path: Path | None = None) -> dict[str, dict[str, float | None]]:
    p = path or MANUAL_TABLE
    if not p.is_file():
        p = OUT_DIR / "pilot90_acc_table.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8")).get("tasks") or {}
    out: dict[str, dict[str, float | None]] = {}
    for tid, row in raw.items():
        out[str(tid)] = {k: (None if v is None else float(v) / 100.0) for k, v in row.items()}
    return out


def _manual_pose_motion(manual: dict[str, dict[str, float | None]]) -> dict[str, dict[str, list[float | None]]]:
    out: dict[str, dict[str, list[float | None]]] = {}
    for label in MODEL_ORDER:
        pose = [manual.get(t, {}).get(label) for t in POSE_TASKS]
        motion = [manual.get(t, {}).get(label) for t in MOTION_TASKS]
        out[label] = {"pose": pose, "motion": motion}
    return out


def collect_accuracies(suites: list, *, manual: dict[str, dict[str, float | None]] | None = None) -> dict[str, list[float | None]]:
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


def collect_accuracies_merged(suites: list, manual: dict[str, dict[str, float | None]]) -> dict[str, dict[str, list[float | None]]]:
    """File-based accuracies with manual table override per task×model."""
    file_accs = collect_accuracies(suites)
    merged: dict[str, dict[str, list[float | None]]] = {}
    for label in MODEL_ORDER:
        merged[label] = dict(file_accs.get(label, {"pose": [None] * 3, "motion": [None] * 3}))
    if manual:
        for label in MODEL_ORDER:
            for tid in POSE_TASKS:
                if tid in manual and label in manual[tid] and manual[tid][label] is not None:
                    merged[label]["pose"][POSE_TASKS.index(tid)] = manual[tid][label]
            for tid in MOTION_TASKS:
                if tid in manual and label in manual[tid] and manual[tid][label] is not None:
                    merged[label]["motion"][MOTION_TASKS.index(tid)] = manual[tid][label]
    return merged


def _apply_acc_yaxis(ax, values_pct: list[float]) -> None:
    """Zoom y-axis: [min−10, max+10] with ticks every 20%."""
    finite = [float(v) for v in values_pct if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        return
    vmin, vmax = min(finite), max(finite)
    lo, hi = vmin - 10, vmax + 10
    tick_start = int(math.floor(lo / 20) * 20)
    ticks = [t for t in range(tick_start, int(hi) + 21, 20) if t <= hi]
    if not ticks:
        ticks = [int(round(lo)), int(round(hi))]
    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.set_autoscale_on(False)


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
    all_pct: list[float] = []

    for label in MODEL_ORDER:
        ys = accs.get(label, {}).get(key, [])
        if not ys or all(y is None for y in ys):
            continue
        pct = [100 * y for y in ys if y is not None]
        all_pct.extend(pct)
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
    _apply_acc_yaxis(ax, all_pct)
    ax.set_title(title, pad=2)
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.3, borderpad=0.2, columnspacing=0.8)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, out_path, caption=caption)


def _plot_bar_all(
    manual: dict[str, dict[str, float | None]],
    *,
    out_path: Path,
    caption: str,
) -> None:
    apply_paper_style()
    plt.rcParams["hatch.linewidth"] = 0.55
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    xs = [_bar_x_for_task(tid) for tid in BAR_PLOT_TASKS]
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    all_pct: list[float] = []

    warm_hi = xs[2] + 0.55
    cool_lo = xs[3] - 0.55
    ax.axvspan(xs[0] - 0.55, warm_hi, color=WARM_BG, alpha=0.45, zorder=0)
    ax.axvspan(cool_lo, xs[3] + 0.55, color=COOL_BG, alpha=0.45, zorder=0)

    for mi, label in enumerate(MODEL_ORDER):
        style = MODEL_STYLE[label]
        for tid, x in zip(BAR_PLOT_TASKS, xs, strict=True):
            v = manual.get(tid, {}).get(label)
            y = 100 * v if v is not None else float("nan")
            if v is not None:
                all_pct.append(y)
            group = _bar_group_for_task(tid)
            ax.bar(
                x + offsets[mi],
                y,
                width=width,
                label=label if tid == BAR_PLOT_TASKS[0] else None,
                color=_group_bar_color(style, group),
                edgecolor="black",
                linewidth=0.8,
                hatch=style.get("bar_hatch", ""),
                zorder=2,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([BAR_TASK_LABELS[t] for t in BAR_PLOT_TASKS], rotation=0, ha="center")
    ax.set_xlim(xs[0] - 0.75, xs[3] + 0.75)
    ax.set_ylabel("Accuracy (%)")
    _apply_acc_yaxis(ax, all_pct)
    ax.set_title("Comparison (verification)", pad=6)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        handletextpad=0.3,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
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
    manual = load_manual_table()
    accs = collect_accuracies_merged(suites, manual) if manual else collect_accuracies(suites)
    od = Path(out_dir) if out_dir else OUT_DIR
    od.mkdir(parents=True, exist_ok=True)

    summary_path = od / "acc_summary.json"
    summary_path.write_text(json.dumps(accs, indent=2, ensure_ascii=False), encoding="utf-8")

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
    if manual:
        _plot_bar_all(
            manual,
            out_path=od / "barplot_all_tasks.pdf",
            caption=(
                "Pilot-90 tasks 4–6 (pose pairwise, M6, M12) and task 10 (motion pairwise). "
                "Warm background = pose group; cool = motion pairwise. "
                "Qwen: verify-gt SUMMARY 2026-06-22; Gemini: prior pilot90_gemini table."
            ),
        )
        print(f"Wrote {od / 'barplot_all_tasks.pdf'}")

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
