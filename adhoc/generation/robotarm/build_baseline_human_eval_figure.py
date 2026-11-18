#!/usr/bin/env python3
"""Plot human cue-ID accuracy for Motion Code and trajectory baselines."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from openpyxl import load_workbook


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/evaluation/embodiment_cue_id"
REVIEW_XLSX = OUT / "all_methods_human_yesno_c1_30.xlsx"
SOURCE_PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
ORDER = (0, 1, 3, 2, 4)
PLATFORMS = tuple(SOURCE_PLATFORMS[i] for i in ORDER)
DISPLAY = ("Humanoid", "Tiago", "Single Arm", "Stretch", "Roomba")

# Human annotation supplied 2026-08-09. Blank cells mean ``oo``. The first
# character is Joint and the second is xyzrpy. A missing/``?`` character is NA.
RAW = {
    1: ("", "", "", "ox", ""), 2: ("", "", "", "", ""),
    3: ("xx", "ox", "ox", "ox", ""), 4: ("", "", "", "", ""),
    5: ("xx", "", "", "", ""), 6: ("", "ox", "ox", "", "ox"),
    7: ("x", "xx", "", "", "xx"), 8: ("xx", "", "", "xo", ""),
    9: ("", "", "", "xx", ""), 10: ("xx", "xx", "", "xx", "xx"),
    11: ("xx", "ox", "ox", "ox", ""), 12: ("ox", "", "", "xx", ""),
    13: ("xx", "xx", "", "xx", "xx"), 14: ("xx", "xo", "", "xx", ""),
    15: ("ox", "oo", "xx", "xo", "xx"), 16: ("ox", "ox", "xo", "xo", ""),
    17: ("xx", "xx", "xo", "xo", "xx"), 18: ("xx", "xx", "xo", "xx", ""),
    19: ("xx", "xx", "", "xx", "xx"), 20: ("ox", "xx", "", "xx", "xx"),
    21: ("ox", "xo", "", "", "xo"), 22: ("xx", "xx", "xx", "xo", "ox"),
    23: ("xx", "xx", "xx", "", "xo"), 24: ("xx", "xo", "", "", "ox"),
    25: ("ox", "x?", "xo", "ox", "xx"), 26: ("xx", "xx", "ox", "", "xo"),
    27: ("", "xx", "", "xo", "xo"), 28: ("xx", "xx", "ox", "xx", "xx"),
    29: ("ox", "ox", "", "xo", ""), 30: ("ox", "ox", "xx", "xx", "xx"),
}

HUMAN_AUTHORED = (100.0, 96.7, 100.0, 93.3, 96.7)
OURS = (53.3, 56.7, 56.7, 56.7, 40.0)
# 2026-08-11 human yes/no on trajectory sketches rendered with hand + elbow
# Cartesian priority and intermediate wrist easing. Ordered as SOURCE_PLATFORMS.
TRAJECTORY_SKETCH = (70.0, 83.3, 93.3, 73.3, 73.3)
TRAJECTORY_DETAIL_SOURCE = (
    {"yes": 21, "n": 30, "accuracy": 70.0},
    {"yes": 25, "n": 30, "accuracy": 83.3},
    {"yes": 28, "n": 30, "accuracy": 93.3},
    {"yes": 22, "n": 30, "accuracy": 73.3},
    {"yes": 22, "n": 30, "accuracy": 73.3},
)


def score(char_index: int) -> tuple[list[float], list[dict[str, int | float]]]:
    values: list[float] = []
    details: list[dict[str, int | float]] = []
    for platform_index in range(len(SOURCE_PLATFORMS)):
        labels: list[str] = []
        for cue in range(1, 31):
            cell = RAW[cue][platform_index] or "oo"
            labels.append(cell[char_index] if len(cell) > char_index else "?")
        valid = [label for label in labels if label in ("o", "x")]
        yes = sum(label == "o" for label in valid)
        acc = 100.0 * yes / len(valid)
        values.append(acc)
        details.append({"yes": yes, "n": len(valid), "accuracy": round(acc, 1)})
    return values, details


def reviewed_scores(
    method: str,
) -> tuple[list[float], list[dict[str, int | float]], dict[tuple[int, str], str]]:
    """Read the reviewer-edited YES/NO labels from the workbook."""
    workbook = load_workbook(REVIEW_XLSX, data_only=True, read_only=True)
    sheet = workbook.active
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    column = headers.index(method)
    labels: dict[tuple[int, str], str] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        cue = int(str(row[0]).removeprefix("c").split(".", 1)[0])
        platform = str(row[1]).split("·", 1)[1].strip()
        value = str(row[column] or "").strip().upper()
        if value not in ("YES", "NO", "N/A"):
            raise ValueError(
                f"Expected YES/NO/N/A in {REVIEW_XLSX.name}: "
                f"c{cue:02d} {platform} {method} = {value!r}"
            )
        labels[(cue, platform)] = value

    values, details = [], []
    for platform in SOURCE_PLATFORMS:
        valid = [
            labels[(cue, platform)]
            for cue in range(1, 31)
            if labels[(cue, platform)] in ("YES", "NO")
        ]
        yes = sum(value == "YES" for value in valid)
        values.append(100.0 * yes / len(valid))
        details.append({"yes": yes, "n": len(valid), "accuracy": round(values[-1], 1)})
    return values, details, labels


def main() -> None:
    gt_source, gt_detail_source, gt_labels = reviewed_scores("GT")
    joint_source, joint_detail_source, joint_labels = reviewed_scores("Joint-Space")
    xyzrpy_source, xyzrpy_detail_source, xyzrpy_labels = reviewed_scores("Cartesian-Space")
    ours_source, ours_detail_source, ours_labels = reviewed_scores("Motion Program")
    trajectory_source, trajectory_detail_source, trajectory_labels = reviewed_scores("Trajectory Sketching")
    joint = [joint_source[i] for i in ORDER]
    xyzrpy = [xyzrpy_source[i] for i in ORDER]
    joint_detail = [joint_detail_source[i] for i in ORDER]
    xyzrpy_detail = [xyzrpy_detail_source[i] for i in ORDER]
    human_authored = [gt_source[i] for i in ORDER]
    ours = [ours_source[i] for i in ORDER]
    trajectory = [trajectory_source[i] for i in ORDER]
    trajectory_detail = [trajectory_detail_source[i] for i in ORDER]
    payload = {
        "interpretation": "blank=oo; first=joint; second=xyzrpy; ?=missing",
        "platforms": list(PLATFORMS),
        "human_authored": human_authored,
        "ours_gemini_2_5_pro": ours,
        "trajectory_sketch_hand_elbow_wrist_eased": trajectory_detail,
        "joint_gemini_2_5_pro": joint_detail,
        "xyzrpy_gemini_2_5_pro": xyzrpy_detail,
        "overall": {
            "human_authored": {
                "yes": sum(int(v["yes"]) for v in gt_detail_source),
                "n": sum(int(v["n"]) for v in gt_detail_source),
                "accuracy": round(sum(human_authored) / len(human_authored), 1),
            },
            "ours_gemini_2_5_pro": {
                "yes": sum(int(v["yes"]) for v in ours_detail_source),
                "n": sum(int(v["n"]) for v in ours_detail_source),
                "accuracy": round(sum(ours) / len(ours), 1),
            },
            "trajectory_sketch_hand_elbow_wrist_eased": {
                "yes": sum(int(v["yes"]) for v in trajectory_detail_source),
                "n": sum(int(v["n"]) for v in trajectory_detail_source),
                "accuracy": round(sum(trajectory) / len(trajectory), 1),
            },
            "joint_gemini_2_5_pro": {
                "yes": sum(int(v["yes"]) for v in joint_detail),
                "n": sum(int(v["n"]) for v in joint_detail),
                "accuracy": round(sum(joint) / len(joint), 1),
            },
            "xyzrpy_gemini_2_5_pro": {
                "yes": sum(int(v["yes"]) for v in xyzrpy_detail),
                "n": sum(int(v["n"]) for v in xyzrpy_detail),
                "accuracy": round(sum(xyzrpy) / len(xyzrpy), 1),
            },
        },
        "cell_labels": {
            platform: {
                f"g{cue:02d}": {
                    "joint": "o" if joint_labels[(cue, platform)] == "YES" else "x",
                    "xyzrpy": "o" if xyzrpy_labels[(cue, platform)] == "YES" else "x",
                }
                for cue in range(1, 31)
            }
            for platform_index, platform in enumerate(SOURCE_PLATFORMS)
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "cueid_human_cue_identification_with_traj_baselines"
    try:
        stem.with_suffix(".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        # Paper-only regenerations may lack write access under data/results.
        print(f"skip json write ({exc})")

    # Joint gray, Cartesian white, Selective Cartesian white+hatch,
    # Symbolic (LLM) blue, Symbolic (Human) black.
    fig, ax = plt.subplots(figsize=(9.8, 2.85))
    x = list(range(len(DISPLAY)))
    width = 0.15
    joint_bars = ax.bar(
        [v - 2.0 * width for v in x], joint, width,
        color="#b0b0b0", edgecolor="#202020", linewidth=1.15, label="Joint",
    )
    cart_bars = ax.bar(
        [v - 1.0 * width for v in x], xyzrpy, width,
        color="white", edgecolor="#202020", linewidth=1.15, label="Cartesian",
    )
    sel_bars = ax.bar(
        [v + 0.0 * width for v in x], trajectory, width,
        color="white", edgecolor="#202020", linewidth=1.15, hatch="///",
        label="Selective Cartesian",
    )
    llm_bars = ax.bar(
        [v + 1.0 * width for v in x], ours, width,
        color="#1F6FEB", edgecolor="#1F6FEB", linewidth=1.15, label="Symbolic (LLM)",
    )
    hum_bars = ax.bar(
        [v + 2.0 * width for v in x], human_authored, width,
        color="#111111", edgecolor="#111111", linewidth=1.15, label="Symbolic (Human)",
    )
    labeled = (
        (joint_bars, joint, "#202020"),
        (cart_bars, xyzrpy, "#202020"),
        (sel_bars, trajectory, "#202020"),
        (llm_bars, ours, "white"),
        (hum_bars, human_authored, "white"),
    )
    for bars, values, text_color in labeled:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 2.0,
                f"{value:.1f}",
                ha="center",
                va="top",
                rotation=90,
                fontsize=10.5,
                color=text_color,
            )
    for bar, hum, llm in zip(llm_bars, human_authored, ours):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.4,
            f"+{hum - llm:.1f}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color="#1F6FEB",
        )
    ax.set_ylabel("Accuracy (%)", fontsize=10.5)
    ax.set_xticks(x, DISPLAY, fontsize=10.5)
    # The top margin leaves a small but unambiguous gap above the 100% labels.
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="both", labelsize=10.5, width=1.0, length=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=5,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#202020",
        fontsize=10.5,
        borderpad=0.55,
        handlelength=2.4,
        columnspacing=0.9,
        labelspacing=0.45,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.72)
    for path in (
        stem.with_suffix(".png"),
        stem.with_suffix(".pdf"),
        REPO / "papers/icra2027_motion_abstraction/figures/figure1_human_eval.png",
    ):
        try:
            fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.03)
            print(path)
        except OSError as exc:
            print(f"skip save {path} ({exc})")
    plt.close(fig)

    # Keep Figures 1 and 4 on the same requested embodiment order. Their
    # canonical values come from the human-eval JSONs, not duplicated tables.
    gt_json = json.loads((OUT / "cueid_human_yesno_acc_by_platform_gt_made.json").read_text())
    gen_json = json.loads((OUT / "cueid_human_binary_acc_by_platform_gemini_made.json").read_text())
    gt = [100.0 * gt_json["yesno"]["by_platform"][p]["accuracy"] for p in PLATFORMS]
    yesno = [100.0 * gen_json["yesno"]["by_platform"][p]["accuracy"] for p in PLATFORMS]
    easy = [100.0 * gen_json["easy_negative"]["by_platform"][p]["accuracy"] for p in PLATFORMS]
    hard = [100.0 * gen_json["hard_negative"]["by_platform"][p]["accuracy"] for p in PLATFORMS]

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(x, gt, "-o", color="black", linewidth=3.2, markersize=11, label="Human")
    ax.plot(x, yesno, "-s", color="black", linewidth=3.2, markersize=10, label="Gemini-2.5-Pro")
    ax.plot(x, trajectory, "--P", color="#555555", linewidth=2.4, markersize=9,
            label="Trajectory sketch")
    ax.set_title("Cue Identification by Human", fontsize=25, pad=18)
    _style_axes(ax, x)
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=1, fontsize=15)
    fig.tight_layout()
    fig.savefig(OUT / "cueid_human_cue_identification.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "cueid_human_cue_identification.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.plot(x, gt, "-o", color="black", linewidth=3.2, markersize=11,
            label="Human (GT-made yes/no)")
    ax.plot(x, yesno, "-s", color="black", linewidth=3.2, markersize=10,
            label="Gemini-made yes/no")
    ax.plot(x, trajectory, "--P", color="#555555", linewidth=2.4, markersize=9,
            label="Trajectory sketch yes/no")
    ax.plot(x, easy, "-^", color="black", linewidth=2.0, markersize=9,
            label="Gemini-made pairwise (easy)")
    ax.plot(x, hard, "--D", color="black", linewidth=2.0, markersize=8,
            label="Gemini-made pairwise (hard)")
    ax.axhline(50, color="#246bfe", linewidth=1.2)
    ax.text(4.12, 50.5, "random", color="#246bfe", fontsize=15, fontstyle="italic")
    ax.set_title("Cue Identification by Human", fontsize=25, pad=18)
    _style_axes(ax, x, ymin=30)
    ax.legend(loc="lower left", frameon=True, fancybox=False, framealpha=1, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "cueid_human_binary_acc_by_platform_gemini_made.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))


def _style_axes(ax, x: list[int], *, ymin: int = 0) -> None:
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_xticks(x, DISPLAY, fontsize=16)
    ax.set_ylim(ymin, 105)
    ax.set_yticks(range(ymin, 101, 10))
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", color="#d6d6d6", linewidth=1)
    ax.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
