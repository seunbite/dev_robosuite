#!/usr/bin/env python3
"""Count canonical output tokens and render paper-facing comparison options."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import yaml


REPO = Path(__file__).resolve().parents[3]
UNIFIED = REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
DRAG_DRAW = REPO / "data/seed/shots/drag_draw/fewshot_cues_yml_5.yaml"
OUT = REPO / "data/results/drag_draw/token_cost"
PLATFORMS = ("humanoid", "tiago", "single_arm", "stretch", "roomba")
METHODS = ("joint", "xyzrpy", "ours", "drag_draw")
GROUPED_METHODS = ("xyzrpy", "joint", "ours", "drag_draw")
LABELS = {
    "joint": "Joint-Space Demonstration",
    "xyzrpy": "Cartesian-Space Demonstration",
    "ours": "Motion-Program Composition",
    "drag_draw": "Trajectory Sketching",
}
SHORT_LABELS = {
    "joint": "Joint-Space",
    "xyzrpy": "Cartesian-Space",
    "ours": "Motion Program",
    "drag_draw": "Trajectory",
}
COLORS = {
    "joint": "#202020",
    "xyzrpy": "#202020",
    "ours": "#202020",
    "drag_draw": "#1F6FEB",
}
BAR_FACE_COLORS = {
    "joint": "white",
    "xyzrpy": "#202020",
    "drag_draw": "#1F6FEB",
    "ours": "white",
}
HATCHES = {
    "joint": None,
    "xyzrpy": None,
    "drag_draw": None,
    "ours": "//",
}
LINE_STYLES = {
    "joint": ("-", "o"),
    "xyzrpy": ("--", "s"),
    "drag_draw": ("-", "o"),
    "ours": ("--", "s"),
}


FIXED_FRAMES = 24  # paper-facing length-matched comparison


def _canonical_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _resample_frames(seq: list, n: int) -> list:
    """Nearest-neighbor resample a [T, D] numeric sequence to n frames."""
    if not seq:
        return seq
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if len(arr) == n:
        return arr.tolist()
    idx = np.linspace(0, len(arr) - 1, n)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(arr) - 1)
    w = idx - i0
    out = (1.0 - w)[:, None] * arr[i0] + w[:, None] * arr[i1]
    return out.tolist()


def _normalize_xyzrpy(xyzrpy: object, n: int) -> object:
    if isinstance(xyzrpy, list) and xyzrpy and isinstance(xyzrpy[0], list):
        return _resample_frames(xyzrpy, n)
    if isinstance(xyzrpy, dict):
        return {
            key: _resample_frames(val, n) if isinstance(val, list) and val and isinstance(val[0], list) else val
            for key, val in xyzrpy.items()
        }
    return xyzrpy


def _normalize_drag_draw(payload: dict, n: int) -> dict:
    out = json.loads(json.dumps(payload))
    for traj in out.get("trajectories", []):
        if isinstance(traj.get("vectors"), list) and traj["vectors"]:
            traj["vectors"] = _resample_frames(traj["vectors"], n)
    return out


def count_tokens(*, fixed_frames: int | None = None) -> dict:
    encoder = tiktoken.get_encoding("o200k_base")
    unified = json.loads(UNIFIED.read_text(encoding="utf-8"))
    drag = yaml.safe_load(DRAG_DRAW.read_text(encoding="utf-8"))
    result: dict = {
        "meta": {
            "tokenizer": "o200k_base",
            "serialization": "canonical compact JSON",
            "n_cues_per_embodiment": 5,
            "n_outputs": 25,
            "fixed_frames": fixed_frames,
            "source_unified": str(UNIFIED.relative_to(REPO)),
            "source_drag_draw": str(DRAG_DRAW.relative_to(REPO)),
        },
        "by_embodiment": {},
    }
    for platform in PLATFORMS:
        drag_by_cue = {row["cue"]: row for row in drag[platform]}
        values = {method: [] for method in METHODS}
        frame_counts = []
        cue_names = []
        for row in unified[platform]:
            cue = row["cue"]
            cue_names.append(cue)
            hz = int(row.get("hz") or 8)
            joint = row["joint_angles"]
            joints_rad = joint["joints_rad"]
            xyzrpy = row["xyzrpy"]
            drag_payload = drag_by_cue[cue]["drag_draw"]
            n_raw = len(joints_rad) if isinstance(joints_rad, list) else 0
            frame_counts.append(n_raw)
            if fixed_frames is not None:
                # Length-match only dense vector interfaces. Motion Program is
                # compositional; Trajectory Sketching uses sparse handles, so
                # upsampling short strokes would invent control points.
                joints_rad = _resample_frames(joints_rad, fixed_frames)
                xyzrpy = _normalize_xyzrpy(xyzrpy, fixed_frames)
            payloads = {
                "joint": {
                    "cue": cue,
                    "hz": hz,
                    "joint_names": joint["joint_names"],
                    "joints_rad": joints_rad,
                },
                "xyzrpy": {
                    "cue": cue,
                    "hz": hz,
                    "link_names": ["end_effector"],
                    "xyzrpy": xyzrpy,
                },
                "ours": {"cue": cue, "source_code": row["motion_code"]},
                "drag_draw": drag_payload,
            }
            for method, payload in payloads.items():
                values[method].append(len(encoder.encode(_canonical_json(payload))))
        result["by_embodiment"][platform] = {
            method: {
                "total": int(sum(counts)),
                "mean": round(float(np.mean(counts)), 1),
                "per_cue": dict(zip(cue_names, counts)),
            }
            for method, counts in values.items()
        }
        result["by_embodiment"][platform]["raw_frames"] = {
            "mean": round(float(np.mean(frame_counts)), 1),
            "per_cue": dict(zip(cue_names, frame_counts)),
        }
    result["overall"] = {}
    for method in METHODS:
        totals = [result["by_embodiment"][p][method]["total"] for p in PLATFORMS]
        result["overall"][method] = {
            "total": int(sum(totals)),
            "mean": round(float(sum(totals) / 25), 1),
        }
    return result


def _paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 22.5,
        "axes.titlesize": 25.5,
        "axes.labelsize": 22.5,
        "xtick.labelsize": 22.5,
        "ytick.labelsize": 22.5,
        "legend.fontsize": 18,
        "axes.edgecolor": "#202020",
        "axes.linewidth": 1.0,
        "hatch.linewidth": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_option_a(report: dict) -> None:
    x = np.arange(len(PLATFORMS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    for index, method in enumerate(GROUPED_METHODS):
        values = [report["by_embodiment"][p][method]["mean"] for p in PLATFORMS]
        bars = ax.bar(
            x + (index - 1.5) * width,
            values,
            width=width,
            color=BAR_FACE_COLORS[method],
            edgecolor=COLORS[method],
            linewidth=1.3,
            hatch=HATCHES[method],
            label=SHORT_LABELS[method],
        )
        for bar, value in zip(bars, values):
            inside = value > 3000
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value - 60 if inside else value + 70,
                f"{value:,.0f}",
                ha="center",
                va="top" if inside else "bottom",
                rotation=90,
                fontsize=19.5,
                color=(
                    "white"
                    if inside and BAR_FACE_COLORS[method] != "white"
                    else "#202020"
                ),
            )
    ax.set_ylim(0, 4500)
    ax.set_xticks(x, [p.replace("_", " ").title() for p in PLATFORMS])
    ax.set_ylabel("Mean output tokens")
    ax.set_yticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=5)
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#202020",
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderpad=0.55,
        handlelength=2.4,
        columnspacing=0.9,
        labelspacing=0.45,
    )
    fig.tight_layout(pad=0.8)
    _save(fig, "option_a_grouped_vertical_bars")


def plot_option_b(report: dict) -> None:
    x = np.arange(len(PLATFORMS))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    for method in GROUPED_METHODS:
        values = [report["by_embodiment"][p][method]["mean"] for p in PLATFORMS]
        linestyle, marker = LINE_STYLES[method]
        ax.plot(
            x,
            values,
            color=COLORS[method],
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white" if linestyle == "--" else COLORS[method],
            markeredgecolor=COLORS[method],
            markersize=7,
            linewidth=2.0,
            label=SHORT_LABELS[method],
        )
    ax.set_yscale("log")
    ax.set_ylim(35, 7500)
    ax.set_xticks(x, [p.replace("_", " ").title() for p in PLATFORMS])
    ax.set_ylabel("Mean output tokens per motion (log scale)")
    title = "Output Token Cost across Embodiments"
    if report.get("meta", {}).get("fixed_frames"):
        title += f" (length-matched · {report['meta']['fixed_frames']} frames)"
    ax.set_title(title, loc="center", fontweight="normal", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=5)
    ax.legend(
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#202020",
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        borderpad=0.55,
        columnspacing=0.9,
        labelspacing=0.45,
    )
    fig.tight_layout(pad=0.8)
    _save(fig, "option_b_embodiment_lines")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _paper_style()
    raw = count_tokens(fixed_frames=None)
    (OUT / "token_count_report_raw_frames.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Paper-facing: same frame budget across embodiments (removes handmade length bias).
    report = count_tokens(fixed_frames=FIXED_FRAMES)
    (OUT / "token_count_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "token_count_report_fixed_frames.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_option_a(report)
    plot_option_b(report)
    print("raw", json.dumps(raw["overall"], ensure_ascii=False))
    print(f"fixed_{FIXED_FRAMES}", json.dumps(report["overall"], ensure_ascii=False))
    print("cartesian by embodiment (fixed):")
    for platform in PLATFORMS:
        print(
            f"  {platform:12} raw_frames={raw['by_embodiment'][platform]['raw_frames']['mean']:5.1f}  "
            f"xyzrpy_raw={raw['by_embodiment'][platform]['xyzrpy']['mean']:7.1f}  "
            f"xyzrpy_fixed={report['by_embodiment'][platform]['xyzrpy']['mean']:7.1f}"
        )


if __name__ == "__main__":
    main()
