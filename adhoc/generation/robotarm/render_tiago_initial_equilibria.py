#!/usr/bin/env python3
"""Render Tiago initial IK equilibria and compare them with LLM trajectories."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))

from drag_draw import interpolate_program, program_to_xyzrpy  # noqa: E402
from run_embodiment_baseline_traj import RENDERERS  # noqa: E402

SOURCE = REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_latest.json"
OUT = REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/initial_equilibria"
VARIANTS = (
    ("temporal_symmetric", "temporal + symmetric"),
    ("llm_elbow", "closest LLM elbow"),
    ("elbow_up", "elbow-up branch"),
    ("elbow_down", "elbow-down branch"),
)
LINK_TO_HANDLE = {
    "right_hand": "right_ee",
    "left_hand": "left_ee",
    "right_elbow": "right_elbow",
    "left_elbow": "left_elbow",
}


def rmse(values: list[float]) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values else float("nan")


def score(payload: dict, program: dict) -> dict[str, float]:
    names = payload["_landmark_names"]
    actual = np.asarray(payload["_landmarks_xyz"], dtype=float)
    states = interpolate_program(program)
    count = min(len(actual), len(states))
    errors: dict[str, list[float]] = {"hand": [], "elbow": []}
    for frame in range(count):
        for link, handle in LINK_TO_HANDLE.items():
            observed = actual[frame, names.index(link)]
            expected = np.asarray(states[frame][handle], dtype=float)
            errors["hand" if link.endswith("hand") else "elbow"].append(
                float(np.linalg.norm(observed - expected))
            )
    hand, elbow = rmse(errors["hand"]), rmse(errors["elbow"])
    return {
        "hand_rmse_cm": hand * 100,
        "elbow_rmse_cm": elbow * 100,
        "combined_rmse_cm": float(np.sqrt((hand * hand + elbow * elbow) / 2) * 100),
    }


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = sorted(
        (
            row for row in rows
            if row.get("parse_ok") and row["platform"] == "tiago" and int(row["idx"]) in (27, 28)
        ),
        key=lambda row: int(row["idx"]),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    sections = []
    for row in rows:
        results = []
        for variant, label in VARIANTS:
            payload = program_to_xyzrpy(row["program"])
            payload["elbow_selection_mode"] = "previous"
            payload["initial_equilibrium_variant"] = variant
            payload["branch_exploration_seeds"] = 12
            target = OUT / f"c{int(row['idx']):02d}_{row['cue']}_{variant}.gif"
            RENDERERS["tiago"](payload, "xyzrpy", target, {})
            results.append((variant, label, target, score(payload, row["program"])))
        best = min(results, key=lambda result: result[3]["combined_rmse_cm"])[0]
        cards = []
        for variant, label, target, metrics in results:
            best_tag = "<span class='best'>best combined match</span>" if variant == best else ""
            cards.append(
                "<article>"
                f"<h3>{html.escape(label)} {best_tag}</h3>"
                f"<img src='{html.escape(target.name)}?v={target.stat().st_mtime_ns}'>"
                "<dl>"
                f"<dt>hand XYZ RMSE</dt><dd>{metrics['hand_rmse_cm']:.2f} cm</dd>"
                f"<dt>elbow XYZ RMSE</dt><dd>{metrics['elbow_rmse_cm']:.2f} cm</dd>"
                f"<dt>combined RMSE</dt><dd>{metrics['combined_rmse_cm']:.2f} cm</dd>"
                "</dl></article>"
            )
        sections.append(
            f"<section><h2>c{int(row['idx']):02d} · {html.escape(row['cue'])}</h2>"
            "<p class='meta'>Similarity compares rendered hand + elbow XYZ with the LLM vectors; "
            "combined RMSE weights hand and elbow equally.</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    gallery = OUT / "gallery_tiago_initial_equilibria.html"
    gallery.write_text(
        """<!doctype html><html><head><meta charset='utf-8'>
<title>Tiago initial IK equilibria</title><style>
body{font-family:system-ui;margin:24px;background:#0d1117;color:#e6edf3}.grid{display:grid;grid-template-columns:repeat(4,minmax(260px,1fr));gap:12px}
section{margin:30px 0}article{background:#161b22;border:1px solid #30363d;padding:10px}h3{font-size:14px;margin:0 0 8px}
img{display:block;width:100%;background:#000}.meta{color:#9da7b3;font-size:13px}dl{display:grid;grid-template-columns:1fr auto;gap:3px;margin:9px 0 0;font-size:12px}dt{color:#9da7b3}dd{margin:0}.best{color:#3fb950;font-size:11px}
</style></head><body><h1>Tiago initial IK equilibria</h1>"""
        + "".join(sections)
        + "</body></html>",
        encoding="utf-8",
    )
    print(gallery)


if __name__ == "__main__":
    main()
