#!/usr/bin/env python3
"""Build the 150-row, five-method render gallery with human yes/no badges."""

from __future__ import annotations

import html
import csv
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/html/all_methods_human_eval_c1_30.html"
PLATFORMS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
METHODS = (
    ("gt", "GT"),
    ("joint", "Joint-Space"),
    ("xyzrpy", "Cartesian-Space"),
    ("motion_program", "Motion Program"),
    ("trajectory", "Trajectory Sketching"),
)
BASELINE_EVAL = (
    REPO
    / "data/results/evaluation/embodiment_cue_id/"
    "cueid_human_cue_identification_with_traj_baselines.json"
)
GT_EVAL = (
    REPO
    / "data/results/evaluation/embodiment_cue_id/"
    "cueid_human_yesno_acc_by_platform_gt_made.json"
)
MOTION_PROGRAM_EVAL = (
    REPO
    / "data/results/evaluation/embodiment_cue_id/"
    "cueid_human_binary_acc_by_platform_gemini_made.json"
)
TRAJECTORY_EVAL = (
    REPO
    / "data/results/evaluation/embodiment_cue_id/"
    "human_yesno_trajectory_hand_elbow_wrist_eased.md"
)
TRAJECTORY_SOURCES = (
    REPO / "data/results/drag_draw/gemini25pro_c1_25/predictions_rendered.json",
    REPO / "data/results/drag_draw/gemini25pro_c1_5_c26_30/predictions_rendered.json",
    REPO / "data/results/drag_draw/roomba_sophisticated_full30/predictions_rendered.json",
)
MOTION_SCORES = (
    REPO / "data/results/evaluation/motion_similarity_3d/"
    "pairwise_3d_orientation_scores.csv"
)
REVIEW_XLSX = (
    REPO
    / "data/results/evaluation/embodiment_cue_id"
    / "all_methods_human_yesno_c1_30.xlsx"
)


def wrong_indices(path: Path) -> dict[str, set[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        platform: {
            int(row["idx"])
            for row in details["wrong_details"]
        }
        for platform, details in payload["yesno"]["by_platform"].items()
    }


def trajectory_labels() -> dict[tuple[str, int], str]:
    """Parse the user-supplied Markdown table; blank is YES and x is NO."""
    labels: dict[tuple[str, int], str] = {}
    columns = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
    for line in TRAJECTORY_EVAL.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\|\s*c(\d+)\s*\|(.*)\|$", line)
        if not match:
            continue
        idx = int(match.group(1))
        values = [value.strip().lower() for value in match.group(2).split("|")]
        for platform, value in zip(columns, values):
            labels[(platform, idx)] = "x" if value == "x" else "o"
    if len(labels) != 150:
        raise ValueError(f"Expected 150 trajectory labels, found {len(labels)}")
    return labels


def trajectory_renders() -> dict[tuple[str, int], Path]:
    renders: dict[tuple[str, int], Path] = {}
    for source in TRAJECTORY_SOURCES:
        for row in json.loads(source.read_text(encoding="utf-8")):
            if row.get("parse_ok") and row.get("render_gif"):
                renders[(row["platform"], int(row["idx"]))] = REPO / row["render_gif"]
    return renders


def cue_map() -> dict[int, str]:
    rows = json.loads(
        (
            REPO / "data/seed/agent_fill/embodiment_short/cues.json"
        ).read_text(encoding="utf-8")
    )
    return {int(row["idx"]): str(row["cue"]) for row in rows}


def reviewed_labels() -> dict[tuple[str, int, str], str]:
    """Use the manually reviewed workbook as the display-label source."""
    from openpyxl import load_workbook

    workbook = load_workbook(REVIEW_XLSX, data_only=True, read_only=True)
    sheet = workbook.active
    headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
    methods = {
        "GT": "gt",
        "Joint-Space": "joint",
        "Cartesian-Space": "xyzrpy",
        "Motion Program": "motion_program",
        "Trajectory Sketching": "trajectory",
    }
    positions = {name: headers.index(name) for name in methods}
    labels: dict[tuple[str, int, str], str] = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        idx = int(str(row[0]).removeprefix("c").split(".", 1)[0])
        platform = str(row[1]).split("·", 1)[1].strip()
        for column, method in positions.items():
            value = str(row[method] or "").strip().upper()
            if value not in ("YES", "NO"):
                raise ValueError(
                    f"Expected YES/NO in {REVIEW_XLSX.name}: "
                    f"c{idx:02d} {platform} {column} = {value!r}"
                )
            labels[(platform, idx, methods[column])] = "o" if value == "YES" else "x"
    expected = len(PLATFORMS) * 30 * len(methods)
    if len(labels) != expected:
        raise ValueError(f"Expected {expected} reviewed labels, found {len(labels)}")
    return labels


def gt_distances() -> dict[tuple[str, int, str], tuple[float, float, float, float]]:
    """Return per-cell position/orientation pose and dynamics distances to GT.

    The pairwise file stores each unordered pair once.  The gallery's method
    names match its identifiers except ``gt``, whose distance is zero.
    """
    scores: dict[tuple[str, int, str], tuple[float, float, float, float]] = {}
    with MOTION_SCORES.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["a"] != "gt" and row["b"] != "gt":
                continue
            method = row["b"] if row["a"] == "gt" else row["a"]
            scores[(row["robot"], int(row["idx"]), method)] = (
                float(row["pose_dtw"]),
                float(row["orientation_dtw"]),
                float(row["velocity_dtw"]),
                float(row["angular_velocity_dtw"]),
            )
    # Roomba uses the separately rerun sophisticated full-30 outputs.  Their
    # metrics have not been recalculated, so never attach stale values from the
    # prior baseline renders to the replacement GIFs.
    for idx in range(1, 31):
        for method in ("joint", "xyzrpy", "motion_program", "trajectory"):
            scores.pop(("roomba", idx, method), None)
    return scores


def render_path(method: str, platform: str, idx: int, cue: str, trajectory: dict[tuple[str, int], Path]) -> Path | None:
    if method == "trajectory":
        return trajectory.get((platform, idx))
    stem = f"emb_g{idx:02d}_{cue}"
    if method in ("joint", "xyzrpy"):
        if platform == "roomba":
            base = REPO / f"data/results/baseline_traj/roomba_sophisticated_full30/render_{method}/roomba"
            candidate = base / f"{stem}_gemini-2.5-pro.gif"
            return candidate if candidate.is_file() else None
        base = REPO / f"data/results/baseline_traj/ours5/render_{method}/{platform}"
        candidates = (base / f"{stem}_gemini-2.5-pro.gif", base / f"{stem}.gif")
        return next((candidate for candidate in candidates if candidate.is_file()), None)
    if method == "motion_program" and platform == "roomba":
        path = REPO / f"data/results/roomba_sophisticated_full30_motion_code/render/{stem}.gif"
        return path if path.is_file() else None
    root = (
        "embodiment_motion_code_gt"
        if method == "gt"
        else "embodiment_motion_code_gemini_gemini-2.5-pro"
    )
    render_platform = "manipulator" if platform == "single_arm" else platform
    # Humanoid/Tiago Motion Program: prefer EE residual-translate renders when present.
    if method == "motion_program" and platform in ("humanoid", "tiago"):
        ee_base = (
            REPO
            / "data/results/render"
            / f"{root}_ee_translate"
            / render_platform
            / "gif"
        )
        ee_path = ee_base / f"{stem}.gif"
        if ee_path.is_file():
            return ee_path
    base = REPO / f"data/results/render/{root}/{render_platform}/gif"
    if platform == "single_arm":
        base /= "IIWA"
    path = base / f"{stem}.gif"
    return path if path.is_file() else None


def badge(label: str) -> str:
    if label == "o":
        return "<span class='badge yes'>YES</span>"
    if label == "x":
        return "<span class='badge no'>NO</span>"
    return "<span class='badge na'>N/A</span>"


def distance_badge(distance: tuple[float, float, float, float] | None) -> str:
    if distance is None:
        return "<span class='distance na'>P/R/D/A N/A</span>"
    pose, orientation, dynamic, angular = distance
    return (
        "<span class='distance'>"
        f"P {pose:.3f} · R {orientation:.3f}<br>"
        f"D {dynamic:.3f} · A {angular:.3f}"
        "</span>"
    )


def cell(
    method: str,
    platform: str,
    idx: int,
    cue: str,
    label: str,
    trajectory: dict[tuple[str, int], Path],
    distances: dict[tuple[str, int, str], tuple[float, float, float, float]],
) -> str:
    rendered = render_path(method, platform, idx, cue, trajectory)
    if not rendered:
        content = "<div class='missing'>render unavailable</div>"
    else:
        relative = html.escape(os.path.relpath(rendered, OUT.parent))
        content = f"<img loading='lazy' src='{relative}?v={rendered.stat().st_mtime_ns}' alt='{method} render'>"
    distance = (0.0, 0.0, 0.0, 0.0) if method == "gt" else distances.get((platform, idx, method))
    return (
        f"<article class='cell method-{method}' data-label='{'yes' if label == 'o' else 'no' if label == 'x' else 'na'}'>"
        f"{distance_badge(distance)}{badge(label)}{content}</article>"
    )


def main() -> None:
    baseline = json.loads(BASELINE_EVAL.read_text(encoding="utf-8"))
    gt_wrong = wrong_indices(GT_EVAL)
    motion_wrong = wrong_indices(MOTION_PROGRAM_EVAL)
    trajectory = trajectory_renders()
    trajectory_eval = trajectory_labels()
    cues = cue_map()
    distances = gt_distances()
    reviewed = reviewed_labels()

    rows = []
    missing = []
    for idx in range(1, 31):
        cue = cues[idx]
        for platform in PLATFORMS:
            source_labels = {
                "gt": "x" if idx in gt_wrong[platform] else "o",
                "joint": baseline["cell_labels"][platform][f"g{idx:02d}"]["joint"],
                "xyzrpy": baseline["cell_labels"][platform][f"g{idx:02d}"]["xyzrpy"],
                "motion_program": "x" if idx in motion_wrong[platform] else "o",
                "trajectory": trajectory_eval[(platform, idx)],
            }
            labels = {
                method: reviewed[(platform, idx, method)]
                for method, _ in METHODS
            }
            cells = []
            for method, _ in METHODS:
                if not render_path(method, platform, idx, cue, trajectory):
                    missing.append(f"{platform}:c{idx:02d}:{method}")
                cells.append(
                    cell(method, platform, idx, cue, labels[method], trajectory, distances)
                )
            rows.append(
                f"<section class='row' data-platform='{platform}'>"
                f"<div class='meta'><strong>c{idx:02d}</strong><span>{html.escape(cue)}</span>"
                f"<small>{html.escape(platform.replace('_', ' ').title())}</small></div>"
                + "".join(cells)
                + "</section>"
            )

    headers = "".join(f"<div>{label}</div>" for _, label in METHODS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        """<!doctype html><html lang='en'><head><meta charset='utf-8'>
<title>All Methods · Human Cue Evaluation · c1–30</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:22px;background:#0d1117;color:#e6edf3;font-family:ui-sans-serif,system-ui,sans-serif}
h1{margin:0 0 6px}.sub{color:#9da7b3;margin:0 0 18px}.legend{display:flex;gap:16px;margin:0 0 14px;font-size:13px}.badge,.distance{border-radius:4px;font-size:11px;font-weight:800;letter-spacing:.04em;padding:3px 6px}.yes{color:#8ce99a;background:#153d25;border:1px solid #2d7b47}.no{color:#ffb4b4;background:#481f27;border:1px solid #a64654}.na{color:#d1d5db;background:#30363d;border:1px solid #6e7681}.distance{color:#dbeafe;background:#172554;border:1px solid #3b82f6;font-variant-numeric:tabular-nums;line-height:1.5}
.header,.row{display:grid;grid-template-columns:145px repeat(5,minmax(235px,1fr));gap:8px;min-width:1370px}.header{position:sticky;top:0;z-index:2;background:#0d1117;border-bottom:1px solid #30363d;padding:8px 0;font-size:13px;font-weight:750}.scroll{overflow-x:auto}.row{margin:8px 0}.row.platform-filtered{display:none!important}.meta{border:1px solid #30363d;background:#161b22;padding:10px;display:flex;flex-direction:column;gap:5px}.meta span{font-size:12px;word-break:break-word}.meta small{color:#9da7b3;text-transform:capitalize}.cell{position:relative;border:1px solid #30363d;background:#161b22;min-height:142px;overflow:hidden}.cell img{display:block;width:100%;height:100%;min-height:142px;object-fit:contain;background:#000}.cell .badge{position:absolute;top:6px;right:6px;z-index:1;box-shadow:0 1px 4px #000;pointer-events:none}.cell .distance{position:absolute;left:6px;bottom:6px;z-index:1;box-shadow:0 1px 4px #000;pointer-events:none}.missing{height:142px;display:grid;place-items:center;color:#9da7b3;font-size:13px}.method-gt{border-color:#457b9d}.footer{margin:18px 0;color:#9da7b3;font-size:12px}
.filter{display:flex;align-items:center;gap:8px;margin:0 0 14px;font-size:13px}.filter select{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:5px 8px}.cell.filtered{visibility:hidden}
</style></head><body><h1>Rendered method comparison · c1–30</h1>
<p class='sub'>150 rows (30 cues × 5 embodiments) · GT, Joint-Space, Cartesian-Space, Motion Program, and Trajectory Sketching. P = positional Pose-DTW; R = Orientation-DTW; D = linear Velocity-DTW; A = Angular-Velocity-DTW (all vs GT; lower is closer). The upper-right badge is the human cue-identification result.</p>
<div class='legend'><span class='distance'>P 0.000 · R 0.000<br>D 0.000 · A 0.000</span><span>Position / orientation / linear dynamics / angular dynamics distance to GT</span><span class='badge yes'>YES</span><span>recognized cue</span><span class='badge no'>NO</span><span>not recognized</span><span class='badge na'>N/A</span><span>not scored / unavailable metric</span></div><div class='filter'><label>Embodiment <select id='platform-filter'><option value='all'>All</option><option value='humanoid'>Humanoid</option><option value='tiago'>Tiago</option><option value='stretch'>Stretch</option><option value='single_arm'>Single Arm</option><option value='roomba'>Roomba</option></select></label><label>Result <select id='label-filter'><option value='all'>All</option><option value='yes'>YES</option><option value='no'>NO</option></select></label></div><div class='scroll'><div class='header'><div>cue · embodiment</div>"""
        + headers
        + "</div>"
        + "".join(rows)
        + "</div></div>"
        + f"<p class='footer'>Generated {len(rows)} rows. Missing render cells: {len(missing)}"
        + (f" · {html.escape(', '.join(missing))}" if missing else "")
        + """</p><script>
const platformFilter=document.querySelector('#platform-filter');
const labelFilter=document.querySelector('#label-filter');
function applyFilters(){
  document.querySelectorAll('.row').forEach(row=>{
    row.classList.toggle('platform-filtered',platformFilter.value!=='all'&&row.dataset.platform!==platformFilter.value);
  });
  document.querySelectorAll('.cell').forEach(cell=>{
    cell.classList.toggle('filtered',labelFilter.value!=='all'&&cell.dataset.label!==labelFilter.value);
  });
}
platformFilter.addEventListener('change',applyFilters);
labelFilter.addEventListener('change',applyFilters);
</script></body></html>""",
        encoding="utf-8",
    )
    print(OUT)
    print(f"rows={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
