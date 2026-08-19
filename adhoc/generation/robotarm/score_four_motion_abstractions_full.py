#!/usr/bin/env python3
"""Pairwise render-space motion metrics for 4 methods x 5 embodiments x 30 cues."""
from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageSequence

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/evaluation/motion_similarity_full"
ROBOTS = ("humanoid", "tiago", "stretch", "single_arm", "roomba")
METHODS = ("joint", "xyzrpy", "motion_program", "gt")
LABELS = {"joint": "Joint", "xyzrpy": "Cartesian", "motion_program": "Motion Program", "gt": "GT"}
N, SIZE = 24, 96
METRICS = ("rgb_sim", "ssim", "edge_sim", "flow_sim", "energy_sim")


def cue_map() -> dict[int, str]:
    rows = json.loads((REPO / "data/seed/agent_fill/embodiment_short/cues.json").read_text())
    return {int(r["idx"]): str(r["cue"]) for r in rows}


def path_for(method: str, robot: str, idx: int, cue: str) -> Path | None:
    stem = f"emb_g{idx:02d}_{cue}"
    if method in ("joint", "xyzrpy"):
        base = REPO / f"data/results/baseline_traj/ours5/render_{method}/{robot}"
        candidates = (base / f"{stem}_gemini-2.5-pro.gif", base / f"{stem}.gif")
    else:
        root = "embodiment_motion_code_gt" if method == "gt" else "embodiment_motion_code_gemini_gemini-2.5-pro"
        render_robot = "manipulator" if robot == "single_arm" else robot
        base = REPO / f"data/results/render/{root}/{render_robot}/gif"
        if robot == "single_arm":
            base = base / "IIWA"
        candidates = (base / f"{stem}.gif",)
    return next((p for p in candidates if p.is_file()), None)


def load(path: Path) -> np.ndarray:
    im = Image.open(path)
    frames = []
    for fr in ImageSequence.Iterator(im):
        a = np.asarray(fr.convert("RGB"))
        if a.shape[0] > 100:
            a = a[28:, :, :]
        frames.append(cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    arr = np.asarray(frames, dtype=np.uint8)
    ids = np.linspace(0, len(arr) - 1, N).round().astype(int)
    return arr[ids]


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    d = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float32)
    d[0, 0] = 0
    for i in range(1, len(a) + 1):
        costs = np.linalg.norm(b - a[i - 1], axis=1)
        for j in range(1, len(b) + 1):
            d[i, j] = costs[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[-1, -1] / (len(a) + len(b)))


def features(x: np.ndarray) -> dict[str, np.ndarray]:
    g = np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in x])
    edges, flow, energy = [], [], []
    for fr in g:
        e = cv2.Canny(fr, 65, 130) > 0
        edges.append((e, cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3)))
    for a, b in zip(g[:-1], g[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
        u = cv2.resize(f[..., 0], (6, 6), interpolation=cv2.INTER_AREA)
        v = cv2.resize(f[..., 1], (6, 6), interpolation=cv2.INTER_AREA)
        flow.append(np.r_[u.ravel(), v.ravel()])
        energy.append([float(np.mean(np.linalg.norm(f, axis=2)))])
    return {"rgb": x.astype(np.float32), "gray": g.astype(np.float32), "edges": edges,
            "flow": np.asarray(flow), "energy": np.asarray(energy)}


def global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Framewise luminance SSIM; no external skimage dependency.
    vals = []
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    for x, y in zip(a, b):
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = np.mean((x - mx) * (y - my))
        vals.append(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return float(np.clip(np.mean(vals), 0, 1))


def compare(a: dict, b: dict) -> dict[str, float]:
    mae = float(np.mean(np.abs(a["rgb"] - b["rgb"])))
    chamfers = []
    for (ea, da), (eb, db) in zip(a["edges"], b["edges"]):
        ab = float(db[ea].mean()) if ea.any() else SIZE
        ba = float(da[eb].mean()) if eb.any() else SIZE
        chamfers.append((ab + ba) / 2)
    edge = float(np.mean(chamfers))
    fd, ed = dtw(a["flow"], b["flow"]), dtw(a["energy"], b["energy"])
    return {
        "rgb_sim": math.exp(-mae / 12.0),
        "ssim": global_ssim(a["gray"], b["gray"]),
        "edge_sim": math.exp(-edge / 4.0),
        "flow_sim": math.exp(-fd / 2.0),
        "energy_sim": math.exp(-ed / 0.15),
    }


def matrix(rows: list[dict], metric: str, robot: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    z, n = np.full((4, 4), np.nan), np.zeros((4, 4), dtype=int)
    for i, a in enumerate(METHODS):
        z[i, i], n[i, i] = 1.0, 30 if robot else 150
        for j in range(i + 1, 4):
            b = METHODS[j]
            vals = [float(r[metric]) for r in rows if r["method_a"] == a and r["method_b"] == b
                    and (robot is None or r["robot"] == robot)]
            if vals:
                z[i, j] = z[j, i] = np.mean(vals)
                n[i, j] = n[j, i] = len(vals)
    return z, n


def plot_tile(z: np.ndarray, n: np.ndarray, metric: str, scope: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(z, vmin=0, vmax=1, cmap="Blues")
    names = [LABELS[m] for m in METHODS]
    ax.set_xticks(range(4), names, rotation=25, ha="right")
    ax.set_yticks(range(4), names)
    ax.set_title(f"{scope} · {metric}", fontsize=12)
    for i in range(4):
        for j in range(4):
            txt = "—" if np.isnan(z[i, j]) else f"{z[i,j]:.3f}\n(n={n[i,j]})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if not np.isnan(z[i, j]) and z[i, j] > .62 else "black")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Similarity (higher is closer)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cues, rows, missing = cue_map(), [], []
    for robot in ROBOTS:
        for idx in range(1, 31):
            cue = cues[idx]
            paths = {m: path_for(m, robot, idx, cue) for m in METHODS}
            for m, p in paths.items():
                if p is None:
                    missing.append({"robot": robot, "idx": idx, "cue": cue, "method": m})
            feats = {m: features(load(p)) for m, p in paths.items() if p is not None}
            for a, b in itertools.combinations(METHODS, 2):
                if a not in feats or b not in feats:
                    continue
                row = {"robot": robot, "idx": idx, "cue": cue, "method_a": a, "method_b": b}
                row.update(compare(feats[a], feats[b])); rows.append(row)
            print(f"[{robot:10s}] {idx:02d}/30 methods={len(feats)} pairs={len(feats)*(len(feats)-1)//2}", flush=True)

    with (OUT / "pairwise_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "missing.json").write_text(json.dumps(missing, indent=2) + "\n")

    summary = {}
    for metric in METRICS:
        z, n = matrix(rows, metric)
        summary.setdefault("all", {})[metric] = {"similarity": z.tolist(), "n": n.tolist()}
        plot_tile(z, n, metric, "All embodiments", OUT / f"tile_all_{metric}.png")
        for robot in ROBOTS:
            z, n = matrix(rows, metric, robot)
            summary.setdefault(robot, {})[metric] = {"similarity": z.tolist(), "n": n.tolist()}
            plot_tile(z, n, metric, robot.replace("_", " ").title(), OUT / f"tile_{robot}_{metric}.png")
    (OUT / "matrix_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    print(f"wrote {len(rows)} pair comparisons, {len(missing)} missing method-cells to {OUT}")


if __name__ == "__main__":
    main()
