#!/usr/bin/env python3
"""Sample four-way motion similarity against GT using common-camera renders.

This is a practical proxy study before exporting common 3-D keypoints from all
four decoders. Lower distances are better; similarities are reported in [0,1].
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence
from scipy.stats import pointbiserialr

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "data/results/evaluation/motion_similarity_sample"
SAMPLE = (1, 3, 5, 10, 15, 18, 22, 28)
ROBOTS = ("humanoid", "tiago")
METHODS = ("joint", "xyzrpy", "motion_program", "gt")
N = 24
SIZE = 96


def cue_map() -> dict[int, str]:
    rows = json.loads((REPO / "data/seed/agent_fill/embodiment_short/cues.json").read_text())
    return {int(r["idx"]): str(r["cue"]) for r in rows}


def path_for(method: str, robot: str, idx: int, cue: str) -> Path:
    if method == "joint":
        return REPO / f"data/results/baseline_traj/ours5/render_joint/{robot}/emb_g{idx:02d}_{cue}_gemini-2.5-pro.gif"
    if method == "xyzrpy":
        return REPO / f"data/results/baseline_traj/ours5/render_xyzrpy/{robot}/emb_g{idx:02d}_{cue}_gemini-2.5-pro.gif"
    root = "embodiment_motion_code_gt" if method == "gt" else "embodiment_motion_code_gemini_gemini-2.5-pro"
    return REPO / f"data/results/render/{root}/{robot}/gif/emb_g{idx:02d}_{cue}.gif"


def load(path: Path) -> np.ndarray:
    im = Image.open(path)
    frames = []
    for fr in ImageSequence.Iterator(im):
        a = np.asarray(fr.convert("RGB"))
        # Remove renderer progress strip; retain exactly the shared scene.
        a = a[28:, :, :]
        a = cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        frames.append(a)
    arr = np.asarray(frames, dtype=np.uint8)
    ids = np.linspace(0, len(arr) - 1, N).round().astype(int)
    return arr[ids]


def gray(x: np.ndarray) -> np.ndarray:
    return np.asarray([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in x])


def dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            d[i, j] = cost + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m] / (n + m))


def flow_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = gray(x)
    feats, energy = [], []
    for a, b in zip(g[:-1], g[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
        # 6x6 spatial flow field preserves which body region moves.
        u = cv2.resize(f[..., 0], (6, 6), interpolation=cv2.INTER_AREA)
        v = cv2.resize(f[..., 1], (6, 6), interpolation=cv2.INTER_AREA)
        feats.append(np.concatenate([u.ravel(), v.ravel()]))
        energy.append(float(np.mean(np.linalg.norm(f, axis=2))))
    return np.asarray(feats), np.asarray(energy)[:, None]


def edge_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for aa, bb in zip(gray(a), gray(b)):
        ea = cv2.Canny(aa, 65, 130) > 0
        eb = cv2.Canny(bb, 65, 130) > 0
        da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
        vals.append((float(db[ea].mean()) if ea.any() else SIZE) + (float(da[eb].mean()) if eb.any() else SIZE))
    return float(np.mean(vals) / 2.0)


def scores(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    # Same camera/background: RGB MAE is a strict pose+appearance distance.
    mae = float(np.mean(np.abs(pred.astype(float) - gt.astype(float))))
    pf, pe = flow_features(pred); gf, ge = flow_features(gt)
    fd = dtw(pf, gf)
    ed = dtw(pe, ge)
    chamfer = edge_chamfer(pred, gt)
    # Bounded kernels use interpretable scene-scale constants.
    return {
        "rgb_mae": mae,
        "edge_chamfer_px": chamfer,
        "flow_dtw": fd,
        "energy_dtw": ed,
        "rgb_sim": math.exp(-mae / 12.0),
        "edge_sim": math.exp(-chamfer / 4.0),
        "flow_sim": math.exp(-fd / 2.0),
        "energy_sim": math.exp(-ed / 0.15),
    }


def auc(y: list[int], s: list[float]) -> float:
    pos = [v for v, yy in zip(s, y) if yy == 1]
    neg = [v for v, yy in zip(s, y) if yy == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + .5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cues = cue_map()
    rows = []
    for robot in ROBOTS:
        for idx in SAMPLE:
            cue = cues[idx]
            paths = {m: path_for(m, robot, idx, cue) for m in METHODS}
            if not all(p.is_file() for p in paths.values()):
                print("[skip missing]", robot, idx, {m: str(p) for m, p in paths.items() if not p.is_file()})
                continue
            videos = {m: load(p) for m, p in paths.items()}
            for method in METHODS:
                row = {"robot": robot, "idx": idx, "cue": cue, "method": method}
                row.update(scores(videos[method], videos["gt"]))
                rows.append(row)
                print(f"{robot:8s} g{idx:02d} {method:14s} edge={row['edge_sim']:.3f} flow={row['flow_sim']:.3f} energy={row['energy_sim']:.3f}")

    fields = list(rows[0])
    with (OUT / "scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    labels = json.loads((REPO / "data/results/evaluation/embodiment_cue_id/cueid_human_cue_identification_with_traj_baselines.json").read_text())["cell_labels"]
    validation = {}
    for metric in ("rgb_sim", "edge_sim", "flow_sim", "energy_sim"):
        y, s = [], []
        for r in rows:
            if r["method"] not in ("joint", "xyzrpy"):
                continue
            lab = labels[r["robot"]][f"g{r['idx']:02d}"][r["method"]]
            if lab not in ("o", "x"):
                continue
            y.append(1 if lab == "o" else 0); s.append(float(r[metric]))
        validation[metric] = {
            "n": len(y), "yes": sum(y), "auc": auc(y, s),
            "point_biserial_r": float(pointbiserialr(y, s).statistic),
        }
    (OUT / "metric_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

    summary = {}
    for method in METHODS:
        rr = [r for r in rows if r["method"] == method]
        summary[method] = {k: float(np.mean([r[k] for r in rr])) for k in ("rgb_sim", "edge_sim", "flow_sim", "energy_sim")}
    (OUT / "method_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\nVALIDATION", json.dumps(validation, indent=2))
    print("\nSUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
