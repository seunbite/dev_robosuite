#!/usr/bin/env python3
"""Tiago-only posture-aware Cartesian few-shot baseline for four pilot cues.

The arm representation contains the usual hand pose plus an elbow-link position
feature.  The latter removes the ordinary redundant-arm IK branch ambiguity.
Target Motion Codes are extracted only for round-trip diagnostics and are never
included in the inference prompt.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, HERE, REPO / "adhoc/generation/tiago"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from correction_lib import _import_tiago_render  # noqa: E402
from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_baseline_traj import (  # noqa: E402
    HZ,
    MAX_FRAMES,
    _index_gt,
    _quat_wxyz_to_rpy,
    _round_rows,
    _subsample,
    _write_gif,
)

TARGETS = (1, 2, 29, 30)
LINKS = (
    "base",
    "torso",
    "head",
    "right_hand",
    "left_hand",
    "right_elbow_posture",
    "left_elbow_posture",
)
BODY_NAMES = {
    "base": "robot0_base_link",
    "torso": "robot0_torso_lift_link",
    "head": "robot0_head_2_link",
    "right_hand": "robot0_right_hand",
    "left_hand": "robot0_left_hand",
    "right_elbow_posture": "robot0_arm_right_4_link",
    "left_elbow_posture": "robot0_arm_left_4_link",
}
LINK_DOFS = {
    "base": (0, 1, 2),
    "torso": (3,),
    "head": (4, 5),
    "right_hand": tuple(range(6, 12)),
    "left_hand": tuple(range(18, 24)),
    "right_elbow_posture": tuple(range(6, 12)),
    "left_elbow_posture": tuple(range(18, 24)),
}
SOURCE_JSON = REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
SOURCE_XLSX = REPO / "data/seed/shots/embodiment_unified/fewshot_cues_yml_5.xlsx"
GT_RENDER = REPO / "data/results/render/embodiment_motion_code_gt/tiago/gif"
DEFAULT_OUT = REPO / "data/results/baseline_traj/tiago_posture_cartesian_four"


def body_pose(env: Any, body_id: int) -> list[float]:
    import mujoco

    pos = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
    mat = np.asarray(env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.ravel())
    return [*map(float, pos), *map(float, _quat_wxyz_to_rpy(quat))]


def extract_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Execute Motion Code and record the seven-link Cartesian representation."""
    make_env, render_config, mod = _import_tiago_render()
    env = make_env()
    bids = [int(env.sim.model.body_name2id(BODY_NAMES[name])) for name in LINKS]
    rows: list[list[list[float]]] = []
    original_capture = mod._capture

    def capture(cap_env: Any, q: np.ndarray, camera: str = mod.CAM):
        image = original_capture(cap_env, q, camera)
        rows.append([body_pose(cap_env, bid) for bid in bids])
        return image

    mod._capture = capture
    try:
        render_config(cell, env=env, overlay_progress_bar=False)
    finally:
        mod._capture = original_capture
        env.close()
    if not rows:
        raise RuntimeError(f"no Tiago states for {cell.get('cue')}")
    sampled = _subsample(rows)
    links: dict[str, Any] = {}
    for i, name in enumerate(LINKS):
        width = 3 if name.endswith("_elbow_posture") else 6
        links[name] = {"xyz" if width == 3 else "xyzrpy": _round_rows(
            [frame[i][:width] for frame in sampled]
        )}
    return {
        "cue": str(cell["cue"]),
        "hz": HZ,
        "link_names": list(LINKS),
        "links": links,
    }


def bounds(qidx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(len(qidx), -3.0)
    hi = np.full(len(qidx), 3.0)
    for i, qi in enumerate(qidx):
        if qi == 3:
            lo[i], hi[i] = 0.0, 0.35
        elif qi == 2:
            lo[i], hi[i] = -math.pi, math.pi
        elif qi in (0, 1):
            lo[i], hi[i] = -1.0, 1.0
    return lo, hi


def angle_delta(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (actual - target + math.pi) % (2.0 * math.pi) - math.pi


def render_and_measure(payload: dict[str, Any], gif_path: Path) -> dict[str, Any]:
    """Posture-aware IK replay, returning target-versus-rendered errors."""
    from scipy.optimize import least_squares

    make_env, _render_config, mod = _import_tiago_render()
    env = make_env()
    names = list(payload["link_names"])
    if names != list(LINKS):
        raise ValueError(f"link_names must be {list(LINKS)}")
    bids = [int(env.sim.model.body_name2id(BODY_NAMES[name])) for name in names]
    n_frames = len(payload["links"]["base"]["xyzrpy"])
    rows = []
    for fi in range(n_frames):
        frame = []
        for name in LINKS:
            key = "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
            frame.append(np.asarray(payload["links"][name][key][fi], dtype=float))
        rows.append(frame)
    spans = np.asarray([
        np.ptp(np.asarray([frame[i] for frame in rows]), axis=0).max()
        for i in range(len(LINKS))
    ])
    moving = {name for name, span in zip(names, spans) if float(span) > 1e-3}
    frames = []
    actual_rows: list[list[list[float]]] = []
    q = np.asarray(mod._default_qpos(env), dtype=float)

    def residual(x: np.ndarray, qidx: np.ndarray, target: np.ndarray) -> np.ndarray:
        q[qidx] = x
        env.sim.data.qpos[:] = q
        env.sim.forward()
        chunks: list[np.ndarray] = []
        for name, bid, want in zip(names, bids, target):
            got = np.asarray(body_pose(env, bid), dtype=float)
            if name.endswith("_elbow_posture"):
                # The semantic posture feature is elbow position, not elbow-frame orientation.
                chunks.append(1.5 * (got[:3] - want[:3]))
            else:
                chunks.append(got[:3] - want[:3])
                chunks.append(0.25 * angle_delta(got[3:], want[3:]))
        return np.concatenate(chunks)

    try:
        for frame_i, target in enumerate(rows):
            if frame_i == 0:
                free = sorted({qi for name in names for qi in LINK_DOFS[name]})
            else:
                free = sorted({qi for name in moving for qi in LINK_DOFS[name]})
                if not free:
                    free = [4, 5]
            qidx = np.asarray(free, dtype=int)
            lo, hi = bounds(qidx)
            q[qidx] = least_squares(
                residual,
                np.clip(q[qidx], lo, hi),
                args=(qidx, target),
                bounds=(lo, hi),
                max_nfev=140 if frame_i == 0 else 90,
            ).x
            env.sim.data.qpos[:] = q
            env.sim.forward()
            actual = [body_pose(env, bid) for bid in bids]
            actual_rows.append(actual)
            frames.append(mod._capture(env, q, mod.CAM))
        _write_gif(frames, gif_path, hz=int(payload.get("hz", HZ)))
    finally:
        env.close()

    actual = np.asarray(actual_rows, dtype=float)
    pose_pos: list[float] = []
    pose_rot: list[float] = []
    elbow_pos: list[float] = []
    per_link: dict[str, Any] = {}
    for i, name in enumerate(names):
        target_i = np.asarray([frame[i] for frame in rows], dtype=float)
        pos = np.linalg.norm(actual[:, i, :3] - target_i[:, :3], axis=1)
        per_link[name] = {
            "position_rmse_m": float(np.sqrt(np.mean(pos**2))),
            "position_max_m": float(np.max(pos)),
        }
        if name.endswith("_elbow_posture"):
            elbow_pos.extend(pos.tolist())
        else:
            rot = np.linalg.norm(angle_delta(actual[:, i, 3:], target_i[:, 3:]), axis=1)
            per_link[name].update(
                orientation_rmse_rad=float(np.sqrt(np.mean(rot**2))),
                orientation_max_rad=float(np.max(rot)),
            )
            pose_pos.extend(pos.tolist())
            pose_rot.extend(rot.tolist())
    return {
        "n_frames": int(len(rows)),
        "target_cartesian_pose": {
            "position_rmse_m": float(np.sqrt(np.mean(np.square(pose_pos)))),
            "position_max_m": float(np.max(pose_pos)),
            "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(pose_rot)))),
            "orientation_max_rad": float(np.max(pose_rot)),
        },
        "elbow_posture_feature": {
            "position_rmse_m": float(np.sqrt(np.mean(np.square(elbow_pos)))),
            "position_max_m": float(np.max(elbow_pos)),
        },
        "per_link": per_link,
    }


def compact_links(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in LINKS:
        key = "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
        block = payload["links"][name][key]
        if len(block) > 1 and float(np.ptp(np.asarray(block), axis=0).max()) < 1e-4:
            result[name] = {"constant": True, "n_frames": len(block), key: [block[0]]}
        else:
            result[name] = {key: block}
    return result


def expand_links(obj: dict[str, Any]) -> dict[str, Any]:
    links = obj.get("links")
    if not isinstance(links, dict) or set(links) != set(LINKS):
        raise ValueError(f"links must contain exactly {list(LINKS)}")
    def key_for(name: str) -> str:
        return "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
    n = max(len((links[name] or {}).get(key_for(name)) or []) for name in LINKS)
    n = max(n, *(int((links[name] or {}).get("n_frames") or 0) for name in LINKS))
    if not 12 <= n <= MAX_FRAMES:
        raise ValueError(f"n_frames={n}, expected 12..{MAX_FRAMES}")
    expanded: dict[str, Any] = {}
    for name in LINKS:
            block = links[name] or {}
            key = key_for(name)
            vals = block.get(key) or []
            if not isinstance(vals, list) or not vals:
                raise ValueError(f"empty links.{name}.{key}")
            width = 3 if key == "xyz" else 6
            out_rows = []
            for fi in range(n):
                pose = vals[0] if block.get("constant") and len(vals) == 1 else vals[min(fi, len(vals) - 1)]
                if not isinstance(pose, list) or len(pose) != width:
                    raise ValueError(f"links.{name}.{key}[{fi}] must have {width} floats")
                out_rows.append([float(v) for v in pose])
            expanded[name] = {key: _round_rows(out_rows)}
    return {"cue": str(obj.get("cue") or ""), "hz": int(obj.get("hz") or HZ),
            "link_names": list(LINKS), "links": expanded}


def build_prompt(cue: str, description: str, shots: list[dict[str, Any]]) -> str:
    examples = []
    for shot in shots:
        examples.append(json.dumps({
            "cue": shot["cue"], "hz": shot["hz"],
            "n_frames": len(shot["links"]["base"]["xyzrpy"]),
            "links": compact_links(shot),
        }, ensure_ascii=False))
    schema = {
        name: ({"xyz": [[0.0] * 3]} if name.endswith("_elbow_posture")
               else {"xyzrpy": [[0.0] * 6]})
        for name in LINKS
    }
    return f"""You are an expert Tiago robot motion designer. Generate one posture-aware Cartesian trajectory.

Representation (world frame): ordinary xyzrpy rows are [x,y,z,roll,pitch,yaw].
- base, torso, head, right_hand, left_hand are Cartesian link poses.
- right_elbow_posture and left_elbow_posture are strictly XYZ-only arm-posture features. Their 3D position fixes the elbow branch and must remain anatomically consistent with the corresponding hand. Do not provide elbow RPY.

Return ONE JSON object only, no markdown:
{{"cue":"{cue}","hz":{HZ},"links":{json.dumps(schema)}}}

Rules:
- Return exactly these links in this semantic order: {list(LINKS)}.
- Each ordinary xyzrpy series has 12–{MAX_FRAMES} rows of exactly 6 floats. Each elbow_posture xyz series has exactly 3 floats per row. You may encode a still link as constant=true, n_frames=N, and one row.
- All links share the same n_frames. Keep still links at demonstrated rest poses.
- Head nod/shake changes head orientation, not its height or the torso.
- For arm gestures, move each hand and its matching elbow_posture together; never move a hand while leaving its elbow feature inconsistent.
- Use smooth, reachable motions within the demonstrated numeric workspace. Do not copy an example.

Few-shot examples extracted from the Excel-derived Motion Codes:
{chr(10).join(examples)}

Target cue: {cue}
Description: {description}
""".strip()


def infer(prompt: str, cue: str, model: str, retries: int) -> tuple[dict[str, Any], str]:
    from google import genai
    from google.genai import types

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=key)
    error = ""
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=16384),
            )
            raw = (response.text or "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end < start:
                raise ValueError("no JSON object in response")
            payload = expand_links(json.loads(raw[start:end + 1]))
            payload["cue"] = cue
            return payload, raw
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(error)


def make_gallery(out: Path, targets: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    cards = []
    for target in targets:
        idx, cue = target["idx"], target["cue"]
        rt = report["target_roundtrip"][cue]
        pred = report["predictions"][cue]
        def media(rel: str, label: str) -> str:
            return f'<div><h3>{html.escape(label)}</h3><img src="{html.escape(rel)}"></div>'
        cards.append(
            f'<section><h2>g{idx:02d} · {html.escape(cue)}</h2>'
            f'<p>{html.escape(target.get("description") or "")}</p><div class="grid">'
            + media(f"source_motion_code/emb_g{idx:02d}_{cue}.gif", "Source Motion Code")
            + media(f"target_roundtrip/emb_g{idx:02d}_{cue}.gif", "Extracted posture-Cartesian round-trip")
            + media(f"predictions/emb_g{idx:02d}_{cue}.gif", "Posture-aware Cartesian prediction")
            + '</div><p class="metric">Round-trip: pose pos RMSE '
            + f'{rt["target_cartesian_pose"]["position_rmse_m"]:.4f} m; pose rot RMSE '
            + f'{rt["target_cartesian_pose"]["orientation_rmse_rad"]:.4f} rad; elbow RMSE '
            + f'{rt["elbow_posture_feature"]["position_rmse_m"]:.4f} m. Prediction replay: pose pos RMSE '
            + f'{pred["target_cartesian_pose"]["position_rmse_m"]:.4f} m; elbow RMSE '
            + f'{pred["elbow_posture_feature"]["position_rmse_m"]:.4f} m.</p></section>'
        )
    path = out / "gallery_tiago_posture_cartesian_four.html"
    path.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Tiago posture-aware Cartesian baseline</title>
<style>body{{margin:0;background:#11151b;color:#edf2f7;font:15px system-ui,sans-serif}}header{{padding:24px;position:sticky;top:0;background:#11151bee;z-index:2;border-bottom:1px solid #39424e}}section{{margin:18px;padding:18px;background:#1a2029;border:1px solid #39424e;border-radius:12px}}h1,h2,h3{{margin:.25em 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000;border-radius:8px}}.metric{{color:#b9c3cf;font-family:ui-monospace,monospace}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><header><h1>Tiago · posture-aware Cartesian baseline</h1><div>4 requested cues · Excel-derived Motion Code few-shots · hand pose + elbow-posture constraint</div></header>{''.join(cards)}</body></html>""", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    fewshot_entries = source["tiago"]
    if len(fewshot_entries) != 5:
        raise RuntimeError(f"expected five Tiago few-shots, got {len(fewshot_entries)}")
    artifact_path = out / "fewshot_posture_cartesian.json"
    if artifact_path.is_file():
        shots = json.loads(artifact_path.read_text(encoding="utf-8"))["shots"]
    else:
        shots = []
    fewshot_metrics: dict[str, Any] = {}
    if not shots:
        for rank, entry in enumerate(fewshot_entries, 1):
            parsed = convert_code(entry["motion_code"], platform="tiago")
            cell = {"idx": rank, "cue": entry["cue"], "description": entry.get("notes", ""),
                    "movements": parsed.get("movements") or [], "source_code": entry["motion_code"]}
            shot = extract_cell(cell)
            shot["motion_code"] = entry["motion_code"]
            shots.append(shot)
    for rank, (entry, shot) in enumerate(zip(fewshot_entries, shots), 1):
        gif = out / "fewshot_roundtrip" / f"{rank:02d}_{entry['cue']}.gif"
        metric_path = out / "metrics" / f"fewshot_{rank:02d}_{entry['cue']}.json"
        if metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metric = render_and_measure(shot, gif)
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
        fewshot_metrics[entry["cue"]] = metric

    artifact = {
        "meta": {"description": "Tiago posture-aware Cartesian few-shot baseline artifact",
                 "source_xlsx": str(SOURCE_XLSX.relative_to(REPO)),
                 "source_json": str(SOURCE_JSON.relative_to(REPO)), "n_shots": 5,
                 "representation": list(LINKS), "posture_feature": "elbow-link XYZ position only"},
        "shots": shots,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    gt = _index_gt()
    targets = []
    target_metrics: dict[str, Any] = {}
    prediction_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for idx in TARGETS:
        cell = dict(gt[(idx, "tiago")])
        targets.append(cell)
        cue = cell["cue"]
        source_gif = GT_RENDER / f"emb_g{idx:02d}_{cue}.gif"
        (out / "source_motion_code").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_gif, out / "source_motion_code" / source_gif.name)
        (out / "target_extract").mkdir(parents=True, exist_ok=True)
        extracted_path = out / "target_extract" / f"g{idx:02d}_{cue}.json"
        if extracted_path.is_file():
            extracted = json.loads(extracted_path.read_text(encoding="utf-8"))
        else:
            extracted = extract_cell(cell)
            extracted_path.write_text(json.dumps(extracted, indent=2) + "\n", encoding="utf-8")
        target_metric_path = out / "metrics" / f"target_g{idx:02d}_{cue}.json"
        if target_metric_path.is_file():
            target_metric = json.loads(target_metric_path.read_text(encoding="utf-8"))
        else:
            target_metric = render_and_measure(
                extracted, out / "target_roundtrip" / f"emb_g{idx:02d}_{cue}.gif")
            target_metric_path.parent.mkdir(parents=True, exist_ok=True)
            target_metric_path.write_text(json.dumps(target_metric, indent=2) + "\n", encoding="utf-8")
        target_metrics[cue] = target_metric

        prompt = build_prompt(cue, cell.get("description") or "", shots)
        (out / "prompts").mkdir(parents=True, exist_ok=True)
        (out / "prompts" / f"g{idx:02d}_{cue}.txt").write_text(prompt + "\n", encoding="utf-8")
        if args.skip_inference:
            continue
        prediction_path = out / "prediction_payloads" / f"g{idx:02d}_{cue}.json"
        if prediction_path.is_file():
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            raw = ""
        else:
            payload, raw = infer(prompt, cue, args.model, args.retries)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (out / "raw_responses").mkdir(parents=True, exist_ok=True)
        raw_path = out / "raw_responses" / f"g{idx:02d}_{cue}.txt"
        if raw:
            raw_path.write_text(raw + "\n", encoding="utf-8")
        prediction_metric_path = out / "metrics" / f"prediction_g{idx:02d}_{cue}.json"
        if prediction_metric_path.is_file():
            prediction_metric = json.loads(prediction_metric_path.read_text(encoding="utf-8"))
        else:
            prediction_metric = render_and_measure(
                payload, out / "predictions" / f"emb_g{idx:02d}_{cue}.gif")
            prediction_metric_path.parent.mkdir(parents=True, exist_ok=True)
            prediction_metric_path.write_text(json.dumps(prediction_metric, indent=2) + "\n", encoding="utf-8")
        prediction_metrics[cue] = prediction_metric
        prediction_rows.append({"job_id": f"xyzrpy_posture.tiago.g{idx:02d}", "platform": "tiago",
                                "idx": idx, "cue": cue, "model": args.model, "parse_ok": True,
                                "payload": payload})

    if not args.skip_inference:
        if len(prediction_rows) != 4:
            raise RuntimeError(f"expected exactly 4 predictions, got {len(prediction_rows)}")
        (out / "predictions_posture_cartesian.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8")
    report = {"fewshot_roundtrip": fewshot_metrics, "target_roundtrip": target_metrics,
              "predictions": prediction_metrics, "inference_jobs": len(prediction_rows)}
    (out / "roundtrip_error_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.skip_inference:
        gallery = make_gallery(out, targets, report)
        print(f"complete: {gallery}")


if __name__ == "__main__":
    main()
