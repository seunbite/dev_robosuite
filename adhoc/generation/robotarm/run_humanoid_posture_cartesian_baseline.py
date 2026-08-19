#!/usr/bin/env python3
"""Humanoid posture-aware Cartesian baseline for exactly four pilot cues.

The representation is root/torso/head/right_hand/left_hand XYZRPY plus
right_elbow_posture/left_elbow_posture XYZ. Target Motion Code is used only
for source and round-trip diagnostics and is never included in inference.
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from motion_code_to_config import convert_code  # noqa: E402
from run_embodiment_baseline_traj import (  # noqa: E402
    HZ,
    MAX_FRAMES,
    _index_gt,
    _quat_wxyz_to_rpy,
    _round_rows,
    _subsample,
)

TARGETS = (1, 2, 29, 30)
LINKS = (
    "root", "torso", "head", "right_hand", "left_hand",
    "right_elbow_posture", "left_elbow_posture",
)
BODY_CANDIDATES = {
    "root": ("pelvis", "torso_link", "pelvis_contour_link"),
    "torso": ("torso_link", "waist_yaw_link", "pelvis"),
    "head": ("neck_pitch_link", "neck_yaw_link", "torso_link"),
    "right_hand": ("right_rubber_hand", "right_palm", "right_wrist_yaw_link"),
    "left_hand": ("left_rubber_hand", "left_palm", "left_wrist_yaw_link"),
    "right_elbow_posture": ("right_elbow_link",),
    "left_elbow_posture": ("left_elbow_link",),
}
SOURCE_JSON = REPO / "data/results/shots/embodiment_unified/shot_cues_yml_5.json"
SOURCE_XLSX = REPO / "data/seed/shots/embodiment_unified/fewshot_cues_yml_5.xlsx"
DEFAULT_OUT = REPO / "data/results/baseline_traj/humanoid_posture_cartesian_four"


def _model_context():
    import mujoco
    from embodiment_renderers import _G1_XML, _make_g1_scene_xml

    if not _G1_XML.is_file():
        raise FileNotFoundError(_G1_XML)
    scene = _make_g1_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(scene))
    return scene, model, mujoco.MjData(model)


def _body_ids(model: Any) -> list[int]:
    names = {model.body(i).name: i for i in range(model.nbody)}
    result = []
    for label in LINKS:
        found = next((names[n] for n in BODY_CANDIDATES[label] if n in names), None)
        if found is None:
            raise KeyError(f"no body for {label}: {BODY_CANDIDATES[label]}")
        result.append(int(found))
    return result


def _body_pose(data: Any, bid: int) -> list[float]:
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(data.xmat[bid], dtype=float).ravel())
    return [*map(float, data.xpos[bid]), *map(float, _quat_wxyz_to_rpy(quat))]


def extract_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Execute humanoid Motion Code and capture the seven-link representation."""
    import mujoco
    from embodiment_renderers import _g1_motion_qpos

    scene, model, data = _model_context()
    try:
        bids = _body_ids(model)
        qseq = _g1_motion_qpos(cell, model, fps=HZ)
        rows: list[list[list[float]]] = []
        for q in qseq:
            data.qpos[:] = q
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            rows.append([_body_pose(data, bid) for bid in bids])
        if not rows:
            raise RuntimeError(f"no humanoid states for {cell.get('cue')}")
        sampled = _subsample(rows)
        links = {}
        for i, name in enumerate(LINKS):
            key = "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
            width = 3 if key == "xyz" else 6
            links[name] = {key: _round_rows([frame[i][:width] for frame in sampled])}
        return {"cue": str(cell["cue"]), "hz": HZ, "link_names": list(LINKS), "links": links}
    finally:
        scene.unlink(missing_ok=True)


def _angle_delta(actual: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (actual - target + math.pi) % (2.0 * math.pi) - math.pi


def render_and_measure(payload: dict[str, Any], gif_path: Path) -> dict[str, Any]:
    """Replay with posture-aware IK and measure target-versus-replayed links."""
    import mujoco
    from scipy.optimize import least_squares
    from embodiment_renderers import _g1_home, _joint_qpos_map, _progress_overlay, _save_media

    if list(payload.get("link_names") or []) != list(LINKS):
        raise ValueError(f"link_names must be {list(LINKS)}")
    scene, model, data = _model_context()
    renderer = None
    try:
        bids = _body_ids(model)
        jmap = _joint_qpos_map(model)
        q = np.asarray(_g1_home(model, jmap), dtype=float)
        gesture_names = [
            n for n in jmap
            if any(tok in n for tok in ("waist", "head", "shoulder", "elbow", "wrist"))
            and int(model.jnt_type[model.joint(n).id]) in (2, 3)
            and model.jnt_range[model.joint(n).id, 1] - model.jnt_range[model.joint(n).id, 0] > 1e-8
        ]
        qidx = np.asarray([jmap[n] for n in gesture_names], dtype=int)
        lo = np.asarray([model.jnt_range[model.joint(n).id, 0] for n in gesture_names], dtype=float)
        hi = np.asarray([model.jnt_range[model.joint(n).id, 1] for n in gesture_names], dtype=float)
        n_frames = len(payload["links"]["root"]["xyzrpy"])
        targets = []
        for fi in range(n_frames):
            targets.append([
                np.asarray(payload["links"][name]["xyz" if name.endswith("_elbow_posture") else "xyzrpy"][fi], dtype=float)
                for name in LINKS
            ])

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.85)
        camera.distance = 2.7
        camera.azimuth = 145
        camera.elevation = -8
        renderer = mujoco.Renderer(model, height=480, width=480)
        frames = []
        actual_rows = []

        def residual(x: np.ndarray, target: list[np.ndarray]) -> np.ndarray:
            q[qidx] = x
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            chunks = []
            for name, bid, want in zip(LINKS, bids, target):
                got = np.asarray(_body_pose(data, bid), dtype=float)
                if name.endswith("_elbow_posture"):
                    chunks.append(1.5 * (got[:3] - want[:3]))
                else:
                    chunks.append(got[:3] - want[:3])
                    chunks.append(0.35 * _angle_delta(got[3:], want[3:]))
            return np.concatenate(chunks)

        for fi, target in enumerate(targets):
            q[qidx] = least_squares(
                residual, np.clip(q[qidx], lo, hi), args=(target,), bounds=(lo, hi),
                max_nfev=160 if fi == 0 else 100,
            ).x
            data.qpos[:] = q
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            actual_rows.append([_body_pose(data, bid) for bid in bids])
            renderer.update_scene(data, camera)
            frames.append(_progress_overlay(renderer.render(), fi, n_frames))
        _save_media(frames, gif_path, fps=int(payload.get("hz", HZ)))

        actual = np.asarray(actual_rows, dtype=float)
        pose_pos: list[float] = []
        pose_rot: list[float] = []
        elbow_pos: list[float] = []
        per_link = {}
        for i, name in enumerate(LINKS):
            want = np.asarray([frame[i] for frame in targets], dtype=float)
            pos = np.linalg.norm(actual[:, i, :3] - want[:, :3], axis=1)
            per_link[name] = {"position_rmse_m": float(np.sqrt(np.mean(pos**2))), "position_max_m": float(np.max(pos))}
            if name.endswith("_elbow_posture"):
                elbow_pos.extend(pos.tolist())
            else:
                rot = np.linalg.norm(_angle_delta(actual[:, i, 3:], want[:, 3:]), axis=1)
                per_link[name].update(orientation_rmse_rad=float(np.sqrt(np.mean(rot**2))), orientation_max_rad=float(np.max(rot)))
                pose_pos.extend(pos.tolist())
                pose_rot.extend(rot.tolist())
        return {
            "n_frames": n_frames,
            "source_to_extract": {"position_rmse_m": 0.0, "orientation_rmse_rad": 0.0, "note": "direct state capture from source execution"},
            "extract_to_replay": {
                "target_cartesian_pose": {
                    "position_rmse_m": float(np.sqrt(np.mean(np.square(pose_pos)))), "position_max_m": float(np.max(pose_pos)),
                    "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(pose_rot)))), "orientation_max_rad": float(np.max(pose_rot)),
                },
                "elbow_posture_feature": {"position_rmse_m": float(np.sqrt(np.mean(np.square(elbow_pos)))), "position_max_m": float(np.max(elbow_pos))},
            },
            "per_link": per_link,
        }
    finally:
        if renderer is not None:
            renderer.close()
        scene.unlink(missing_ok=True)


def compact_links(payload: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in LINKS:
        key = "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
        rows = payload["links"][name][key]
        if len(rows) > 1 and float(np.ptp(np.asarray(rows), axis=0).max()) < 1e-4:
            result[name] = {"constant": True, "n_frames": len(rows), key: [rows[0]]}
        else:
            result[name] = {key: rows}
    return result


def expand_links(obj: dict[str, Any]) -> dict[str, Any]:
    links = obj.get("links")
    if not isinstance(links, dict) or set(links) != set(LINKS):
        raise ValueError(f"links must contain exactly {list(LINKS)}")
    key_for = lambda name: "xyz" if name.endswith("_elbow_posture") else "xyzrpy"
    n = max(max(len((links[name] or {}).get(key_for(name)) or []), int((links[name] or {}).get("n_frames") or 0)) for name in LINKS)
    if not 12 <= n <= MAX_FRAMES:
        raise ValueError(f"n_frames={n}, expected 12..{MAX_FRAMES}")
    expanded = {}
    for name in LINKS:
        block = links[name] or {}
        key = key_for(name)
        vals = block.get(key) or []
        if not isinstance(vals, list) or not vals:
            raise ValueError(f"empty links.{name}.{key}")
        width = 3 if key == "xyz" else 6
        rows = []
        for fi in range(n):
            row = vals[0] if block.get("constant") and len(vals) == 1 else vals[min(fi, len(vals) - 1)]
            if not isinstance(row, list) or len(row) != width:
                raise ValueError(f"links.{name}.{key}[{fi}] must have {width} floats")
            rows.append([float(v) for v in row])
        expanded[name] = {key: _round_rows(rows)}
    return {"cue": str(obj.get("cue") or ""), "hz": int(obj.get("hz") or HZ), "link_names": list(LINKS), "links": expanded}


def build_prompt(cue: str, description: str, shots: list[dict[str, Any]]) -> str:
    examples = [json.dumps({"cue": s["cue"], "hz": s["hz"], "n_frames": len(s["links"]["root"]["xyzrpy"]), "links": compact_links(s)}, ensure_ascii=False) for s in shots]
    schema = {name: ({"xyz": [[0.0] * 3]} if name.endswith("_elbow_posture") else {"xyzrpy": [[0.0] * 6]}) for name in LINKS}
    return f"""You are an expert Unitree G1 humanoid motion designer. Generate one posture-aware Cartesian trajectory.

Representation (world frame): ordinary XYZRPY rows are [x,y,z,roll,pitch,yaw].
- root, torso, head, right_hand, left_hand are Cartesian link poses.
- right_elbow_posture and left_elbow_posture are strictly XYZ-only arm-posture features. Their 3D positions fix each redundant arm's elbow branch. Never provide elbow RPY or any direct joint angle.

Return ONE JSON object only, no markdown:
{{"cue":"{cue}","hz":{HZ},"links":{json.dumps(schema)}}}

Rules:
- Return exactly these links in this semantic order: {list(LINKS)}.
- Ordinary xyzrpy series have 12–{MAX_FRAMES} rows of 6 floats. Elbow-posture xyz series have 3 floats per row. A still link may use constant=true, n_frames=N, and one row.
- All links share n_frames. Keep still links at demonstrated rest poses; keep the root/legs grounded.
- Head nod/shake changes head orientation, not height or torso pose.
- Arm gestures coordinate each hand with its matching elbow_posture; do not move a hand with an inconsistent elbow feature.
- Use smooth, reachable motion inside the demonstrated workspace. Do not copy an example.

Five few-shot examples extracted from Excel-synchronized Motion Code:
{chr(10).join(examples)}

Target cue: {cue}
Description: {description}
""".strip()


def infer(prompt: str, cue: str, model: str) -> tuple[dict[str, Any], str]:
    from google import genai
    from google.genai import types

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    # Keep the client alive for the duration of the request.  The SDK's
    # ``models`` facade does not retain a strong reference to a temporary
    # client, so chaining construction could close its HTTP transport early.
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=16384),
    )
    raw = (response.text or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in response")
    payload = expand_links(json.loads(raw[start:end + 1]))
    payload["cue"] = cue
    return payload, raw


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def make_gallery(out: Path, targets: list[dict[str, Any]], report: dict[str, Any]) -> Path:
    cards = []
    for target in targets:
        idx, cue = target["idx"], target["cue"]
        rt = report["target_roundtrip"][cue]["extract_to_replay"]
        pred = report["predictions"][cue]["extract_to_replay"]
        def media(path: Path, label: str) -> str:
            return f'<div><h3>{html.escape(label)}</h3><img alt="{html.escape(label)}" src="{_data_uri(path)}"></div>'
        cards.append(
            f'<section><h2>g{idx:02d} · {html.escape(cue)}</h2><p>{html.escape(target.get("description") or "")}</p><div class="grid">'
            + media(out / "source_motion_code" / f"emb_g{idx:02d}_{cue}.gif", "Source Motion Code")
            + media(out / "target_roundtrip" / f"emb_g{idx:02d}_{cue}.gif", "Extracted posture-Cartesian round-trip")
            + media(out / "predictions" / f"emb_g{idx:02d}_{cue}.gif", "Posture-aware Cartesian prediction")
            + f'</div><p class="metric">Source→extract RMSE: 0 (direct capture). Extract→replay: pose {rt["target_cartesian_pose"]["position_rmse_m"]:.4f} m / {rt["target_cartesian_pose"]["orientation_rmse_rad"]:.4f} rad; elbow {rt["elbow_posture_feature"]["position_rmse_m"]:.4f} m. Prediction replay: pose {pred["target_cartesian_pose"]["position_rmse_m"]:.4f} m; elbow {pred["elbow_posture_feature"]["position_rmse_m"]:.4f} m.</p></section>'
        )
    path = out / "gallery_humanoid_posture_cartesian_four_self_contained.html"
    path.write_text(f'''<!doctype html><html><head><meta charset="utf-8"><title>Humanoid posture-aware Cartesian baseline</title><style>body{{margin:0;background:#11151b;color:#edf2f7;font:15px system-ui,sans-serif}}header{{padding:24px;border-bottom:1px solid #39424e}}section{{margin:18px;padding:18px;background:#1a2029;border:1px solid #39424e;border-radius:12px}}h1,h2,h3{{margin:.25em 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}img{{width:100%;aspect-ratio:1;object-fit:cover;background:#000;border-radius:8px}}.metric{{color:#b9c3cf;font-family:ui-monospace,monospace}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><header><h1>Humanoid · posture-aware Cartesian baseline</h1><div>Exactly 4 cues · 5 Excel-derived Motion Code few-shots · XYZ-only elbow posture</div></header>{''.join(cards)}</body></html>''', encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    entries = source["humanoid"]
    if len(entries) != 5:
        raise RuntimeError(f"expected five humanoid few-shots, got {len(entries)}")
    artifact_path = out / "fewshot_posture_cartesian.json"
    shots = json.loads(artifact_path.read_text(encoding="utf-8"))["shots"] if artifact_path.is_file() else []
    if not shots:
        for rank, entry in enumerate(entries, 1):
            parsed = convert_code(entry["motion_code"], platform="humanoid")
            cell = {"idx": rank, "cue": entry["cue"], "description": entry.get("notes", ""), "movements": parsed.get("movements") or [], "source_code": entry["motion_code"]}
            shot = extract_cell(cell)
            shot["motion_code"] = entry["motion_code"]
            shots.append(shot)
    fewshot_metrics = {}
    for rank, (entry, shot) in enumerate(zip(entries, shots), 1):
        metric_path = out / "metrics" / f"fewshot_{rank:02d}_{entry['cue']}.json"
        if metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metric = render_and_measure(shot, out / "fewshot_roundtrip" / f"{rank:02d}_{entry['cue']}.gif")
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
        fewshot_metrics[entry["cue"]] = metric
    artifact = {"meta": {"description": "Humanoid posture-aware Cartesian five-shot artifact", "source_xlsx": str(SOURCE_XLSX.relative_to(REPO)), "source_json": str(SOURCE_JSON.relative_to(REPO)), "n_shots": 5, "representation": {"xyzrpy": list(LINKS[:5]), "xyz_only": list(LINKS[5:])}, "direct_joint_angle_features": False}, "shots": shots}
    artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    gt = _index_gt()
    # Render the source Motion Codes with the same current G1 runtime used by
    # extraction. Historical GT media can predate renderer/pose-resolution
    # updates and is therefore not a reliable source-side round-trip visual.
    from embodiment_renderers import render_unitree_g1
    source_dir = out / "source_motion_code"
    source_targets = [dict(gt[(idx, "humanoid")]) for idx in TARGETS]
    render_unitree_g1(source_targets, source_dir, force=True, fps=HZ)
    targets = []
    target_metrics = {}
    prediction_metrics = {}
    prediction_rows = []
    for idx in TARGETS:
        cell = dict(gt[(idx, "humanoid")])
        targets.append(cell)
        cue = cell["cue"]
        extract_path = out / "target_extract" / f"g{idx:02d}_{cue}.json"
        if extract_path.is_file():
            extracted = json.loads(extract_path.read_text(encoding="utf-8"))
        else:
            extracted = extract_cell(cell)
            extract_path.parent.mkdir(parents=True, exist_ok=True)
            extract_path.write_text(json.dumps(extracted, indent=2) + "\n", encoding="utf-8")
        metric_path = out / "metrics" / f"target_g{idx:02d}_{cue}.json"
        if metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            metric = render_and_measure(extracted, out / "target_roundtrip" / f"emb_g{idx:02d}_{cue}.gif")
            metric_path.parent.mkdir(parents=True, exist_ok=True)
            metric_path.write_text(json.dumps(metric, indent=2) + "\n", encoding="utf-8")
        target_metrics[cue] = metric
        prompt = build_prompt(cue, cell.get("description") or "", shots)
        prompt_path = out / "prompts" / f"g{idx:02d}_{cue}.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        if args.skip_inference:
            continue
        payload_path = out / "prediction_payloads" / f"g{idx:02d}_{cue}.json"
        if payload_path.is_file():
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        else:
            payload, raw = infer(prompt, cue, args.model)
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            raw_path = out / "raw_responses" / f"g{idx:02d}_{cue}.txt"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw + "\n", encoding="utf-8")
        pred_metric_path = out / "metrics" / f"prediction_g{idx:02d}_{cue}.json"
        if pred_metric_path.is_file():
            pred_metric = json.loads(pred_metric_path.read_text(encoding="utf-8"))
        else:
            pred_metric = render_and_measure(payload, out / "predictions" / f"emb_g{idx:02d}_{cue}.gif")
            pred_metric_path.parent.mkdir(parents=True, exist_ok=True)
            pred_metric_path.write_text(json.dumps(pred_metric, indent=2) + "\n", encoding="utf-8")
        prediction_metrics[cue] = pred_metric
        prediction_rows.append({"job_id": f"xyzrpy_posture.humanoid.g{idx:02d}", "platform": "humanoid", "idx": idx, "cue": cue, "model": args.model, "parse_ok": True, "payload": payload})

    if not args.skip_inference:
        expected = [(1, "nod_yes"), (2, "shake_head_no"), (29, "depict_small_object"), (30, "depict_flying_bird")]
        actual = [(row["idx"], row["cue"]) for row in prediction_rows]
        if actual != expected:
            raise RuntimeError(f"prediction job set mismatch: {actual} != {expected}")
        (out / "predictions_posture_cartesian.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows), encoding="utf-8")
    report = {"fewshot_roundtrip": fewshot_metrics, "target_roundtrip": target_metrics, "predictions": prediction_metrics, "inference_jobs": len(prediction_rows), "exact_job_set": [[1, "nod_yes"], [2, "shake_head_no"], [29, "depict_small_object"], [30, "depict_flying_bird"]]}
    (out / "roundtrip_error_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.skip_inference:
        print(f"complete: {make_gallery(out, targets, report)}")


if __name__ == "__main__":
    main()
