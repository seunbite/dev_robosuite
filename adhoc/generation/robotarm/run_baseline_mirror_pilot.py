#!/usr/bin/env python3
"""Five-cue bilateral mirror pilot for humanoid and Tiago trajectory baselines.

The LLM predicts only the primary (right) arm/hand plus the non-arm body parts
and a binary ``left_arm_mirror`` switch.  Post-processing restores the omitted
left arm by exact robot-specific reflection before replay/render.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import run_embodiment_baseline_traj as base  # noqa: E402
from trajectory_parts import compact_links_for_prompt, compact_parts_for_prompt  # noqa: E402

PILOT = (18, 19, 22, 26, 28)
ROBOTS = ("humanoid", "tiago")
MODES = ("joint", "xyzrpy")
G1_SIGNS = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)


def _mirror_rpy(rpy: list[float]) -> list[float]:
    """Reflect an orientation across the world x-z (sagittal) plane."""
    from scipy.spatial.transform import Rotation

    reflect = np.diag([1.0, -1.0, 1.0])
    rot = Rotation.from_euler("xyz", rpy).as_matrix()
    mirrored = reflect @ rot @ reflect
    return Rotation.from_matrix(mirrored).as_euler("xyz").tolist()


def expand_mirror(payload: dict[str, Any], robot: str, mode: str) -> dict[str, Any]:
    """Restore the left arm/hand omitted from the model output."""
    mirror = int(payload.pop("left_arm_mirror", 0)) == 1
    if mode == "joint":
        parts = payload["parts"]
        right = parts["right_arm"]
        rows = [[float(v) for v in row] for row in right["joints_rad"]]
        width = 7 if robot == "humanoid" else 6
        if any(len(row) != width for row in rows):
            raise ValueError(f"{robot} right_arm row width must be {width}")
        if mirror:
            signs = G1_SIGNS if robot == "humanoid" else (1.0,) * 6
            left_rows = [[signs[i] * row[i] for i in range(width)] for row in rows]
        else:
            rest = base.REST_JOINTS.get(robot, {}).get("left_arm")
            if rest is None:
                # G1 home: symmetric relaxed arms.
                rest = [0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0]
            left_rows = [list(rest) for _ in rows]
        parts["left_arm"] = {
            "joint_names": base.PART_JOINTS[robot]["left_arm"],
            "joints_rad": left_rows,
        }
        names, values = base.flatten_joint_parts(parts, robot)
        return {
            "cue": payload["cue"], "hz": int(payload.get("hz", base.HZ)),
            "left_arm_mirror": int(mirror), "parts": parts,
            "joint_names": names, "joints_rad": values,
        }

    links = payload["links"]
    right_rows = [[float(v) for v in row] for row in links["right_hand"]["xyzrpy"]]
    if any(len(row) != 6 for row in right_rows):
        raise ValueError("right_hand xyzrpy rows must have width 6")
    if mirror:
        left_rows = [
            [row[0], -row[1], row[2], *_mirror_rpy(row[3:6])]
            for row in right_rows
        ]
    else:
        # Keep a stable left-hand pose from the first few-shot example.
        left_rows = [list(right_rows[0]) for _ in right_rows]
        for row in left_rows:
            row[1] *= -1.0
            row[3:6] = _mirror_rpy(row[3:6])
    links["left_hand"] = {"xyzrpy": left_rows}
    names, values = base.flatten_link_parts(links, robot)
    return {
        "cue": payload["cue"], "hz": int(payload.get("hz", base.HZ)),
        "left_arm_mirror": int(mirror), "links": links,
        "link_names": names, "xyzrpy": values,
    }


def build_prompt(robot: str, mode: str, target: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    examples: list[str] = []
    if mode == "joint":
        keep = [p for p in base.part_order(robot) if p != "left_arm"]
        for shot in shots:
            shot = base._enrich_art_parts(robot, dict(shot))
            parts = compact_parts_for_prompt({p: shot["parts"][p] for p in keep})
            examples.append(json.dumps({"cue": shot["cue"], "hz": shot.get("hz", 8), "parts": parts}, ensure_ascii=False))
        schema = {
            p: {"joint_names": base.PART_JOINTS[robot][p], "joints_rad": [[0.0] * len(base.PART_JOINTS[robot][p])]}
            for p in keep
        }
        representation = "per-part joint angles in radians"
        body = f'"parts": {json.dumps(schema, ensure_ascii=False)}'
        rule = "Every part must have the same 16-32 frames. Keep joint_names exact."
    else:
        keep = [p for p in base.link_order(robot) if p != "left_hand"]
        for shot in shots:
            shot = base._enrich_art_parts(robot, dict(shot))
            links = compact_links_for_prompt({p: shot["links"][p] for p in keep})
            examples.append(json.dumps({"cue": shot["cue"], "hz": shot.get("hz", 8), "links": links}, ensure_ascii=False))
        schema = {p: {"xyzrpy": [[0.0] * 6]} for p in keep}
        representation = "per-link world-frame xyzrpy"
        body = f'"links": {json.dumps(schema, ensure_ascii=False)}'
        rule = "Every link must have the same 16-32 frames; each xyzrpy row has exactly 6 floats."
    return f"""Generate a {representation} trajectory for robot `{robot}`.
To reduce output dimensionality, NEVER output left_arm or left_hand. Output only the primary right side.
Set left_arm_mirror=1 when the cue is inherently bilateral and the left arm should be an exact sagittal-plane mirror of the right; otherwise 0.
The renderer deterministically reconstructs the omitted left side after inference.

Examples:
{chr(10).join(examples)}

Target cue: {target['cue']}
Description: {target.get('description', '')}

Return JSON only:
{{"cue":"{target['cue']}","hz":8,"left_arm_mirror":1,{body}}}

Rules: {rule} Numeric arrays only. Smooth motion. Do not output markdown or commentary."""


def validate_reduced(obj: dict[str, Any], robot: str, mode: str, cue: str) -> dict[str, Any]:
    if int(obj.get("left_arm_mirror", -1)) not in (0, 1):
        raise ValueError("left_arm_mirror must be 0 or 1")
    obj["cue"] = cue
    obj["hz"] = int(obj.get("hz", 8))
    if mode == "joint":
        parts = obj.get("parts")
        expected = [p for p in base.part_order(robot) if p != "left_arm"]
        if not isinstance(parts, dict) or any(p not in parts for p in expected) or "left_arm" in parts:
            raise ValueError(f"parts must be exactly reduced schema {expected}")
        # Reuse the strict flattener before mirror expansion.
        for p in expected:
            rows = parts[p].get("joints_rad")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"empty parts.{p}.joints_rad")
    else:
        links = obj.get("links")
        expected = [p for p in base.link_order(robot) if p != "left_hand"]
        if not isinstance(links, dict) or any(p not in links for p in expected) or "left_hand" in links:
            raise ValueError(f"links must be exactly reduced schema {expected}")
        for p in expected:
            rows = links[p].get("xyzrpy")
            if not isinstance(rows, list) or not rows or any(len(x) != 6 for x in rows):
                raise ValueError(f"bad links.{p}.xyzrpy")
    return obj


def generate_one(client: Any, robot: str, mode: str, idx: int, cell: dict[str, Any], shots: list[dict[str, Any]], model: str) -> dict[str, Any]:
    from google.genai import types

    prompt = build_prompt(robot, mode, cell, shots)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3, max_output_tokens=16384,
            thinking_config=types.ThinkingConfig(thinking_budget=128 if "pro" in model else 0),
        ),
    )
    raw = (response.text or "").strip()
    reduced = validate_reduced(base._extract_json(raw), robot, mode, cell["cue"])
    payload = expand_mirror(json.loads(json.dumps(reduced)), robot, mode)
    return {
        "job_id": f"mirror.{mode}.{robot}.g{idx:02d}", "mode": mode,
        "platform": robot, "idx": idx, "cue": cell["cue"],
        "description": cell.get("description", ""), "model": model,
        "parse_ok": True, "reduced_payload": reduced, "payload": payload,
        "raw_output": raw,
    }


def write_gallery(out: Path, rows: list[dict[str, Any]]) -> Path:
    original = REPO / "data/results/baseline_traj/ours5"
    idx = {(r["platform"], r["idx"], r["mode"]): r for r in rows if r.get("parse_ok")}
    cues = json.loads(base.CUES_JSON.read_text())
    cue_map = {int(x["idx"]): x["cue"] for x in cues}
    trs = []
    for gi in PILOT:
        for robot in ROBOTS:
            cells = []
            for mode in MODES:
                old_pat = original / f"render_{mode}" / robot / f"emb_g{gi:02d}_{cue_map[gi]}_gemini-2.5-pro.gif"
                old_rel = os.path.relpath(old_pat, out)
                new = out / f"render_{mode}" / robot / f"emb_g{gi:02d}_{cue_map[gi]}_mirror_gemini-2.5-pro.gif"
                new_rel = os.path.relpath(new, out)
                cells.append(f'<td><div>{mode} · original</div><img src="{html.escape(old_rel)}"></td>')
                flag = idx.get((robot, gi, mode), {}).get("payload", {}).get("left_arm_mirror", "?")
                cells.append(f'<td><div>{mode} · mirror <b>flag={flag}</b></div><img src="{html.escape(new_rel)}"></td>')
            trs.append(f'<tr><th>{robot}<br>g{gi:02d} {cue_map[gi]}</th>{"".join(cells)}</tr>')
    page = f"""<!doctype html><meta charset=utf-8><title>Baseline mirror pilot</title>
<style>body{{font-family:system-ui;margin:20px;background:#111;color:#eee}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #444;padding:8px;vertical-align:top}}th{{width:13%;text-align:left}}img{{display:block;width:100%;max-width:280px;background:#000;margin-top:5px}}div{{font-size:13px;color:#bbb}}b{{color:#8fda9d}}</style>
<h2>Baseline–Mirror Pilot</h2><p>LLM output omits the left arm/hand. <code>left_arm_mirror=1</code> reconstructs it deterministically.</p>
<table><thead><tr><th>Robot / cue</th><th>Joint original</th><th>Joint mirror</th><th>xyzrpy original</th><th>xyzrpy mirror</th></tr></thead><tbody>{''.join(trs)}</tbody></table>"""
    path = out / "gallery_baseline_mirror_pilot.html"
    path.write_text(page)
    return path


def seed_from_existing(out: Path) -> list[dict[str, Any]]:
    """Post-hoc decoder sanity check from the latest valid full-body Pro rows."""
    source = REPO / "data/results/baseline_traj/ours5"
    chosen: dict[tuple[str, str, int], dict[str, Any]] = {}
    for mode in MODES:
        path = source / f"predictions_{mode}.jsonl"
        for line in path.read_text().splitlines():
            row = json.loads(line)
            robot, gi = row.get("platform"), int(row.get("idx", 0))
            payload = row.get("payload") or {}
            structured = "parts" in payload if mode == "joint" else "links" in payload
            if robot in ROBOTS and gi in PILOT and row.get("parse_ok") and structured and row.get("model") == "gemini-2.5-pro":
                chosen[(robot, mode, gi)] = row
    rows: list[dict[str, Any]] = []
    for (robot, mode, gi), old in sorted(chosen.items()):
        payload = json.loads(json.dumps(old["payload"]))
        if mode == "joint":
            payload["parts"].pop("left_arm", None)
        else:
            payload["links"].pop("left_hand", None)
        payload.pop("joint_names", None)
        payload.pop("joints_rad", None)
        payload.pop("link_names", None)
        payload.pop("xyzrpy", None)
        payload["left_arm_mirror"] = 1
        full = expand_mirror(json.loads(json.dumps(payload)), robot, mode)
        rows.append({
            "job_id": f"mirror.{mode}.{robot}.g{gi:02d}", "mode": mode,
            "platform": robot, "idx": gi, "cue": old["cue"],
            "description": old.get("description", ""), "model": "gemini-2.5-pro",
            "parse_ok": True, "source": "posthoc_existing_pro_right_side_only",
            "reduced_payload": payload, "payload": full,
        })
    pred = out / "predictions_mirror_posthoc.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
    print(f"[seed-existing] {len(rows)}/20 rows -> {pred}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=REPO / "data/results/baseline_traj/mirror_pilot_5")
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--seed-existing", action="store_true", help="Decode latest Pro right side with mirror; no API call")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "predictions_mirror.jsonl"
    gt = base._index_gt()
    rows = seed_from_existing(out) if args.seed_existing else ([json.loads(x) for x in pred.read_text().splitlines() if x.strip()] if pred.exists() else [])
    done = {r["job_id"] for r in rows if r.get("parse_ok")}
    if not args.render_only and not args.seed_existing:
        client = base._client()
        jobs = []
        for robot in ROBOTS:
            shots = base._fewshot_artifacts(REPO / "data/results/baseline_traj/ours5", robot, gt, shots_path=base.DEFAULT_FEWSHOT_SHOTS)
            for mode in MODES:
                for gi in PILOT:
                    jid = f"mirror.{mode}.{robot}.g{gi:02d}"
                    if jid not in done:
                        jobs.append((robot, mode, gi, gt[(gi, robot)], shots))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_one, client, *job, args.model): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    with pred.open("a") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.append(row)
                    print(f"[infer] {row['job_id']} parse=PASS mirror={row['payload']['left_arm_mirror']}", flush=True)
                except Exception as exc:
                    print(f"[infer] {job[1]}.{job[0]}.g{job[2]:02d} FAIL {exc}", flush=True)
    # Render every valid result; deterministic and safe to overwrite.
    for row in rows:
        if not row.get("parse_ok"):
            continue
        gif = out / f"render_{row['mode']}" / row["platform"] / f"emb_g{row['idx']:02d}_{row['cue']}_mirror_{base._model_slug(row['model'])}.gif"
        print(f"[render] {row['job_id']} -> {gif}", flush=True)
        base.RENDERERS[row["platform"]](row["payload"], row["mode"], gif, {})
    print(write_gallery(out, rows), flush=True)


if __name__ == "__main__":
    main()
