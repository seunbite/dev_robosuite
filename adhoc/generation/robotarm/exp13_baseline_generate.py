#!/usr/bin/env python3
"""
Exp13 baseline few-shot generation (one cue):
  1) Load few-shot xyz/joint artifacts (exp1 handmade shots).
  2) Build few-shot prompts (xyz and/or joint mode).
  3) Call Gemini to generate trajectories for a target cue NOT in few-shot.
  4) Render GIFs from generated trajectories + HTML preview.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

from google import genai

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
for p in (_REPO, _HERE, _REPO / "adhoc" / "generation" / "robotarm" / "legacy"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp13_baseline_render_one import (  # noqa: E402
    _write_gif,
    render_from_joints,
    render_from_xyz,
)
from legacy.motion_generation_core import MotionGenerator  # noqa: E402
from pilot90_paths import PROMPT_EXP_DIR, load_gt_by_cue  # noqa: E402

PROMPT_EXP13_XYZ = PROMPT_EXP_DIR / "prompt_exp13_xyz.txt"
PROMPT_EXP13_JOINT = PROMPT_EXP_DIR / "prompt_exp13_joint.txt"

FEWSHOT_CUES = {
    "Raise one hand briefly",
    "Bow slightly from the waist (Respect / Greeting)",
    "Bring one hand to forehead in salute position (Salute)",
    "Rotate finger near temple (Crazy)",
}


def _esc(x: object) -> str:
    return html.escape(str(x) if x is not None else "")


def _safe_slug(text: str, *, max_len: int = 80) -> str:
    s = re.sub(r"[^\w.-]+", "_", (text or "").strip()).strip("_")
    return s[:max_len] if s else "cue"


def _round_traj(vals: list[float], *, ndigits: int = 4) -> list[float]:
    return [round(float(v), ndigits) for v in vals]


def _extract_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("no JSON object in LLM response")
    raw = m.group(0)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([\]}])", r"\1", raw)
        obj = json.loads(fixed)
    if not isinstance(obj, dict):
        raise ValueError("parsed JSON is not an object")
    return obj


def _load_gt_description(cue: str) -> str:
    row = load_gt_by_cue().get(cue)
    if not row:
        return cue.replace("_", " ")
    desc = str(row.get("description") or "").strip()
    if desc:
        return desc
    pose = str(row.get("pose_gt") or "").strip()
    if pose:
        return pose
    return cue.replace("_", " ")


def load_fewshot_artifacts(fewshot_dir: Path) -> list[dict[str, Any]]:
    xyz_dir = fewshot_dir / "xyz"
    joint_dir = fewshot_dir / "joints"
    if not xyz_dir.is_dir() or not joint_dir.is_dir():
        raise FileNotFoundError(f"missing xyz/ or joints/ under {fewshot_dir}")

    artifacts: list[dict[str, Any]] = []
    for xyz_path in sorted(xyz_dir.glob("*.json")):
        xyz_data = json.loads(xyz_path.read_text(encoding="utf-8"))
        cue = str(xyz_data.get("cue") or "")
        stem = xyz_path.stem.replace("_xyz", "")
        joint_path = joint_dir / f"{stem}_joints.json"
        if not joint_path.is_file():
            raise FileNotFoundError(f"missing joint artifact for {xyz_path.name}")
        joint_data = json.loads(joint_path.read_text(encoding="utf-8"))
        artifacts.append(
            {
                "idx": int(xyz_data.get("idx", 0)),
                "cue": cue,
                "hz": int(xyz_data.get("hz", 8)),
                "xyz": xyz_data.get("xyz") or [],
                "joint_names": joint_data.get("joint_names") or [],
                "joints_rad": joint_data.get("joints_rad") or [],
                "gif": fewshot_dir / "gifs" / f"{stem}.gif",
            }
        )
    artifacts.sort(key=lambda a: int(a["idx"]))
    return artifacts


def _fewshot_xyz_block(artifacts: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Few-shot EE xyz trajectory examples:"]
    for art in artifacts:
        compact_xyz = [_round_traj(pt) for pt in art["xyz"]]
        payload = {
            "cue": art["cue"],
            "hz": art["hz"],
            "n_frames": len(compact_xyz),
            "xyz": compact_xyz,
        }
        lines.append(f"\n### Example: {art['cue']}")
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def _fewshot_joint_block(artifacts: list[dict[str, Any]]) -> str:
    lines: list[str] = ["Few-shot joint-angle trajectory examples (radians):"]
    for art in artifacts:
        compact = [_round_traj(row) for row in art["joints_rad"]]
        payload = {
            "cue": art["cue"],
            "hz": art["hz"],
            "joint_names": art["joint_names"],
            "n_frames": len(compact),
            "joints_rad": compact,
        }
        lines.append(f"\n### Example: {art['cue']}")
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)


def build_prompt(
    template_path: Path,
    *,
    mode: Literal["xyz", "joint"],
    artifacts: list[dict[str, Any]],
    cue: str,
    description: str,
) -> str:
    tpl = template_path.read_text(encoding="utf-8")
    fewshot = _fewshot_xyz_block(artifacts) if mode == "xyz" else _fewshot_joint_block(artifacts)
    joint_names = artifacts[0]["joint_names"] if artifacts else []
    out = tpl.replace("{{FEW_SHOT_EXAMPLES}}", fewshot)
    out = out.replace("{{CUE_NAME}}", cue)
    out = out.replace("{{CUE_DESCRIPTION}}", description)
    out = out.replace("{{JOINT_NAMES_JSON}}", json.dumps(joint_names, ensure_ascii=False))
    return out


def call_gemini(*, prompt: str, model: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    text = getattr(resp, "text", None) or ""
    if not text.strip():
        raise RuntimeError("empty Gemini response")
    return text


def _validate_xyz(obj: dict[str, Any], *, cue: str) -> dict[str, Any]:
    xyz = obj.get("xyz")
    if not isinstance(xyz, list) or not xyz:
        raise ValueError("xyz must be a non-empty list")
    clean: list[list[float]] = []
    for i, pt in enumerate(xyz):
        if not isinstance(pt, (list, tuple)) or len(pt) != 3:
            raise ValueError(f"xyz[{i}] must be [x,y,z]")
        clean.append([float(pt[0]), float(pt[1]), float(pt[2])])
    hz = int(obj.get("hz", 8))
    return {"cue": str(obj.get("cue") or cue), "hz": hz, "xyz": clean}


def _validate_joints(obj: dict[str, Any], *, cue: str, joint_names: list[str]) -> dict[str, Any]:
    names = obj.get("joint_names") or joint_names
    if list(names) != list(joint_names):
        names = joint_names
    rows = obj.get("joints_rad")
    if not isinstance(rows, list) or not rows:
        raise ValueError("joints_rad must be a non-empty list")
    n = len(joint_names)
    clean: list[list[float]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError(f"joints_rad[{i}] must have {n} values")
        clean.append([float(x) for x in row])
    hz = int(obj.get("hz", 8))
    return {"cue": str(obj.get("cue") or cue), "hz": hz, "joint_names": list(joint_names), "joints_rad": clean}


def write_preview_html(
    out_path: Path,
    *,
    cue: str,
    description: str,
    model: str,
    xyz_gif: Path | None,
    joint_gif: Path | None,
    fewshot_dir: Path,
) -> None:
    fewshot_preview = fewshot_dir / "fewshot_preview.html"
    cards: list[str] = []
    if xyz_gif and xyz_gif.is_file():
        cards.append(
            f'<div class="card"><h3>Generated — fewshot_xyz</h3>'
            f'<img src="{_esc(xyz_gif.name)}"/><p><code>{_esc(xyz_gif.with_suffix(".json").name)}</code></p></div>'
        )
    if joint_gif and joint_gif.is_file():
        cards.append(
            f'<div class="card"><h3>Generated — fewshot_joint</h3>'
            f'<img src="{_esc(joint_gif.name)}"/><p><code>{_esc(joint_gif.with_suffix(".json").name)}</code></p></div>'
        )

    out_path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'/>
<title>Exp13 generate — {_esc(cue)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px 48px;background:#fafafa;color:#111}}
h1{{font-size:22px}}
.row{{display:flex;gap:16px;flex-wrap:wrap}}
.card{{flex:1;min-width:320px;border:1px solid #ddd;border-radius:12px;background:#fff;padding:12px}}
img{{max-width:100%}}
code{{background:#f0f0f2;padding:2px 6px;border-radius:6px}}
.meta{{color:#555;margin:8px 0 14px}}
a{{color:#06c}}
</style></head><body>
<h1>Exp13 baseline few-shot generation</h1>
<div class="meta">
target cue: <code>{_esc(cue)}</code> (NOT in few-shot)<br/>
description: {_esc(description)}<br/>
model: <code>{_esc(model)}</code><br/>
few-shot artifacts: <a href="{_esc(fewshot_preview.resolve().as_uri() if fewshot_preview.is_file() else '#')}">fewshot_preview.html</a>
</div>
<div class="row">
{''.join(cards)}
</div>
</body></html>""",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp13: LLM few-shot xyz/joint generation for one cue.")
    ap.add_argument("--cue", type=str, default="raise_fists_victory")
    ap.add_argument("--model", type=str, default="gemini-2.5-pro")
    ap.add_argument("--hz", type=int, default=8)
    ap.add_argument("--mode", choices=("xyz", "joint", "both"), default="both")
    ap.add_argument(
        "--fewshot-dir",
        type=Path,
        default=Path("data/results/baseline_fewshot/exp13"),
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    cue = args.cue.strip()
    if cue in FEWSHOT_CUES:
        raise SystemExit(f"cue {cue!r} is in few-shot set; pick a pilot cue instead.")

    fewshot_dir = args.fewshot_dir
    artifacts = load_fewshot_artifacts(fewshot_dir)
    description = _load_gt_description(cue)
    joint_names: list[str] = list(artifacts[0]["joint_names"])

    slug = _safe_slug(cue)
    out_dir = args.out_dir or (fewshot_dir / f"generate_{slug}")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = MotionGenerator(
        robot_name="IIWA",
        capture_image_width=512,
        capture_image_height=512,
        has_renderer=False,
        has_offscreen_renderer=True,
        hz=int(args.hz),
    )

    xyz_gif: Path | None = None
    joint_gif: Path | None = None

    if args.mode in ("xyz", "both"):
        prompt = build_prompt(PROMPT_EXP13_XYZ, mode="xyz", artifacts=artifacts, cue=cue, description=description)
        (out_dir / "prompt_xyz.txt").write_text(prompt, encoding="utf-8")
        raw = call_gemini(prompt=prompt, model=args.model)
        (out_dir / "llm_response_xyz.txt").write_text(raw, encoding="utf-8")
        payload = _validate_xyz(_extract_json(raw), cue=cue)
        payload["hz"] = int(args.hz)
        xyz_path = out_dir / "xyz_pred.json"
        xyz_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        xyz_gif = out_dir / f"gif_{slug}_fewshot_xyz.gif"
        render_from_xyz(gen=gen, xyz=payload["xyz"], out_gif=xyz_gif, hz=int(args.hz))
        print(f"xyz: {len(payload['xyz'])} frames -> {xyz_gif}")

    if args.mode in ("joint", "both"):
        prompt = build_prompt(
            PROMPT_EXP13_JOINT,
            mode="joint",
            artifacts=artifacts,
            cue=cue,
            description=description,
        )
        (out_dir / "prompt_joint.txt").write_text(prompt, encoding="utf-8")
        raw = call_gemini(prompt=prompt, model=args.model)
        (out_dir / "llm_response_joint.txt").write_text(raw, encoding="utf-8")
        payload = _validate_joints(_extract_json(raw), cue=cue, joint_names=joint_names)
        payload["hz"] = int(args.hz)
        joints_path = out_dir / "joints_pred.json"
        joints_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        joint_gif = out_dir / f"gif_{slug}_fewshot_joint.gif"
        render_from_joints(gen=gen, joints=payload["joints_rad"], out_gif=joint_gif, hz=int(args.hz))
        print(f"joint: {len(payload['joints_rad'])} frames -> {joint_gif}")

    preview = out_dir / "preview.html"
    write_preview_html(
        preview,
        cue=cue,
        description=description,
        model=args.model,
        xyz_gif=xyz_gif,
        joint_gif=joint_gif,
        fewshot_dir=fewshot_dir,
    )
    print(f"Wrote preview: {preview}")
    if args.open and sys.platform == "darwin":
        subprocess.run(["open", str(preview)], check=False)


if __name__ == "__main__":
    main()
