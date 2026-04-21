import json
from copy import deepcopy
from html import escape
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTION_DIR = ROOT / "data" / "motions"
OUT_ROOT = SEED / "prompt19_sophisticated_path_trajectory_compare"
HTML_PATH = SEED / "prompt19_sophisticated_path_trajectory_compare_20260402_ko.html"
SCALE = 1.8
ROBOT = "IIWA"
HZ = 8

sys.path.insert(0, str(ROOT / "adhoc" / "test"))
from testset_utils import (  # noqa: E402
    build_samples,
    _estimate_step_frame_counts,
    _find_latest_single_gif,
    _load_cue_config_from_sample,
    _offset_overlapping_projected_path,
    _save_png,
    _step_ranges_from_trajectory_len,
)


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _has_path(cfg: dict) -> bool:
    return any(step.get("type") == "path" for step in cfg.get("movements", []))


def _path_shapes(cfg: dict) -> list[str]:
    return [
        step.get("parameters", {}).get("shape", "?")
        for step in cfg.get("movements", [])
        if step.get("type") == "path"
    ]


def _scale_path_magnitudes(cfg: dict, factor: float) -> dict:
    out = deepcopy(cfg)
    for step in out.get("movements", []):
        if step.get("type") != "path":
            continue
        params = step.get("parameters", {})
        shape = params.get("shape")
        if shape == "line":
            distance = params.get("distance")
            if isinstance(distance, (int, float)):
                params["distance"] = distance * factor
            elif isinstance(distance, dict):
                params["distance"] = {
                    axis: value * factor if isinstance(value, (int, float)) else value
                    for axis, value in distance.items()
                }
        elif shape == "arc":
            radius = params.get("radius")
            if isinstance(radius, (int, float)):
                params["radius"] = radius * factor
    return out


def _sample_row(base_sample: dict, config_path: Path) -> dict:
    row = dict(base_sample)
    row["config_path"] = str(config_path)
    return row


def _resolve_gif_path(testset: str, cue: str) -> str:
    subdir = "v19_sophisticated" if testset == "iconic" else "v19_sophisticated_contextual"
    return _find_latest_single_gif(MOTION_DIR / subdir / ROBOT, cue) or ""


def _fallback_sample(testset: str, cfg: dict, config_path: Path) -> dict:
    return {
        "sample_id": f"{testset}_c{int(cfg['idx'])}",
        "testset": testset,
        "cue_idx": int(cfg["idx"]),
        "cue": cfg["cue"],
        "gif_path": _resolve_gif_path(testset, cfg["cue"]),
        "config_path": str(config_path),
        "meta": {},
    }


def _build_path_only_png(sample: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return

    robotarm_dir = ROOT / "adhoc" / "robotarm"
    if str(robotarm_dir) not in sys.path:
        sys.path.insert(0, str(robotarm_dir))

    from motion_generation import MotionGenerator, _select_initial_poses
    from vlm_pose_benchmark import _project_3d, _simulate_trajectory

    cfg = _load_cue_config_from_sample(sample)
    first_pose_def = None
    for movement in cfg.get("movements", []):
        if movement.get("type") == "pose":
            first_pose_def = movement.get("parameters", {}).get("pose")
            break

    jsonl_path = ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    temp_output_root = (ROOT / "adhoc" / "test" / "_traj_tmp").resolve()
    temp_output_root.mkdir(parents=True, exist_ok=True)

    gen = None
    try:
        gen = MotionGenerator(
            robot_name=ROBOT,
            jsonl_path=str(jsonl_path),
            output_dir=str(temp_output_root),
            capture_image_width=512,
            capture_image_height=512,
            hz=HZ,
        )
        pose_id = sample.get("selected_pose_id") or sample.get("pose_index")
        if pose_id is None and first_pose_def is not None:
            matching = gen._find_matching_poses(first_pose_def)
            selected = _select_initial_poses(matching, first_pose_def, 1)
            if selected:
                pose_id = selected[0]["pose_id"]

        frames, trajectory, cam_pos, cam_rot, fovy = _simulate_trajectory(
            gen,
            sample["cue"],
            int(sample["cue_idx"]),
            str(sample["config_path"]),
            hz=HZ,
            pose_index=pose_id,
        )
        if not frames:
            raise RuntimeError(f"No frames captured for {sample['sample_id']}")

        frame = frames[len(frames) // 2]
        base = Image.fromarray(frame if isinstance(frame, np.ndarray) else np.array(frame)).convert("RGB")
        draw = ImageDraw.Draw(base)

        pts = []
        for t in trajectory:
            projected = _project_3d(t["pos"], cam_pos, cam_rot, fovy, base.size[0])
            if projected is not None:
                pts.append(projected)
        pts = _offset_overlapping_projected_path(pts)
        if len(pts) < 2:
            _save_png(base, str(output_path))
            return

        step_counts = _estimate_step_frame_counts(cfg, HZ)
        ranges = _step_ranges_from_trajectory_len(step_counts, len(pts))
        palette = [
            ((20, 107, 196), (14, 73, 141)),
            ((221, 95, 32), (172, 70, 20)),
            ((43, 131, 78), (27, 91, 53)),
            ((123, 53, 176), (92, 34, 133)),
        ]

        path_idx = 0
        for step, (start, end) in zip(cfg.get("movements", []), ranges):
            if step.get("type") != "path":
                continue
            seg = pts[start:end]
            if len(seg) < 2:
                continue
            line_color, end_color = palette[path_idx % len(palette)]
            for i in range(1, len(seg)):
                draw.line([seg[i - 1], seg[i]], fill=line_color, width=7)
            sx, sy = seg[0]
            ex, ey = seg[-1]
            draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=(255, 255, 255), outline=line_color, width=2)
            draw.ellipse([ex - 6, ey - 6, ex + 6, ey + 6], fill=end_color, outline=(255, 255, 255), width=2)
            path_idx += 1

        _save_png(base, str(output_path))
    finally:
        if gen is not None:
            try:
                gen.env.close()
            except Exception:
                pass


def _build_png(sample: dict, output_path: Path):
    _build_path_only_png(sample, output_path)


def _gif_rel_from_abs(path_str: str) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    return path.relative_to(ROOT).as_posix()


def _build_html(groups: list[tuple[str, list[dict]]]) -> str:
    sections = []
    for title, rows in groups:
        cards = []
        for row in rows:
            cards.append(
                f"""
<article class="card">
  <div class="card-header">
    <div class="title">c{row["idx"]} {escape(row["cue"])}</div>
    <div class="meta">paths: {escape(", ".join(row["path_shapes"]))} | line distance + arc radius x1.8 | path-only trajectory</div>
  </div>
  <div class="card-body">
    <div class="compare-grid">
      <div class="media-block">
        <div class="media-label">Original Path Only</div>
        <img src="{escape(row["orig_rel"])}" alt="original path trajectory c{row["idx"]}">
      </div>
      <div class="media-block">
        <div class="media-label">Scaled Path Only</div>
        <img src="{escape(row["scaled_rel"])}" alt="scaled path trajectory c{row["idx"]}">
      </div>
      <div class="media-block">
        <div class="media-label">Original GIF</div>
        <img src="{escape(row["gif_rel"])}" alt="original gif c{row["idx"]}">
      </div>
    </div>
  </div>
</article>
""".strip()
            )
        sections.append(
            f"""
<section class="section">
  <h2>{escape(title)} <span>{len(rows)} cues</span></h2>
  <div class="grid">
    {"".join(cards)}
  </div>
</section>
""".strip()
        )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sophisticated Path Trajectory Compare</title>
<style>
:root {{
  --bg:#ffffff;
  --surface:#fcfcfc;
  --border:#e3e7eb;
  --text:#1f2328;
  --muted:#59636e;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,'SF Pro Text','Segoe UI',sans-serif; }}
.wrap {{ max-width:1920px; margin:0 auto; padding:20px; }}
.hero {{ margin-bottom:18px; }}
.hero h1 {{ margin:0 0 8px; font-size:28px; }}
.hero p {{ margin:0; color:var(--muted); }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:12px; }}
.chip {{ padding:4px 9px; border:1px solid var(--border); border-radius:999px; font-size:13px; color:var(--muted); background:transparent; }}
.section {{ margin:0 0 18px; }}
.section h2 {{ margin:0 0 10px; font-size:18px; }}
.section h2 span {{ color:var(--muted); font-size:13px; font-weight:500; margin-left:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(860px, 1fr)); gap:12px; }}
.card {{ border:1px solid var(--border); border-radius:10px; background:var(--surface); overflow:hidden; }}
.card-header {{ padding:10px 12px; border-bottom:1px solid #edf0f2; background:#fcfcfc; }}
.title {{ font-weight:600; }}
.meta {{ margin-top:4px; color:var(--muted); font-size:12px; }}
.card-body {{ padding:12px; }}
.compare-grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; align-items:start; }}
.media-block {{ display:grid; gap:6px; }}
.media-label {{ font-size:11px; font-weight:700; color:var(--muted); letter-spacing:.04em; text-transform:uppercase; }}
img {{ width:100%; display:block; border:1px solid #eceff2; border-radius:6px; background:#fafafa; }}
@media (max-width: 1100px) {{
  .wrap {{ padding:14px; }}
  .grid {{ grid-template-columns:1fr; }}
  .compare-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>Sophisticated Path Trajectory Compare</h1>
    <p>Path가 들어간 sophisticated cue에 대해 현재 path trajectory, line distance + arc radius x1.8 path trajectory, 그리고 원본 GIF를 같이 비교합니다.</p>
    <div class="chips">
      <span class="chip">robot: {ROBOT}</span>
      <span class="chip">render: middle frame + path-only overlay</span>
      <span class="chip">right panel: original gif</span>
      <span class="chip">scaled rule: line distance + arc radius x{SCALE}</span>
      <span class="chip">iconic path cues: 30</span>
      <span class="chip">contextual path cues: 11</span>
    </div>
  </section>
  {"".join(sections)}
</div>
</body>
</html>
"""


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    orig_dir = OUT_ROOT / "orig_png"
    scaled_dir = OUT_ROOT / "scaled_png"
    scaled_cfg_dir = OUT_ROOT / "scaled_configs"
    for path in (orig_dir, scaled_dir, scaled_cfg_dir):
        path.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "title": "Sophisticated Iconic",
            "testset": "iconic",
            "config_path": SEED / "motion_configs_prompt_v19_sophisticated.json",
            "motion_subdir": "v19_sophisticated",
            "scaled_config_path": scaled_cfg_dir / "motion_configs_prompt_v19_sophisticated_scaled_distance_x1p8.json",
        },
        {
            "title": "Sophisticated Contextual",
            "testset": "contextual",
            "config_path": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
            "motion_subdir": "v19_sophisticated_contextual",
            "scaled_config_path": scaled_cfg_dir / "motion_configs_prompt_v19_sophisticated_contextual_scaled_distance_x1p8.json",
        },
    ]

    html_groups = []

    for spec in specs:
        configs = _load_json(spec["config_path"])
        path_cfgs = [cfg for cfg in configs if _has_path(cfg)]
        scaled_cfgs = [_scale_path_magnitudes(cfg, SCALE) for cfg in configs]
        _write_json(spec["scaled_config_path"], scaled_cfgs)

        samples = build_samples(
            testset=spec["testset"],
            robot=ROBOT,
            config_json=str(spec["config_path"]),
            iconic_motion_subdir=spec["motion_subdir"],
            contextual_motion_subdir=spec["motion_subdir"],
        )
        sample_by_idx = {int(sample["cue_idx"]): sample for sample in samples}

        rows = []
        for cfg in sorted(path_cfgs, key=lambda x: int(x["idx"])):
            idx = int(cfg["idx"])
            sample = sample_by_idx.get(idx)
            if not sample:
                sample = _fallback_sample(spec["testset"], cfg, spec["config_path"])
            if not sample.get("gif_path"):
                sample["gif_path"] = _resolve_gif_path(spec["testset"], cfg["cue"])

            stem = f'{spec["testset"]}_c{idx:02d}_{cfg["cue"]}'
            orig_png = orig_dir / f"{stem}__orig.png"
            scaled_png = scaled_dir / f"{stem}__scaled_distance_x1p8.png"

            _build_png(_sample_row(sample, spec["config_path"]), orig_png)
            _build_png(_sample_row(sample, spec["scaled_config_path"]), scaled_png)

            rows.append(
                {
                    "idx": idx,
                    "cue": cfg["cue"],
                    "path_shapes": _path_shapes(cfg),
                    "orig_rel": f'./prompt19_sophisticated_path_trajectory_compare/orig_png/{orig_png.name}',
                    "scaled_rel": f'./prompt19_sophisticated_path_trajectory_compare/scaled_png/{scaled_png.name}',
                    "gif_rel": f'../{_gif_rel_from_abs(sample["gif_path"])}' if sample.get("gif_path") else "",
                }
            )

        html_groups.append((spec["title"], rows))

    HTML_PATH.write_text(_build_html(html_groups), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
