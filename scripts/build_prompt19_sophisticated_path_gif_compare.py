import json
import os
import shutil
import sys
from datetime import datetime
from html import escape
from pathlib import Path


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
SEED = ROOT / "data" / "seed"
MOTION_DIR = ROOT / "data" / "motions"
OUT_ROOT = ROOT / "data" / "prompt19_sophisticated_path_gif_compare"
RENDER_ROOT = MOTION_DIR / "prompt19_sophisticated_path_gif_compare"
HTML_PATH = OUT_ROOT / "prompt19_sophisticated_path_gif_compare_20260402_ko.html"
ROBOT = "IIWA"
HZ = 8
SCALE = 1.8

sys.path.insert(0, str(ROOT / "adhoc" / "robotarm"))
sys.path.insert(0, str(ROOT / "adhoc" / "test"))

from motion_generation import MotionGenerator, _select_initial_poses  # noqa: E402
from testset_utils import _find_contextual_gif, _find_iconic_gif, _find_latest_single_gif  # noqa: E402


SELECTIONS = {
    "line": [
        ("iconic", 2, "handshake_offer"),
        ("iconic", 20, "rub_eye_tired"),
        ("iconic", 50, "firm_accept_forward_reach"),
        ("contextual", 14, "curl_fingers_give_me"),
        ("contextual", 39, "commit_action_fast_reach"),
    ],
    "arc": [
        ("iconic", 1, "raising_hand_greeting"),
        ("iconic", 5, "big_heart_above_head"),
        ("iconic", 22, "circle_temple_crazy"),
        ("iconic", 31, "draw_circle_repeat"),
        ("contextual", 3, "self_hug"),
        ("contextual", 43, "shame_shy_substitute"),
    ],
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _first_pose_def(row: dict):
    for movement in row.get("movements", []):
        if movement.get("type") == "pose":
            return movement.get("parameters", {}).get("pose")
    return None


def _scale_path_magnitudes(cfg: dict, factor: float) -> dict:
    out = json.loads(json.dumps(cfg))
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


def _spec_for_testset(testset: str) -> dict:
    if testset == "iconic":
        return {
            "config_path": SEED / "motion_configs_prompt_v19_sophisticated.json",
            "motion_subdir": "v19_sophisticated",
            "finder": lambda idx: _find_iconic_gif(ROBOT, idx, motion_subdir="v19_sophisticated"),
            "single_finder": lambda cue: _find_latest_single_gif(MOTION_DIR / "v19_sophisticated" / ROBOT, cue),
        }
    return {
        "config_path": SEED / "motion_configs_prompt_v19_sophisticated_contextual.json",
        "motion_subdir": "v19_sophisticated_contextual",
        "finder": lambda idx: _find_contextual_gif(ROBOT, idx, motion_subdir="v19_sophisticated_contextual"),
        "single_finder": lambda cue: _find_latest_single_gif(MOTION_DIR / "v19_sophisticated_contextual" / ROBOT, cue),
    }


def _find_row(configs: list[dict], cue_idx: int) -> dict:
    for row in configs:
        if int(row["idx"]) == int(cue_idx):
            return row
    raise KeyError(f"cue idx not found: {cue_idx}")


def _latest_new_gif(out_dir: Path, before_names: set[str], cue: str) -> Path | None:
    created = sorted(
        [p for p in out_dir.glob("*.gif") if p.name not in before_names],
        key=lambda p: p.stat().st_mtime,
    )
    if created:
        return created[-1]
    safe_cue = _safe_name(cue)
    matches = sorted(out_dir.glob(f"*_{safe_cue}_p*.gif"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _render_selected_gifs() -> list[dict]:
    jsonl_path = ROOT / "data" / "seed" / "closest_poses_results.jsonl"
    scaled_cfg_dir = OUT_ROOT / "scaled_configs"
    rendered_dir = RENDER_ROOT / "rendered_gifs"
    scaled_cfg_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir.mkdir(parents=True, exist_ok=True)

    selected_rows: dict[str, list[dict]] = {"iconic": [], "contextual": []}
    for testset in ("iconic", "contextual"):
        spec = _spec_for_testset(testset)
        configs = _load_json(spec["config_path"])
        all_idxs = {idx for items in SELECTIONS.values() for ts, idx, _ in items if ts == testset}
        rows = [_find_row(configs, idx) for idx in sorted(all_idxs)]
        scaled_rows = [_scale_path_magnitudes(row, SCALE) for row in rows]
        scaled_path = scaled_cfg_dir / f"{testset}_scaled_x1p8.json"
        _write_json(scaled_path, scaled_rows)
        selected_rows[testset] = [{"row": row, "scaled_config_path": scaled_path} for row in rows]

    generator = MotionGenerator(
        robot_name=ROBOT,
        jsonl_path=str(jsonl_path),
        output_dir=str(rendered_dir),
        has_renderer=False,
        has_offscreen_renderer=True,
    )

    results = []
    try:
        for bucket_name, items in SELECTIONS.items():
            for testset, cue_idx, cue in items:
                row_info = next(item for item in selected_rows[testset] if int(item["row"]["idx"]) == cue_idx)
                row = row_info["row"]
                pose_def = _first_pose_def(row)
                if pose_def is None:
                    raise ValueError(f"no pose movement for c{cue_idx} {cue}")
                matching = generator._find_matching_poses(pose_def)
                selected = _select_initial_poses(matching, pose_def, top_k=1)
                if not selected:
                    raise ValueError(f"no matching pose for c{cue_idx} {cue}")

                testset_out_dir = rendered_dir / testset
                testset_out_dir.mkdir(parents=True, exist_ok=True)
                generator.output_dir = str(testset_out_dir)
                before_names = {p.name for p in testset_out_dir.glob("*.gif")}
                generator._set_joint_positions(generator.initial_joint_pos)
                generator.execute_cue(
                    cue=cue,
                    pose_index=selected[0]["pose_id"],
                    config_path=str(row_info["scaled_config_path"]),
                    hz=HZ,
                    cue_idx=cue_idx,
                    save_gif=True,
                )
                latest = _latest_new_gif(testset_out_dir, before_names, cue)
                if latest is None:
                    raise FileNotFoundError(f"rendered gif not found for c{cue_idx} {cue}")

                tiled_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ROBOT}_{_safe_name(cue)}_c{cue_idx}_after_scaled_tiled.gif"
                tiled_path = testset_out_dir / tiled_name
                shutil.copy2(latest, tiled_path)

                original = _spec_for_testset(testset)["single_finder"](cue)
                if not original:
                    original = _spec_for_testset(testset)["finder"](cue_idx)
                if not original:
                    raise FileNotFoundError(f"original gif not found for c{cue_idx} {cue}")

                results.append(
                    {
                        "bucket": bucket_name,
                        "testset": testset,
                        "idx": cue_idx,
                        "cue": cue,
                        "before_abs": original,
                        "after_abs": str(tiled_path),
                    }
                )
    finally:
        generator.close()

    return results


def _rel(path_str: str) -> str:
    return os.path.relpath(path_str, HTML_PATH.parent)


def _build_html(rows: list[dict]) -> str:
    sections = []
    for bucket in ("line", "arc"):
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        cards = []
        for row in bucket_rows:
            cards.append(
                f"""
<article class="card">
  <div class="card-header">
    <div class="title">{bucket.upper()} · c{row["idx"]} {escape(row["cue"])}</div>
    <div class="meta">{escape(row["testset"])}</div>
  </div>
  <div class="card-body">
    <div class="compare-grid">
      <div class="media-block">
        <div class="media-label">Before GIF Top 1</div>
        <img src="{escape(_rel(row["before_abs"]))}" alt="before c{row["idx"]}">
      </div>
      <div class="media-block">
        <div class="media-label">After GIF Top 1</div>
        <img src="{escape(_rel(row["after_abs"]))}" alt="after c{row["idx"]}">
      </div>
    </div>
  </div>
</article>
""".strip()
            )
        sections.append(
            f"""
<section class="section">
  <h2>{bucket.title()} <span>{len(bucket_rows)} cues</span></h2>
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
<title>Sophisticated Path GIF Before After</title>
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
.wrap {{ max-width:1600px; margin:0 auto; padding:20px; }}
.hero {{ margin-bottom:18px; }}
.hero h1 {{ margin:0 0 8px; font-size:28px; }}
.hero p {{ margin:0; color:var(--muted); }}
.chips {{ display:flex; gap:6px; flex-wrap:wrap; margin-top:12px; }}
.chip {{ padding:4px 9px; border:1px solid var(--border); border-radius:999px; font-size:13px; color:var(--muted); }}
.section {{ margin:0 0 18px; }}
.section h2 {{ margin:0 0 10px; font-size:18px; }}
.section h2 span {{ color:var(--muted); font-size:13px; font-weight:500; margin-left:6px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(620px, 1fr)); gap:12px; }}
.card {{ border:1px solid var(--border); border-radius:10px; background:var(--surface); overflow:hidden; }}
.card-header {{ padding:10px 12px; border-bottom:1px solid #edf0f2; }}
.title {{ font-weight:600; }}
.meta {{ margin-top:4px; color:var(--muted); font-size:12px; text-transform:capitalize; }}
.card-body {{ padding:12px; }}
.compare-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
.media-block {{ display:grid; gap:6px; }}
.media-label {{ font-size:11px; font-weight:700; color:var(--muted); letter-spacing:.04em; text-transform:uppercase; }}
img {{ width:100%; display:block; border:1px solid #eceff2; border-radius:6px; background:#fafafa; }}
@media (max-width: 960px) {{
  .wrap {{ padding:14px; }}
  .grid {{ grid-template-columns:1fr; }}
  .compare-grid {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <h1>Sophisticated Path GIF Before / After</h1>
    <p>line 5개, arc 5개를 골라 기존 GIF와 path magnitude 조정 후 GIF를 나란히 비교합니다.</p>
    <div class="chips">
      <span class="chip">robot: {ROBOT}</span>
      <span class="chip">selection: line 5 + arc 5</span>
      <span class="chip">after rule: line distance + arc radius x{SCALE}</span>
    </div>
  </section>
  {"".join(sections)}
</div>
</body>
</html>
"""


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = _render_selected_gifs()
    HTML_PATH.write_text(_build_html(rows), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
