import html
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from adhoc.robotarm.motion_generation import MotionGenerator

ROBOTS = ["IIWA", "Panda", "XArm7"]

TARGETS = [
    {"section": "iconic", "idx": 2, "cue": "handshake_offer"},
    {"section": "iconic", "idx": 50, "cue": "firm_accept_forward_reach"},
    {"section": "iconic", "idx": 52, "cue": "cancel_previous_offer_retract"},
    {"section": "contextual", "idx": 31, "cue": "start_action_forward_snap"},
    {"section": "contextual", "idx": 39, "cue": "commit_action_fast_reach"},
]

VARIANTS = [
    {
        "slug": "sophisticated",
        "label": "Sophisticated",
        "config_paths": {
            "iconic": ROOT / "data" / "seed" / "motion_configs_prompt_v19_sophisticated.json",
            "contextual": ROOT / "data" / "seed" / "motion_configs_prompt_v19_sophisticated_contextual.json",
        },
        "before_motion_roots": {
            "iconic": [ROOT / "data" / "motions" / "v19_sophisticated"],
            "contextual": [
                ROOT / "data" / "motions" / "v19_sophisticated_contextual_q4filled",
                ROOT / "data" / "motions" / "v19_sophisticated_contextual",
            ],
        },
        "after_motion_roots": {
            "iconic": ROOT / "data" / "motions" / "path_x_fix_probe" / "sophisticated_iconic",
            "contextual": ROOT / "data" / "motions" / "path_x_fix_probe" / "sophisticated_contextual",
        },
    },
    {
        "slug": "no_reasoning",
        "label": "No Reasoning",
        "config_paths": {
            "iconic": ROOT / "data" / "seed" / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_iconic.json",
            "contextual": ROOT / "data" / "seed" / "baseline_prompt19_full_no_reasoning" / "motion_configs_prompt_v19_sophisticated_no_reasoning_contextual.json",
        },
        "before_motion_roots": {
            "iconic": [ROOT / "data" / "motions" / "baseline_prompt19_full_no_reasoning" / "no_reasoning_iconic"],
            "contextual": [ROOT / "data" / "motions" / "baseline_prompt19_full_no_reasoning" / "no_reasoning_contextual"],
        },
        "after_motion_roots": {
            "iconic": ROOT / "data" / "motions" / "path_x_fix_probe" / "no_reasoning_iconic",
            "contextual": ROOT / "data" / "motions" / "path_x_fix_probe" / "no_reasoning_contextual",
        },
    },
]


def _safe_name(text: str) -> str:
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _rel(path: Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _find_target_row(rows: list[dict], idx: int, cue: str) -> dict:
    for row in rows:
        if int(row["idx"]) == int(idx) and row["cue"] == cue:
            return row
    raise KeyError(f"Missing cue idx={idx} cue={cue}")


def _line_x_steps(row: dict) -> list[str]:
    out = []
    for i, step in enumerate(row.get("movements", [])):
        if step.get("type") != "path":
            continue
        p = step.get("parameters", {})
        if p.get("shape") != "line":
            continue
        distance = p.get("distance")
        if p.get("axis") == "x":
            out.append(f"s{i}: x={distance}")
        elif isinstance(distance, dict) and "x" in distance:
            out.append(f"s{i}: x={distance['x']} in {distance}")
    return out


def _latest_single_gif(roots: list[Path], robot: str, cue: str) -> Path | None:
    safe_cue = _safe_name(cue)
    matches: list[Path] = []
    for root in roots:
        robot_dir = root / robot
        if robot_dir.exists():
            matches.extend(robot_dir.glob(f"*_{safe_cue}_p*.gif"))
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _latest_single_gif_one_root(root: Path, robot: str, cue: str) -> Path | None:
    robot_dir = root / robot
    if not robot_dir.exists():
        return None
    safe_cue = _safe_name(cue)
    matches = sorted(robot_dir.glob(f"*_{safe_cue}_p*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _clear_after_dirs() -> None:
    base = ROOT / "data" / "motions" / "path_x_fix_probe"
    if base.exists():
        shutil.rmtree(base)


def _render_targets() -> None:
    config_cache = {}
    for variant in VARIANTS:
        for section, config_path in variant["config_paths"].items():
            config_cache[(variant["slug"], section)] = _load_rows(config_path)

    for robot in ROBOTS:
        for variant in VARIANTS:
            for target in TARGETS:
                section = target["section"]
                rows = config_cache[(variant["slug"], section)]
                row = _find_target_row(rows, target["idx"], target["cue"])
                out_root = variant["after_motion_roots"][section]
                out_root.mkdir(parents=True, exist_ok=True)
                gen = MotionGenerator(
                    robot_name=robot,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    hz=8,
                    output_dir=str(out_root),
                )
                print(f"[render] robot={robot} variant={variant['slug']} cue={row['cue']} idx={row['idx']}", flush=True)
                gen.execute_cue(
                    cue=row["cue"],
                    config_path=str(variant["config_paths"][section]),
                    cue_idx=int(row["idx"]),
                    hz=8,
                    save_gif=True,
                )


def _build_html() -> Path:
    out_dir = ROOT / "data" / "seed" / "path_x_fix_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompt19_path_x_fix_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}_ko.html"

    config_cache = {
        (variant["slug"], section): _load_rows(config_path)
        for variant in VARIANTS
        for section, config_path in variant["config_paths"].items()
    }

    parts = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Prompt19 Path X Fix Probe</title>",
        "<style>",
        ":root{--bg:#ffffff;--panel:#ffffff;--ink:#171717;--muted:#666;--line:#dddddd;--soft:#f7f7f7;--accent:#1d4ed8;--before:#a16207;--after:#047857;}",
        "*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}",
        ".wrap{max-width:2100px;margin:0 auto;padding:20px 20px 60px}",
        ".hero{border:1px solid var(--line);border-radius:18px;padding:18px 20px;margin-bottom:18px;background:var(--panel)}",
        ".hero h1{margin:0 0 8px;font-size:28px}.hero p{margin:0;color:var(--muted);line-height:1.5}",
        ".card{border:1px solid var(--line);border-radius:18px;margin:18px 0;background:var(--panel);overflow:hidden}",
        ".card-hd{padding:16px 18px;border-bottom:1px solid var(--line)}",
        ".eyebrow{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:6px}",
        ".title{font-size:24px;font-weight:750;margin:0 0 8px}",
        ".meta{color:var(--muted);font-size:14px;line-height:1.5}",
        ".variant{padding:14px 18px 18px;border-top:1px solid var(--line)}",
        ".variant:first-of-type{border-top:none}",
        ".variant h3{margin:0 0 10px;font-size:18px}",
        ".grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}",
        ".tile{border:1px solid var(--line);border-radius:14px;padding:10px;background:var(--soft)}",
        ".tile .lab{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}",
        ".before{color:var(--before)} .after{color:var(--after)}",
        ".media{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff;aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}",
        ".media img{display:block;width:100%;height:100%;object-fit:contain;background:#fff}",
        ".missing{color:var(--muted);font-size:13px;font-weight:600;text-align:center;padding:14px}",
        ".submeta{margin-top:8px;font-size:12px;color:var(--muted);line-height:1.4;word-break:break-word}",
        ".steps{margin-top:8px;font-size:12px;color:var(--muted)}",
        ".steps code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}",
        "@media (max-width:1700px){.grid{grid-template-columns:repeat(3,minmax(0,1fr))}}",
        "</style>",
        "</head>",
        "<body><div class='wrap'>",
        "<section class='hero'>",
        "<h1>Path X Sign Fix Probe</h1>",
        "<p>We changed only the path-axis sign handling so authored <code>x</code> values follow the same visible convention as movement. This page keeps existing renders as <strong>Before</strong> and shows newly rendered <strong>After</strong> results for 5 representative x-line cues across IIWA, Panda, and XArm7.</p>",
        "</section>",
    ]

    for target in TARGETS:
        parts.append("<article class='card'>")
        parts.append("<div class='card-hd'>")
        parts.append(f"<div class='eyebrow'>{_esc(target['section'])} cue {target['idx']}</div>")
        parts.append(f"<h2 class='title'>{_esc(target['cue'])}</h2>")
        parts.append("</div>")
        for variant in VARIANTS:
            row = _find_target_row(config_cache[(variant["slug"], target["section"])], target["idx"], target["cue"])
            step_text = ", ".join(_line_x_steps(row)) or "no x-line steps found"
            parts.append("<section class='variant'>")
            parts.append(f"<h3>{_esc(variant['label'])}</h3>")
            parts.append(f"<div class='steps'><code>{_esc(step_text)}</code></div>")
            parts.append("<div class='grid'>")
            for phase in ("before", "after"):
                for robot in ROBOTS:
                    if phase == "before":
                        gif_path = _latest_single_gif(variant["before_motion_roots"][target["section"]], robot, target["cue"])
                    else:
                        gif_path = _latest_single_gif_one_root(variant["after_motion_roots"][target["section"]], robot, target["cue"])
                    parts.append("<div class='tile'>")
                    parts.append(f"<div class='lab {phase}'>{_esc(phase)} · {_esc(robot)}</div>")
                    if gif_path is None:
                        parts.append("<div class='media'><div class='missing'>No image</div></div>")
                    else:
                        rel = _rel(gif_path, out_path.parent)
                        parts.append(f"<div class='media'><img src='{_esc(rel)}' alt='{_esc(target['cue'])} {_esc(robot)} {_esc(phase)}'></div>")
                        parts.append(f"<div class='submeta'><a href='{_esc(rel)}'>open gif</a></div>")
                    parts.append("</div>")
            parts.append("</div>")
            parts.append("</section>")
        parts.append("</article>")

    parts.append("</div></body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def main() -> None:
    _clear_after_dirs()
    _render_targets()
    out_path = _build_html()
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "targets": TARGETS,
        "robots": ROBOTS,
        "variants": [variant["slug"] for variant in VARIANTS],
        "html": str(out_path),
    }
    manifest_path = out_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
