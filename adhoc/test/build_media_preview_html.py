import os
from datetime import datetime
from pathlib import Path

import fire

from testset_utils import build_binary_tasks, build_samples, prepare_normalized_eval_media, prepare_test_media


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _media_html(media_path: str, media_type: str, html_dir: Path) -> str:
    rel = os.path.relpath(media_path, html_dir)
    if media_type == "mp4":
        return f'<video src="{_esc(rel)}" controls muted playsinline preload="metadata"></video>'
    if media_type == "gif":
        return f'<img src="{_esc(rel)}" alt="gif">'
    return f'<img src="{_esc(rel)}" alt="{_esc(media_type)}">'


def main(
    testset: str = "iconic",
    robot: str = "IIWA",
    prompt_version: int = 18,
    persona_output_dir: str = "data/seed/persona_tag_dataset_v1",
    limit: int | None = None,
    output_dir: str = "adhoc/test/results",
    normalize_media: bool = True,
    render_hz: int = 8,
    render_speed_scale: float = 1.0,
    render_hold_scale: float = 1.0,
    render_top_k: int = 1,
    variants: str = "first_frame_only,gif,alpha_stack_final,alpha_stack_final_arrow,middle_frame_only,middle_frame_trajectory,mp4",
    force_regen: bool = False,
    task_family: str = "binary_match",
    seed: int = 42,
):
    if limit is not None and int(limit) <= 0:
        limit = None
    samples = build_samples(
        testset=testset,
        robot=robot,
        prompt_version=prompt_version,
        persona_output_dir=persona_output_dir,
        limit=limit,
    )
    if normalize_media:
        samples = prepare_normalized_eval_media(
            samples,
            robot=robot,
            render_hz=render_hz,
            preview_speed_scale=render_speed_scale,
            preview_hold_scale=render_hold_scale,
            top_k=render_top_k,
        )

    if isinstance(variants, (list, tuple)):
        raw_items = [str(item) for item in variants]
    else:
        raw_items = str(variants).split(",")
    variant_list = [item.strip().strip("()[]'\" ") for item in raw_items if item and str(item).strip("()[]'\" ")]
    media_by_variant = {
        variant: prepare_test_media(samples, test_type=variant, robot=robot, hz=render_hz, force=force_regen)
        for variant in variant_list
    }
    binary_tasks = build_binary_tasks(samples, seed=seed) if task_family == "binary_match" else []
    neg_by_sample_id = {
        task["sample"]["sample_id"]: task["display_cue"]
        for task in binary_tasks
        if task.get("label") == 0
    }

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{testset}_media_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Media Preview</title>
<style>
:root {
  --bg: #f4f7fb; --surface: #ffffff; --surface2: #eef3f8; --border: #d6dde7; --text: #17202a; --muted: #62707f;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.wrap { max-width: 1800px; margin: 0 auto; padding: 24px; }
.hero { margin-bottom: 18px; }
.hero h1 { margin: 0 0 6px; font-size: 28px; }
.hero p { margin: 0; color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; margin-bottom: 20px; }
.hdr { padding: 12px 16px; background: var(--surface2); border-bottom: 1px solid var(--border); }
.title { font-weight: 700; font-size: 18px; }
.meta { margin-top: 4px; color: var(--muted); font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 14px; padding: 16px; }
.cell { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: #fff; }
.label { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 12px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); background: #fafcff; }
.media { min-height: 220px; background: #edf2f7; display: flex; align-items: center; justify-content: center; overflow: hidden; }
.media img, .media video { max-width: 100%; display: block; }
@media (max-width: 1100px) { .grid { grid-template-columns: 1fr 1fr; } }
@media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body><div class="wrap">"""]

    parts.append(
        f"<section class=\"hero\"><h1>Media Preview</h1><p>testset={_esc(testset)} | n_samples={len(samples)} | variants={_esc(', '.join(variant_list))} | task_family={_esc(task_family)} | seed={seed}</p></section>"
    )

    for idx, sample in enumerate(samples):
        neg_cue = neg_by_sample_id.get(sample["sample_id"], "")
        parts.append(
            f"<section class=\"card\"><div class=\"hdr\"><div class=\"title\">{idx + 1}. {_esc(sample['cue'])}</div><div class=\"meta\">sample_id={_esc(sample['sample_id'])} | cue_idx={_esc(sample['cue_idx'])} | positive cue={_esc(sample['cue'])}"
            + (f" | negative cue={_esc(neg_cue)}" if neg_cue else "")
            + "</div></div><div class=\"grid\">"
        )
        for variant in variant_list:
            media_sample = media_by_variant[variant][idx]
            parts.append(
                f"<div class=\"cell\"><div class=\"label\">{_esc(variant)}</div><div class=\"media\">{_media_html(media_sample['media_path'], media_sample['test_media_type'], html_path.parent)}</div></div>"
            )
        parts.append("</div></section>")

    parts.append("</div></body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))

    print(f"HTML: {html_path}")
    print(f"HTML URL: file://{html_path}")


if __name__ == "__main__":
    fire.Fire(main)
