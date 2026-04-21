import json
import math
import os
from datetime import datetime
from pathlib import Path

import fire
from google import genai
from google.genai import types
from PIL import Image, ImageDraw

from testset_utils import (
    _build_alpha_stack_image,
    build_samples,
    prepare_normalized_eval_media,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _load_media_part(media_path: str, media_mime: str) -> types.Part:
    with open(media_path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=media_mime)


def _extract_variation_frames_from_tiled(
    tiled_gif_path: str,
    *,
    num_variations: int = 5,
    max_per_row: int = 10,
) -> list[list[Image.Image]]:
    gif = Image.open(tiled_gif_path)
    frames: list[Image.Image] = []
    try:
        while True:
            frames.append(gif.copy().convert("RGB"))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    finally:
        gif.close()

    if not frames:
        raise ValueError(f"No frames in tiled gif: {tiled_gif_path}")

    total_width = frames[0].width
    total_height = frames[0].height
    cols = min(num_variations, max_per_row)
    rows = max(1, math.ceil(num_variations / max_per_row))
    tile_w = total_width // cols
    tile_h = total_height // rows

    all_variations: list[list[Image.Image]] = []
    for variation_index in range(num_variations):
        row = variation_index // max_per_row
        col = variation_index % max_per_row
        v_frames = []
        for frame in frames:
            left = col * tile_w
            top = row * tile_h
            cropped = frame.crop((left, top, left + tile_w, top + tile_h)).convert("RGB")
            v_frames.append(cropped)
        all_variations.append(v_frames)
    return all_variations


def _alpha_from_frames(frames: list[Image.Image], *, stack_count: int = 12) -> Image.Image:
    if not frames:
        raise ValueError("Empty variation frames")
    total_frames = len(frames)
    if total_frames == 1:
        return frames[0].convert("RGB")

    indices = sorted(set(int(round(i * (total_frames - 1) / max(1, stack_count - 1))) for i in range(stack_count)))
    final_rgb = frames[-1].convert("RGB")
    canvas = final_rgb.convert("RGBA")

    import numpy as np

    final_np = np.asarray(final_rgb).astype(np.int16)
    trail_indices = [idx for idx in indices if idx != total_frames - 1]
    total_trails = len(trail_indices)
    for order, frame_idx in enumerate(trail_indices):
        frame_rgb = frames[frame_idx].convert("RGB")
        frame_np = np.asarray(frame_rgb).astype(np.int16)
        diff = np.abs(frame_np - final_np).sum(axis=2)
        from PIL import ImageFilter

        mask = Image.fromarray(np.where(diff > 28, 255, 0).astype(np.uint8), mode="L")
        if order == 0:
            mask = mask.filter(ImageFilter.MaxFilter(size=13))
            alpha = 255
        else:
            grow = max(3, 11 - min(8, order))
            if grow % 2 == 0:
                grow += 1
            mask = mask.filter(ImageFilter.MaxFilter(size=grow)).filter(ImageFilter.GaussianBlur(radius=1.2))
            frac = 1.0 - (order / max(1, total_trails - 1))
            alpha = int(90 + 90 * frac)
        frame_rgba = frame_rgb.convert("RGBA")
        frame_rgba.putalpha(mask.point(lambda px: min(255, int(px * alpha / 255.0))))
        canvas.alpha_composite(frame_rgba)

    final_overlay = final_rgb.convert("RGBA")
    final_overlay.putalpha(32)
    canvas.alpha_composite(final_overlay)
    return canvas.convert("RGB")


def _alpha_tiles_from_tiled(tiled_gif_path: str, *, num_variations: int = 5) -> list[Image.Image]:
    variations = _extract_variation_frames_from_tiled(tiled_gif_path, num_variations=num_variations)
    return [_alpha_from_frames(v_frames) for v_frames in variations]


def _top1_pixel_diffs(alpha_tiles: list[Image.Image]) -> list[float]:
    import numpy as np

    if not alpha_tiles:
        return []
    ref = np.asarray(alpha_tiles[0].convert("RGB")).astype(np.float32)
    ref_fg = (ref < 245).any(axis=2)
    diffs: list[float] = []
    for tile in alpha_tiles:
        arr = np.asarray(tile.convert("RGB")).astype(np.float32)
        fg = (arr < 245).any(axis=2)
        mask = ref_fg | fg
        if not mask.any():
            diffs.append(0.0)
            continue
        mad = np.abs(ref - arr)[mask].mean() / 255.0
        diffs.append(float(mad * 100.0))
    return diffs


def _tile_pixel_diff(a: Image.Image, b: Image.Image) -> float:
    import numpy as np

    arr_a = np.asarray(a.convert("RGB")).astype(np.float32)
    arr_b = np.asarray(b.convert("RGB")).astype(np.float32)
    fg_a = (arr_a < 245).any(axis=2)
    fg_b = (arr_b < 245).any(axis=2)
    mask = fg_a | fg_b
    if not mask.any():
        return 0.0
    return float(np.abs(arr_a - arr_b)[mask].mean() / 255.0 * 100.0)


def _greedy_filter_alpha_tiles(alpha_tiles: list[Image.Image], *, diff_threshold: float) -> tuple[list[Image.Image], list[int], list[float]]:
    if not alpha_tiles:
        return [], [], []
    kept_tiles = [alpha_tiles[0]]
    kept_indices = [1]
    kept_pair_diffs = [0.0]
    last_kept = alpha_tiles[0]
    for idx, tile in enumerate(alpha_tiles[1:], start=2):
        diff = _tile_pixel_diff(last_kept, tile)
        if diff < diff_threshold:
            continue
        kept_tiles.append(tile)
        kept_indices.append(idx)
        kept_pair_diffs.append(diff)
        last_kept = tile
    return kept_tiles, kept_indices, kept_pair_diffs


def _build_topk_alpha_strip(
    tiled_gif_path: str,
    output_path: Path,
    *,
    num_variations: int = 5,
    diff_threshold: float | None = None,
) -> Path:
    alpha_tiles = _alpha_tiles_from_tiled(tiled_gif_path, num_variations=num_variations)
    display_indices = list(range(1, len(alpha_tiles) + 1))
    if diff_threshold is not None:
        alpha_tiles, display_indices, _ = _greedy_filter_alpha_tiles(alpha_tiles, diff_threshold=diff_threshold)
    tile_w, tile_h = alpha_tiles[0].size
    actual_variations = len(alpha_tiles)
    label_h = max(38, tile_h // 10)
    canvas = Image.new("RGB", (tile_w * actual_variations, tile_h + label_h), color="white")
    draw = ImageDraw.Draw(canvas)
    for i, tile in enumerate(alpha_tiles):
        x = i * tile_w
        canvas.paste(tile, (x, 0))
        draw.rectangle([x, 0, x + tile_w - 1, tile_h + label_h - 1], outline="black", width=2)
        badge = str(i + 1)
        draw.text((x + 14, tile_h + 7), badge, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    return output_path


def _build_top5_alpha_strip(tiled_gif_path: str, output_path: Path, *, num_variations: int = 5) -> Path:
    return _build_topk_alpha_strip(tiled_gif_path, output_path, num_variations=num_variations)


def _choice_prompt(cue: str, num_variations: int = 5) -> str:
    numbers = ", ".join(str(i) for i in range(1, num_variations + 1))
    return (
        f"You will see one PNG containing {num_variations} robot gesture candidates arranged left to right and labeled 1 to {num_variations}. "
        f"All {num_variations} are intended to represent the same cue, but they differ in initial pose. "
        f"Which candidate best communicates this cue?\n\nCue: {cue}\n\n"
        f"Answer with exactly one number from: {numbers}."
    )


def _parse_choice(text: str, num_variations: int = 5) -> int | None:
    import re

    allowed = {str(i) for i in range(1, num_variations + 1)}
    for token in re.findall(r"\d+", str(text)):
        if token in allowed:
            return int(token)
    return None


def _write_html_report(html_path: Path, rows: list[dict], summary: dict, *, num_variations: int = 5) -> None:
    parts = ["""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Top-K Alpha Choice Eval</title>
<style>
:root {
  --bg: #f4f7fb; --surface: #fff; --surface2: #eef3f8; --border: #d6dde7;
  --text: #17202a; --muted: #62707f; --green: #1b8f3a; --greenbg: #ecfbef; --red: #c93b36; --redbg: #fff1f1;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.wrap { max-width: 1480px; margin: 0 auto; padding: 24px; }
.hero h1 { margin: 0 0 8px; font-size: 28px; }
.hero p { margin: 0; color: var(--muted); }
.summary { display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0 22px; }
.chip { background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 8px 12px; font-size: 13px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; margin-bottom: 18px; }
.hdr { padding: 12px 16px; background: var(--surface2); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.title { font-weight: 700; }
.badge { border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }
.badge.correct { background: var(--greenbg); color: var(--green); }
.badge.wrong { background: var(--redbg); color: var(--red); }
.body { display: grid; grid-template-columns: minmax(520px, 760px) 1fr; gap: 18px; padding: 16px; }
.media-col { display: grid; grid-template-columns: 1fr; gap: 12px; }
.media { background: #edf2f7; border: 1px solid var(--border); border-radius: 12px; padding: 8px; }
.media img, .media video { width: 100%; display: block; }
.media-label { font-size: 12px; font-weight: 700; color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 8px; }
.info { display: flex; flex-direction: column; gap: 10px; }
.cue { font-size: 16px; color: var(--muted); }
.answer { background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: 12px; }
.label { font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); margin-bottom: 8px; }
.option { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; margin-bottom: 8px; }
.option.correct { background: var(--greenbg); border-color: #7fd39a; }
.option.wrong { background: var(--redbg); border-color: #ef9a9a; }
.raw { white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
@media (max-width: 980px) { .body { grid-template-columns: 1fr; } }
</style>
</head>
<body><div class="wrap">"""]
    parts.append(f"<section class=\"hero\"><h1>Top-{num_variations} Alpha Choice Eval</h1><p>VLM chooses which initial-pose variation best communicates the given cue. Ground truth assumes tile 1 is the best candidate.</p></section>")
    parts.append("<section class=\"summary\">")
    for key in ["testset", "n_tasks", "correct", "accuracy", "model_name"]:
        if key in summary:
            value = summary[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            parts.append(f"<div class=\"chip\">{_esc(key)}: {_esc(value)}</div>")
    parts.append("</section>")

    for idx, row in enumerate(rows, start=1):
        correct = bool(row.get("correct"))
        status_class = "correct" if correct else "wrong"
        status_text = "Correct" if correct else "Wrong"
        row_variations = len(row.get("kept_indices", [])) or num_variations
        rel = os.path.relpath(row["media_path"], html_path.parent)
        gif_rel = os.path.relpath(row["gif_path"], html_path.parent)
        parts.append(f"<section class=\"card\"><div class=\"hdr\"><div class=\"title\">{idx}. {_esc(row['sample_id'])}</div><div class=\"badge {status_class}\">{status_text}</div></div><div class=\"body\">")
        parts.append("<div class=\"media-col\">")
        parts.append(f"<div class=\"media\"><div class=\"media-label\">Filtered Top-{row_variations} Alpha Strip</div><img src=\"{_esc(rel)}\" alt=\"topk alpha strip\"></div>")
        parts.append(f"<div class=\"media\"><div class=\"media-label\">Source Tiled GIF</div><img src=\"{_esc(gif_rel)}\" alt=\"source tiled gif\"></div>")
        parts.append("</div>")
        parts.append("<div class=\"info\">")
        parts.append(f"<div class=\"cue\">Cue: <strong>{_esc(row['cue'])}</strong></div>")
        if row.get("kept_indices"):
            parts.append(f"<div class=\"cue\">Kept original tiles: <strong>{_esc(row['kept_indices'])}</strong></div>")
        parts.append("<div class=\"answer\"><div class=\"label\">Choice</div>")
        for i in range(1, row_variations + 1):
            classes = []
            if i == 1:
                classes.append("correct")
            if row.get("pred_choice") == i and i != 1:
                classes.append("wrong")
            parts.append(f"<div class=\"option {' '.join(classes)}\">Tile {i}</div>")
        parts.append(f"<div class=\"option\">Ground truth: Tile 1 | Model picked: {_esc(row.get('pred_choice'))}</div>")
        parts.append("</div>")
        if row.get("top1_pixel_diffs"):
            parts.append("<div class=\"answer\"><div class=\"label\">Pixel Diff vs Tile 1</div>")
            for i, diff in enumerate(row["top1_pixel_diffs"], start=1):
                parts.append(f"<div class=\"option\">Tile {i}: {diff:.2f}%</div>")
            parts.append("</div>")
        if row.get("kept_pair_diffs"):
            parts.append("<div class=\"answer\"><div class=\"label\">Greedy Keep Diff</div>")
            for i, diff in enumerate(row["kept_pair_diffs"], start=1):
                parts.append(f"<div class=\"option\">Kept Tile {i}: {diff:.2f}% vs previous kept</div>")
            parts.append("</div>")
        parts.append(f"<div class=\"answer\"><div class=\"label\">Model Raw Response</div><div class=\"raw\">{_esc(row.get('raw_response', ''))}</div></div>")
        parts.append("</div></div></section>")

    parts.append("</div></body></html>")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


def main(
    testset: str = "iconic",
    robot: str = "IIWA",
    prompt_version: int = 18,
    config_json: str | None = None,
    iconic_motion_subdir: str = "v18",
    contextual_motion_subdir: str = "v18_contextual",
    persona_output_dir: str = "data/seed/persona_tag_dataset_v1",
    model_name: str = "gemini-2.5-flash",
    limit: int = 10,
    output_dir: str = "adhoc/test/results",
    seed: int = 42,
    temperature: float = 0.0,
    render_hz: int = 8,
    render_speed_scale: float = 1.0,
    render_hold_scale: float = 1.0,
    normalize_media: bool = False,
    force_render: bool = False,
    force_png: bool = False,
    open_html: bool = True,
    top_k: int = 5,
    greedy_diff_threshold: float | None = None,
):
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY or GEMINI_API_KEY.")

    samples = build_samples(
        testset=testset,
        robot=robot,
        prompt_version=prompt_version,
        config_json=config_json,
        iconic_motion_subdir=iconic_motion_subdir,
        contextual_motion_subdir=contextual_motion_subdir,
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
            top_k=top_k,
            output_dir="adhoc/test/eval_media_top5",
            force=force_render,
        )

    out_dir = (REPO_ROOT / output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / f"top{top_k}_alpha_png"
    png_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{testset}_top{top_k}_alpha_choice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    summary_path = out_dir / f"{stem}_summary.json"
    html_path = out_dir / f"{stem}_report.html"

    print(f"HTML report: {html_path.resolve()}")
    print(f"HTML URL: file://{html_path.resolve()}")

    prepared = []
    for sample in samples:
        media_path = png_dir / f"{sample['sample_id']}.png"
        alpha_tiles = _alpha_tiles_from_tiled(sample["gif_path"], num_variations=top_k)
        kept_indices = list(range(1, top_k + 1))
        kept_pair_diffs = [0.0]
        if greedy_diff_threshold is not None:
            alpha_tiles, kept_indices, kept_pair_diffs = _greedy_filter_alpha_tiles(alpha_tiles, diff_threshold=greedy_diff_threshold)
        pixel_diffs = _top1_pixel_diffs(alpha_tiles)
        if force_png or not media_path.exists():
            _build_topk_alpha_strip(
                sample["gif_path"],
                media_path,
                num_variations=top_k,
                diff_threshold=greedy_diff_threshold,
            )
        item = dict(sample)
        item["media_path"] = str(media_path)
        item["media_mime"] = "image/png"
        item["top1_pixel_diffs"] = pixel_diffs
        item["kept_indices"] = kept_indices
        item["kept_pair_diffs"] = kept_pair_diffs
        prepared.append(item)

    client = genai.Client(api_key=api_key)
    rows: list[dict] = []
    with open(jsonl_path, "w", encoding="utf-8", buffering=1) as f:
        for idx, sample in enumerate(prepared, start=1):
            prompt = _choice_prompt(sample["cue"], len(sample.get("kept_indices", [])) or top_k)
            media = _load_media_part(sample["media_path"], sample["media_mime"])
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, media],
                    config=types.GenerateContentConfig(temperature=float(temperature)),
                )
                raw_text = response.text.strip()
            except Exception as exc:
                raw_text = f"ERROR: {exc}"

            pred_choice = _parse_choice(raw_text, len(sample.get("kept_indices", [])) or top_k)
            row = {
                "sample_id": sample["sample_id"],
                "testset": testset,
                "cue": sample["cue"],
                "cue_idx": sample["cue_idx"],
                "media_path": sample["media_path"],
                "gif_path": sample["gif_path"],
                "raw_response": raw_text,
                "pred_choice": pred_choice,
                "top1_pixel_diffs": sample.get("top1_pixel_diffs", []),
                "kept_indices": sample.get("kept_indices", []),
                "kept_pair_diffs": sample.get("kept_pair_diffs", []),
                "correct": pred_choice == 1,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            summary = {
                "testset": testset,
                "model_name": model_name,
                "n_tasks": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
                "jsonl_path": str(jsonl_path),
            }
            _write_html_report(html_path, rows, summary, num_variations=(len(sample.get("kept_indices", [])) or top_k))
            print(f"[{idx}/{len(prepared)}] refreshed HTML: file://{html_path.resolve()}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results JSONL: {jsonl_path}")
    print(f"Summary JSON: {summary_path}")
    print(f"Final HTML: {html_path.resolve()}")
    print(f"Final HTML URL: file://{html_path.resolve()}")
    if open_html:
        os.system(f"open '{html_path.resolve()}'")


if __name__ == "__main__":
    fire.Fire(main)
