"""Top-of-frame GIF progress overlays (typed step segments vs simple scrubber).

Shared by mobile / quadruped render so we do not import the full manipulator MotionGenerator."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

STEP_PROGRESS_COLORS = {
    "pose": (88, 166, 255, 255),
    "movement": (63, 185, 80, 255),
    "path": (188, 140, 255, 255),
}

STEP_PROGRESS_BG = {
    "pose": (88, 166, 255, 64),
    "movement": (63, 185, 80, 64),
    "path": (188, 140, 255, 64),
}


def overlay_typed_progress_bar_on_frames(
    frames: list[Image.Image],
    step_spans: list[dict[str, Any]],
) -> list[Image.Image]:
    """Manipulator-style segmented bar keyed by span ``type``: pose | movement | path."""
    if not frames or not step_spans:
        return frames

    total_frames = len(frames)
    width, height = frames[0].size
    outer_pad = max(8, width // 48)
    bar_height = max(12, height // 36)
    bar_left = outer_pad
    bar_top = outer_pad
    bar_width = width - outer_pad * 2
    bar_right = bar_left + bar_width
    bar_bottom = bar_top + bar_height
    marker_width = max(2, width // 180)

    overlaid: list[Image.Image] = []
    for frame_idx, frame in enumerate(frames):
        base = frame.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rounded_rectangle(
            [bar_left - 4, bar_top - 4, bar_right + 4, bar_bottom + 4],
            radius=bar_height // 2 + 4,
            fill=(8, 12, 18, 170),
            outline=(230, 237, 243, 40),
            width=1,
        )

        for span in step_spans:
            start = int(span["start"])
            end = int(span["end"])
            if end <= start:
                continue
            step_type = span["type"]
            bg_color = STEP_PROGRESS_BG.get(step_type, (139, 148, 158, 64))
            fg_color = STEP_PROGRESS_COLORS.get(step_type, (139, 148, 158, 255))

            x0 = bar_left + int(round((start / total_frames) * bar_width))
            x1 = bar_left + int(round((end / total_frames) * bar_width))
            if x1 <= x0:
                x1 = min(bar_right, x0 + 1)

            draw.rounded_rectangle(
                [x0, bar_top, x1, bar_bottom],
                radius=bar_height // 2,
                fill=bg_color,
            )

            progress_end = min(frame_idx + 1, end)
            if progress_end > start:
                fill_x1 = bar_left + int(round((progress_end / total_frames) * bar_width))
                if fill_x1 <= x0:
                    fill_x1 = min(bar_right, x0 + 1)
                draw.rounded_rectangle(
                    [x0, bar_top, fill_x1, bar_bottom],
                    radius=bar_height // 2,
                    fill=fg_color,
                )

        progress_x = bar_left + int(round(((frame_idx + 1) / total_frames) * bar_width))
        progress_x = max(bar_left, min(bar_right, progress_x))
        draw.rectangle(
            [progress_x - marker_width, bar_top - 2, progress_x + marker_width, bar_bottom + 2],
            fill=(255, 255, 255, 220),
        )

        draw.rounded_rectangle(
            [bar_left, bar_top, bar_right, bar_bottom],
            radius=bar_height // 2,
            outline=(230, 237, 243, 120),
            width=1,
        )

        overlaid.append(Image.alpha_composite(base, overlay).convert("RGB"))

    return overlaid


def overlay_simple_progress_bar_on_frames(frames: list[Image.Image]) -> list[Image.Image]:
    """Neutral single-segment scrubber."""
    if not frames:
        return frames

    total_frames = len(frames)
    width, height = frames[0].size
    outer_pad = max(8, width // 48)
    bar_height = max(10, height // 42)
    bar_left = outer_pad
    bar_top = outer_pad
    bar_width = width - outer_pad * 2
    bar_right = bar_left + bar_width
    bar_bottom = bar_top + bar_height

    overlaid = []
    for frame_idx, frame in enumerate(frames):
        base = frame.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        draw.rounded_rectangle(
            [bar_left - 3, bar_top - 3, bar_right + 3, bar_bottom + 3],
            radius=bar_height // 2 + 3,
            fill=(12, 16, 22, 110),
        )
        draw.rounded_rectangle(
            [bar_left, bar_top, bar_right, bar_bottom],
            radius=bar_height // 2,
            fill=(235, 239, 244, 180),
        )

        progress = (frame_idx + 1) / max(1, total_frames)
        fill_x1 = bar_left + int(round(progress * bar_width))
        fill_x1 = min(max(fill_x1, bar_left + 1), bar_right)
        draw.rounded_rectangle(
            [bar_left, bar_top, fill_x1, bar_bottom],
            radius=bar_height // 2,
            fill=(30, 30, 30, 230),
        )

        composed = Image.alpha_composite(base, overlay).convert("RGBA")
        overlaid.append(composed)

    return [im.convert("RGB") for im in overlaid]
