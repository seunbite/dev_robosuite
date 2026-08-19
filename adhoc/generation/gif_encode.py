"""Shared GIF encoder — stable palette for motion + wide pose-pool strips."""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def _sample_frame_indices(n: int, *, cap: int = 28) -> list[int]:
    if n <= 0:
        return []
    if n <= cap:
        return list(range(n))
    idx = {0, n - 1}
    step = max(1, n // (cap - 2))
    idx.update(range(0, n, step))
    return sorted(idx)


def _palette_ref_sheet(rgb: list[Image.Image]) -> Image.Image:
    """Build a reference palette image from representative frames (and tile columns)."""
    w, h = rgb[0].size
    samples = [rgb[i] for i in _sample_frame_indices(len(rgb))]
    strips: list[Image.Image] = []
    ncol = w // h if h > 0 and w >= h * 2 and w % h == 0 else 1
    for s in samples:
        if ncol > 1:
            for c in range(ncol):
                strips.append(s.crop((c * h, 0, (c + 1) * h, h)))
        else:
            strips.append(s)
    tw, th = strips[0].size
    sheet = Image.new("RGB", (tw, th * len(strips)))
    for i, im in enumerate(strips):
        sheet.paste(im, (0, i * th))
    return sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)


def save_gif_frames(
    frames: list,
    gif_path: Path,
    *,
    duration_ms: int = 50,
) -> None:
    """Save RGB frames as GIF with one shared palette (motion-safe sampling)."""
    if not frames:
        return
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = [f.convert("RGB") for f in frames]
    palette_ref = _palette_ref_sheet(rgb)
    frames_q = [im.quantize(palette=palette_ref, dither=Image.Dither.NONE) for im in rgb]
    frames_q[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames_q[1:],
        duration=duration_ms,
        loop=0,
        # disposal=2 → each frame is a full canvas (no sub-rect tiles).
        # disposal=0 + Pillow's tile optimizer leaves frame0 full + later
        # progress-bar-only tiles, which flickers badly in browsers.
        disposal=2,
        optimize=False,
    )
