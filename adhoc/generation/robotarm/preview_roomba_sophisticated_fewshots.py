#!/usr/bin/env python3
"""Create a milestone contact sheet for the Roomba sophisticated GIFs."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_cues_yml_5.json"
OUT = REPO / "data/results/shots/embodiment_unified/roomba_sophisticated_milestones.png"


def main() -> None:
    entries = json.loads(SOURCE.read_text(encoding="utf-8"))
    thumb_w, thumb_h = 208, 168
    label_h, margin = 28, 12
    canvas = Image.new(
        "RGB",
        (margin * 2 + thumb_w * 4, margin * 2 + len(entries) * (thumb_h + label_h)),
        "#10151d",
    )
    draw = ImageDraw.Draw(canvas)
    for row, entry in enumerate(entries):
        gif = Image.open(REPO / entry["render_gif"])
        indices = [round((gif.n_frames - 1) * phase) for phase in (0.0, 0.33, 0.67, 1.0)]
        y = margin + row * (thumb_h + label_h)
        draw.text((margin, y), entry["cue"], fill="#9cc9ff")
        for col, index in enumerate(indices):
            gif.seek(index)
            frame = gif.convert("RGB")
            frame.thumbnail((thumb_w, thumb_h))
            x = margin + col * thumb_w
            canvas.paste(frame, (x, y + label_h))
            draw.text((x + 4, y + label_h + 4), f"{index + 1}/{gif.n_frames}", fill="white")
    canvas.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
