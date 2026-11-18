from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = ROOT / "papers/icra2027_motion_abstraction"
OUT = ROOT / "tmp/final_paper_video/iiwa_bird_attribute_variants"
RAW = OUT / "raw"
NORMALIZED = OUT / "normalized"
BLUE = (31, 111, 235)
TRACK = (45, 48, 55)

os.environ.setdefault("MUJOCO_GL", "glfw")
sys.path[:0] = [
    str(PAPER / "figures"),
    str(PAPER / "motion_code"),
    str(PAPER),
    str(ROOT),
    str(ROOT / "adhoc/generation/robotarm"),
]

from build_attr_buildup_grid import _entry  # noqa: E402
from motion_code_to_config import render_manipulator  # noqa: E402


PROGRAMS = {
    "orientation": (
        "pose(dir=up, grip=horiz, y=100, z=65) "
        "+ move(shoulder, z, [-25, 25], rep=4, spd=2.5)"
    ),
    "location": (
        "pose(dir=out, grip=horiz, y=100, z=90) "
        "+ move(shoulder, z, [-25, 25], rep=4, spd=2.5)"
    ),
}


def bottom_bar(frame: Image.Image, index: int, total: int) -> Image.Image:
    result = frame.convert("RGB")
    draw = ImageDraw.Draw(result)
    margin = 10
    y0 = result.height - 14
    x1 = margin + int((result.width - 2 * margin) * (index + 1) / max(1, total))
    draw.rounded_rectangle((margin, y0, result.width - margin, y0 + 5), radius=2, fill=TRACK)
    draw.rounded_rectangle((margin, y0, x1, y0 + 5), radius=2, fill=BLUE)
    return result


def normalize_top_bar(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        frames: list[Image.Image] = []
        durations: list[int] = []
        total = getattr(image, "n_frames", 1)
        for index in range(total):
            image.seek(index)
            frame = image.convert("RGB").crop((0, 30, image.width, image.height))
            square = ImageOps.fit(frame, (320, 320), method=Image.Resampling.LANCZOS)
            frames.append(bottom_bar(square, index, total))
            durations.append(max(20, int(image.info.get("duration", 80))))
    target.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    rows = []
    base_index = 9400
    for offset, (tag, program) in enumerate(PROGRAMS.items()):
        entry = _entry(base_index + offset, "depict_flying_bird", "single_arm", program, tag)
        rows.append(entry["platforms"]["single_arm"])

    rendered = render_manipulator(rows, out_root=RAW, force=True)
    for tag in PROGRAMS:
        cue = f"depict_flying_bird__{tag}"
        source = rendered.get(cue)
        if source is None or not source.is_file():
            raise FileNotFoundError(f"Renderer did not produce {cue}")
        target = NORMALIZED / f"30_depict_flying_bird__single_arm__{tag}.gif"
        normalize_top_bar(source, target)
        print(target)


if __name__ == "__main__":
    main()
