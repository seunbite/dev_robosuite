from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = ROOT / "papers/icra2027_motion_abstraction"
SRC_LLM = PAPER / ".codex_pptx/optimized_gifs_320"
SRC_GT = PAPER / ".codex_pptx/balanced_15/optimized_gt_gifs_320"
OUT_LLM = PAPER / ".codex_pptx/normalized_gifs_v4_320"
OUT_GT = PAPER / ".codex_pptx/balanced_15/normalized_gt_gifs_v4_320"
SIM = ROOT / "tmp/final_paper_video/v4_sim"
ROBOTARM = ROOT / "adhoc/generation/robotarm"
BLUE = (31, 111, 235)
TRACK = (45, 48, 55)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROBOTARM))
from embodiment_renderers import render_roomba, render_stretch  # noqa: E402


def rows_from(path: Path, platform: str) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [record["platforms"][platform] for record in records if platform in record.get("platforms", {})]


def save_gif(frames: list[Image.Image], durations: list[int], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )


def bottom_bar(frame: Image.Image, index: int, total: int) -> Image.Image:
    out = frame.convert("RGB")
    draw = ImageDraw.Draw(out)
    margin = 10
    y0 = out.height - 14
    x1 = margin + int((out.width - 2 * margin) * (index + 1) / max(1, total))
    draw.rounded_rectangle((margin, y0, out.width - margin, y0 + 5), radius=2, fill=TRACK)
    draw.rounded_rectangle((margin, y0, x1, y0 + 5), radius=2, fill=BLUE)
    return out


def normalize_top_bar(src: Path, dst: Path) -> None:
    im = Image.open(src)
    frames: list[Image.Image] = []
    durations: list[int] = []
    total = getattr(im, "n_frames", 1)
    for i in range(total):
        im.seek(i)
        frame = im.convert("RGB")
        crop = frame.crop((0, 20, frame.width, frame.height))
        square = ImageOps.fit(crop, (320, 320), method=Image.Resampling.LANCZOS)
        frames.append(bottom_bar(square, i, total))
        durations.append(max(20, int(im.info.get("duration", 80))))
    save_gif(frames, durations, dst)


def normalize_sim(src: Path, dst: Path) -> None:
    im = Image.open(src)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        frames.append(ImageOps.fit(im.convert("RGB"), (320, 320), method=Image.Resampling.LANCZOS))
        durations.append(max(20, int(im.info.get("duration", 83))))
    save_gif(frames, durations, dst)


def render_sets() -> None:
    llm_main = ROOT / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
    llm_ext = ROOT / "data/results/motion_configs/embodiment_extension20_gemini/embodiment_motion_code_all.json"
    gt_main = ROOT / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
    for platform, fn in (("stretch", render_stretch), ("roomba", render_roomba)):
        llm_rows = rows_from(llm_main, platform) + rows_from(llm_ext, platform)
        gt_rows = rows_from(gt_main, platform)
        fn(llm_rows, SIM / "llm" / platform, force=True, fps=12)
        fn(gt_rows, SIM / "gt" / platform, force=True, fps=12)


def assemble(src_root: Path, dst_root: Path, sim_root: Path) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    for src in sorted(src_root.glob("*.gif")):
        dst = dst_root / src.name
        platform = src.stem.rsplit("__", 1)[-1]
        if platform in {"tiago", "single_arm"}:
            normalize_top_bar(src, dst)
            continue
        if platform in {"stretch", "roomba"}:
            prefix = src.stem.split("__", 1)[0]
            idx, cue = prefix.split("_", 1)
            rendered = sim_root / platform / f"emb_g{int(idx):02d}_{cue}.gif"
            if rendered.exists():
                normalize_sim(rendered, dst)
                continue
        shutil.copy2(src, dst)


if __name__ == "__main__":
    render_sets()
    assemble(SRC_LLM, OUT_LLM, SIM / "llm")
    assemble(SRC_GT, OUT_GT, SIM / "gt")
    print(OUT_LLM)
    print(OUT_GT)
