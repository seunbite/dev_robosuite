from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path("/Users/sb/Downloads/workspace/dev_robosuite")
PAPER = ROOT / "papers/icra2027_motion_abstraction"
SRC_LLM = PAPER / ".codex_pptx/normalized_gifs_v4_320"
SRC_GT = PAPER / ".codex_pptx/balanced_15/normalized_gt_gifs_v4_320"
OUT_LLM = PAPER / ".codex_pptx/normalized_gifs_v5_320"
OUT_GT = PAPER / ".codex_pptx/balanced_15/normalized_gt_gifs_v5_320"
SIM = ROOT / "tmp/final_paper_video/v5_sim"
ROBOTARM = ROOT / "adhoc/generation/robotarm"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROBOTARM))
from embodiment_renderers import render_roomba  # noqa: E402


def rows_from(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return [record["platforms"]["roomba"] for record in records if "roomba" in record.get("platforms", {})]


def normalize_sim(src: Path, dst: Path) -> None:
    im = Image.open(src)
    frames: list[Image.Image] = []
    durations: list[int] = []
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        frames.append(ImageOps.fit(im.convert("RGB"), (320, 320), method=Image.Resampling.LANCZOS))
        durations.append(max(20, int(im.info.get("duration", 83))))
    frames[0].save(dst, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True, disposal=2)


def replace_roomba(dst_root: Path, sim_root: Path) -> None:
    for dst in sorted(dst_root.glob("*__roomba.gif")):
        prefix = dst.stem.split("__", 1)[0]
        idx, cue = prefix.split("_", 1)
        rendered = sim_root / f"emb_g{int(idx):02d}_{cue}.gif"
        if not rendered.exists():
            raise FileNotFoundError(rendered)
        normalize_sim(rendered, dst)


if __name__ == "__main__":
    shutil.copytree(SRC_LLM, OUT_LLM, dirs_exist_ok=True)
    shutil.copytree(SRC_GT, OUT_GT, dirs_exist_ok=True)
    llm_main = ROOT / "data/results/motion_configs/embodiment_motion_code_gemini_gemini-2.5-pro/embodiment_motion_code_all.json"
    llm_ext = ROOT / "data/results/motion_configs/embodiment_extension20_gemini/embodiment_motion_code_all.json"
    gt_main = ROOT / "data/results/motion_configs/embodiment_motion_code_gt/embodiment_motion_code_all.json"
    render_roomba(rows_from(llm_main) + rows_from(llm_ext), SIM / "llm", force=True, fps=12)
    render_roomba(rows_from(gt_main), SIM / "gt", force=True, fps=12)
    replace_roomba(OUT_LLM, SIM / "llm")
    replace_roomba(OUT_GT, SIM / "gt")
    print(OUT_LLM)
    print(OUT_GT)
