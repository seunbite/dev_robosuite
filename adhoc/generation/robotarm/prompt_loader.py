"""Load pilot-90 prompt templates from data/seed/prompt/manipulator/exp/."""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
PROMPT_DIR = _REPO / "data/seed/prompt/manipulator/exp"
# Legacy pilot-40 verify snippets (shared blocks copied into exp/)
LEGACY_PILOT40 = _REPO / "data/seed/prompt/pilot40"

# exp02_pose_verify_vlm.txt → prompt_exp2.txt
_LEGACY_ALIASES: dict[str, str] = {
    "exp02_pose_verify_vlm.txt": "prompt_exp2.txt",
    "exp03_pose_verify_text.txt": "prompt_exp3.txt",
    "exp04_pose_pairwise_2way.txt": "prompt_exp4.txt",
    "exp05_pose_multitile_grid.txt": "prompt_exp5.txt",
    "exp08_motion_verify_vlm.txt": "prompt_exp8.txt",
    "exp09_motion_verify_text.txt": "prompt_exp9.txt",
    "exp10_motion_pairwise_mp4.txt": "prompt_exp10.txt",
}


def prompt_path(name: str) -> Path:
    resolved = _LEGACY_ALIASES.get(name, name)
    p = PROMPT_DIR / resolved
    if p.is_file():
        return p
    legacy = LEGACY_PILOT40 / name
    if legacy.is_file():
        return legacy
    return p


def load_snippet(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8").strip()


def fill_template(name: str, mapping: dict[str, str]) -> str:
    text = load_snippet(name)
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text.strip()


def exp_prompt_path(exp_id: str | int) -> Path:
    return PROMPT_DIR / f"prompt_exp{int(exp_id)}.txt"
