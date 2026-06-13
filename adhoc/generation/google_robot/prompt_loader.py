"""Load pilot-40 Google Robot prompt templates from data/seed/prompt/google_robot/exp/."""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
PROMPT_DIR = _REPO / "data/seed/prompt" / "google_robot" / "exp"


def prompt_path(name: str) -> Path:
    return PROMPT_DIR / name


def load_snippet(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8").strip()


def fill_template(name: str, mapping: dict[str, str]) -> str:
    text = load_snippet(name)
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text.strip()


def exp_prompt_path(exp_id: str | int) -> Path:
    return PROMPT_DIR / f"prompt_exp{int(exp_id)}.txt"
