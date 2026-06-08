"""Load pilot-40 prompt templates from data/seed/prompt/pilot40/."""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
PROMPT_DIR = _REPO / "data/seed/prompt/pilot40"


def prompt_path(name: str) -> Path:
    return PROMPT_DIR / name


def load_snippet(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8").strip()


def fill_template(name: str, mapping: dict[str, str]) -> str:
    text = load_snippet(name)
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text.strip()
