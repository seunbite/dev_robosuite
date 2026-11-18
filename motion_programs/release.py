"""Validation helpers for the compact paper release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROGRAM_FILES = {
    "human_nonanthropomorphic30.json",
    "gemini_nonanthropomorphic30.json",
    "human_humanlike90.json",
    "gemini_humanlike90.json",
    "qwen32b_humanlike90.json",
    "qwen7b_humanlike90.json",
    "gemini_synthetic.json",
    "human_synthetic_reference.json",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_release(root: str | Path) -> dict[str, int]:
    """Validate required release files and return lightweight counts."""
    root = Path(root)
    schema_path = root / "motion_program_schema.yaml"
    cues_path = root / "cues50_synthetic.yaml"
    if not schema_path.is_file() or not cues_path.is_file():
        raise FileNotFoundError("paper schemas are missing")
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    cues = yaml.safe_load(cues_path.read_text(encoding="utf-8"))
    if str(schema.get("meta", {}).get("version")) != "1.0":
        raise ValueError("unexpected Motion Program schema version")
    if int(cues.get("meta", {}).get("n_cues", 0)) != 50:
        raise ValueError("expected the 50-cue release inventory")

    programs = root / "programs"
    missing = sorted(PROGRAM_FILES - {p.name for p in programs.glob("*.json")})
    if missing:
        raise FileNotFoundError(f"missing program files: {missing}")
    rows = 0
    for name in sorted(PROGRAM_FILES):
        payload = _load_json(programs / name)
        if not isinstance(payload, list):
            raise ValueError(f"{name}: expected a JSON list")
        rows += len(payload)

    evaluations = list((root / "evaluation").glob("*.json"))
    for path in evaluations:
        _load_json(path)
    return {
        "program_files": len(PROGRAM_FILES),
        "program_rows": rows,
        "evaluation_files": len(evaluations),
        "synthetic_cues": len(cues.get("cues", {})),
    }
