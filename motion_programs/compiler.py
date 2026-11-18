"""Portable entry point for compiling the paper's Motion Code DSL."""

from __future__ import annotations

from typing import Any


def compile_motion_code(code: str, platform: str) -> dict[str, Any]:
    """Compile one Motion Code string into renderer-ready movements."""
    from adhoc.generation.robotarm.motion_code_to_config import convert_code

    return convert_code(code, platform=platform)
