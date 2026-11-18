from pathlib import Path

from motion_programs.compiler import compile_motion_code
from motion_programs.release import validate_release


def test_curated_paper_release_is_complete() -> None:
    root = Path(__file__).resolve().parents[1] / "data/paper"
    counts = validate_release(root)
    assert counts["program_files"] == 8
    assert counts["program_rows"] > 100
    assert counts["evaluation_files"] >= 7
    assert counts["synthetic_cues"] == 50


def test_motion_code_compiles_for_roomba() -> None:
    result = compile_motion_code(
        "pose(dir=front) + path(base, line, x=-0.15) + hold(t=1.0)",
        platform="roomba",
    )
    assert result["parse_ok"] is True
    assert result["movements"]
