from pathlib import Path

from motion_programs.compiler import compile_motion_code
from motion_programs.pose_envelopes import is_valid_configuration, validate_pose_envelopes
from motion_programs.release import validate_release


def test_curated_paper_release_is_complete() -> None:
    root = Path(__file__).resolve().parents[1] / "data/paper"
    counts = validate_release(root)
    assert counts["program_files"] == 8
    assert counts["program_rows"] > 100
    assert counts["evaluation_files"] >= 7
    assert counts["synthetic_cues"] == 50
    assert counts["g1_valid_poses"] == 206
    assert counts["tiago_valid_poses"] == 47


def test_motion_code_compiles_for_roomba() -> None:
    result = compile_motion_code(
        "pose(dir=front) + path(base, line, x=-0.15) + hold(t=1.0)",
        platform="roomba",
    )
    assert result["parse_ok"] is True
    assert result["movements"]


def test_pose_envelopes_are_closed_world_allowlists() -> None:
    counts = validate_pose_envelopes()
    assert counts["unitree_g1"] == {
        "candidate_count": 1458,
        "valid_count": 206,
        "invalid_count": 1252,
    }
    assert counts["tiago"] == {
        "candidate_count": 729,
        "valid_count": 47,
        "invalid_count": 682,
    }
    assert is_valid_configuration("unitree_g1", 14)
    assert not is_valid_configuration("unitree_g1", 43)
    assert is_valid_configuration("tiago", "front|horizontal|2")
    assert not is_valid_configuration("tiago", "front|horizontal|10")
