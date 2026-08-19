from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "adhoc/generation/robotarm/drag_draw.py"
SPEC = importlib.util.spec_from_file_location("drag_draw", MODULE_PATH)
drag_draw = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(drag_draw)


@pytest.mark.parametrize("platform", drag_draw.PLATFORMS)
def test_fewshots_convert_to_valid_vectors(platform: str) -> None:
    source = json.loads(drag_draw.SHOT_PATH.read_text(encoding="utf-8"))
    schema = drag_draw.load_schema()
    for row in source[platform]:
        program = drag_draw.row_to_vector_program(platform, row, schema)
        drag_draw.validate_program(program, schema)
        assert program["trajectories"]
        for track in program["trajectories"]:
            assert set(track) == {"handle", "vectors"}
            assert all(len(vector) == 4 for vector in track["vectors"])


def test_interpolation_encodes_speed_and_hold_in_frames() -> None:
    program = {
        "version": "2.0", "cue": "nod_yes", "platform": "humanoid", "fps": 20,
        "trajectories": [{
            "handle": "head",
            "vectors": [
                [-0.004, 0.0, 1.224, 0],
                [0.010, 0.0, 1.170, 4],
                [-0.010, 0.0, 1.280, 8],
                [-0.010, 0.0, 1.280, 14],
            ],
        }],
    }
    frames = drag_draw.interpolate_program(program)
    assert len(frames) == 15
    assert frames[8]["head"] == [-0.01, 0.0, 1.28]
    assert frames[14]["head"] == frames[8]["head"]  # hold has no separate attribute


def test_rejects_shape_or_speed_style_vector() -> None:
    program = {
        "version": "2.0", "cue": "bad", "platform": "roomba", "fps": 20,
        "trajectories": [{"handle": "head", "vectors": [[0, 0, 0, 0]]}],
    }
    with pytest.raises(ValueError, match="unavailable handle"):
        drag_draw.validate_program(program)


def test_roomba_front_point_recovers_yaw() -> None:
    program = {
        "version": "2.0", "cue": "turn", "platform": "roomba", "fps": 20,
        "trajectories": [
            {"handle": "base", "vectors": [[0.0, 0.0, 0.09, 0], [0.0, 0.0, 0.09, 10]]},
            {"handle": "base_front", "vectors": [[0.15, 0.0, 0.09, 0], [0.0, 0.15, 0.09, 10]]},
        ],
    }
    payload = drag_draw.program_to_xyzrpy(program)
    assert payload["xyzrpy"][-1][5] == pytest.approx(1.57079632679)


def test_humanoid_payload_only_contains_authored_handles() -> None:
    schema = drag_draw.load_schema()
    neutral = schema["humanoid"]["neutral_handles"]["right_ee"]
    program = {
        "version": "2.0", "cue": "raising_hand_greeting", "platform": "humanoid", "fps": 20,
        "trajectories": [{
            "handle": "right_ee",
            "vectors": [[*neutral, 0], [0.22, -0.29, 1.30, 12]],
        }],
    }
    payload = drag_draw.program_to_xyzrpy(program, schema)
    assert payload["link_names"] == ["right_hand"]
    assert len(payload["xyzrpy"][0]) == 6


def test_three_point_hand_frame_is_orthogonalized_without_moving_center() -> None:
    schema = drag_draw.load_schema()
    hand0 = schema["humanoid"]["neutral_handles"]["right_ee"]
    tip0 = schema["humanoid"]["neutral_handles"]["right_fingertip"]
    palm0 = schema["humanoid"]["neutral_handles"]["right_palm_normal"]
    hand1 = [0.20, -0.25, 1.20]
    program = {
        "version": "2.0", "cue": "point", "platform": "humanoid", "fps": 20,
        "trajectories": [
            {"handle": "right_ee", "vectors": [[*hand0, 0], [*hand1, 10]]},
            # Deliberately mismatched frames and an arbitrary 0.4 m anchor length.
            {"handle": "right_fingertip", "vectors": [[*tip0, 0], [0.60, -0.25, 1.20, 7]]},
            # Deliberately collinear with the fingertip at a different frame.
            {"handle": "right_palm_normal", "vectors": [[*palm0, 0], [0.60, -0.25, 1.20, 9]]},
        ],
    }
    payload = drag_draw.program_to_xyzrpy(program, schema)
    assert payload["link_names"] == ["right_hand"]
    assert payload["xyzrpy"][-1][:3] == hand1
    direction = payload["fingertip_directions"]["right_hand"][-1]
    palm = payload["palm_directions"]["right_hand"][-1]
    assert sum(v * v for v in direction) == pytest.approx(1.0)
    assert sum(v * v for v in palm) == pytest.approx(1.0)
    assert sum(a * b for a, b in zip(direction, palm)) == pytest.approx(0.0, abs=1e-6)


def test_fingertip_requires_matching_hand_center() -> None:
    schema = drag_draw.load_schema()
    tip0 = schema["tiago"]["neutral_handles"]["right_fingertip"]
    program = {
        "version": "2.0", "cue": "bad", "platform": "tiago", "fps": 20,
        "trajectories": [{"handle": "right_fingertip", "vectors": [[*tip0, 0]]}],
    }
    with pytest.raises(ValueError, match="requires matching right_ee"):
        drag_draw.validate_program(program, schema)


def test_two_point_hand_orientation_is_rejected() -> None:
    schema = drag_draw.load_schema()
    hand0 = schema["humanoid"]["neutral_handles"]["right_ee"]
    tip0 = schema["humanoid"]["neutral_handles"]["right_fingertip"]
    program = {
        "version": "2.0", "cue": "bad", "platform": "humanoid", "fps": 20,
        "trajectories": [
            {"handle": "right_ee", "vectors": [[*hand0, 0]]},
            {"handle": "right_fingertip", "vectors": [[*tip0, 0]]},
        ],
    }
    with pytest.raises(ValueError, match="also requires right_palm_normal"):
        drag_draw.validate_program(program, schema)


def test_tiago_uses_two_point_hand_direction() -> None:
    schema = drag_draw.load_schema()
    hand0 = schema["tiago"]["neutral_handles"]["right_ee"]
    tip0 = schema["tiago"]["neutral_handles"]["right_fingertip"]
    program = {
        "version": "2.0", "cue": "point", "platform": "tiago", "fps": 20,
        "trajectories": [
            {"handle": "right_ee", "vectors": [[*hand0, 0], [0.2, -0.3, 1.0, 10]]},
            {"handle": "right_fingertip", "vectors": [[*tip0, 0], [0.3, -0.3, 1.0, 10]]},
        ],
    }
    payload = drag_draw.program_to_xyzrpy(program, schema)
    assert "right_hand" in payload["fingertip_directions"]
    assert not payload["palm_directions"]


def test_single_arm_uses_four_point_frame() -> None:
    schema = drag_draw.load_schema()
    ee0 = schema["single_arm"]["neutral_handles"]["ee"]
    elbow0 = schema["single_arm"]["neutral_handles"]["elbow"]
    tip0 = schema["single_arm"]["neutral_handles"]["fingertip"]
    palm0 = schema["single_arm"]["neutral_handles"]["palm_normal"]
    program = {
        "version": "2.0", "cue": "reach", "platform": "single_arm", "fps": 20,
        "trajectories": [
            {"handle": "ee", "vectors": [[*ee0, 0], [0.6, 0.1, 1.2, 10]]},
            {"handle": "elbow", "vectors": [[*elbow0, 0], [0.3, 0.2, 1.5, 8]]},
            {"handle": "fingertip", "vectors": [[*tip0, 0], [0.6, 0.1, 1.08, 10]]},
            {"handle": "palm_normal", "vectors": [[*palm0, 0], [0.6, 0.18, 1.2, 10]]},
        ],
    }
    payload = drag_draw.program_to_xyzrpy(program, schema)
    assert payload["xyzrpy"][-1][:3] == [0.6, 0.1, 1.2]
    assert payload["elbow_targets"][-1] == [0.3, 0.2, 1.5]
    assert "upper_arm_targets" not in payload
    assert "forearm_targets" not in payload
    assert "wrist_targets" not in payload
    assert "end_effector" in payload["fingertip_directions"]
    assert "end_effector" in payload["palm_directions"]


def test_stretch_uses_ee_plus_fingertip() -> None:
    schema = drag_draw.load_schema()
    ee0 = schema["stretch"]["neutral_handles"]["ee"]
    tip0 = schema["stretch"]["neutral_handles"]["fingertip"]
    program = {
        "version": "2.0", "cue": "point", "platform": "stretch", "fps": 20,
        "trajectories": [
            {"handle": "ee", "vectors": [[*ee0, 0], [0.5, 0.0, 0.8, 10]]},
            {"handle": "fingertip", "vectors": [[*tip0, 0], [0.5, 0.0, 0.92, 10]]},
        ],
    }
    drag_draw.validate_program(program, schema)
    payload = drag_draw.program_to_xyzrpy(program, schema)
    assert "end_effector" in payload["fingertip_directions"]
