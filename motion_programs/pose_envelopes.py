"""Closed-world pose-envelope allowlists for anthropomorphic embodiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = _REPO / "data/paper/pose_envelopes/manifest.json"


def load_pose_envelopes(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load the canonical paper pose-envelope manifest."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalise_id(embodiment: str, configuration_id: int | str) -> int | str:
    if embodiment == "unitree_g1":
        return int(configuration_id)
    if embodiment == "tiago":
        return str(configuration_id).strip().lower().replace("_", "|")
    raise KeyError(f"unsupported pose envelope: {embodiment}")


def is_valid_configuration(
    embodiment: str,
    configuration_id: int | str,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> bool:
    """Return True only when the configuration ID is in the paper allowlist."""
    embodiment = embodiment.strip().lower()
    manifest = load_pose_envelopes(manifest_path)
    block = manifest["embodiments"][embodiment]
    requested = _normalise_id(embodiment, configuration_id)
    return requested in set(block["valid_configuration_ids"])


def validate_pose_envelopes(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, dict[str, int]]:
    """Cross-check allowlists against the full checked-in configuration records."""
    manifest_path = Path(manifest_path)
    manifest = load_pose_envelopes(manifest_path)
    if manifest.get("validity_policy") != "closed_world_allowlist":
        raise ValueError("pose-envelope manifest must use closed_world_allowlist")

    root = manifest_path.parent
    result: dict[str, dict[str, int]] = {}
    for embodiment, block in manifest["embodiments"].items():
        expected = block["valid_configuration_ids"]
        if len(expected) != len(set(expected)):
            raise ValueError(f"{embodiment}: duplicate configuration IDs")
        if len(expected) != int(block["valid_count"]):
            raise ValueError(f"{embodiment}: valid_count does not match allowlist")

        records_path = (root / block["configuration_records"]).resolve()
        if embodiment == "unitree_g1":
            payload = json.loads(records_path.read_text(encoding="utf-8"))
            actual = [int(row["pose_id"]) for row in payload["poses"]]
            if int(payload["n_poses"]) != len(actual):
                raise ValueError("unitree_g1: n_poses does not match pose records")
        elif embodiment == "tiago":
            rows = [
                json.loads(line)
                for line in records_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            actual = [
                f"{row['arm_position']}|{row['gripper_orientation']}|{row['tile_idx']}"
                for row in rows
            ]
        else:
            raise KeyError(f"unsupported pose envelope: {embodiment}")

        if set(actual) != set(expected):
            raise ValueError(f"{embodiment}: configuration records differ from allowlist")
        manual_invalid = set(block.get("explicit_manual_invalid_configuration_ids", []))
        if manual_invalid.intersection(expected):
            raise ValueError(f"{embodiment}: valid and manual-invalid IDs overlap")
        candidate_count = int(block["candidate_count"])
        valid_count = int(block["valid_count"])
        if int(block["invalid_count"]) != candidate_count - valid_count:
            raise ValueError(f"{embodiment}: invalid_count is inconsistent")
        result[embodiment] = {
            "candidate_count": candidate_count,
            "valid_count": valid_count,
            "invalid_count": candidate_count - valid_count,
        }
    return result
