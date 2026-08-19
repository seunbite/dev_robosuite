"""Shared world-axis joint convention for embodiment Motion Code.

DSL ``move(shoulder|elbow|wrist, x|y|z, ±deg)`` means:
  - pick the DOF in that joint group most aligned with the **world** axis
  - **+deg** moves the EE in the **+** world direction (+x forward, +y left, +z up)
  - **−deg** moves the EE in the **−** world direction
"""
from __future__ import annotations


def jac_world_sign(jac_component: float) -> float:
    """+1 if a positive joint delta increases the world coordinate, else −1."""
    return 1.0 if float(jac_component) >= 0.0 else -1.0


def align_offset_to_world_axis(offset: float, jac_component: float) -> float:
    """Map config offset so +config moves EE along +world axis.

    ``offset`` is already in the same units as the joint qpos delta (rad or m).
    """
    return float(offset) * jac_world_sign(jac_component)
