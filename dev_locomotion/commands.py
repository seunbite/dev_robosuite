from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .config import JOINT_ORDER

JointMap = dict[str, float]
FootTargetMap = dict[str, tuple[float, float, float]]


def expand_joint_value(value: float | Mapping[str, float], default: float = 0.0) -> JointMap:
    if isinstance(value, Mapping):
        return {joint: float(value.get(joint, default)) for joint in JOINT_ORDER}
    return {joint: float(value) for joint in JOINT_ORDER}


@dataclass
class LowLevelCommand:
    q: JointMap = field(default_factory=dict)
    dq: JointMap = field(default_factory=dict)
    tau: JointMap = field(default_factory=dict)
    kp: JointMap = field(default_factory=dict)
    kd: JointMap = field(default_factory=dict)
    mask: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def hold_position(
        cls,
        joint_positions: Mapping[str, float],
        kp: float | Mapping[str, float],
        kd: float | Mapping[str, float],
        tau: float | Mapping[str, float] = 0.0,
    ) -> "LowLevelCommand":
        return cls(
            q={joint: float(joint_positions[joint]) for joint in JOINT_ORDER},
            dq=expand_joint_value(0.0),
            tau=expand_joint_value(tau),
            kp=expand_joint_value(kp),
            kd=expand_joint_value(kd),
            mask={joint: True for joint in JOINT_ORDER},
        )


@dataclass(frozen=True)
class VelocityCommand:
    vx: float
    vy: float = 0.0
    wz: float = 0.0
    body_height: float | None = None
    cycle_period: float = 0.55
    step_height: float = 0.055
    stride_scale: float = 1.0
