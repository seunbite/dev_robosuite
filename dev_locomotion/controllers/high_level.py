from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..commands import FootTargetMap, LowLevelCommand, VelocityCommand
from ..config import LEG_ORDER
from ..gestures import GestureLibrary
from ..kinematics import QuadrupedKinematics, add_vec, lerp
from .low_level import LowLevelController


@dataclass
class TrotPlanner:
    kinematics: QuadrupedKinematics = field(default_factory=QuadrupedKinematics)
    stance_ratio: float = 0.62
    cycle_period: float = 0.55
    max_stride_x: float = 0.12
    max_stride_y: float = 0.06
    default_step_height: float = 0.055
    nominal_feet: FootTargetMap = field(init=False)
    phase_offsets: dict[str, float] = field(
        default_factory=lambda: {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
    )

    def __post_init__(self) -> None:
        self.nominal_feet = self.kinematics.nominal_stance()

    def foot_targets(self, command: VelocityCommand, t: float) -> FootTargetMap:
        body_height = command.body_height or self.kinematics.config.nominal_base_height
        nominal_feet = self.kinematics.nominal_stance(body_height=body_height)
        cycle_period = max(command.cycle_period, 1e-3)
        step_height = max(command.step_height, 0.0)

        feet: FootTargetMap = {}
        for leg in LEG_ORDER:
            displacement = self._foot_displacement(command, leg, cycle_period)
            phase = ((t / cycle_period) + self.phase_offsets[leg]) % 1.0

            if phase < self.stance_ratio:
                alpha = phase / self.stance_ratio
                foot_delta = (
                    lerp(displacement[0], -displacement[0], alpha),
                    lerp(displacement[1], -displacement[1], alpha),
                    0.0,
                )
            else:
                alpha = (phase - self.stance_ratio) / (1.0 - self.stance_ratio)
                foot_delta = (
                    lerp(-displacement[0], displacement[0], alpha),
                    lerp(-displacement[1], displacement[1], alpha),
                    step_height * math.sin(math.pi * alpha),
                )

            feet[leg] = add_vec(nominal_feet[leg], foot_delta)
        return feet

    def _foot_displacement(self, command: VelocityCommand, leg: str, cycle_period: float) -> tuple[float, float]:
        nominal = self.nominal_feet[leg]
        swing_window = 0.5 * cycle_period * command.stride_scale
        forward = command.vx * swing_window
        lateral = command.vy * swing_window
        yaw_dx = -command.wz * nominal[1] * swing_window
        yaw_dy = command.wz * nominal[0] * swing_window
        dx = max(-self.max_stride_x, min(self.max_stride_x, forward + yaw_dx))
        dy = max(-self.max_stride_y, min(self.max_stride_y, lateral + yaw_dy))
        return (dx, dy)


@dataclass
class HighLevelController:
    low_level: LowLevelController = field(default_factory=LowLevelController)
    gait: TrotPlanner = field(init=False)
    gestures: GestureLibrary = field(init=False)

    def __post_init__(self) -> None:
        self.gait = TrotPlanner(self.low_level.kinematics)
        self.gestures = GestureLibrary(self.low_level.kinematics)

    def command_pose(self, name: str, kp_scale: float = 1.0, kd_scale: float = 1.0) -> LowLevelCommand:
        foot_targets = self.gestures.pose(name)
        return self.low_level.build_foot_command(foot_targets, kp_scale=kp_scale, kd_scale=kd_scale)

    def command_gesture(self, name: str, t: float, kp_scale: float = 1.0, kd_scale: float = 1.0) -> LowLevelCommand:
        foot_targets = self.gestures.sequence(name).sample(t)
        return self.low_level.build_foot_command(foot_targets, kp_scale=kp_scale, kd_scale=kd_scale)

    def command_walk(self, command: VelocityCommand, t: float, kp_scale: float = 1.0, kd_scale: float = 1.0) -> LowLevelCommand:
        foot_targets = self.gait.foot_targets(command, t)
        return self.low_level.build_foot_command(foot_targets, kp_scale=kp_scale, kd_scale=kd_scale)
