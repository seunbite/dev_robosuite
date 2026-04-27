from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..commands import FootTargetMap, JointMap, LowLevelCommand
from ..config import JOINT_ORDER, JOINT_TYPES, LEG_ORDER
from ..kinematics import QuadrupedKinematics, clamp


@dataclass
class LowLevelController:
    kinematics: QuadrupedKinematics = field(default_factory=QuadrupedKinematics)
    torque_limit: float | None = None

    def __post_init__(self) -> None:
        if self.torque_limit is None:
            self.torque_limit = self.kinematics.config.torque_limit

    def default_gain_map(self, gain_type: str, scale: float = 1.0) -> JointMap:
        gains = (
            self.kinematics.config.default_joint_kp
            if gain_type == "kp"
            else self.kinematics.config.default_joint_kd
        )
        gain_map: JointMap = {}
        for leg in LEG_ORDER:
            for joint_type, gain in zip(JOINT_TYPES, gains, strict=True):
                gain_map[f"{leg}_{joint_type}_joint"] = gain * scale
        return gain_map

    def build_joint_position_command(
        self,
        joint_positions: Mapping[str, float],
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        feedforward_tau: Mapping[str, float] | None = None,
    ) -> LowLevelCommand:
        tau = {joint: 0.0 for joint in JOINT_ORDER}
        if feedforward_tau is not None:
            tau.update({joint: float(feedforward_tau.get(joint, 0.0)) for joint in JOINT_ORDER})
        return LowLevelCommand(
            q={joint: float(joint_positions[joint]) for joint in JOINT_ORDER},
            dq={joint: 0.0 for joint in JOINT_ORDER},
            tau=tau,
            kp=self.default_gain_map("kp", kp_scale),
            kd=self.default_gain_map("kd", kd_scale),
            mask={joint: True for joint in JOINT_ORDER},
        )

    def build_foot_command(
        self,
        foot_targets: FootTargetMap,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> LowLevelCommand:
        joint_positions = self.kinematics.feet_to_joint_map(foot_targets)
        return self.build_joint_position_command(joint_positions, kp_scale=kp_scale, kd_scale=kd_scale)

    def compute_torques(
        self,
        command: LowLevelCommand,
        qpos: Mapping[str, float],
        qvel: Mapping[str, float],
    ) -> JointMap:
        torques: JointMap = {}
        for joint in JOINT_ORDER:
            if not command.mask.get(joint, True):
                torques[joint] = 0.0
                continue

            q = float(qpos.get(joint, 0.0))
            dq = float(qvel.get(joint, 0.0))
            q_des = float(command.q.get(joint, q))
            dq_des = float(command.dq.get(joint, 0.0))
            kp = float(command.kp.get(joint, 0.0))
            kd = float(command.kd.get(joint, 0.0))
            tau_ff = float(command.tau.get(joint, 0.0))

            tau = kp * (q_des - q) + kd * (dq_des - dq) + tau_ff
            torques[joint] = clamp(tau, -float(self.torque_limit), float(self.torque_limit))
        return torques
