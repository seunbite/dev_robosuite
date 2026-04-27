from __future__ import annotations

import math
from dataclasses import dataclass

from .commands import FootTargetMap, JointMap
from .config import DEFAULT_CONFIG, JOINT_TYPES, LEG_ORDER, LEG_SIDE_SIGN, QuadrupedConfig

Vector3 = tuple[float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def add_vec(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_vec(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def lerp_vec(a: Vector3, b: Vector3, alpha: float) -> Vector3:
    return (lerp(a[0], b[0], alpha), lerp(a[1], b[1], alpha), lerp(a[2], b[2], alpha))


@dataclass
class QuadrupedKinematics:
    config: QuadrupedConfig = DEFAULT_CONFIG

    def forward_local(self, leg: str, joint_angles: tuple[float, float, float]) -> Vector3:
        q_abd, q_thigh, q_knee = joint_angles
        side = LEG_SIDE_SIGN[leg]
        l_hip = self.config.hip_offset
        l_thigh = self.config.thigh_length
        l_calf = self.config.calf_length

        x = l_thigh * math.sin(q_thigh) + l_calf * math.sin(q_thigh + q_knee)
        z_plane_down = l_thigh * math.cos(q_thigh) + l_calf * math.cos(q_thigh + q_knee)
        lateral = side * l_hip

        y = lateral * math.cos(q_abd) - z_plane_down * math.sin(q_abd)
        z_down = lateral * math.sin(q_abd) + z_plane_down * math.cos(q_abd)
        return (x, y, -z_down)

    def inverse_local(self, leg: str, foot_local: Vector3) -> tuple[float, float, float]:
        x, y, z = foot_local
        z_down = -z
        side = LEG_SIDE_SIGN[leg]
        l_hip = self.config.hip_offset
        l_thigh = self.config.thigh_length
        l_calf = self.config.calf_length

        radial_sq = y * y + z_down * z_down
        min_radius_sq = l_hip * l_hip + 1e-8
        if radial_sq < min_radius_sq:
            raise ValueError(f"Unreachable lateral target for {leg}: {foot_local}")

        z_plane_down = math.sqrt(radial_sq - l_hip * l_hip)
        lateral = side * l_hip
        q_abd = math.atan2(lateral * z_down - z_plane_down * y, lateral * y + z_plane_down * z_down)

        reach_sq = x * x + z_plane_down * z_plane_down
        cos_knee = clamp((reach_sq - l_thigh * l_thigh - l_calf * l_calf) / (2.0 * l_thigh * l_calf), -1.0, 1.0)
        q_knee = -math.acos(cos_knee)

        swing_angle = math.atan2(x, z_plane_down)
        knee_coupling = math.atan2(l_calf * math.sin(q_knee), l_thigh + l_calf * math.cos(q_knee))
        q_thigh = swing_angle - knee_coupling

        return (q_abd, q_thigh, q_knee)

    def foot_position(self, leg: str, joint_angles: tuple[float, float, float]) -> Vector3:
        return add_vec(self.config.hip_origin(leg), self.forward_local(leg, joint_angles))

    def inverse_leg(self, leg: str, foot_position_body: Vector3) -> tuple[float, float, float]:
        foot_local = sub_vec(foot_position_body, self.config.hip_origin(leg))
        return self.inverse_local(leg, foot_local)

    def nominal_stance(self, body_height: float | None = None) -> FootTargetMap:
        nominal: FootTargetMap = {}
        for leg in LEG_ORDER:
            nominal[leg] = self.foot_position(leg, self.config.default_joint_angles_for_leg())

        if body_height is None:
            return nominal

        target_z = -abs(body_height)
        current_z = nominal["FR"][2]
        z_shift = target_z - current_z
        return {leg: (pos[0], pos[1], pos[2] + z_shift) for leg, pos in nominal.items()}

    def feet_to_joint_map(self, foot_targets: FootTargetMap) -> JointMap:
        joint_map: JointMap = {}
        for leg in LEG_ORDER:
            q_abd, q_thigh, q_knee = self.inverse_leg(leg, foot_targets[leg])
            values = (q_abd, q_thigh, q_knee)
            for joint_type, value in zip(JOINT_TYPES, values, strict=True):
                joint_map[f"{leg}_{joint_type}_joint"] = value
        return joint_map

    def joint_map_to_feet(self, joint_map: JointMap) -> FootTargetMap:
        feet: FootTargetMap = {}
        for leg in LEG_ORDER:
            q_abd = joint_map[f"{leg}_hip_joint"]
            q_thigh = joint_map[f"{leg}_thigh_joint"]
            q_knee = joint_map[f"{leg}_calf_joint"]
            feet[leg] = self.foot_position(leg, (q_abd, q_thigh, q_knee))
        return feet
