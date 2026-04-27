from __future__ import annotations

from dataclasses import dataclass

LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_TYPES = ("hip", "thigh", "calf")
JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in LEG_ORDER for joint in JOINT_TYPES)
LEG_SIDE_SIGN = {"FR": -1, "RR": -1, "FL": 1, "RL": 1}
LEG_FRONT_SIGN = {"FR": 1, "FL": 1, "RR": -1, "RL": -1}


@dataclass(frozen=True)
class QuadrupedConfig:
    base_length: float = 0.3868
    base_width: float = 0.1840
    hip_offset: float = 0.0955
    thigh_length: float = 0.2130
    calf_length: float = 0.2300
    nominal_base_height: float = 0.30
    default_abduction: float = 0.0
    default_thigh: float = 0.85
    default_calf: float = -1.65
    default_joint_kp: tuple[float, float, float] = (70.0, 150.0, 150.0)
    default_joint_kd: tuple[float, float, float] = (3.0, 3.5, 3.5)
    torque_limit: float = 33.0
    simulation_dt: float = 0.002
    control_dt: float = 0.02
    body_mass: float = 8.5

    def hip_origin(self, leg: str) -> tuple[float, float, float]:
        x = LEG_FRONT_SIGN[leg] * self.base_length / 2.0
        y = LEG_SIDE_SIGN[leg] * self.base_width / 2.0
        return (x, y, 0.0)

    def default_joint_angles_for_leg(self) -> tuple[float, float, float]:
        return (self.default_abduction, self.default_thigh, self.default_calf)

    def default_joint_map(self) -> dict[str, float]:
        joint_map: dict[str, float] = {}
        for leg in LEG_ORDER:
            abduction, thigh, calf = self.default_joint_angles_for_leg()
            joint_map[f"{leg}_hip_joint"] = abduction
            joint_map[f"{leg}_thigh_joint"] = thigh
            joint_map[f"{leg}_calf_joint"] = calf
        return joint_map


DEFAULT_CONFIG = QuadrupedConfig()
