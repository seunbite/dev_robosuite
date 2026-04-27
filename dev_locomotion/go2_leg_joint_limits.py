"""Unitree Go2 한 다리(3 DoF) 관절 한계 — `dev_robosuite/.../go2/robot.xml` 기준 (라디안).

앞다리(FR/FL): abduction(hip), front thigh pitch, knee.
뒷다리(RR/RL): abduction, back thigh pitch, knee (thigh 범위만 다름).
"""
from __future__ import annotations

# (low, high) rad
GO2_ABDUCTION_LIMIT = (-1.0472, 1.0472)
GO2_FRONT_THIGH_LIMIT = (-1.5708, 3.4907)
GO2_BACK_THIGH_LIMIT = (-0.5236, 4.5379)
GO2_KNEE_LIMIT = (-2.7227, -0.83776)


def joint_limits_for_leg(leg: str) -> dict[str, tuple[float, float]]:
    """leg별 MuJoCo joint 이름 → (lower, upper)."""
    if leg not in {"FR", "FL", "RR", "RL"}:
        raise ValueError(f"Unknown leg: {leg}")
    thigh_lim = GO2_FRONT_THIGH_LIMIT if leg in ("FR", "FL") else GO2_BACK_THIGH_LIMIT
    return {
        f"{leg}_hip_joint": GO2_ABDUCTION_LIMIT,
        f"{leg}_thigh_joint": thigh_lim,
        f"{leg}_calf_joint": GO2_KNEE_LIMIT,
    }


def describe_actuated_leg() -> str:
    return (
        "한 앞발(FR 등)을 움직일 때 **3개 관절**만 해당 다리에 대해 독립적으로 설정할 수 있습니다.\n"
        "  • hip_joint (abduction): 축 +X, 좌우로 벌림\n"
        "  • thigh_joint (hip pitch): 축 +Y, 몸통 대비 허벅지 앞뒤 스윙\n"
        "  • calf_joint (knee): 축 +Y, 무릎 굽힘 (항상 굽힌 방향의 음수 구간)\n"
        "나머지 세 다리는 기본 서보 자세를 유지합니다."
    )
