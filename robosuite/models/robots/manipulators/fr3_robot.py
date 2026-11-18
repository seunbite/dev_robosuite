"""Franka Research 3 (FR3) Arm.

Kinematics and control match the 7-DoF Franka family used by DROID
(Panda joint order, 855 mm reach, 3 kg payload). Pose lookup reuses the
Panda bank. Hardware on the user's desk is FR3 Arm v2.0 (manual R02210)
with a Robotiq 2F-85 on the flange (not the Panda Hand).
"""

import numpy as np

from robosuite.models.robots.manipulators.panda_robot import Panda

# Panda pose-bank joints were recorded with Panda Hand (right_hand already
# -45°). DroidRobotiq85Gripper adds +45° on the adapter, so bank q7 must
# shift by the opposite amount for motion-code grip=horiz/vert to keep the
# same visual opening (floor-horizontal vs up-down).
BANK_Q7_OFFSET_RAD = -np.pi / 4.0
_Q7_LO = -2.8973
_Q7_HI = 2.8973


def apply_bank_gripper_clock(q: np.ndarray) -> np.ndarray:
    """Shift wrist q7 so Panda-bank poses match the DROID Robotiq clock.

    2F-85 opening is 180°-symmetric, so wrap by π when q7 would saturate.
    """
    out = np.asarray(q, dtype=np.float64).copy()
    if out.ndim == 1:
        q7 = out[6] + BANK_Q7_OFFSET_RAD
        if q7 < _Q7_LO:
            q7 += np.pi
        elif q7 > _Q7_HI:
            q7 -= np.pi
        out[6] = np.clip(q7, _Q7_LO, _Q7_HI)
    else:
        q7 = out[..., 6] + BANK_Q7_OFFSET_RAD
        q7 = np.where(q7 < _Q7_LO, q7 + np.pi, q7)
        q7 = np.where(q7 > _Q7_HI, q7 - np.pi, q7)
        out[..., 6] = np.clip(q7, _Q7_LO, _Q7_HI)
    return out


class FR3(Panda):
    """Franka Research 3 arm registered as a robosuite robot.

    Default gripper is the DROID Robotiq 2F-85, clocked +45° about the
    tool axis so the jaws follow the Franka flange (Panda ``right_hand``
    is already -45°). ``init_qpos`` is the measured desk-neutral, not
    Panda HOME. Pose-bank q7 is shifted by ``BANK_Q7_OFFSET_RAD``.
    """

    @property
    def default_gripper(self):
        return {"right": "DroidRobotiq85Gripper"}

    @property
    def init_qpos(self):
        # Live FR3 desk-neutral, 2026-09-09 (DROID RobotEnv observation).
        return np.array(
            [
                -0.03462224826216698,
                -0.6726680397987366,
                0.044236764311790466,
                -2.5911834239959717,
                0.048997968435287476,
                1.9046064615249634,
                0.05532360449433327,
            ]
        )
