"""
6-DoF gripper with its open/close variant
"""
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import find_elements, string_to_array, xml_path_completion


class Robotiq85GripperBase(GripperModel):
    """
    6-DoF Robotiq gripper.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/robotiq_gripper_85.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([-0.026, -0.267, -0.200, -0.026, -0.267, -0.200])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [
                "left_outer_finger_collision",
                "left_inner_finger_collision",
                "left_fingertip_collision",
                "left_fingerpad_collision",
            ],
            "right_finger": [
                "right_outer_finger_collision",
                "right_inner_finger_collision",
                "right_fingertip_collision",
                "right_fingerpad_collision",
            ],
            "left_fingerpad": ["left_fingerpad_collision"],
            "right_fingerpad": ["right_fingerpad_collision"],
        }


class Robotiq85Gripper(Robotiq85GripperBase):
    """
    1-DoF variant of RobotiqGripperBase.
    """

    def format_action(self, action):
        """
        Maps continuous action into binary output
        -1 => open, 1 => closed

        Args:
            action (np.array): gripper-specific action

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        assert len(action) == 1
        self.current_action = np.clip(
            self.current_action * np.array([1.0, 1.0]) + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.20

    @property
    def dof(self):
        return 1


class DroidRobotiq85Gripper(Robotiq85Gripper):
    """Robotiq 2F-85 with an extra +45° yaw about the tool axis.

    Panda ``right_hand`` is already -45°. This adapter clock cancels that
    so the 2F-85 opening follows the Franka flange (left-right at the
    measured desk-neutral). Panda pose-bank joints still need
    ``FR3.apply_bank_gripper_clock`` (−45° on q7) so grip=horiz/vert
    keep the same visual opening.
    """

    _CLOCK_QUAT_WXYZ = "0.9238795 0 0 0.3826834"  # +45° about Z

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        self.worldbody[0].set("quat", self._CLOCK_QUAT_WXYZ)
        base_quat = np.fromstring(self.worldbody[0].get("quat", "1 0 0 0"), dtype=np.float64, sep=" ")[
            [1, 2, 3, 0]
        ]
        eef_element = find_elements(
            root=self.root, tags="body", attribs={"name": self.correct_naming("eef")}, return_first=True
        )
        eef_relative_quat = string_to_array(eef_element.get("quat", "1 0 0 0"))[[1, 2, 3, 0]]
        self.rotation_offset = T.quat_multiply(eef_relative_quat, base_quat)
