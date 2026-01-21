import numpy as np
from robosuite.models.robots.manipulators.legged_manipulator_model import LeggedManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion

class Go2(LeggedManipulatorModel):
    """
    Go2 is a quadruped robot from Unitree.
    This model represents the robot without an arm.
    """

    arms = []

    def __init__(self, idn=0):
        # We use the same XML as the base since it's a pure quadruped
        super().__init__(xml_path_completion("robots/go2/robot.xml"), idn=idn)

    @property
    def legs_joints(self):
        return self.correct_naming([
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ])

    @property
    def legs_actuators(self):
        return self.correct_naming([
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ])

    @property
    def default_base(self):
        return "NullBase"

    @property
    def default_controller_config(self):
        return {
            "legs": "default_go2",
        }

    @property
    def init_qpos(self):
        # 12 leg joints
        return np.array([0.0, 0.9, -1.8] * 4)

    @property
    def arm_type(self):
        return "none"

    @property
    def eef_name(self):
        return {}

    @property
    def base_xpos_offset(self):
        return {
            "bins": (0, 0, 0),
            "empty": (0, 0, 0),
            "table": lambda table_length: (0, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 0.3))

    @property
    def _horizontal_radius(self):
        return 0.3
