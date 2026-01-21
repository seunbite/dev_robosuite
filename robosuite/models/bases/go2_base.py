import numpy as np

from robosuite.models.bases.leg_base_model import LegBaseModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Go2Base(LegBaseModel):
    """
    Unitree Go2 Quadruped Robot Base.
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/go2/robot.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.array((0, 0, 0))

    @property
    def horizontal_radius(self):
        return 0.25

    @property
    def init_qpos(self):
        # Initial angles from the XML home keyframe: 0 0.9 -1.8 for each leg
        return np.array([0.0, 0.9, -1.8] * 4)
