"""Environment wrapper for Code as Policies."""

import numpy as np
from shapely.geometry import box
from shapely.geometry import *
from shapely.affinity import *

class LMPWrapper:
    """Wrapper class for environment to support Language Model Programming."""

    def __init__(self, env, cfg, render=False):
        self.env = env
        self._cfg = cfg
        self.object_names = list(self._cfg['env']['init_objs'])
        
        self._min_xy = np.array(self._cfg['env']['coords']['bottom_left'])
        self._max_xy = np.array(self._cfg['env']['coords']['top_right'])
        self._range_xy = self._max_xy - self._min_xy

        self._table_z = self._cfg['env']['coords']['table_z']
        self.render = render

    def is_obj_visible(self, obj_name):
        """Check if object is visible in the scene."""
        return obj_name in self.object_names

    def get_obj_names(self):
        """Get list of all object names."""
        return self.object_names[::]

    def denormalize_xy(self, pos_normalized):
        """Convert normalized coordinates to robot base frame."""
        return pos_normalized * self._range_xy + self._min_xy

    def get_corner_positions(self):
        """Get corner positions in robot base frame."""
        unit_square = box(0, 0, 1, 1)
        normalized_corners = np.array(list(unit_square.exterior.coords))[:4]
        corners = np.array(([self.denormalize_xy(corner) for corner in normalized_corners]))
        return corners

    def get_side_positions(self):
        """Get side positions in robot base frame."""
        side_xs = np.array([0, 0.5, 0.5, 1])
        side_ys = np.array([0.5, 0, 1, 0.5])
        normalized_side_positions = np.c_[side_xs, side_ys]
        side_positions = np.array(([self.denormalize_xy(corner) for corner in normalized_side_positions]))
        return side_positions

    def get_obj_pos(self, obj_name):
        """Get object position in robot base frame."""
        return self.env.get_obj_pos(obj_name)[:2]

    def get_obj_position_np(self, obj_name):
        """Get object position as numpy array."""
        return self.get_pos(obj_name)

    def get_bbox(self, obj_name):
        """Get object bounding box in robot base frame."""
        bbox = self.env.get_bounding_box(obj_name)
        return bbox

    def get_color(self, obj_name):
        """Get object color."""
        for color, rgb in COLORS.items():
            if color in obj_name:
                return rgb

    def pick_place(self, pick_pos, place_pos):
        """Execute pick and place action."""
        pick_pos_xyz = np.r_[pick_pos, [self._table_z]]
        place_pos_xyz = np.r_[place_pos, [self._table_z]]
        self.env.step(action={'pick': pick_pos_xyz, 'place': place_pos_xyz})

    def put_first_on_second(self, arg1, arg2):
        """Put first object on top of second object/position."""
        pick_pos = self.get_obj_pos(arg1) if isinstance(arg1, str) else arg1
        place_pos = self.get_obj_pos(arg2) if isinstance(arg2, str) else arg2
        self.pick_place(pick_pos, place_pos)

    def get_robot_pos(self):
        """Get robot end-effector position."""
        return self.env.get_ee_pos()

    def goto_pos(self, position_xy):
        """Move robot end-effector to desired position."""
        ee_xyz = self.env.get_ee_pos()
        position_xyz = np.concatenate([position_xy, ee_xyz[-1]])
        while np.linalg.norm(position_xyz - ee_xyz) > 0.01:
            self.env.movep(position_xyz)
            self.env.step_sim_and_render()
            ee_xyz = self.env.get_ee_pos()

    def follow_traj(self, traj):
        """Follow trajectory of positions."""
        for pos in traj:
            self.goto_pos(pos)

    def get_corner_name(self, pos):
        """Get name of corner position."""
        corner_positions = self.get_corner_positions()
        corner_idx = np.argmin(np.linalg.norm(corner_positions - pos, axis=1))
        return ['top left corner', 'top right corner', 'bottom left corner', 'bottom right corner'][corner_idx]

    def get_side_name(self, pos):
        """Get name of side position."""
        side_positions = self.get_side_positions()
        side_idx = np.argmin(np.linalg.norm(side_positions - pos, axis=1))
        return ['top side', 'right side', 'bottom side', 'left side'][side_idx]
