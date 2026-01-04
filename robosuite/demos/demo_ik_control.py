"""
This demo shows how to use the IKSolver class for robot control.
It divides the workspace into a 3x3x3 grid (27 cells) and allows selecting
a cell to move the end effector to using IK.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.utils.input_utils import *
from robosuite.utils.ik_utils import IKSolver
import robosuite.utils.transform_utils as T
import mujoco

MAX_FR = 25  # max frame rate for running simulation

GRID_SIZE = 3  # 3x3x3 grid = 27 cells
# Voxel size (each voxel is a cube with this side length)
VOXEL_SIZE = 10  # 10cm cubes

class WorkspaceGrid:
    """Manages 3x3x3 workspace grid division with cubic voxels"""
    
    def __init__(self, center_pos, voxel_size=0.1, grid_size=3):
        """
        Initialize workspace grid centered at end effector position
        
        Args:
            center_pos: np.array of shape (3,) with [x, y, z] center position
            voxel_size: size of each cubic voxel (default: 0.1m = 10cm)
            grid_size: number of divisions per axis (default: 3)
        """
        self.center_pos = np.array(center_pos)
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.num_cells = grid_size ** 3
        
        # Calculate total workspace size
        total_size = grid_size * voxel_size
        
        # Calculate bounds centered at center_pos
        half_size = total_size / 2
        self.bounds = np.array([
            [center_pos[0] - half_size, center_pos[0] + half_size],  # X
            [center_pos[1] - half_size, center_pos[1] + half_size],  # Y
            [center_pos[2] - half_size, center_pos[2] + half_size],  # Z
        ])
        
        # Cell size is the same for all dimensions (cubic voxels)
        self.cell_size = np.array([voxel_size, voxel_size, voxel_size])
    
    def get_cell_corners(self, cell_id):
        """
        Get the 8 corners of a cell (voxel)
        
        Args:
            cell_id: cell ID from 1 to 27
            
        Returns:
            np.array: (8, 3) array of corner positions
        """
        info = self.get_cell_info(cell_id)
        bounds = info['bounds']
        
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']
        z_min, z_max = bounds['z']
        
        # 8 corners of a box
        corners = np.array([
            [x_min, y_min, z_min],  # 0: bottom-left-back
            [x_max, y_min, z_min],  # 1: bottom-right-back
            [x_max, y_max, z_min],  # 2: bottom-right-front
            [x_min, y_max, z_min],  # 3: bottom-left-front
            [x_min, y_min, z_max],  # 4: top-left-back
            [x_max, y_min, z_max],  # 5: top-right-back
            [x_max, y_max, z_max],  # 6: top-right-front
            [x_min, y_max, z_max],  # 7: top-left-front
        ])
        
        return corners
    
    def get_cell_edges(self, cell_id):
        """
        Get the 12 edges of a cell (voxel) as line segments
        
        Args:
            cell_id: cell ID from 1 to 27
            
        Returns:
            list: List of 12 edge pairs, each edge is (start_pos, end_pos)
        """
        corners = self.get_cell_corners(cell_id)
        
        # 12 edges of a box (connect corners)
        edges = [
            # Bottom face (4 edges)
            (corners[0], corners[1]),  # back edge
            (corners[1], corners[2]),  # right edge
            (corners[2], corners[3]),  # front edge
            (corners[3], corners[0]),  # left edge
            # Top face (4 edges)
            (corners[4], corners[5]),  # back edge
            (corners[5], corners[6]),  # right edge
            (corners[6], corners[7]),  # front edge
            (corners[7], corners[4]),  # left edge
            # Vertical edges (4 edges)
            (corners[0], corners[4]),  # back-left
            (corners[1], corners[5]),  # back-right
            (corners[2], corners[6]),  # front-right
            (corners[3], corners[7]),  # front-left
        ]
        
        return edges
        
    def get_cell_center(self, cell_id):
        """
        Get the center position of a cell
        
        Args:
            cell_id: cell ID from 1 to 26 (1-indexed) or 0 to 25 (0-indexed)
            
        Returns:
            np.array: [x, y, z] position of cell center
        """
        # Convert to 0-indexed if needed (assuming 1-indexed input)
        if 1 <= cell_id <= self.num_cells:
            idx = cell_id - 1
        elif 0 <= cell_id < self.num_cells:
            idx = cell_id
        else:
            raise ValueError(f"Cell ID must be between 1 and {self.num_cells} (1-indexed) or 0 and {self.num_cells-1} (0-indexed)")
        
        # Convert linear index to 3D grid coordinates
        i = idx // (self.grid_size ** 2)  # Z index
        j = (idx // self.grid_size) % self.grid_size  # Y index
        k = idx % self.grid_size  # X index
        
        # Calculate cell center position (cubic voxels)
        # Offset from center: (k - 1, j - 1, i - 1) * voxel_size for grid_size=3
        # This makes cell 14 (middle) at center_pos
        offset = np.array([
            (k - (self.grid_size - 1) / 2) * self.voxel_size,
            (j - (self.grid_size - 1) / 2) * self.voxel_size,
            (i - (self.grid_size - 1) / 2) * self.voxel_size,
        ])
        
        return self.center_pos + offset
    
    def get_cell_info(self, cell_id):
        """Get information about a cell"""
        center = self.get_cell_center(cell_id)
        # Convert to 0-indexed for calculation
        if 1 <= cell_id <= self.num_cells:
            idx = cell_id - 1
        else:
            idx = cell_id
        i = idx // (self.grid_size ** 2)
        j = (idx // self.grid_size) % self.grid_size
        k = idx % self.grid_size
        
        # Calculate bounds for cubic voxel
        half_voxel = self.voxel_size / 2
        bounds = {
            "x": [center[0] - half_voxel, center[0] + half_voxel],
            "y": [center[1] - half_voxel, center[1] + half_voxel],
            "z": [center[2] - half_voxel, center[2] + half_voxel]
        }
        
        return {
            "cell_id": cell_id,
            "grid_coords": (k, j, i),  # (x, y, z) in grid
            "center": center,
            "bounds": bounds
        }
    
    def print_all_cells(self):
        """Print information about all cells"""
        print("\nAll cells in workspace:")
        print("-" * 60)
        for cell_id in range(1, self.num_cells + 1):
            info = self.get_cell_info(cell_id)
            print(f"Cell {cell_id:2d}: Grid({info['grid_coords']}) -> Pos{info['center']}")

class IKController:
    def __init__(self, env, robot_config):
        """
        Initialize IK controller
        
        Args:
            env: robosuite environment
            robot_config: robot configuration dictionary containing:
                - joint_names: list of joint names
                - end_effector_sites: list of end effector site names
                - initial_keyframe: name of initial keyframe (optional)
                - nullspace_gains: list of nullspace gains for each joint
        """
        self.env = env
        self.model = env.sim.model
        self.data = env.sim.data
        
        # Initialize IK solver
        self.ik_solver = IKSolver(
            model=self.model,
            data=self.data,
            robot_config=robot_config,
            damping=0.05,  # damping coefficient for pseudo-inverse
            integration_dt=1.0 / env.control_freq,
            max_dq=0.5,  # maximum joint velocity
            input_type="keyboard",  # we'll use keyboard input mode
            debug=False,
            input_action_repr="absolute",  # use absolute positioning
            input_rotation_repr="axis_angle"  # use axis-angle for rotation
        )
        
        # Visualization state
        self.visualize_grid_enabled = True
        self.grid_visualized = False  # Track if grid has been visualized
        
    def reset(self):
        """Reset robot to initial pose"""
        self.ik_solver.reset_to_initial_state()
        
    def move_to_position(self, target_pos, target_ori=None, steps=100):
        """
        Move end effector to target position using IK
        
        Args:
            target_pos: target position [x, y, z]
            target_ori: target orientation as axis-angle [rx, ry, rz] (optional, defaults to current orientation)
            steps: number of steps for smooth movement
        """
        # Get current orientation if not provided
        if target_ori is None:
            current_rot = self.data.site(self.ik_solver.site_ids[0]).xmat
            current_quat = T.mat2quat(current_rot.reshape(3, 3))
            # Convert to axis-angle (zero rotation)
            target_ori = np.array([0.0, 0.0, 0.0])
        
        # Create target action: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        target_action = np.concatenate([target_pos, target_ori])
        
        # Get current position for interpolation
        current_pos = self.data.site(self.ik_solver.site_ids[0]).xpos
        
        # Smooth interpolation
        for i in range(steps):
            alpha = (i + 1) / steps
            # Interpolate position
            interp_pos = current_pos * (1 - alpha) + target_pos * alpha
            # Keep orientation constant (or interpolate if needed)
            interp_ori = target_ori
            
            # Create action for this step
            action = np.concatenate([interp_pos, interp_ori])
            
            # Solve IK
            q_des = self.ik_solver.solve(action)
            
            # Set joint positions
            self.data.qpos[self.ik_solver.dof_ids] = q_des
            self.env.sim.forward()  # Update simulation
            
            # Step simulation
            self.env.step(np.zeros(self.env.robots[0].dof))  # Zero action since we directly set qpos
            
            # Draw coordinate axes at origin (if enabled)
            if self.visualize_grid_enabled and self.env.viewer is not None:
                self.draw_coordinate_axes(origin=np.array([0, 0, 0]), axis_length=0.05, axis_width=0.003)
            
            # Draw grid visualization BEFORE viewer update
            # We need to prepare markers before render() is called
            # For interactive viewer: we'll add markers directly to scene after mjv_updateScene
            # For passive viewer: we add markers after sync()
            if (self.visualize_grid_enabled and hasattr(self, 'grid') and 
                hasattr(self, 'target_cell_id') and self.env.viewer is not None):
                # Store marker info for later use
                if not hasattr(self, '_pending_markers'):
                    self._pending_markers = []
                
                target_info = self.grid.get_cell_info(self.target_cell_id)
                center = target_info['center']
                # Make sphere visible - use absolute size instead of relative to voxel_size
                # Voxel size might be very large (10), so use a reasonable absolute size
                sphere_radius = 5  # 5cm radius for visibility
                target_rgba = np.array([0.0, 1.0, 0.0, 1.0])
                
                self._pending_markers = [{
                    'pos': center,
                    'mat': np.eye(3),
                    'type': mujoco.mjtGeom.mjGEOM_SPHERE,
                    'size': np.array([sphere_radius, sphere_radius, sphere_radius]),
                    'rgba': target_rgba,
                }]
            
            # Update viewer (this calls viewer.sync() or viewer.render())
            # For interactive viewer, we need to hook into render() to add markers
            if hasattr(self.env.viewer, 'update'):
                self.env.viewer.update()
            
            # For interactive viewer, add markers after render() updates scene
            # For passive viewer, add markers after sync()
            if (self.visualize_grid_enabled and hasattr(self, '_pending_markers') and 
                self._pending_markers and self.env.viewer is not None):
                self.draw_grid_visualization(self.grid, self.target_cell_id)
                self._pending_markers = []  # Clear after adding
            
            # Render (for passive viewer, this is needed)
            self.env.render()
            
            # Limit frame rate
            time.sleep(1.0 / MAX_FR)
    
    def _add_marker_to_scene(self, scn, marker):
        """
        Add a marker to the scene (based on mujoco_custom_viewer.py)
        
        Args:
            scn: MuJoCo scene object
            marker: dict with marker parameters (pos, mat, type, size, rgba, etc.)
        """
        if scn.ngeom >= scn.maxgeom:
            return False  # No room for more markers
        
        g = scn.geoms[scn.ngeom]
        # Set default values (based on mujoco_custom_viewer.py)
        # Note: Some attributes were removed in MuJoCo 3.x (texid, texuniform, etc.)
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        
        # Set texture-related attributes only if they exist (MuJoCo 3.x removed some)
        if hasattr(g, 'texuniform'):
            g.texuniform = 0
        if hasattr(g, 'texrepeat'):
            g.texrepeat[0] = 1
            g.texrepeat[1] = 1
        
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)
        
        # Apply marker parameters
        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                if key == "label":
                    if value is None:
                        g.label[0] = 0
                    else:
                        g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
        
        scn.ngeom += 1
        return True
    
    def draw_grid_visualization(self, grid, target_cell_id=None):
        """
        Draw workspace grid visualization - simplified version using sphere for target
        
        Args:
            grid: WorkspaceGrid instance
            target_cell_id: cell ID to highlight (1-27), None to highlight none
        """
        if not hasattr(self.env, 'viewer') or self.env.viewer is None:
            return
        
        # Get the actual MuJoCo viewer object
        actual_viewer = None
        if hasattr(self.env.viewer, 'viewer'):
            actual_viewer = self.env.viewer.viewer
        else:
            return
        
        # Note: mujoco_viewer's _markers has compatibility issues with MuJoCo 3.x (texid removed)
        # So we'll use scene-based marker addition for both interactive and passive viewers
        # We add markers AFTER viewer.update() which calls render() or sync()
        if not hasattr(actual_viewer, 'scn'):
            return
        
        scn = actual_viewer.scn
        if scn is None:
            return
        
        # For interactive viewer, scene is already updated by render() in viewer.update()
        # For passive viewer, scene is updated by sync() in viewer.update()
        # So we can directly add markers to the scene without calling mjv_updateScene again
        
        # Check if we have room for markers
        if scn.ngeom >= scn.maxgeom - 10:
            return
        
        # Draw sphere for target cell (if specified)
        if target_cell_id is not None:
            target_info = grid.get_cell_info(target_cell_id)
            center = target_info['center']
            # Make sphere visible - use absolute size instead of relative to voxel_size
            sphere_radius = 5  # 5cm radius for visibility
            target_rgba = np.array([0.0, 1.0, 0.0, 1.0])  # Green, fully opaque
            
            marker = {
                'pos': center,
                'mat': np.eye(3),
                'type': mujoco.mjtGeom.mjGEOM_SPHERE,
                'size': np.array([sphere_radius, sphere_radius, sphere_radius]),
                'rgba': target_rgba,
            }
            success = self._add_marker_to_scene(scn, marker)
    
    def _clear_grid_markers(self, actual_viewer=None):
        """Clear all grid visualization markers by resetting scene ngeom"""
        # For MuJoCo passive viewer, we don't need to clear markers
        # because we add them fresh each frame and they're automatically cleared
        # when the scene is rebuilt
        pass
    
    def draw_coordinate_axes(self, origin=np.array([0, 0, 0]), axis_length=0.1, axis_width=0.005):
        """
        Draw coordinate axes (X, Y, Z) at specified origin
        
        Args:
            origin: np.array of shape (3,) with [x, y, z] origin position
            axis_length: length of each axis
            axis_width: width (radius) of each axis
        """
        if not hasattr(self.env, 'viewer') or self.env.viewer is None:
            return
        
        # Get the actual MuJoCo viewer object
        actual_viewer = None
        if hasattr(self.env.viewer, 'viewer'):
            actual_viewer = self.env.viewer.viewer
        else:
            return
        
        # MuJoCo passive viewer uses scene to add markers
        if not hasattr(actual_viewer, 'scn'):
            return
        
        scn = actual_viewer.scn
        if scn is None:
            return
        
        # Get model and data from viewer or environment
        model = None
        data = None
        if hasattr(actual_viewer, 'model') and hasattr(actual_viewer, 'data'):
            model = actual_viewer.model
            data = actual_viewer.data
        else:
            # Fallback: use environment's model and data
            model = self.env.sim.model._model if hasattr(self.env.sim.model, '_model') else self.env.sim.model
            data = self.env.sim.data._data if hasattr(self.env.sim.data, '_data') else self.env.sim.data
        
        # Get opt and cam from viewer
        opt = actual_viewer.opt if hasattr(actual_viewer, 'opt') else None
        cam = actual_viewer.cam if hasattr(actual_viewer, 'cam') else None
        
        if opt is None or cam is None or model is None or data is None:
            return
        
        # Update scene first
        try:
            mujoco.mjv_updateScene(
                model,
                data,
                opt,
                None,  # pert
                cam,
                mujoco.mjtCatBit.mjCAT_ALL.value if hasattr(mujoco.mjtCatBit.mjCAT_ALL, 'value') else mujoco.mjtCatBit.mjCAT_ALL,
                scn
            )
        except Exception:
            pass
        
        # Check if we have room for 3 axis markers
        if scn.ngeom >= scn.maxgeom - 3:
            return
        
        # X-axis: Red, pointing in +X direction
        # Create rotation matrix to align cylinder with X-axis
        # X-axis is already aligned with world X, so we need to rotate cylinder (which points in Z) to point in X
        R_x = np.array([
            [0, 0, 1],  # cylinder's Z becomes world's X
            [1, 0, 0],  # cylinder's X becomes world's Y
            [0, 1, 0]   # cylinder's Y becomes world's Z
        ])
        p_x = origin + np.array([axis_length / 2, 0, 0])  # Center of cylinder at half length along X
        
        marker_x = {
            'pos': p_x,
            'mat': R_x,
            'type': mujoco.mjtGeom.mjGEOM_CYLINDER,
            'size': np.array([axis_width, axis_width, axis_length / 2]),
            'rgba': np.array([1.0, 0.0, 0.0, 0.9]),  # Red
        }
        self._add_marker_to_scene(scn, marker_x)
        
        # Y-axis: Green, pointing in +Y direction
        # Rotate cylinder to point in Y direction
        R_y = np.array([
            [0, 1, 0],  # cylinder's Z becomes world's Y
            [0, 0, 1],  # cylinder's X becomes world's Z
            [1, 0, 0]   # cylinder's Y becomes world's X
        ])
        p_y = origin + np.array([0, axis_length / 2, 0])  # Center of cylinder at half length along Y
        
        marker_y = {
            'pos': p_y,
            'mat': R_y,
            'type': mujoco.mjtGeom.mjGEOM_CYLINDER,
            'size': np.array([axis_width, axis_width, axis_length / 2]),
            'rgba': np.array([0.0, 1.0, 0.0, 0.9]),  # Green
        }
        self._add_marker_to_scene(scn, marker_y)
        
        # Z-axis: Blue, pointing in +Z direction
        # Cylinder already points in Z direction, so identity rotation
        R_z = np.eye(3)
        p_z = origin + np.array([0, 0, axis_length / 2])  # Center of cylinder at half length along Z
        
        marker_z = {
            'pos': p_z,
            'mat': R_z,
            'type': mujoco.mjtGeom.mjGEOM_CYLINDER,
            'size': np.array([axis_width, axis_width, axis_length / 2]),
            'rgba': np.array([0.0, 0.0, 1.0, 0.9]),  # Blue
        }
        self._add_marker_to_scene(scn, marker_z)

def main(
    selected_cell: int = 17,
    robot: str = "Panda",
    visualize_grid: bool = True,
):
    # Create dict to hold options that will be passed to env creation call
    options = {}

    # Choose environment and add it to options
    options["env_name"] = "Stack"
    options["robots"] = robot

    # Initialize the task
    env = suite.make(
        **options,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    
    # Reset environment
    env.reset()
    env.viewer.set_camera(camera_id=0)
    
    # Get robot and joint names
    robot = env.robots[0]
    # Get joint names from robot model
    try:
        joint_names = list(robot.robot_model.joints)
    except:
        # Fallback: use joint indexes and convert to names
        joint_names = [env.sim.model.joint_id2name(idx) for idx in robot.joint_indexes]
    
    # Create robot configuration
    # Get end effector site name from gripper
    try:
        eef_site_name = robot.gripper["right"].important_sites["grip_site"]
    except:
        # Fallback: use standard naming convention
        eef_site_name = "gripper0_right_grip_site"
    
    robot_config = {
        "joint_names": joint_names,
        "end_effector_sites": [eef_site_name],
        "nullspace_gains": [1.0] * len(joint_names),
    }
    
    # Initialize controller
    controller = IKController(env, robot_config)
    controller.reset()
    # Store controller reference in environment for viewer access
    env._ik_controller = controller
    
    # Get initial end effector position to center the workspace
    # robot._hand_pos returns a dict with arm names as keys
    initial_eef_pos = robot._hand_pos["right"]
    print(f"Initial end effector position: {initial_eef_pos}")
    
    # Create workspace grid centered at initial end effector position
    grid = WorkspaceGrid(initial_eef_pos, voxel_size=VOXEL_SIZE, grid_size=GRID_SIZE)
    
    # Store grid and target cell for visualization
    controller.grid = grid
    controller.target_cell_id = selected_cell
    controller.visualize_grid_enabled = visualize_grid
    
    print("="*60)
    print("Workspace Grid IK Control Demo")
    print("="*60)
    print(f"Workspace center (initial EE position): {grid.center_pos}")
    print(f"Workspace bounds: X={grid.bounds[0]}, Y={grid.bounds[1]}, Z={grid.bounds[2]}")
    print(f"Grid size: {GRID_SIZE}x{GRID_SIZE}x{GRID_SIZE} = {grid.num_cells} cells")
    print(f"Voxel size (cubic): {grid.voxel_size}m x {grid.voxel_size}m x {grid.voxel_size}m")
    print("="*60)
    
    # Print all cells for reference
    grid.print_all_cells()
    
    # Example: Move to different cells
    # You can select cells from 1 to 26
    # To select a specific cell, change the cell_id below
    
    print(f"\n{'='*60}")
    print(f"Moving to cell {selected_cell}")
    print(f"{'='*60}")
    
    try:
        cell_info = grid.get_cell_info(selected_cell)
        print(f"Cell {selected_cell} information:")
        print(f"  Grid coordinates (x, y, z): {cell_info['grid_coords']}")
        print(f"  Target position: {cell_info['center']}")
        print(f"  Cell bounds:")
        print(f"    X: [{cell_info['bounds']['x'][0]:.3f}, {cell_info['bounds']['x'][1]:.3f}]")
        print(f"    Y: [{cell_info['bounds']['y'][0]:.3f}, {cell_info['bounds']['y'][1]:.3f}]")
        print(f"    Z: [{cell_info['bounds']['z'][0]:.3f}, {cell_info['bounds']['z'][1]:.3f}]")
        
        controller.move_to_position(
            target_pos=cell_info['center'],
            steps=75
        )
        
        print(f"\nSuccessfully moved to cell {selected_cell}!")
        
    except Exception as e:
        print(f"Error moving to cell {selected_cell}: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Demo finished!")
    print("="*60)
    print("\nTo select a different cell, modify 'selected_cell' in the code.")
    print("Or use:")
    print("  cell_info = grid.get_cell_info(cell_id)  # cell_id from 1 to 26")
    print("  controller.move_to_position(cell_info['center'])")
    
    # Keep window open for a moment
    time.sleep(2.0)
    env.close()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
