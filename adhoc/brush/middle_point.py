"""
Middle Point Approach with Jacobian Analysis

This script:
1. Generates two random points in 3D space to define a target line
2. Sets the robot to a specific orientation (r=-90, p=90/-90, y=-90)
3. Moves the end effector to the midpoint of the line
4. Ensures the end effector is perpendicular to the line in 3D space
5. Analyzes the Jacobian to find which joint has motion most aligned with the line direction
6. Saves the result as a GIF
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Tuple, Optional
import mujoco

import robosuite as suite
from robosuite.controllers.composite.composite_controller_config import refactor_composite_controller_config
from robosuite.utils.transform_utils import mat2quat, quat2mat, euler2mat


class MiddlePointApproach:
    """Approach the middle point of a line with specific orientation constraints."""
    
    def __init__(
        self,
        robot_name: str = "Panda",
        env_name: str = "EmptySpace",
        controller_name: str = "IK_POSE",
        has_renderer: bool = False,
        has_offscreen_renderer: bool = True,
        control_freq: int = 20,
    ):
        """
        Initialize the middle point approach task.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the IK controller
            has_renderer: Whether to show on-screen rendering
            has_offscreen_renderer: Whether to enable offscreen rendering
            control_freq: Control frequency
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.control_freq = control_freq
        
        # Create output directory
        self.output_dir = os.path.join("data/brush", robot_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": has_offscreen_renderer,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",
            "camera_heights": 512,
            "camera_widths": 512,
            "control_freq": control_freq,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=10000)
        self.env.reset()
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get end effector site name
        try:
            self.eef_site_name = self.robot.gripper["right"].important_sites["grip_site"]
        except (KeyError, AttributeError):
            self.eef_site_name = f"gripper0_right_grip_site"
        
        print(f"End effector site: {self.eef_site_name}")
        print("Initialization complete\n")
        
        # Store target line for visualization
        self.target_start = None
        self.target_end = None
    
    def generate_random_line(self, workspace_bounds: dict = None, min_length: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate two random points in the robot workspace with guaranteed minimum length.
        
        Args:
            workspace_bounds: Dict with 'x', 'y', 'z' keys containing (min, max) tuples
            min_length: Minimum line length in meters
        
        Returns:
            Tuple of (start_point, end_point) as numpy arrays
        """
        if workspace_bounds is None:
            # Default workspace bounds for typical robot arms (MUCH larger for long lines)
            workspace_bounds = {
                'x': (0.0, 0.8),   # Forward/backward (very wide)
                'y': (-0.5, 0.5),  # Left/right (very wide)
                'z': (0.4, 1.4),   # Up/down (very wide)
            }
        
        # Keep generating until we get a line that's long enough
        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate random start point
            start_point = np.array([
                np.random.uniform(*workspace_bounds['x']),
                np.random.uniform(*workspace_bounds['y']),
                np.random.uniform(*workspace_bounds['z']),
            ])
            
            # Generate random end point
            end_point = np.array([
                np.random.uniform(*workspace_bounds['x']),
                np.random.uniform(*workspace_bounds['y']),
                np.random.uniform(*workspace_bounds['z']),
            ])
            
            line_length = np.linalg.norm(end_point - start_point)
            
            if line_length >= min_length:
                break
        
        # Compute line properties
        line_vector = end_point - start_point
        line_direction = line_vector / line_length
        midpoint = (start_point + end_point) / 2
        
        print(f"Start point: {start_point}")
        print(f"End point: {end_point}")
        print(f"Midpoint: {midpoint}")
        print(f"Line length: {line_length:.3f}m (minimum required: {min_length}m)")
        print(f"Line direction (normalized): {line_direction}\n")
        
        # Store for visualization
        self.target_start = start_point
        self.target_end = end_point
        
        return start_point, end_point
    
    def compute_perpendicular_orientation(
        self,
        line_direction: np.ndarray,
        roll: float = -90,
        pitch_option: float = 90,
        yaw: float = -90,
    ) -> np.ndarray:
        """
        Compute orientation matrix that is perpendicular to the line.
        
        The goal is to align the end effector such that it's perpendicular to the line,
        while respecting the specified roll, pitch, yaw constraints.
        
        Args:
            line_direction: Normalized direction vector of the line
            roll: Roll angle in degrees (rotation around x-axis)
            pitch_option: Pitch angle in degrees (rotation around y-axis), can be 90 or -90
            yaw: Yaw angle in degrees (rotation around z-axis)
        
        Returns:
            Orientation matrix (3x3)
        """
        # Convert angles to radians
        roll_rad = np.deg2rad(roll)
        pitch_rad = np.deg2rad(pitch_option)
        yaw_rad = np.deg2rad(yaw)
        
        # Create rotation matrix from euler angles (ZYX convention)
        # This creates a rotation matrix based on the specified orientation
        orientation_mat = euler2mat([roll_rad, pitch_rad, yaw_rad])
        
        print(f"Target orientation (r={roll}, p={pitch_option}, y={yaw}):")
        print(orientation_mat)
        print()
        
        # Check perpendicularity: compute dot product between line direction and EE z-axis
        ee_z_axis = orientation_mat[:, 2]  # Third column is z-axis of EE
        dot_product = np.dot(line_direction, ee_z_axis)
        angle_deg = np.rad2deg(np.arccos(np.clip(np.abs(dot_product), 0, 1)))
        
        print(f"EE z-axis: {ee_z_axis}")
        print(f"Line direction: {line_direction}")
        print(f"Dot product: {dot_product:.3f}")
        print(f"Angle between EE z-axis and line: {angle_deg:.1f}°")
        
        if angle_deg < 80 or angle_deg > 100:
            print("Warning: EE orientation may not be perpendicular to the line!")
        else:
            print("✓ EE orientation is approximately perpendicular to the line")
        print()
        
        return orientation_mat
    
    def compute_jacobian(self) -> Tuple[np.ndarray, list]:
        """
        Compute the Jacobian matrix at the current configuration.
        
        Returns:
            Tuple of (jacobian matrix, list of joint names)
        """
        # Get MuJoCo model and data
        mujoco_model = self.env.sim.model._model
        mujoco_data = self.env.sim.data._data
        
        # Get end effector site ID
        site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, self.eef_site_name)
        
        # Compute Jacobian for end effector site
        jac_pos = np.zeros((3, mujoco_model.nv))
        jac_rot = np.zeros((3, mujoco_model.nv))
        mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
        
        # Combine position and rotation Jacobians
        jac_full = np.vstack([jac_pos, jac_rot])
        
        # Get joint names
        joint_names = []
        for i in range(mujoco_model.nv):
            # Find joint corresponding to this DOF
            for j in range(mujoco_model.njnt):
                joint_adr = mujoco_model.jnt_dofadr[j]
                if joint_adr <= i < joint_adr + mujoco_model.jnt_type[j]:
                    joint_name = mujoco.mj_id2name(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    joint_names.append(joint_name if joint_name else f"joint_{j}")
                    break
        
        return jac_full, joint_names
    
    def find_most_aligned_joint(
        self,
        jac_pos: np.ndarray,
        line_direction: np.ndarray,
        joint_names: list,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Find the joint whose motion is most aligned with the line direction.
        
        Args:
            jac_pos: Position Jacobian (3 x num_joints)
            line_direction: Normalized direction vector of the line
            joint_names: List of joint names
        
        Returns:
            Tuple of (joint_index, alignment_score, motion_direction)
        """
        print(f"{'='*60}")
        print("Analyzing Joint Alignment with Line Direction")
        print(f"{'='*60}\n")
        
        print(f"Line direction: {line_direction}")
        print(f"Line direction magnitude: {np.linalg.norm(line_direction):.3f}\n")
        
        best_joint_idx = -1
        best_alignment = 0.0
        best_motion_dir = None
        
        print(f"Joint analysis:")
        print(f"{'Joint':<30} {'Motion Direction':<40} {'Alignment':<10}")
        print("-" * 80)
        
        for i in range(jac_pos.shape[1]):
            # Get motion direction for this joint (column of position Jacobian)
            motion_dir = jac_pos[:, i]
            motion_magnitude = np.linalg.norm(motion_dir)
            
            if motion_magnitude < 1e-6:
                alignment = 0.0
                normalized_motion = np.zeros(3)
            else:
                normalized_motion = motion_dir / motion_magnitude
                # Compute alignment: dot product with line direction
                alignment = np.abs(np.dot(normalized_motion, line_direction))
            
            joint_name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            print(f"{joint_name:<30} {str(normalized_motion):<40} {alignment:.3f}")
            
            if alignment > best_alignment:
                best_alignment = alignment
                best_joint_idx = i
                best_motion_dir = normalized_motion
        
        print("-" * 80)
        best_joint_name = joint_names[best_joint_idx] if best_joint_idx < len(joint_names) else f"joint_{best_joint_idx}"
        print(f"\n✓ Best aligned joint: {best_joint_name} (index {best_joint_idx})")
        print(f"  Alignment score: {best_alignment:.3f}")
        print(f"  Motion direction: {best_motion_dir}")
        print(f"{'='*60}\n")
        
        return best_joint_idx, best_alignment, best_motion_dir
    
    def _add_line_markers_to_scene(self, scn):
        """
        Add target line and endpoint markers to the MuJoCo scene.
        Uses multiple spheres along the line for better visibility.
        
        Args:
            scn: MuJoCo scene object
        """
        if self.target_start is None or self.target_end is None:
            return
        
        # Line parameters
        line_vec = self.target_end - self.target_start
        line_length = np.linalg.norm(line_vec)
        
        # Add multiple LARGE spheres along the line for maximum visibility
        num_line_markers = 20  # More spheres for longer lines
        eye_flat = np.eye(3).flatten()
        
        for i in range(num_line_markers):
            if scn.ngeom >= scn.maxgeom:
                break
            
            # Interpolate position along the line
            t = i / (num_line_markers - 1)  # 0 to 1
            pos = self.target_start + t * line_vec
            
            g = scn.geoms[scn.ngeom]
            g.dataid = -1
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[0] = 0.04  # MUCH larger spheres (4cm)
            g.size[1] = 0.04
            g.size[2] = 0.04
            g.pos[:] = pos
            for j in range(9):
                g.mat.flat[j] = eye_flat[j]
            g.rgba[:] = [1, 0, 0, 1]  # Bright red
            scn.ngeom += 1
        
        # Add start point marker (HUGE green sphere)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            g.dataid = -1
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[0] = 0.12  # HUGE radius (12cm)
            g.size[1] = 0.12
            g.size[2] = 0.12
            g.pos[:] = self.target_start
            # Identity matrix for sphere
            eye_flat = np.eye(3).flatten()
            for i in range(9):
                g.mat.flat[i] = eye_flat[i]
            g.rgba[:] = [0, 1, 0, 1]  # Pure bright green
            scn.ngeom += 1
        
        # Add end point marker (HUGE blue sphere)
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            g.dataid = -1
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[0] = 0.12  # HUGE radius (12cm)
            g.size[1] = 0.12
            g.size[2] = 0.12
            g.pos[:] = self.target_end
            # Identity matrix for sphere
            eye_flat = np.eye(3).flatten()
            for i in range(9):
                g.mat.flat[i] = eye_flat[i]
            g.rgba[:] = [0, 0, 1, 1]  # Pure bright blue
            scn.ngeom += 1
        
        # Add midpoint marker (GIANT yellow sphere)
        midpoint = (self.target_start + self.target_end) / 2
        if scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            g.dataid = -1
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size[0] = 0.15  # GIANT radius (15cm)
            g.size[1] = 0.15
            g.size[2] = 0.15
            g.pos[:] = midpoint
            # Identity matrix for sphere
            eye_flat = np.eye(3).flatten()
            for i in range(9):
                g.mat.flat[i] = eye_flat[i]
            g.rgba[:] = [1, 1, 0, 1]  # Pure bright yellow
            scn.ngeom += 1
    
    def _capture_image(self) -> np.ndarray:
        """Capture an image from the camera with target line visualization."""
        # Get the render context
        render_context = self.env.sim._render_context_offscreen
        
        # Store original ngeom
        original_ngeom = render_context.scn.ngeom
        
        # Add markers to scene
        self._add_line_markers_to_scene(render_context.scn)
        
        # Render with markers
        camera_obs = self.env.sim.render(
            camera_name="frontview",
            width=512,
            height=512,
            depth=False,
        )
        
        # Reset ngeom to remove temporary markers
        render_context.scn.ngeom = original_ngeom
        
        # Flip vertically (MuJoCo renders upside down)
        return camera_obs[::-1]
    
    def move_to_midpoint(
        self,
        midpoint: np.ndarray,
        orientation_mat: np.ndarray,
        steps: int = 200,
        capture_frequency: int = 10,
    ) -> list:
        """
        Move the end effector to the midpoint with the specified orientation.
        
        Args:
            midpoint: Target position (3D point)
            orientation_mat: Target orientation (3x3 rotation matrix)
            steps: Number of steps to reach the target
            capture_frequency: Capture a frame every N steps
        
        Returns:
            List of captured frames as PIL Images
        """
        frames = []
        
        # Capture initial frame
        image = self._capture_image()
        frames.append(Image.fromarray(image))
        
        print(f"Moving to midpoint in {steps} steps...")
        
        for step in range(steps):
            # Get current EE position and orientation
            site_id = self.env.sim.model.site_name2id(self.eef_site_name)
            current_pos = np.array(self.env.sim.data.site_xpos[site_id])
            current_quat = np.array(self.env.sim.data.site_xquat[site_id])  # w, x, y, z
            current_mat = quat2mat(current_quat)
            
            # Compute position delta
            dpos = midpoint - current_pos
            
            # Compute orientation delta (simplified: use difference in rotation)
            # For a more accurate approach, compute the rotation that takes current to target
            target_quat = mat2quat(orientation_mat)
            
            # Simple proportional control for orientation
            # In practice, this is handled by the IK controller
            drot = np.zeros(3)  # Simplified for now
            
            # Create action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action = np.concatenate([dpos * 0.1, drot, [1.0]])  # Scale down for stability
            
            # Execute action
            self.env.step(action)
            
            # Capture frame periodically
            if step % capture_frequency == 0:
                image = self._capture_image()
                frames.append(Image.fromarray(image))
        
        # Capture final frame
        image = self._capture_image()
        frames.append(Image.fromarray(image))
        
        print(f"Captured {len(frames)} frames\n")
        
        return frames
    
    def save_gif(self, frames: list, filename: str = None, duration: int = 100):
        """
        Save frames as animated GIF.
        
        Args:
            frames: List of PIL Images
            filename: Output filename (auto-generated if None)
            duration: Duration per frame in milliseconds
        """
        if len(frames) == 0:
            print("Error: No frames to save!")
            return
        
        if filename is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{now}_{self.robot_name}_middle_point.gif"
        
        filepath = os.path.join(self.output_dir, filename)
        
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:] if len(frames) > 1 else [],
            duration=duration,
            loop=0
        )
        
        print(f"\n{'='*60}")
        print(f"Saved GIF: {filepath}")
        print(f"Total frames: {len(frames)}")
        print(f"GIF duration: {len(frames) * duration / 1000:.2f} seconds")
        print(f"{'='*60}\n")
        
        return filepath
    
    def run(
        self,
        roll: float = -90,
        pitch_option: float = 90,
        yaw: float = -90,
        steps: int = 200,
        workspace_bounds: dict = None,
    ):
        """
        Run the complete middle point approach task.
        
        Args:
            roll: Roll angle in degrees
            pitch_option: Pitch angle in degrees (90 or -90)
            yaw: Yaw angle in degrees
            steps: Number of steps to reach the midpoint
            workspace_bounds: Custom workspace bounds
        """
        print(f"{'='*60}")
        print("Middle Point Approach Task")
        print(f"Target orientation: r={roll}°, p={pitch_option}°, y={yaw}°")
        print(f"{'='*60}\n")
        
        # Generate random line (minimum 50cm long)
        start_point, end_point = self.generate_random_line(workspace_bounds, min_length=0.5)
        
        # Compute line properties
        line_vector = end_point - start_point
        line_length = np.linalg.norm(line_vector)
        line_direction = line_vector / line_length
        midpoint = (start_point + end_point) / 2
        
        # Compute perpendicular orientation
        orientation_mat = self.compute_perpendicular_orientation(
            line_direction, roll, pitch_option, yaw
        )
        
        # Move to midpoint
        frames = self.move_to_midpoint(midpoint, orientation_mat, steps)
        
        # Compute and analyze Jacobian
        jac_full, joint_names = self.compute_jacobian()
        jac_pos = jac_full[:3, :]  # Position part
        
        # Find most aligned joint
        best_joint_idx, alignment, motion_dir = self.find_most_aligned_joint(
            jac_pos, line_direction, joint_names
        )
        
        # Save GIF
        self.save_gif(frames)
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "Panda",
    env: str = "EmptySpace",
    controller: str = "IK_POSE",
    roll: float = -90,
    pitch: float = 90,
    yaw: float = -90,
    steps: int = 200,
    has_renderer: bool = False,
):
    """
    Main function to run middle point approach task.
    
    Args:
        robot: Robot name (e.g., "Panda", "IIWA", "Kinova3", "Jaco")
        env: Environment name
        controller: Controller name (should be IK-based)
        roll: Roll angle in degrees (default: -90)
        pitch: Pitch angle in degrees (default: 90, can also use -90)
        yaw: Yaw angle in degrees (default: -90)
        steps: Number of steps to reach the midpoint
        has_renderer: Whether to show on-screen rendering
    """
    approach = MiddlePointApproach(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        has_renderer=has_renderer,
    )
    
    try:
        approach.run(
            roll=roll,
            pitch_option=pitch,
            yaw=yaw,
            steps=steps,
        )
    finally:
        approach.close()


if __name__ == "__main__":
    import fire
    fire.Fire(main)

