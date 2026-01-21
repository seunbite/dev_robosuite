"""
Inverse Kinematics Brush Drawing

This script:
1. Generates two random points in 3D space
2. Creates a target line connecting them
3. Divides the line into 50 segments
4. Uses inverse kinematics to follow the line with the robot's end effector
5. Saves the result as a GIF
"""

import os
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Tuple
import mujoco

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class IKBrushDrawer:
    """Use IK to draw a line in 3D space."""
    
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
        Initialize the IK brush drawer.
        
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
        
        print(f"Start point: {start_point}")
        print(f"End point: {end_point}")
        print(f"Line length: {line_length:.3f}m (minimum required: {min_length}m)\n")
        
        # Store for visualization
        self.target_start = start_point
        self.target_end = end_point
        
        return start_point, end_point
    
    def interpolate_line(self, start: np.ndarray, end: np.ndarray, num_points: int = 50) -> np.ndarray:
        """
        Interpolate points along a line.
        
        Args:
            start: Start point (3D)
            end: End point (3D)
            num_points: Number of interpolation points
        
        Returns:
            Array of shape (num_points, 3) with interpolated points
        """
        # Create linear interpolation
        t = np.linspace(0, 1, num_points)
        points = start[None, :] + t[:, None] * (end - start)[None, :]
        
        return points
    
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
    
    def draw_line_with_ik(
        self,
        target_points: np.ndarray,
        orientation: np.ndarray = None,
        steps_per_point: int = 20,
    ) -> list:
        """
        Follow the target line using IK control.
        
        Args:
            target_points: Array of shape (N, 3) with target positions
            orientation: Target orientation as rotation matrix (3x3). If None, keeps current orientation
            steps_per_point: Number of simulation steps per waypoint
        
        Returns:
            List of captured frames as PIL Images
        """
        frames = []
        
        print(f"Drawing line with {len(target_points)} waypoints...")
        print(f"Steps per point: {steps_per_point}")
        
        # Get current end effector position
        current_pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.eef_site_name)])
        print(f"Initial EE position: {current_pos}\n")
        
        # If orientation not specified, use fixed downward-pointing orientation
        if orientation is None:
            # Default: gripper pointing down
            orientation = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
        
        # Move to each target point
        for i, target_pos in enumerate(target_points):
            if i % 10 == 0:
                print(f"Waypoint {i}/{len(target_points)}: {target_pos}")
            
            # Get current EE position
            current_pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.eef_site_name)])
            
            # Compute position delta
            dpos = target_pos - current_pos
            
            # Orientation stays fixed (drot = 0)
            drot = np.zeros(3)
            
            # Create action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            action = np.concatenate([dpos, drot, [1.0]])  # gripper open
            
            # Execute action for multiple steps
            for _ in range(steps_per_point):
                self.env.step(action)
                
                # Update current position for next iteration
                current_pos = np.array(self.env.sim.data.site_xpos[self.env.sim.model.site_name2id(self.eef_site_name)])
                dpos = target_pos - current_pos
                action = np.concatenate([dpos, drot, [1.0]])
            
            # Capture frame every few waypoints
            image = self._capture_image()
            frames.append(Image.fromarray(image))
        
        print(f"\nCaptured {len(frames)} frames")
        
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
            filename = f"{now}_{self.robot_name}_ik_brush.gif"
        
        filepath = os.path.join(self.output_dir, filename)
        
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:] if len(frames) > 1 else [],
            duration=duration,
            loop=0  # Infinite loop
        )
        
        print(f"\n{'='*60}")
        print(f"Saved GIF: {filepath}")
        print(f"Total frames: {len(frames)}")
        print(f"GIF duration: {len(frames) * duration / 1000:.2f} seconds")
        print(f"{'='*60}\n")
        
        return filepath
    
    def run(
        self,
        num_points: int = 50,
        steps_per_point: int = 20,
        workspace_bounds: dict = None,
    ):
        """
        Run the complete IK brush drawing task.
        
        Args:
            num_points: Number of points to divide the line into
            steps_per_point: Number of simulation steps per waypoint
            workspace_bounds: Custom workspace bounds
        """
        print(f"{'='*60}")
        print("IK Brush Drawing Task")
        print(f"{'='*60}\n")
        
        # Generate random line (minimum 50cm long)
        start_point, end_point = self.generate_random_line(workspace_bounds, min_length=0.5)
        
        # Interpolate line
        target_points = self.interpolate_line(start_point, end_point, num_points)
        print(f"Generated {len(target_points)} target points\n")
        
        # Draw line with IK
        frames = self.draw_line_with_ik(target_points, steps_per_point=steps_per_point)
        
        # Save GIF
        self.save_gif(frames)
    
    def close(self):
        """Close the environment."""
        self.env.close()


def main(
    robot: str = "Panda",
    env: str = "EmptySpace",
    controller: str = "IK_POSE",
    num_points: int = 50,
    steps_per_point: int = 20,
    has_renderer: bool = False,
):
    """
    Main function to run IK brush drawing.
    
    Args:
        robot: Robot name (e.g., "Panda", "IIWA", "Kinova3", "Jaco")
        env: Environment name
        controller: Controller name (should be IK-based)
        num_points: Number of points to divide the line into
        steps_per_point: Number of simulation steps per waypoint
        has_renderer: Whether to show on-screen rendering
    """
    drawer = IKBrushDrawer(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        has_renderer=has_renderer,
    )
    
    try:
        drawer.run(
            num_points=num_points,
            steps_per_point=steps_per_point,
        )
    finally:
        drawer.close()


if __name__ == "__main__":
    import fire
    fire.Fire(main)

