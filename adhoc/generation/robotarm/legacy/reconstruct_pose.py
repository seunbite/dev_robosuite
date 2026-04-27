"""
Reconstruct robot pose from image filename.

This script:
1. Parses pose information from image filename (e.g., Panda_pose_000708_j0+090_j1+090_j2+090_j3-090_j4+090_j5-090.png)
2. Initializes robot environment
3. Sets robot to the exact pose specified in the filename
4. Displays the robot in that pose
"""

import fire
import os
import re
import numpy as np
import time
from typing import Dict, List, Tuple

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config

# Fixed joint indices (matching stack_preset.py)
FIXED_JOINT_INDICES = {
    'GR1': "0-2, 20-31"
}


def parse_pose_from_filename(filename: str) -> Dict:
    """
    Parse pose information from filename.
    
    Expected format: {robot_name}_pose_{pose_id}_j{joint_idx}{angle}...png
    Example: Panda_pose_000708_j0+090_j1+090_j2+090_j3-090_j4+090_j5-090.png
    
    Returns:
        Dictionary with:
            - robot_name: str
            - pose_id: str
            - joint_angles_deg: list of floats
            - active_joint_indices: list of ints
    """
    # Extract base name without extension
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Pattern: {robot}_pose_{id}_j{idx}{sign}{angle}_j{idx}{sign}{angle}...
    # Match robot name and pose_id
    match = re.match(r'^([^_]+)_pose_(\d+)_(.+)$', name_without_ext)
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")
    
    robot_name = match.group(1)
    pose_id = match.group(2)
    joint_angles_str = match.group(3)
    
    # Parse joint angles: j0+090, j1+090, j2+090, etc.
    # Pattern: j{digit}{+ or -}{digits}
    joint_pattern = r'j(\d+)([+-])(\d+)'
    joint_matches = re.findall(joint_pattern, joint_angles_str)
    
    if not joint_matches:
        raise ValueError(f"Could not parse joint angles from: {joint_angles_str}")
    
    active_joint_indices = []
    joint_angles_deg = []
    
    for joint_idx_str, sign, angle_str in joint_matches:
        joint_idx = int(joint_idx_str)
        angle = int(angle_str)
        
        if sign == '-':
            angle = -angle
        
        active_joint_indices.append(joint_idx)
        joint_angles_deg.append(angle)
    
    return {
        'robot_name': robot_name,
        'pose_id': pose_id,
        'active_joint_indices': active_joint_indices,
        'joint_angles_deg': joint_angles_deg,
    }


class PoseReconstructor:
    """Reconstruct and display robot pose from filename."""
    
    def __init__(
        self,
        robot_name: str = 'Panda',
        env_name: str = "EmptySpace",
        controller_name: str = "OSC_POSE",
        has_renderer: bool = True,
    ):
        """
        Initialize pose reconstructor.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment (default: EmptySpace)
            controller_name: Name of the controller
            has_renderer: Whether to show renderer window
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.has_renderer = has_renderer
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": has_renderer,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "renderer": "mjviewer",  # Explicitly set renderer
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Initialize viewer if renderer is enabled
        if has_renderer:
            # Give viewer time to initialize
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                try:
                    self.env.viewer.set_camera(camera_id=0)
                    # Call update() to initialize and render the viewer window
                    # (render() alone doesn't work because MjviewerRenderer.render() is empty)
                    self.env.viewer.update()
                    time.sleep(0.3)  # Give time for window to open
                except Exception as e:
                    print(f"Warning: Could not set camera or render: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions (matching stack_preset.py exactly)
        self.initial_joint_pos = self.robot._joint_positions.copy()
        self.num_joints = len(self.initial_joint_pos)
        
        # Parse fixed joint indices (matching stack_preset.py)
        self.fixed_joint_indices = []
        if robot_name in FIXED_JOINT_INDICES:
            fixed_indices_str = FIXED_JOINT_INDICES[robot_name]
            try:
                for part in fixed_indices_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        start, end = part.split("-", 1)
                        start = int(start.strip())
                        end = int(end.strip())
                        if start > end:
                            continue
                        self.fixed_joint_indices.extend(range(start, end + 1))
                    else:
                        self.fixed_joint_indices.append(int(part))
                self.fixed_joint_indices = sorted(list(set([idx for idx in self.fixed_joint_indices if 0 <= idx < self.num_joints])))
            except ValueError:
                self.fixed_joint_indices = []
        
        print(f"Total joints: {self.num_joints}")
        if self.fixed_joint_indices:
            print(f"Fixed joints: {len(self.fixed_joint_indices)}")
        print(f"Robot initialized successfully!")
    
    def set_pose_from_filename(self, filename: str):
        """
        Set robot to pose specified in filename.
        
        Uses the EXACT same logic as stack_preset.py's generate_combination_poses:
        - stack_preset.py line 803: joint_pos = self.initial_joint_pos.copy()
        - stack_preset.py line 813: joint_pos[active_joint_idx] = angle_value
        - stack_preset.py line 323: self.robot.set_robot_joint_positions(robot_joint_pos)
        
        Args:
            filename: Image filename containing pose information
        """
        # Parse pose from filename
        pose_data = parse_pose_from_filename(filename)
        
        robot_name = pose_data['robot_name']
        pose_id = pose_data['pose_id']
        active_joint_indices = pose_data['active_joint_indices']
        joint_angles_deg = pose_data['joint_angles_deg']
        
        # Check robot name matches
        if robot_name != self.robot_name:
            print(f"Warning: Filename specifies robot '{robot_name}' but initialized with '{self.robot_name}'")
        
        # Convert to radians
        joint_angles_rad = [np.deg2rad(angle_deg) for angle_deg in joint_angles_deg]
        
        print(f"\n{'='*60}")
        print(f"RECONSTRUCTING POSE FROM FILENAME")
        print(f"{'='*60}")
        print(f"Robot: {robot_name}")
        print(f"Pose ID: {pose_id}")
        print(f"Active joint indices: {active_joint_indices}")
        print(f"Joint angles (deg): {joint_angles_deg}")
        print(f"Joint angles (rad): {[f'{r:.4f}' for r in joint_angles_rad]}")
        
        # Reconstruct full joint position array (matching stack_preset.py line 803 exactly)
        joint_pos = self.initial_joint_pos.copy()
        
        # Set positions for active joints (matching stack_preset.py line 810-813 exactly)
        for i, active_joint_idx in enumerate(active_joint_indices):
            if i < len(joint_angles_rad):
                if active_joint_idx < len(joint_pos):
                    joint_pos[active_joint_idx] = joint_angles_rad[i]
                else:
                    print(f"Warning: active_joint_idx {active_joint_idx} >= len(joint_pos) {len(joint_pos)}")
        
        # Set joint positions (matching stack_preset.py line 323 exactly)
        self.robot.set_robot_joint_positions(joint_pos)
        self.env.sim.forward()
        
        # Verify by reading back joint positions
        current_joint_pos = self.robot._joint_positions.copy()
        print(f"\nVerification - Current joint positions (deg) for active joints:")
        for i, active_idx in enumerate(active_joint_indices):
            if active_idx < len(current_joint_pos):
                current_deg = np.rad2deg(current_joint_pos[active_idx])
                expected_deg = joint_angles_deg[i]
                match = "✓" if abs(current_deg - expected_deg) < 1.0 else "✗"
                print(f"  [{active_idx}] {current_deg:+.1f}° (expected {expected_deg:+.1f}°) {match}")
        
        print(f"{'='*60}\n")
        
        # Render the pose continuously
        if self.has_renderer:
            print("Rendering pose. Close the window or press Ctrl+C to exit.")
            # Use viewer.update() directly since env.render() calls viewer.render() which is empty
            if hasattr(self.env, 'viewer') and self.env.viewer is not None:
                # Initial update to ensure window is open
                self.env.viewer.update()
                time.sleep(0.3)  # Give time for window to open
                
                try:
                    while True:
                        self.env.viewer.update()  # Update renders and keeps window alive
                        time.sleep(0.01)
                except KeyboardInterrupt:
                    print("\nExiting...")
            else:
                print("Warning: Viewer is not available")
        else:
            print("Renderer not enabled. Set has_renderer=True to visualize the pose.")


def main(
    filename: str = 'Panda_pose_000721_j0+090_j1+090_j2+090_j3+090_j4-090_j5+000.png',
    robot: str = 'Panda',
    env: str = "EmptySpace",
    controller: str = "OSC_POSE",
    has_renderer: bool = True,
):
    """
    Reconstruct pose from image filename.
    
    Args:
        filename: Path to image file (e.g., Panda_pose_000708_j0+090_j1+090_j2+090_j3-090_j4+090_j5-090.png)
        robot: Robot name (if None, will be parsed from filename)
        env: Environment name (default: EmptySpace)
        controller: Controller name (default: OSC_POSE)
        has_renderer: Whether to show renderer window (default: True)
    """
    # Parse robot name from filename if not provided
    if robot is None:
        pose_data = parse_pose_from_filename(filename)
        robot = pose_data['robot_name']
    
    # Create reconstructor
    reconstructor = PoseReconstructor(
        robot_name=robot,
        env_name=env,
        controller_name=controller,
        has_renderer=has_renderer,
    )
    
    # Set pose from filename
    reconstructor.set_pose_from_filename(filename)
    
    print("\nPose reconstruction complete!")
    if has_renderer:
        print("Close the renderer window to exit.")


if __name__ == "__main__":
    fire.Fire(main)

