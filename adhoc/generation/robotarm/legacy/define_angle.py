FIXED_JOINT_INDICES = {
    'GR1' : "0-2, 20-31"
}

import fire
import os
import numpy as np
from PIL import Image
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config


class AngleDefiner:
    """Define and save a single robot pose with specified angles."""
    
    def __init__(
        self,
        robot_name: str = "IIWA",
        env_name: str = "Lift",
        controller_name: str = "OSC_POSE",
        output_dir: str = "data/poses",
        capture_image_width: int = 1024,
        capture_image_height: int = 1024,
        camera_fov: float = 60.0,
        except_last_joint: bool = True,
    ):
        """
        Initialize the angle definer.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            output_dir: Directory to save pose images
            capture_image_width: Width of captured images
            capture_image_height: Height of captured images
            camera_fov: Camera field of view in degrees
            except_last_joint: If True, exclude the last joint (gripper) from active joints
        """
        self.robot_name = robot_name
        self.env_name = env_name
        self.controller_name = controller_name
        self.output_dir = os.path.join(output_dir, robot_name)
        self.camera_fov = camera_fov
        self.capture_image_width = capture_image_width
        self.capture_image_height = capture_image_height
        self.except_last_joint = except_last_joint
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")
        
        print(f"Initializing robot: {robot_name}")
        
        # Setup environment with offscreen rendering
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_names": "frontview",
            "camera_heights": 1024,
            "camera_widths": 1024,
            "control_freq": 20,
        }
        
        # Load controller config
        arm_controller_config = suite.load_part_controller_config(default_controller=controller_name)
        options["controller_configs"] = refactor_composite_controller_config(
            arm_controller_config, robot_name, ["right", "left"]
        )
        
        # Create environment
        self.env = suite.make(**options, horizon=1000)
        self.env.reset()
        
        # Set camera FOV
        camera_name = "frontview"
        try:
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.sim.model.cam_fovy[cam_id] = camera_fov
            print(f"Camera FOV set to {camera_fov} degrees")
        except Exception as e:
            print(f"Warning: Could not set camera FOV: {e}")
        
        # Get robot
        self.robot = self.env.robots[0]
        
        # Get initial joint positions
        self.initial_joint_pos = self._get_joint_positions()
        self.num_joints = len(self.initial_joint_pos)
        
        # Build joint names list
        try:
            robot_model_joints = list(self.robot.robot_model.joints)
            if len(robot_model_joints) >= len(self.initial_joint_pos):
                base_joint_names = robot_model_joints[:len(self.initial_joint_pos)]
            else:
                base_joint_names = robot_model_joints + [f"joint_{i}" for i in range(len(robot_model_joints), len(self.initial_joint_pos))]
            self.joint_names = base_joint_names
        except:
            self.joint_names = [f"joint_{i}" for i in range(len(self.initial_joint_pos))]
        
        # Parse fixed joint indices from FIXED_JOINT_INDICES
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
            except ValueError as e:
                print(f"Warning: Could not parse fixed_joint_indices: {e}")
                self.fixed_joint_indices = []
        
        # Active joints: all joints except last (gripper) if except_last_joint is True, and fixed joints
        if self.except_last_joint:
            base_active_joint_indices = list(range(self.num_joints - 1))
        else:
            base_active_joint_indices = list(range(self.num_joints))
        self.active_joint_indices = [idx for idx in base_active_joint_indices if idx not in self.fixed_joint_indices]
        
        # Print indices and active angles
        self._print_indices_and_angles()
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        joint_pos = self.robot._joint_positions.copy()
        return joint_pos
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation."""
        robot_joint_pos = joint_positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        self.env.sim.forward()
    
    def _print_indices_and_angles(self):
        """Print indices and active angles."""
        print("\n" + "="*60)
        print("JOINT INDICES AND ACTIVE ANGLES")
        print("="*60)
        print(f"Total joints: {self.num_joints}")
        print(f"\nActive joint indices: {self.active_joint_indices}")
        print(f"Fixed joint indices: {self.fixed_joint_indices}")
        print(f"\nJoint names and initial positions (degrees):")
        for i, (name, pos) in enumerate(zip(self.joint_names, np.rad2deg(self.initial_joint_pos))):
            if i in self.fixed_joint_indices:
                marker = " [FIXED]"
            elif i in self.active_joint_indices:
                marker = " [ACTIVE]"
            else:
                marker = ""
            print(f"  [{i:2d}] {name:40s}: {pos:+.2f}°{marker}")
        
        print(f"\nActive angles (degrees) for active joints:")
        for i, active_idx in enumerate(self.active_joint_indices):
            angle_deg = np.rad2deg(self.initial_joint_pos[active_idx])
            print(f"  Active joint [{active_idx}] ({self.joint_names[active_idx]}): {angle_deg:+.2f}°")
        print("="*60 + "\n")
    
    def _capture_image(self, width: int = None, height: int = None):
        """Capture current camera view as numpy array."""
        if width is None:
            width = self.capture_image_width
        if height is None:
            height = self.capture_image_height
        obs = self.env.sim.render(
            camera_name="frontview",
            width=width,
            height=height,
            depth=False
        )
        return obs[::-1]
    
    def _save_image(self, image_array, filename):
        """Save image array as PNG."""
        img = Image.fromarray(image_array)
        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath)
        return filepath
    
    def set_angles_and_save(
        self,
        angle_dict: dict,
        filename: str = None
    ):
        """
        Set joint angles and save one frame.
        
        Args:
            angle_dict: Dictionary mapping joint indices to angles in degrees (e.g., {3: -90, 5: 45})
                        Only specified joints will be updated; others keep current angles.
            filename: Optional filename for saved image. If None, auto-generates from angles.
        """
        # Get current joint positions
        current_joint_pos = self._get_joint_positions()
        
        # Create joint position array (fixed joints keep initial values)
        joint_pos = current_joint_pos.copy()
        
        # Validate and set positions for specified joints
        updated_indices = []
        for joint_idx, angle_deg in angle_dict.items():
            # Check if index is valid
            if joint_idx < 0 or joint_idx >= self.num_joints:
                raise ValueError(f"Joint index {joint_idx} is out of range [0, {self.num_joints-1}]")
            
            # Check if index is fixed
            if joint_idx in self.fixed_joint_indices:
                print(f"Warning: Joint index {joint_idx} is fixed and cannot be changed. Skipping.")
                continue
            
            # Set position (convert degrees to radians)
            joint_pos[joint_idx] = np.deg2rad(angle_deg)
            updated_indices.append(joint_idx)
        
        # Set joint positions
        self._set_joint_positions(joint_pos)
        
        # Capture image
        image = self._capture_image()
        
        # Generate filename if not provided
        if filename is None:
            angles_str = "_".join([f"j{idx}{int(angle_dict[idx]):+04d}" 
                                    for idx in sorted(updated_indices)])
            filename = f"{self.robot_name}_angle_{angles_str}.png"
        
        # Save image
        filepath = self._save_image(image, filename)
        
        print(f"\nSaved image to: {filepath}")
        print(f"Angles updated (degrees):")
        for idx in sorted(updated_indices):
            print(f"  [{idx}] {self.joint_names[idx]}: {angle_dict[idx]:+.2f}°")
        
        return filepath
    
    def close(self):
        """Close the environment."""
        self.env.close()


def _parse_angle_input(angles_str: str) -> dict:
    """
    Parse angle input string in format "index: angle, index: angle, ..."
    
    Args:
        angles_str: Input string like "3: -90, 5: 45"
    
    Returns:
        Dictionary mapping joint indices to angles (e.g., {3: -90.0, 5: 45.0})
    """
    angle_dict = {}
    
    if not angles_str or not angles_str.strip():
        return angle_dict
    
    # Split by comma
    parts = [part.strip() for part in angles_str.split(",")]
    
    for part in parts:
        if not part:
            continue
        
        # Split by colon
        if ":" not in part:
            raise ValueError(f"Invalid format: '{part}'. Expected format: 'index: angle'")
        
        idx_str, angle_str = part.split(":", 1)
        idx_str = idx_str.strip()
        angle_str = angle_str.strip()
        
        try:
            joint_idx = int(idx_str)
            angle_deg = float(angle_str)
            angle_dict[joint_idx] = angle_deg
        except ValueError as e:
            raise ValueError(f"Invalid format: '{part}'. Index and angle must be numbers. Error: {e}")
    
    return angle_dict


def main(
    robot: str = "GR1",
    env: str = "Lift",
    # angles: str = "6: -60, 7: -15, 8: 5, 9: -15, 10: 90, 11: 0, 12: 0",
    # angles: str = "6: -70.0, 8: 10.0, 9: -30.0, 10: 80.0, 12: 20.0",
    # angles: str = "6: -65.0, 8: 30.0, 9: -25.0, 10: 90.0, 12: 35.0",
    # angles: str = "6: -85.0, 7: -30.0, 8: 40.0, 9: -15.0, 10: 90.0, 12: 30.0",
    # angles: str = "7: -15, 8: 60, 10: 90, 11: -20",
    # angles: str = "7: -20, 8: 45, 9: -20, 10: 90, 12: 30",
    # angles: str = "6: -80.0, 7: -20.0, 8: 45.0, 9: -20.0, 10: 90.0, 12: 30.0",
    # angles: str = "6: -100.0, 8: 65.0, 9: -45.0, 10: 95.0, 12: 35.0",
    # angles: str = "6: -75.0, 7: -10.0, 8: 35.0, 9: -40.0, 10: 90.0, 12: 40.0",
    # angles: str = "6: -75.0, 7: -10.0, 8: 35.0, 9: -55.0, 10: 90.0, 12: 50.0",
    # angles: str = "0: 10, 1: 5, 2: 15, 6: -110, 8: 40, 9: -110, 10: 30, 13: 15, 14: 45, 16: -100, 17: -20",
    angles: str = "0: 15, 2: 20, 6: -105, 7: -20, 8: 70, 9: -120, 10: 40, 13: 25, 14: 60, 15: -20, 16: -105",
    filename: str = None,
    output_dir: str = "data/poses",
    capture_image_width: int = 1024,
    capture_image_height: int = 1024,
    camera_fov: float = 60.0,
    except_last_joint: bool = True,
):
    """
    Define and save a single robot pose with specified angles.
    
    Args:
        robot: Robot name (IIWA, Panda, GR1, etc.)
        env: Environment name
        angles: Angles in format "index: angle, index: angle" (e.g., "3: -90, 5: 45")
               Unspecified joints will keep their current angles.
        filename: Optional filename for saved image
        output_dir: Directory to save pose images
        capture_image_width: Width of captured images
        capture_image_height: Height of captured images
        camera_fov: Camera field of view in degrees
        except_last_joint: If True, exclude the last joint (gripper) from active joints (default: True)
    
    Examples:
        # Interactive mode (will prompt for angles)
        python define_angle.py --robot GR1
        
        # With angles specified (index: angle format)
        python define_angle.py --robot GR1 --angles "3: -90, 5: 45"
        
        # Include last joint (gripper) in active joints
        python define_angle.py --robot GR1 --angles "3: -90" --except-last-joint False
        
        # With custom filename
        python define_angle.py --robot GR1 --angles "3: -90" --filename "my_pose.png"
    """
    
    print("="*60)
    print("ANGLE DEFINER - SINGLE POSE")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Output directory: {output_dir}")
    print(f"Except last joint: {except_last_joint}")
    print("="*60)
    
    # Create definer
    definer = AngleDefiner(
        robot_name=robot,
        env_name=env,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
        except_last_joint=except_last_joint,
    )
    
    try:
        # Parse angles from input
        if angles is None:
            # Interactive mode: prompt for angles
            print(f"\nEnter angles in format 'index: angle' (e.g., '3: -90, 5: 45'):")
            print(f"Active joint indices: {definer.active_joint_indices}")
            print(f"Unspecified joints will keep their current angles.")
            angles_input = input("Angles: ").strip()
            if not angles_input:
                print("No angles provided. Exiting.")
                return
            angle_dict = _parse_angle_input(angles_input)
        else:
            # Parse from command line argument
            angle_dict = _parse_angle_input(angles)
        
        if not angle_dict:
            print("No valid angles provided. Exiting.")
            return
        
        # Set angles and save
        definer.set_angles_and_save(angle_dict, filename)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        definer.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)

