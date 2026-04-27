FIXED_JOINT_INDICES = {
    'GR1' : "0-2, 20-31"
}

import fire
import os
import numpy as np
from PIL import Image
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from datetime import datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")

class ConnectedMotionGenerator:
    """Generate connected motion animations from multiple poses and save as GIF."""
    
    def __init__(
        self,
        robot_name: str = "GR1",
        env_name: str = "Lift",
        controller_name: str = "OSC_POSE",
        output_dir: str = "data/motions",
        capture_image_width: int = 512,
        capture_image_height: int = 512,
        camera_fov: float = 60.0,
        except_last_joint: bool = True,
    ):
        """
        Initialize the motion generator.
        
        Args:
            robot_name: Name of the robot
            env_name: Name of the environment
            controller_name: Name of the controller
            output_dir: Directory to save motion GIFs
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
            "camera_heights": capture_image_height,
            "camera_widths": capture_image_width,
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
    
    def _get_joint_positions(self):
        """Get current joint positions."""
        joint_pos = self.robot._joint_positions.copy()
        return joint_pos
    
    def _set_joint_positions(self, joint_positions):
        """Set joint positions and update simulation."""
        robot_joint_pos = joint_positions
        self.robot.set_robot_joint_positions(robot_joint_pos)
        self.env.sim.forward()
    
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
    
    def _parse_pose_input(self, pose_str: str) -> dict:
        """
        Parse pose input string in format "index: angle, index: angle, ..."
        
        Args:
            pose_str: Input string like "3: -90, 5: 45"
        
        Returns:
            Dictionary mapping joint indices to angles in degrees
        """
        pose_dict = {}
        
        if not pose_str or not pose_str.strip():
            return pose_dict
        
        # Split by comma
        parts = [part.strip() for part in pose_str.split(",")]
        
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
                
                # Validate joint index
                if joint_idx < 0 or joint_idx >= self.num_joints:
                    raise ValueError(f"Joint index {joint_idx} is out of range [0, {self.num_joints-1}]")
                
                if joint_idx in self.fixed_joint_indices:
                    print(f"Warning: Joint index {joint_idx} is fixed and will be skipped.")
                    continue
                
                pose_dict[joint_idx] = angle_deg
            except ValueError as e:
                raise ValueError(f"Invalid format: '{part}'. Error: {e}")
        
        return pose_dict
    
    def _pose_dict_to_joint_positions(self, pose_dict: dict, base_positions: np.ndarray = None) -> np.ndarray:
        """
        Convert pose dictionary to joint positions array.
        
        Args:
            pose_dict: Dictionary mapping joint indices to angles in degrees
            base_positions: Base joint positions (default: initial positions)
        
        Returns:
            Joint positions array in radians
        """
        if base_positions is None:
            joint_pos = self.initial_joint_pos.copy()
        else:
            joint_pos = base_positions.copy()
        
        # Set positions for specified joints (convert degrees to radians)
        for joint_idx, angle_deg in pose_dict.items():
            joint_pos[joint_idx] = np.deg2rad(angle_deg)
        
        return joint_pos
    
    def _interpolate_poses(self, start_pose: np.ndarray, end_pose: np.ndarray, num_steps: int) -> list:
        """
        Interpolate between two poses.
        
        Args:
            start_pose: Starting joint positions (radians)
            end_pose: Ending joint positions (radians)
            num_steps: Number of interpolation steps
        
        Returns:
            List of interpolated joint positions
        """
        poses = []
        for i in range(num_steps + 1):
            alpha = i / num_steps
            # Use smooth interpolation (ease-in-out)
            smooth_alpha = alpha * alpha * (3 - 2 * alpha)
            interpolated = start_pose * (1 - smooth_alpha) + end_pose * smooth_alpha
            poses.append(interpolated)
        return poses
    
    def generate_motion_gif(
        self,
        poses_input: list,
        repetition: int = 2,
        speed: float = 1.0,
        pause_duration: float = 0.5,
        frames_per_transition: int = 20,
        pause_frames: int = None,
        output_filename: str = None,
        gif_duration: int = 100,  # milliseconds per frame
    ):
        """
        Generate motion GIF from multiple poses.
        
        Args:
            poses_input: List of pose strings in format "index: angle, index: angle, ..."
            repetition: Number of times to repeat the motion sequence
            speed: Speed multiplier (higher = faster, default: 1.0)
            pause_duration: Duration to pause at each pose in seconds
            frames_per_transition: Number of frames for each transition (before speed adjustment)
            pause_frames: Number of frames to pause at each pose (overrides pause_duration if set)
            output_filename: Output filename for GIF. If None, auto-generates.
            gif_duration: Duration of each frame in milliseconds for GIF
        """
        if not poses_input or len(poses_input) < 2:
            raise ValueError("At least 2 poses are required to create a motion.")
        
        print("\n" + "="*60)
        print("CONNECTED MOTION GENERATION")
        print("="*60)
        print(f"Number of poses: {len(poses_input)}")
        print(f"Repetition: {repetition}")
        print(f"Speed: {speed}x")
        print(f"Pause duration: {pause_duration}s")
        print("="*60 + "\n")
        
        # Parse all poses
        parsed_poses = []
        for i, pose_str in enumerate(poses_input):
            pose_dict = self._parse_pose_input(pose_str)
            if not pose_dict:
                raise ValueError(f"Pose {i+1} is empty or invalid: '{pose_str}'")
            parsed_poses.append(pose_dict)
            print(f"Pose {i+1}: {pose_dict}")
        
        # Convert poses to joint positions
        joint_poses = []
        for pose_dict in parsed_poses:
            joint_pos = self._pose_dict_to_joint_positions(pose_dict)
            joint_poses.append(joint_pos)
        
        # Calculate frame counts
        transition_frames = max(1, int(frames_per_transition / speed))
        if pause_frames is None:
            pause_frames = max(1, int(pause_duration * 20))  # 20 Hz control frequency
        
        print(f"Transition frames: {transition_frames}")
        print(f"Pause frames: {pause_frames}")
        print(f"\nGenerating frames...")
        
        # Generate frames
        frames = []
        
        # Repeat the motion
        for rep in range(repetition):
            print(f"Repetition {rep + 1}/{repetition}...")
            
            # Go through all poses
            for i in range(len(joint_poses)):
                current_pose = joint_poses[i]
                
                # Move to current pose (if not first pose, interpolate from previous)
                if i == 0:
                    # First pose: set directly
                    self._set_joint_positions(current_pose)
                else:
                    # Interpolate from previous pose
                    previous_pose = joint_poses[i - 1]
                    interpolated = self._interpolate_poses(previous_pose, current_pose, transition_frames)
                    
                    for interp_pose in interpolated:
                        self._set_joint_positions(interp_pose)
                        image = self._capture_image()
                        frames.append(Image.fromarray(image))
                
                # Pause at current pose
                for _ in range(pause_frames):
                    self._set_joint_positions(current_pose)
                    image = self._capture_image()
                    frames.append(Image.fromarray(image))
            
            # Return to first pose for next repetition (if not last repetition)
            if rep < repetition - 1:
                first_pose = joint_poses[0]
                last_pose = joint_poses[-1]
                interpolated = self._interpolate_poses(last_pose, first_pose, transition_frames)
                
                for interp_pose in interpolated:
                    self._set_joint_positions(interp_pose)
                    image = self._capture_image()
                    frames.append(Image.fromarray(image))
        
        print(f"Generated {len(frames)} frames")
        
        # Generate output filename if not provided
        if output_filename is None:
            pose_summary = f"{len(parsed_poses)}poses"
            output_filename = f"{now}_{self.robot_name}_motion_r{repetition}_s{speed:.1f}_p{pause_duration:.1f}_{pose_summary}.gif"
        
        # Save GIF
        filepath = os.path.join(self.output_dir, output_filename)
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration,
            loop=0  # Infinite loop
        )
        
        print(f"\n{'='*60}")
        print(f"Saved motion GIF: {filepath}")
        print(f"Total frames: {len(frames)}")
        print(f"GIF duration: {len(frames) * gif_duration / 1000:.2f} seconds")
        print(f"{'='*60}\n")
        
        return filepath
    
    def generate_variations(
        self,
        poses_input: list,
        repetitions: list = [2, 3],
        speeds: list = [0.5, 1.0, 2.0],
        pause_durations: list = [0.3, 0.5, 1.0],
        frames_per_transition: int = 20,
        gif_duration: int = 100,
    ):
        """
        Generate multiple motion GIFs with different parameter combinations.
        
        Args:
            poses_input: List of pose strings
            repetitions: List of repetition counts to try
            speeds: List of speed multipliers to try
            pause_durations: List of pause durations to try
            frames_per_transition: Number of frames for each transition (before speed adjustment)
            gif_duration: Duration of each frame in milliseconds for GIF
        """
        print("\n" + "="*60)
        print("GENERATING MOTION VARIATIONS")
        print("="*60)
        print(f"Repetitions: {repetitions}")
        print(f"Speeds: {speeds}")
        print(f"Pause durations: {pause_durations}")
        print(f"Total variations: {len(repetitions) * len(speeds) * len(pause_durations)}")
        print("="*60 + "\n")
        
        generated_files = []
        
        for repetition in repetitions:
            for speed in speeds:
                for pause_duration in pause_durations:
                    output_filename = f"{now}_{self.robot_name}_motion_r{repetition}_s{speed:.1f}_p{pause_duration:.1f}.gif"
                    
                    print(f"\nGenerating: {output_filename}")
                    try:
                        filepath = self.generate_motion_gif(
                            poses_input=poses_input,
                            repetition=repetition,
                            speed=speed,
                            pause_duration=pause_duration,
                            frames_per_transition=frames_per_transition,
                            output_filename=output_filename,
                            gif_duration=gif_duration,
                        )
                        generated_files.append(filepath)
                    except Exception as e:
                        print(f"Error generating {output_filename}: {e}")
        
        print(f"\n{'='*60}")
        print(f"Generated {len(generated_files)} motion GIFs")
        print(f"{'='*60}\n")
        
        return generated_files
    
    def close(self):
        """Close the environment."""
        self.env.close()


def _normalize_poses_input(poses_input):
    """
    Normalize poses input. Fire may parse comma-separated strings as tuples.
    
    Args:
        poses_input: String, list, or tuple of pose strings
    
    Returns:
        List of pose strings
    """
    if isinstance(poses_input, str):
        # Single string - split by semicolon or newline
        if ";" in poses_input:
            return [p.strip() for p in poses_input.split(";") if p.strip()]
        elif "\n" in poses_input:
            return [p.strip() for p in poses_input.split("\n") if p.strip()]
        else:
            return [poses_input]
    elif isinstance(poses_input, (list, tuple)):
        # Already a list/tuple
        return [str(p) for p in poses_input if p]
    else:
        return [str(poses_input)]


def main(
    robot: str = "GR1",
    env: str = "Lift",
    poses: str = "6: -30, 7: 0, 8: 5, 9: -100, 11: 35; 6: -30, 7: 0, 8: 5, 9: -70, 11: 10",
    # poses: str = "6: -30, 7: 0, 8: 5, 9: -60, 11: 60, 12: -20; 6: -30, 7: 0, 8: 5, 9: -90, 11: 60, 12: 30",
    # poses: str = "6: -30, 7: 0, 8: 5, 9: -65, 11: 60, 12: 35; 6: -30, 7: 0, 8: 5, 9: -65, 11: 60, 12: 10",
    frames_per_transition: int = 5,
    output_filename: str = None,
    gif_duration: int = 100,
    generate_variations_flag: bool = True,
    repetitions: str = "2,3",
    speeds: str = "1.0,2.0",
    pause_durations: str = "0.0,0.1",
    output_dir: str = "data/motions",
    capture_image_width: int = 512,
    capture_image_height: int = 512,
    camera_fov: float = 60.0,
    except_last_joint: bool = True,
):
    """
    Generate connected motion animations from multiple poses and save as GIF.
    
    Args:
        robot: Robot name (IIWA, Panda, GR1, etc.)
        env: Environment name
        poses: Poses in format "index: angle, index: angle; index: angle, index: angle" 
               (separated by semicolons or newlines)
        repetition: Number of times to repeat the motion sequence
        speed: Speed multiplier (higher = faster, default: 1.0)
        pause_duration: Duration to pause at each pose in seconds
        frames_per_transition: Number of frames for each transition (before speed adjustment)
        output_filename: Output filename for GIF. If None, auto-generates.
        gif_duration: Duration of each frame in milliseconds for GIF
        generate_variations_flag: If True, generate multiple variations with different parameters
        repetitions: Comma-separated list of repetition counts (e.g., "2,3")
        speeds: Comma-separated list of speeds (e.g., "0.5,1.0,2.0")
        pause_durations: Comma-separated list of pause durations (e.g., "0.3,0.5,1.0")
        output_dir: Directory to save motion GIFs
        capture_image_width: Width of captured images
        capture_image_height: Height of captured images
        camera_fov: Camera field of view in degrees
        except_last_joint: If True, exclude the last joint (gripper) from active joints
    
    Examples:
        # Single motion with 2 poses
        python connected_motion.py --poses "3: -90, 5: 45; 3: 90, 5: -45"
        
        # With custom parameters
        python connected_motion.py --poses "3: -90; 3: 90" --repetition 3 --speed 2.0 --pause-duration 1.0
        
        # Generate variations
        python connected_motion.py --poses "3: -90; 3: 90" --generate-variations-flag True
    """
    
    print("="*60)
    print("CONNECTED MOTION GENERATOR")
    print("="*60)
    print(f"Robot: {robot}")
    print(f"Environment: {env}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create generator
    generator = ConnectedMotionGenerator(
        robot_name=robot,
        env_name=env,
        output_dir=output_dir,
        capture_image_width=capture_image_width,
        capture_image_height=capture_image_height,
        camera_fov=camera_fov,
        except_last_joint=except_last_joint,
    )
    
    try:
        # Parse poses input
        poses_input = poses
        
        # Normalize poses input
        poses_list = _normalize_poses_input(poses_input)
        
        if generate_variations_flag:
            # Parse variation parameters
            repetitions_list = [int(x.strip()) for x in repetitions.split(",")]
            speeds_list = [float(x.strip()) for x in speeds.split(",")]
            pause_durations_list = [float(x.strip()) for x in pause_durations.split(",")]
            
            generator.generate_variations(
                poses_input=poses_list,
                repetitions=repetitions_list,
                speeds=speeds_list,
                pause_durations=pause_durations_list,
                frames_per_transition=frames_per_transition,
                gif_duration=gif_duration,
            )
        else:
            # Single motion
            generator.generate_motion_gif(
                poses_input=poses_list,
                repetition=repetition,
                speed=speed,
                pause_duration=pause_duration,
                frames_per_transition=frames_per_transition,
                output_filename=output_filename,
                gif_duration=gif_duration,
            )
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        generator.close()
        print("\nDone!")


if __name__ == "__main__":
    fire.Fire(main)

