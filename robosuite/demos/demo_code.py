"""
Demo script that interprets natural language commands and executes them as robot actions.

This script provides an abstraction layer over demo_control.py, allowing users to:
1. Input natural language commands (e.g., "인사해" - "Say hello")
2. Have an LLM interpret the command into a control sequence
3. Execute the control sequence using the robot controller

The LLM inference is mocked for now, returning pre-defined control sequences.
"""

import fire

import time
import json
import random
import os
from typing import Dict, List, Tuple
import numpy as np

import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils import transform_utils as T

# Add mylmeval to path if not installed
import sys
sys.path.insert(0, '/Users/sb/Downloads/workspace/mylmeval')
from mylmeval.llm_class import MyLLMEval

MAX_FR = 25  # max frame rate for running simulation


class RobotController:
    """Abstraction layer for robot control."""
    
    def __init__(self, env_name="Lift", robot_name="IIWA", controller_name="OSC_POSE", 
                 robots_dir="data/robots"):
        """
        Initialize the robot controller.
        
        Args:
            env_name: Name of the environment
            robot_name: Name of the robot
            controller_name: Name of the controller to use
            robots_dir: Directory for robot info files
        """
        self.env_name = env_name
        self.robot_name = robot_name
        self.controller_name = controller_name
        self.robots_dir = robots_dir
        
        # Setup environment options
        options = {
            "env_name": env_name,
            "robots": robot_name,
            "has_renderer": True,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_camera_obs": False,
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
        self.env.viewer.set_camera(camera_id=0)
        
        # Get robot and action dimensions
        self.robot = self.env.robots[0]
        self.gripper_dim = self.robot.gripper["right"].dof if hasattr(self.robot, "gripper") else 0
        
        # Controller settings
        controller_settings = {
            "OSC_POSE": 6,
            "OSC_POSITION": 3,
            "IK_POSE": 6,
            "JOINT_POSITION": 7,
            "JOINT_VELOCITY": 7,
            "JOINT_TORQUE": 7,
        }
        self.action_dim = controller_settings.get(controller_name, 6)
        self.neutral = np.zeros(self.action_dim + self.gripper_dim)
        
        # Stabilize the robot
        print("Stabilizing robot...")
        for _ in range(50):
            self.env.step(self.neutral)
            self.env.render()
        
        print(f"Robot controller initialized: {robot_name} with {controller_name}")
        print(f"Action dimension: {self.action_dim}, Gripper dimension: {self.gripper_dim}")

        self.robot_info = self.load_robot_info()
            
    
    def execute_action(self, action: np.ndarray, steps: int = 75):
        """
        Execute a single action for a specified number of steps.
        
        Args:
            action: Action vector (should match action_dim + gripper_dim)
            steps: Number of simulation steps to execute
        """
        for _ in range(steps):
            start = time.time()
            self.env.step(action)
            self.env.render()
            
            # Limit frame rate
            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)
    
    def rest(self, steps: int = 75):
        """
        Execute neutral action (rest) for a specified number of steps.
        
        Args:
            steps: Number of simulation steps to rest
        """
        self.execute_action(self.neutral, steps)
    
    def execute_sequence(self, action_sequence: List[Tuple[np.ndarray, int]]):
        """
        Execute a sequence of actions.
        
        Args:
            action_sequence: List of (action, steps) tuples
        """
        for i, (action, steps) in enumerate(action_sequence):
            print(f"Executing action {i+1}/{len(action_sequence)}: {action[:self.action_dim]}")
            self.execute_action(action, steps)
    
    def get_ee_position(self, arm="right") -> np.ndarray:
        """
        Get current end-effector position (x, y, z).
        
        Args:
            arm: Which arm to get position for ("right" or "left")
        
        Returns:
            np.ndarray: [x, y, z] position in meters
        """
        positions = self.robot._hand_pos
        if arm in positions:
            return positions[arm].copy()
        else:
            raise ValueError(f"Arm '{arm}' not found. Available arms: {list(positions.keys())}")
    
    def get_ee_orientation(self, arm="right", return_type="rpy") -> np.ndarray:
        """
        Get current end-effector orientation.
        
        Args:
            arm: Which arm to get orientation for ("right" or "left")
            return_type: "rpy" for roll-pitch-yaw (radians), "quat" for quaternion, "mat" for rotation matrix
        
        Returns:
            np.ndarray: Orientation in requested format
                - "rpy": [roll, pitch, yaw] in radians
                - "quat": [x, y, z, w] quaternion
                - "mat": 3x3 rotation matrix
        """
        if return_type == "quat":
            quats = self.robot._hand_quat
            if arm in quats:
                return quats[arm].copy()
        elif return_type == "mat":
            mats = self.robot._hand_orn
            if arm in mats:
                return mats[arm].copy()
        elif return_type == "rpy":
            mats = self.robot._hand_orn
            if arm in mats:
                # Convert rotation matrix to euler angles (roll, pitch, yaw)
                return np.array(T.mat2euler(mats[arm]))
        else:
            raise ValueError(f"Invalid return_type '{return_type}'. Use 'rpy', 'quat', or 'mat'.")
        
        raise ValueError(f"Arm '{arm}' not found. Available arms: {list(self.robot._hand_pos.keys())}")
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions for all robot joints.
        
        Returns:
            np.ndarray: Joint positions in radians
        """
        return self.robot._joint_positions.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities for all robot joints.
        
        Returns:
            np.ndarray: Joint velocities in radians/second
        """
        return self.robot._joint_velocities.copy()
    
    def get_robot_state(self, arm="right") -> Dict:
        """
        Get complete robot state information.
        
        Args:
            arm: Which arm to get state for ("right" or "left")
        
        Returns:
            Dict containing:
                - 'ee_pos': End-effector position [x, y, z] in meters
                - 'ee_ori_rpy': End-effector orientation [roll, pitch, yaw] in radians
                - 'ee_ori_quat': End-effector orientation as quaternion [x, y, z, w]
                - 'joint_pos': Joint positions in radians
                - 'joint_vel': Joint velocities in radians/second
        """
        return {
            'ee_pos': self.get_ee_position(arm),
            'ee_ori_rpy': self.get_ee_orientation(arm, return_type="rpy"),
            'ee_ori_quat': self.get_ee_orientation(arm, return_type="quat"),
            'joint_pos': self.get_joint_positions(),
            'joint_vel': self.get_joint_velocities(),
        }
    
    def print_robot_state(self, arm="right"):
        """
        Print current robot state in a readable format.
        
        Args:
            arm: Which arm to print state for ("right" or "left")
        """
        state = self.get_robot_state(arm)
        
        print(f"\n{'='*60}")
        print(f"Robot State (Arm: {arm})")
        print(f"{'='*60}")
        
        print(f"\n[End Effector Position]")
        print(f"  X: {state['ee_pos'][0]:8.4f} m")
        print(f"  Y: {state['ee_pos'][1]:8.4f} m")
        print(f"  Z: {state['ee_pos'][2]:8.4f} m")
        
        print(f"\n[End Effector Orientation (RPY)]")
        rpy_deg = np.rad2deg(state['ee_ori_rpy'])
        print(f"  Roll:  {state['ee_ori_rpy'][0]:8.4f} rad ({rpy_deg[0]:7.2f}°)")
        print(f"  Pitch: {state['ee_ori_rpy'][1]:8.4f} rad ({rpy_deg[1]:7.2f}°)")
        print(f"  Yaw:   {state['ee_ori_rpy'][2]:8.4f} rad ({rpy_deg[2]:7.2f}°)")
        
        print(f"\n[End Effector Orientation (Quaternion)]")
        print(f"  x: {state['ee_ori_quat'][0]:7.4f}")
        print(f"  y: {state['ee_ori_quat'][1]:7.4f}")
        print(f"  z: {state['ee_ori_quat'][2]:7.4f}")
        print(f"  w: {state['ee_ori_quat'][3]:7.4f}")
        
        print(f"\n[Joint Positions]")
        for i, (pos, vel) in enumerate(zip(state['joint_pos'], state['joint_vel'])):
            pos_deg = np.rad2deg(pos)
            print(f"  Joint {i}: {pos:7.4f} rad ({pos_deg:7.2f}°) | vel: {vel:7.4f} rad/s")
        
        print(f"\n{'='*60}\n")
    
    def return_to_initial_pose(self, steps: int = 100):
        """
        Return robot to its initial/home position as defined in robot_info.
        
        Args:
            steps: Number of steps for smooth transition back to initial pose
        """
        if not self.robot_info:
            print("[Robot] No robot_info available. Using neutral action instead.")
            self.rest(steps)
            return
        
        initial_state = self.robot_info.get('initial_state', {})
        initial_joint_pos = initial_state.get('joint_positions', None)
        
        if initial_joint_pos is None:
            print("[Robot] No initial joint positions in robot_info. Using neutral action.")
            self.rest(steps)
            return
        
        print("[Robot] Returning to initial pose...")
        
        # Set target joint positions
        target_pos = np.array(initial_joint_pos)
        
        # Gradually move to initial position
        self.robot.set_robot_joint_positions(target_pos)
        self.env.sim.forward()
        
        # Stabilize at initial position
        for _ in range(steps):
            start = time.time()
            self.env.step(self.neutral)
            self.env.render()
            
            # Limit frame rate
            elapsed = time.time() - start
            diff = 1 / MAX_FR - elapsed
            if diff > 0:
                time.sleep(diff)
        
        print("[Robot] Initial pose reached.")
    
    def close(self):
        """Close the environment."""
        self.env.close()

    def load_robot_info(self) -> Dict:
        """
        Load robot kinematic information from JSON file.
        
        Args:
            robot_name: Name of the robot
            robots_dir: Directory containing robot info files
        
        Returns:
            Dictionary with robot information, or None if file doesn't exist
        """
        os.makedirs(self.robots_dir, exist_ok=True)
        robot_file = os.path.join(self.robots_dir, f"{self.robot_name}.json")
        
        if not os.path.exists(robot_file):
            robot_info = self.collect_robot_info()
            self.save_robot_info(robot_info, robot_file)
            return robot_info
        
        with open(robot_file, 'r') as f:
            return json.load(f)

    def save_robot_info(self, info: Dict, file_name: str = None, float_precision: int = 4):
        """
        Save robot kinematic information to JSON file with rounded floats.
        
        Args:
            info: Robot information dictionary
            file_name: Path to save the JSON file
            float_precision: Number of decimal places to round floats (default: 4)
        """
        def round_floats(obj, precision):
            """Recursively round all floats in nested structures."""
            if isinstance(obj, float):
                return round(obj, precision)
            elif isinstance(obj, dict):
                return {k: round_floats(v, precision) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_floats(item, precision) for item in obj]
            else:
                return obj
        
        if float_precision is not None:
            info = round_floats(info, float_precision)
                    
        with open(file_name, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[Robot Info] Saved to {file_name}")


    def collect_robot_info(self, joint_delta_deg: float = 20.0) -> Dict:
        """
        Collect robot kinematic information by testing joint movements.
        
        This function:
        1. Records initial joint positions and EE pose
        2. For each joint, moves it by +delta and -delta degrees
        3. Records how the EE position changes
        4. Returns to initial position
        
        Args:
            controller: RobotController instance
            joint_delta_deg: Amount to move each joint (in degrees)
        
        Returns:
            Dictionary containing robot kinematic information
        """
        print("\n[Robot Info] Collecting kinematic information...")
        print(f"This will test each joint by moving ±{joint_delta_deg}°")
        
        joint_delta_rad = np.deg2rad(joint_delta_deg)
        
        # Get initial state
        initial_joint_pos = self.get_joint_positions()
        initial_ee_pos = self.get_ee_position()
        initial_ee_ori_rpy = self.get_ee_orientation(return_type="rpy")
        
        info = {
            "robot_name": self.robot_name,
            "controller_type": self.controller_name,
            "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joint_delta_deg": joint_delta_deg,
            "initial_state": {
                "joint_positions": initial_joint_pos.tolist(),
                "ee_position": initial_ee_pos.tolist(),
                "ee_orientation_rpy": initial_ee_ori_rpy.tolist(),
            },
            "joint_effects": []
        }
        
        num_joints = len(initial_joint_pos)
        print(f"Testing {num_joints} joints...")
        
        for joint_idx in range(num_joints):
            print(f"\n  Testing Joint {joint_idx}...")
            
            joint_effect = {
                "joint_index": joint_idx,
                "initial_position": float(initial_joint_pos[joint_idx]),
                "positive_delta": {},
                "negative_delta": {}
            }
            
            # Test positive delta
            try:
                target_pos = initial_joint_pos.copy()
                target_pos[joint_idx] += joint_delta_rad
                
                # Move to target position using joint position control
                self.robot.set_robot_joint_positions(target_pos)
                self.env.sim.forward()
                
                # Wait a bit for stabilization
                for _ in range(10):
                    self.env.step(self.neutral)
                    if self.env.has_renderer:
                        self.env.render()
                
                # Record new EE position
                new_ee_pos = self.get_ee_position()
                new_ee_ori = self.get_ee_orientation(return_type="rpy")
                
                joint_effect["positive_delta"] = {
                    "joint_position": float(target_pos[joint_idx]),
                    "ee_position": new_ee_pos.tolist(),
                    "ee_orientation_rpy": new_ee_ori.tolist(),
                    "ee_position_change": (new_ee_pos - initial_ee_pos).tolist(),
                    "ee_orientation_change": (new_ee_ori - initial_ee_ori_rpy).tolist(),
                }
                
            except Exception as e:
                print(f"    Error testing +delta: {e}")
                joint_effect["positive_delta"] = {"error": str(e)}
            
            # Return to initial position
            self.robot.set_robot_joint_positions(initial_joint_pos)
            self.env.sim.forward()
            for _ in range(10):
                self.env.step(self.neutral)
                if self.env.has_renderer:
                    self.env.render()
            
            # Test negative delta
            try:
                target_pos = initial_joint_pos.copy()
                target_pos[joint_idx] -= joint_delta_rad
                
                self.robot.set_robot_joint_positions(target_pos)
                self.env.sim.forward()
                
                for _ in range(10):
                    self.env.step(self.neutral)
                    if self.env.has_renderer:
                        self.env.render()
                
                new_ee_pos = self.get_ee_position()
                new_ee_ori = self.get_ee_orientation(return_type="rpy")
                
                joint_effect["negative_delta"] = {
                    "joint_position": float(target_pos[joint_idx]),
                    "ee_position": new_ee_pos.tolist(),
                    "ee_orientation_rpy": new_ee_ori.tolist(),
                    "ee_position_change": (new_ee_pos - initial_ee_pos).tolist(),
                    "ee_orientation_change": (new_ee_ori - initial_ee_ori_rpy).tolist(),
                }
                
            except Exception as e:
                print(f"    Error testing -delta: {e}")
                joint_effect["negative_delta"] = {"error": str(e)}
            
            # Return to initial position
            self.robot.set_robot_joint_positions(initial_joint_pos)
            self.env.sim.forward()
            for _ in range(10):
                self.env.step(self.neutral)
                if self.env.has_renderer:
                    self.env.render()
            
            info["joint_effects"].append(joint_effect)
            print(f"    ✓ Joint {joint_idx} tested")
        
        print("\n[Robot Info] Collection complete!")
        return info


def robot_info_formatting(robot_info: Dict) -> str:
    """
    Format robot kinematic information into a readable string for LLM prompt.
    
    Args:
        robot_info: Dictionary containing robot kinematic information
    
    Returns:
        Formatted string describing robot kinematics
    """
    if not robot_info:
        return ""
    
    formatted = f"\n{'='*60}\n"
    formatted += f"ROBOT KINEMATIC INFORMATION\n"
    formatted += f"{'='*60}\n\n"
    
    # Robot name and basic info
    formatted += f"Robot: {robot_info.get('robot_name', 'Unknown')}\n"
    formatted += f"Controller: {robot_info.get('controller_type', 'Unknown')}\n"
    formatted += f"Data collected: {robot_info.get('collected_at', 'Unknown')}\n\n"
    
    # Initial state
    initial_state = robot_info.get('initial_state', {})
    if initial_state:
        formatted += f"INITIAL ROBOT STATE (Home Position):\n"
        formatted += f"-" * 60 + "\n"
        
        ee_pos = initial_state.get('ee_position', [])
        if ee_pos:
            formatted += f"  End-Effector Position:\n"
            formatted += f"    X: {ee_pos[0]:.4f} m\n"
            formatted += f"    Y: {ee_pos[1]:.4f} m\n"
            formatted += f"    Z: {ee_pos[2]:.4f} m\n\n"
        
        ee_ori = initial_state.get('ee_orientation_rpy', [])
        if ee_ori:
            formatted += f"  End-Effector Orientation (Roll-Pitch-Yaw):\n"
            formatted += f"    Roll:  {ee_ori[0]:.4f} rad ({np.rad2deg(ee_ori[0]):.2f}°)\n"
            formatted += f"    Pitch: {ee_ori[1]:.4f} rad ({np.rad2deg(ee_ori[1]):.2f}°)\n"
            formatted += f"    Yaw:   {ee_ori[2]:.4f} rad ({np.rad2deg(ee_ori[2]):.2f}°)\n\n"
        
        joint_pos = initial_state.get('joint_positions', [])
        if joint_pos:
            formatted += f"  Initial Joint Positions:\n"
            for i, pos in enumerate(joint_pos):
                formatted += f"    Joint {i}: {pos:.4f} rad ({np.rad2deg(pos):.2f}°)\n"
            formatted += "\n"
    
    # Joint effects
    joint_effects = robot_info.get('joint_effects', [])
    if joint_effects:
        delta_deg = robot_info.get('joint_delta_deg', 30.0)
        formatted += f"JOINT MOVEMENT EFFECTS (±{delta_deg:.0f}° movement):\n"
        formatted += f"-" * 60 + "\n"
        formatted += "How end-effector position changes when each joint moves:\n\n"
        
        for je in joint_effects:
            idx = je['joint_index']
            formatted += f"  Joint {idx}:\n"
            
            # Positive delta
            pos_delta = je.get('positive_delta', {})
            if pos_delta and 'ee_position_change' in pos_delta:
                change = pos_delta['ee_position_change']
                formatted += f"    +{delta_deg:.0f}° → EE moves: "
                formatted += f"X{change[0]:+.4f}m, Y{change[1]:+.4f}m, Z{change[2]:+.4f}m\n"
            
            # Negative delta
            neg_delta = je.get('negative_delta', {})
            if neg_delta and 'ee_position_change' in neg_delta:
                change = neg_delta['ee_position_change']
                formatted += f"    -{delta_deg:.0f}° → EE moves: "
                formatted += f"X{change[0]:+.4f}m, Y{change[1]:+.4f}m, Z{change[2]:+.4f}m\n"
            
            # Analysis: dominant direction
            if pos_delta and 'ee_position_change' in pos_delta:
                change = np.array(pos_delta['ee_position_change'])
                abs_change = np.abs(change)
                max_idx = np.argmax(abs_change)
                axis_names = ['X', 'Y', 'Z']
                if abs_change[max_idx] > 0.01:  # Significant movement
                    formatted += f"    → Primary effect: {axis_names[max_idx]}-axis movement\n"
            
            formatted += "\n"
    
    formatted += f"{'='*60}\n"
    return formatted


def load_codes(codes_path: str = "data/logs/codes.jsonl") -> List[Dict]:
    """
    Load existing code examples from jsonl file.
    
    Args:
        codes_path: Path to the codes.jsonl file
    
    Returns:
        List of code dictionaries
    """
    if not os.path.exists(codes_path):
        return []
    
    codes = []
    with open(codes_path, 'r') as f:
        for line in f:
            if line.strip():
                codes.append(json.loads(line))
    return codes


def save_code(code: Dict, codes_path: str = "data/logs/codes.jsonl"):
    """
    Save a new code example to jsonl file.
    
    Args:
        code: Code dictionary to save
        codes_path: Path to the codes.jsonl file
    """
    os.makedirs(os.path.dirname(codes_path), exist_ok=True)
    with open(codes_path, 'a') as f:
        f.write(json.dumps(code) + '\n')


def interpret_command_llm(
    command: str, 
    controller_type: str = "OSC_POSE", 
    codes_path: str = "data/logs/codes.jsonl",
    model_name: str = "gpt-4o-mini",
    use_cached: bool = True,
    robot_info: Dict = None
) -> Dict:
    """
    Use LLM to interpret natural language commands into control sequences.
    
    Args:
        command: Natural language command (e.g., "bow to greet", "wave hand")
        controller_type: Type of controller being used
        codes_path: Path to codes.jsonl file for examples
        model_name: LLM model to use (default: gpt-4o-mini)
        use_cached: If True, check cached codes first
    
    Returns:
        Dictionary containing:
        - 'command': Original command
        - 'description': Human-readable description of what the robot will do
        - 'controller': Controller type
        - 'sequence': List of (action_vector, num_steps, description) dicts
        - 'rest_steps': Number of steps to rest between actions
    """
    
    # Load existing codes
    codes = load_codes(codes_path)
    
    # Check if command already exists in cache
    if use_cached:
        command_lower = command.lower()
        for code in codes:
            if code.get('command', '').lower() == command_lower:
                print(f"[LLM] Found cached code for command: '{command}'")
                return code
    
    # If no cached code found, use LLM to generate
    print(f"[LLM] Generating new code for command: '{command}'")
    
    # Select a random example from existing codes as one-shot example
    example = None
    if codes:
        example = random.choice(codes)
    
    # Create prompt with task instructions
    system_prompt = """You are a robot control code generator for a robotic arm manipulator.

TASK: Generate a control sequence to perform a given gesture/motion command.

PROCESS (3 Steps):
1. INITIAL POSE: Determine if a specific starting pose is needed for this gesture
   - Consider the robot's current home position
   - Plan any preparatory movements (e.g., raise arm before waving)

2. MOTION PLANNING: Design the movement sequence
   - Break down the gesture into key poses/movements
   - Consider: amplitude (how far to move), speed (steps per movement), timing (pauses)
   - Ensure smooth, natural, human-like motion
   - Think about rhythm and repetition if needed

3. CODE GENERATION: Convert the plan into control actions

CONTROLLER SPECIFICATION:
- Type: OSC_POSE (Operational Space Control - Position and Orientation)
- Action format: [dx, dy, dz, droll, dpitch, dyaw]
  * dx, dy, dz: Linear velocity in meters (-0.2 to 0.2 range)
  * droll, dpitch, dyaw: Angular velocity in radians (-0.5 to 0.5 range)
- Steps: Number of simulation steps to execute the action (20-100 typical)
  * More steps = longer duration = more movement distance
  * Total movement ≈ action_value × steps × 0.05m (per step limit)
- Rest_steps: Pause duration between actions (0-50 typical)

OUTPUT FORMAT (JSON only, no markdown):
{
    "command": "original command",
    "description": "Brief description of what the robot will do",
    "controller": "OSC_POSE",
    "sequence": [
        {"action": [dx, dy, dz, dr, dp, dy], "steps": 50, "description": "what this step does", "rest_steps": 20},
        ...
    ]
}

IMPORTANT GUIDELINES:
- Start from the robot's home position (provided below)
- Keep movements smooth, safe, and natural
- Use appropriate speeds: quick for gestures, slow for precise movements
- Add pauses (rest_steps) between movements for natural rhythm
- Consider the gesture's cultural/social context"""

    # Add robot kinematic information if available
    if robot_info:
        system_prompt += "\n\n" + robot_info_formatting(robot_info)
    
    # Add example code section
    if example:
        system_prompt += f"\n\n{'='*60}\n"
        system_prompt += "EXAMPLE CODE (for reference):\n"
        system_prompt += f"{'='*60}\n"
        system_prompt += f"Command: \"{example['command']}\"\n"
        system_prompt += f"Output:\n{json.dumps(example, indent=2)}\n"
        system_prompt += f"{'='*60}\n"

    user_prompt = f"\n\nNow generate the control sequence for this command:\n\n"
    user_prompt += f"Command: \"{command}\"\n\n"
    user_prompt += "Follow the 3-step process:\n"
    user_prompt += "1. Think about initial pose\n"
    user_prompt += "2. Plan the motion (amplitude, speed, timing)\n"
    user_prompt += "3. Generate the JSON code\n\n"
    
    # Initialize LLM
    llm = MyLLMEval(model_name)
    llm.max_tokens = 1000
    
    # Generate response
    response = llm.generate(user_prompt, system_prompt=system_prompt)
    print(response)
    # Parse JSON response
    # Remove markdown code blocks if present
    response = response.strip()
    if response.startswith("```"):
        lines = response.split('\n')
        response = '\n'.join(lines[1:-1]) if len(lines) > 2 else response
        if response.startswith("json"):
            response = response[4:].strip()
    
    code = json.loads(response)
    
    # Validate required fields
    required_fields = ['description', 'controller', 'sequence']
    for field in required_fields:
        if field not in code:
            raise ValueError(f"Missing required field: {field}")
    
    # Add command to code
    code['command'] = command
    print(f"[LLM] Generated code: {code}")
    save_code(code, codes_path)
    
    return code
        


def execute_command(controller: RobotController, command: str, model: str = 'gpt-4o-mini'):
    """
    High-level function to interpret and execute a natural language command.
    
    Args:
        controller: RobotController instance
        command: Natural language command
        model: LLM model to use
    """
    print(f"\n{'='*60}")
    print(f"User Command: '{command}'")
    print(f"{'='*60}")
    
    # Get LLM interpretation
    print("\n[LLM] Interpreting command...")
    interpretation = interpret_command_llm(
        command, 
        controller.controller_name, 
        model_name=model,
        robot_info=controller.robot_info
    )
    
    if interpretation is None:
        print(f"[LLM] Failed to interpret command. Falling back to default motion.")
        return
    
    print(f"\n[LLM] Interpretation:")
    print(f"  Description: {interpretation['description']}")
    print(f"  Controller: {interpretation['controller']}")
    print(f"  Number of actions: {len(interpretation['sequence'])}")
    
    # Execute the sequence
    print(f"\n[Robot] Executing action sequence...")
    for i, action_info in enumerate(interpretation['sequence']):
        print(f"\n  Step {i+1}/{len(interpretation['sequence'])}: {action_info['description']}")
        
        # Create action array
        action = np.zeros(controller.action_dim + controller.gripper_dim)
        action_values = action_info['action']
        action[:len(action_values)] = action_values
        
        # Execute action
        controller.execute_action(action, action_info['steps'])
        
        # Rest between actions
        if i < len(interpretation['sequence']) - 1:
            controller.rest(action_info.get('rest_steps', 0))
    
    # Return to initial pose
    print("\n[Robot] Returning to initial pose...")
    controller.return_to_initial_pose(steps=100)
    print("[Robot] Command execution completed!\n")


def main(
    mode: str = 'all',  # single, all, interactive
    cues_file: str = 'data/seed/cues.txt', 
    codes_file: str = 'data/logs/codes.jsonl', 
    model: str = 'gpt-4o-mini', 
    no_render: bool = False,
    robot_name: str = 'IIWA',
    env_name: str = 'Lift',
    controller_name: str = 'OSC_POSE',
    robots_dir: str = 'data/robots'
    ):    
    
    print("="*60)
    print("Natural Language Robot Control Demo")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Robot: {robot_name}")
    print(f"Model: {model}")
    print(f"Codes file: {codes_file}")
    print("="*60 + "\n")
    
    
    # Initialize robot controller for normal modes
    controller = RobotController(
        env_name=env_name,
        robot_name=robot_name,
        controller_name=controller_name,
        robots_dir=robots_dir
    )
    
    try:
        if mode == 'single':
            # Test mode: single command
            print(f"\n[TEST MODE] Testing command: '{command}'")
            
            # Print initial robot state
            print("\nInitial Robot State:")
            controller.print_robot_state(arm="right")
            
            # Execute command
            execute_command(controller, command)
            
            # Print final state
            print("\nFinal Robot State:")
            controller.print_robot_state(arm="right")
            
        elif mode == 'all':
            # Batch mode: process all cues from file
            print(f"\n[BATCH MODE] Loading commands from: {cues_file}")
            
            if not os.path.exists(cues_file):
                print(f"Error: Cues file not found: {cues_file}")
                exit(1)
            
            with open(cues_file, "r") as f:
                commands = [line.strip() for line in f.readlines() if line.strip()]
            
            print(f"Found {len(commands)} commands to process\n")
            
            for i, command in enumerate(commands):
                print(f"\n{'='*60}")
                print(f"Command {i+1}/{len(commands)}")
                print(f"{'='*60}")
                
                execute_command(controller, command, model=model)
                
                # Optional: Add a small delay between commands
                time.sleep(1)
            
            print(f"\n[BATCH MODE] Completed processing {len(commands)} commands")
            
        elif mode == 'interactive':
            # Interactive mode: user input
            print("\n[INTERACTIVE MODE]")
            print("Enter commands to control the robot.")
            print("Type 'quit', 'exit', or 'q' to exit.")
            print("Type 'state' to see current robot state.\n")
            
            while True:
                try:
                    command = input("\nEnter command: ").strip()
                    
                    if not command:
                        continue
                    
                    if command.lower() in ['quit', 'exit', 'q']:
                        print("Exiting interactive mode...")
                        break
                    
                    if command.lower() == 'state':
                        controller.print_robot_state(arm="right")
                        continue
                    
                    execute_command(controller, command)
                    
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user. Exiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue
        
    finally:
        controller.close()
        print("\nDemo completed. Environment closed.")




if __name__ == "__main__":
    fire.Fire(main)
