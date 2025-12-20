# Robot Control with Natural Language Commands

This demo allows you to control a robot manipulator using natural language commands. The system uses GPT to interpret commands and generate control sequences.

## Features

- **LLM-Powered Command Interpretation**: Uses GPT-4o-mini to convert natural language to robot control code
- **One-Shot Learning**: Automatically learns from existing examples in `codes.jsonl`
- **Caching**: Previously generated commands are cached to avoid repeated API calls
- **Multiple Modes**: Test single commands, batch process multiple commands, or use interactive mode
- **State Monitoring**: Query robot position, orientation, and joint angles in real-time
- **Robot Kinematic Information**: Automatically collects and caches robot kinematic data to improve LLM code generation accuracy

## Setup

### 1. Install Dependencies

```bash
# Install robosuite
pip install robosuite

# Install mylmeval (for GPT API)
cd /Users/sb/Downloads/workspace/mylmeval
pip install -e .
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Collect Robot Kinematic Information (First Time Setup)

Before using the robot, collect its kinematic information. This only needs to be done once per robot:

```bash
cd /Users/sb/Downloads/workspace/dev_robosuite/robosuite/demos
python demo_code.py --mode collect_robot_info --robot-name IIWA
```

This will:
1. Initialize the robot in neutral position
2. Move each joint by ±30 degrees
3. Record how the end-effector position changes
4. Save the information to `data/robots/IIWA.json`

The collected information helps the LLM generate more accurate control sequences.

### Test Mode (Single Command)

Test a single command to see how it works:

```bash
python demo_code.py --mode single --command "wave hand"
```

### Batch Mode (Process All Cues)

Process all commands from `cues.txt`:

```bash
python demo_code.py --mode all --cues-file data/seed/cues.txt
```

This will:
1. Load all gestures from `cues.txt`
2. For each gesture, check if code exists in `codes.jsonl`
3. If not, use GPT to generate the control sequence (with one random example and robot info)
4. Execute the sequence on the robot
5. Save the new code to `codes.jsonl`

### Interactive Mode

Control the robot interactively:

```bash
python demo_code.py --mode interactive
```

Then enter commands like:
- `wave hand`
- `bow to greet`
- `rotate around`
- `state` - show current robot state
- `quit` - exit

## Command Line Options

```
--mode {single,all,interactive,collect_robot_info}  
                                 Mode to run (default: all)
--cues-file PATH                 Path to cues.txt (default: data/seed/cues.txt)
--codes-file PATH                Path to codes.jsonl (default: data/logs/codes.jsonl)
--model TEXT                     LLM model to use (default: gpt-4o-mini)
--robot-name TEXT                Robot name (default: IIWA)
--env-name TEXT                  Environment name (default: Lift)
--controller-name TEXT           Controller type (default: OSC_POSE)
--robots-dir PATH                Directory for robot info files (default: data/robots)
--no-render                      Disable rendering for faster processing
```

## Files

### `codes.jsonl`

Stores generated control sequences. Format:

```json
{
  "command": "wave hand",
  "description": "Wave hand left and right",
  "controller": "OSC_POSE",
  "sequence": [
    {"action": [0, 0.1, 0, 0, 0, 0], "steps": 40, "description": "Move right", "rest_steps": 0},
    {"action": [0, -0.1, 0, 0, 0, 0], "steps": 40, "description": "Move left", "rest_steps": 0}
  ]
}
```

### `cues.txt`

List of gestures/commands to process in batch mode. One command per line.

### `data/robots/{robot_name}.json`

Stores robot kinematic information. Format:

```json
{
  "robot_name": "IIWA",
  "controller_type": "OSC_POSE",
  "collected_at": "2025-11-20 10:30:00",
  "joint_delta_deg": 30.0,
  "initial_state": {
    "joint_positions": [0.0, -0.785, 0.0, -1.571, 0.0, 0.785, 0.0],
    "ee_position": [0.391, 0.0, 0.412],
    "ee_orientation_rpy": [0.0, 0.0, 1.571]
  },
  "joint_effects": [
    {
      "joint_index": 0,
      "initial_position": 0.0,
      "positive_delta": {
        "joint_position": 0.524,
        "ee_position": [0.350, 0.102, 0.412],
        "ee_position_change": [-0.041, 0.102, 0.0]
      },
      "negative_delta": {
        "joint_position": -0.524,
        "ee_position": [0.350, -0.102, 0.412],
        "ee_position_change": [-0.041, -0.102, 0.0]
      }
    }
  ]
}
```

This information is automatically used by the LLM to generate more accurate control sequences.

## API Functions

### RobotController Class

```python
controller = RobotController(
    env_name="Lift",
    robot_name="IIWA",
    controller_name="OSC_POSE"
)

# Get robot state
position = controller.get_ee_position()  # [x, y, z]
orientation_rpy = controller.get_ee_orientation(return_type="rpy")  # [roll, pitch, yaw]
orientation_quat = controller.get_ee_orientation(return_type="quat")  # [x, y, z, w]
joint_positions = controller.get_joint_positions()  # Joint angles
joint_velocities = controller.get_joint_velocities()  # Joint velocities

# Print current state
controller.print_robot_state(arm="right")

# Get complete state
state = controller.get_robot_state(arm="right")
# Returns: {'ee_pos', 'ee_ori_rpy', 'ee_ori_quat', 'joint_pos', 'joint_vel'}

# Execute actions
action = np.array([dx, dy, dz, droll, dpitch, dyaw, gripper])
controller.execute_action(action, steps=75)
controller.rest(steps=75)  # Return to neutral
```

### LLM Command Interpretation

```python
from demo_code import interpret_command_llm

# Generate control sequence from command
code = interpret_command_llm(
    command="wave hand",
    controller_type="OSC_POSE",
    codes_path="data/logs/codes.jsonl",
    model_name="gpt-4o-mini",
    use_cached=True  # Check cache first
)

# Returns:
# {
#   'command': 'wave hand',
#   'description': 'Wave hand left and right',
#   'controller': 'OSC_POSE',
#   'sequence': [...],
#   'rest_steps': 30
# }
```

## Control Space

The robot uses OSC_POSE controller with 6D control:

- **dx, dy, dz**: Linear movement in meters (typically -0.2 to 0.2)
  - x: forward/backward
  - y: left/right
  - z: up/down

- **droll, dpitch, dyaw**: Angular movement in radians (typically -0.5 to 0.5)
  - roll: rotation around x-axis
  - pitch: rotation around y-axis
  - yaw: rotation around z-axis

### Understanding Steps

Each action is executed for a specified number of simulation steps. **Steps control how long the action is repeated, not the distance:**

- The controller has `output_max/min` limits (typically ±0.05m per step for position)
- If you command `[0, 0, -0.15, 0, 0, 0]`, each step moves at most 0.05m
- With `steps=3`, the robot moves ~0.15m (0.05m × 3)
- With `steps=10`, the robot moves ~0.5m (0.05m × 10)
- With `steps=50`, the robot moves ~2.5m (0.05m × 50)

**Rule of thumb:** 
```
Total movement ≈ action_value (clamped to output_max) × steps
Recommended steps: 20-50 for typical gestures
```

## Examples

### Example 1: Simple Wave

```python
{
  "command": "wave hand",
  "sequence": [
    {"action": [0, 0.1, 0, 0, 0, 0], "steps": 40, "description": "Move right"},
    {"action": [0, -0.1, 0, 0, 0, 0], "steps": 40, "description": "Move left"},
    {"action": [0, 0.1, 0, 0, 0, 0], "steps": 40, "description": "Move right"},
    {"action": [0, -0.05, 0, 0, 0, 0], "steps": 40, "description": "Return center"}
  ]
}
```

### Example 2: Bow

```python
{
  "command": "bow to greet",
  "sequence": [
    {"action": [0, 0, -0.15, 0, 0, 0], "steps": 100, "description": "Move down"},
    {"action": [0, 0, 0, 0, 0, 0], "steps": 50, "description": "Pause"},
    {"action": [0, 0, 0.15, 0, 0, 0], "steps": 100, "description": "Move up"}
  ]
}
```

## Troubleshooting

### API Key Error

Make sure your OpenAI API key is set:
```bash
export OPENAI_API_KEY="sk-..."
```

### JSON Parse Error

If the LLM returns invalid JSON, the system will fall back to a default motion. Check the error message and try again.

### Cached Commands

To force regeneration of a command (ignore cache), delete the corresponding line from `codes.jsonl` or set `use_cached=False` in the code.

### Robot Simulation Issues

If the robot behaves unexpectedly:
1. Check that action values are reasonable (not too large)
2. Increase `rest_steps` between actions
3. Reduce the magnitude of movements
4. Check joint limits with `controller.print_robot_state()`

## Quick Start Example

```bash
# 1. First time: Collect robot information
python demo_code.py --mode collect_robot_info --robot-name IIWA

# 2. Test a single command
python demo_code.py --mode single

# 3. Process all gestures from cues.txt
python demo_code.py --mode all --cues-file data/seed/cues.txt

# 4. Interactive mode
python demo_code.py --mode interactive
```

## Advanced Usage

### Custom Controller Settings

```python
controller = RobotController(
    env_name="Lift",
    robot_name="Panda",  # Try different robots
    controller_name="JOINT_POSITION"  # Try different controllers
)
```

Available controllers:
- `OSC_POSE`: 6D end-effector control (position + orientation)
- `OSC_POSITION`: 3D end-effector control (position only)
- `IK_POSE`: Inverse kinematics pose control
- `JOINT_POSITION`: Direct joint position control
- `JOINT_VELOCITY`: Direct joint velocity control

### Velocity Control

See velocity control options in the main documentation.

## Notes

- The first run may take longer as the LLM generates new codes
- Subsequent runs with the same commands will be much faster (cached)
- Generated codes are automatically saved to `codes.jsonl`
- The system learns from examples - more examples = better performance
- For batch processing, consider using `--no-render` for faster execution

