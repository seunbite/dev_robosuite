# LLM Prompt Structure for Robot Control Code Generation

This document describes how the LLM prompt is structured for generating robot control sequences.

## Prompt Components

The prompt is divided into three main sections:

### 1. TASK INSTRUCTIONS

The system prompt includes a clear 3-step process:

```
PROCESS (3 Steps):
1. INITIAL POSE: Determine if a specific starting pose is needed
   - Consider the robot's current home position
   - Plan any preparatory movements

2. MOTION PLANNING: Design the movement sequence
   - Break down the gesture into key poses/movements
   - Consider: amplitude (how far), speed (steps), timing (pauses)
   - Ensure smooth, natural, human-like motion

3. CODE GENERATION: Convert the plan into control actions
```

This guides the LLM to think through the problem systematically before generating code.

### 2. EXAMPLE CODE

One random example from `codes.jsonl` is provided to demonstrate the expected format:

```json
{
  "command": "bow to greet",
  "description": "...",
  "controller": "OSC_POSE",
  "sequence": [
    {"action": [0, 0, -0.15, 0, 0, 0], "steps": 10, ...}
  ]
}
```

This provides:
- Format reference
- Real working examples
- Context for what "good" code looks like

### 3. ROBOT KINEMATIC INFORMATION

The robot's physical properties and behavior are included via `robot_info_formatting()`:

```
ROBOT KINEMATIC INFORMATION
============================================================

Robot: IIWA
Controller: OSC_POSE

INITIAL ROBOT STATE (Home Position):
  End-Effector Position:
    X: 0.3912 m
    Y: 0.0000 m
    Z: 0.4123 m
  
  End-Effector Orientation (Roll-Pitch-Yaw):
    Roll:  0.0000 rad (0.00°)
    Pitch: 0.0000 rad (0.00°)
    Yaw:   1.5708 rad (90.00°)
  
  Initial Joint Positions:
    Joint 0: 0.0000 rad (0.00°)
    Joint 1: -0.7854 rad (-45.00°)
    ...

JOINT MOVEMENT EFFECTS (±20° movement):
  Joint 0:
    +20° → EE moves: X-0.0412m, Y+0.1023m, Z+0.0000m
    -20° → EE moves: X-0.0412m, Y-0.1023m, Z+0.0000m
    → Primary effect: Y-axis movement
  
  Joint 1:
    +20° → EE moves: X+0.0000m, Y+0.0000m, Z+0.1234m
    -20° → EE moves: X+0.0000m, Y+0.0000m, Z-0.1234m
    → Primary effect: Z-axis movement
  ...
```

This information helps the LLM:
- Understand the robot's workspace
- Know the starting position
- Predict how movements will affect the end-effector
- Generate physically plausible motions

## Complete Prompt Flow

```
[SYSTEM PROMPT]
┌─────────────────────────────────────┐
│ 1. Task Instructions                │
│    - 3-step process                 │
│    - Controller specifications      │
│    - Output format                  │
│    - Guidelines                     │
├─────────────────────────────────────┤
│ 2. Robot Kinematic Information      │
│    - Initial state                  │
│    - Joint effects                  │
├─────────────────────────────────────┤
│ 3. Example Code                     │
│    - One-shot example               │
└─────────────────────────────────────┘

[USER PROMPT]
┌─────────────────────────────────────┐
│ Command: "wave hand"                │
│                                     │
│ Follow the 3-step process:          │
│ 1. Think about initial pose         │
│ 2. Plan the motion                  │
│ 3. Generate the JSON code           │
│                                     │
│ Return ONLY the JSON object         │
└─────────────────────────────────────┘

[LLM OUTPUT]
┌─────────────────────────────────────┐
│ {                                   │
│   "command": "wave hand",           │
│   "description": "...",             │
│   "sequence": [...]                 │
│ }                                   │
└─────────────────────────────────────┘
```

## Benefits of This Structure

1. **Systematic Thinking**: The 3-step process encourages deliberate planning
2. **Context Awareness**: Robot info helps generate physically realistic motions
3. **Format Consistency**: Example code ensures proper JSON structure
4. **Physical Grounding**: Kinematic data prevents impossible movements
5. **Natural Motion**: Guidelines emphasize smooth, human-like gestures

## Implementation

The prompt is constructed in `interpret_command_llm()`:

```python
def interpret_command_llm(command, ..., robot_info=None):
    # 1. Build system prompt with instructions
    system_prompt = """..."""
    
    # 2. Add robot kinematic information
    if robot_info:
        system_prompt += robot_info_formatting(robot_info)
    
    # 3. Add example code
    if example:
        system_prompt += format_example(example)
    
    # 4. Create user prompt with command
    user_prompt = f"Command: {command}..."
    
    # 5. Call LLM
    response = llm.generate(user_prompt, system_prompt)
```

## Return to Initial Pose

After each command execution, the robot returns to its initial pose:

```python
def execute_command(controller, command):
    # 1. Generate code
    code = interpret_command_llm(command, robot_info=controller.robot_info)
    
    # 2. Execute sequence
    for action in code['sequence']:
        controller.execute_action(action)
    
    # 3. Return to initial pose
    controller.return_to_initial_pose()
```

This ensures:
- Consistent starting position for each gesture
- No drift over multiple commands
- Reproducible behavior
- Match with robot_info initial state

## Customization

To modify the prompt:

1. **Change instructions**: Edit the system_prompt string
2. **Adjust robot info format**: Modify `robot_info_formatting()`
3. **Add more examples**: Include multiple examples instead of one
4. **Change process steps**: Update the 3-step breakdown

## Example Output Quality

With this prompt structure, the LLM generates high-quality code:

```json
{
  "command": "Head nodding",
  "description": "The robot will simulate head nodding by pitching forward and back",
  "controller": "OSC_POSE",
  "sequence": [
    {
      "action": [0, 0, -0.05, 0, 0.2, 0],
      "steps": 30,
      "description": "Nod down - move down and pitch forward",
      "rest_steps": 15
    },
    {
      "action": [0, 0, 0.05, 0, -0.2, 0],
      "steps": 30,
      "description": "Nod up - move up and pitch back",
      "rest_steps": 15
    }
  ]
}
```

Notice:
- Appropriate action values for the gesture
- Reasonable step counts for smooth motion
- Rest periods for natural rhythm
- Clear descriptions of each movement



