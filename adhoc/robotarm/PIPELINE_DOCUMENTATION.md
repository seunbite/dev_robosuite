# Automated Pipeline for Robot Expressive Cue Generation

This document describes the automated pipeline for generating expressive robot movements and cues. The pipeline uses a configuration-driven approach where movements are defined as structured JSON, then executed through pose generation and Jacobian-based joint selection.

---

## Motion Configuration Structure

Motions are defined as structured JSON configurations with hierarchical movement definitions. The system supports two types of movements: **pose** (static configurations) and **movement** (dynamic joint rotations).

### Configuration Example

```json
{
  "cue": "waving",
  "movements": [
    {
      "type": "pose",
      "parameters": {
        "pose": "Elbow_down",
        "speed": 1.0,
        "hold_time": 0.5
      }
    },
    {
      "type": "movement",
      "parameters": {
        "repetition": 3,
        "axis": "y",
        "joint": "distal",
        "directions": [
          {"direction": "pos", "degrees": 30, "speed": 1.0, "hold_time": 0.5},
          {"direction": "neg", "degrees": 30, "speed": 1.0, "hold_time": 0.5}
        ]
      }
    }
  ]
}
```

### Movement Types

#### 1. Pose (`type: "pose"`)

Sets the robot to a specific pose from the database. Poses are pre-generated through brute-force search and orientation filtering (see Stage 1 below).

**Parameters**:
- `pose`: Pose name (e.g., "Elbow_down", "Elbow_up", "Stretched_out")
- `speed`: Transition speed (1.0 = 1 second, 2.0 = 0.5 seconds) - only used when transitioning from another pose
- `hold_time`: Duration to hold the pose (seconds)

**Example**:
```json
{
  "type": "pose",
  "parameters": {
    "pose": "Elbow_down",
    "speed": 1.0,
    "hold_time": 0.5
  }
}
```

#### 2. Movement (`type: "movement"`)

Executes joint rotations with repetitions. The system automatically selects the best joint for the specified axis using Jacobian analysis (see Stage 2 below).

**Parameters**:
- `repetition`: Number of times to repeat the movement sequence
- `axis`: Target axis for movement ("x", "y", or "z")
- `joint`: Joint location preference ("proximal" or "distal")
- `directions`: List of movement steps, each with:
  - `direction`: "pos" (positive) or "neg" (negative)
  - `degrees`: Rotation amount in degrees
  - `speed`: Movement speed (1.0 = 1 second, 2.0 = 0.5 seconds)
  - `hold_time`: Hold duration after movement (seconds)

**Example**:
```json
{
  "type": "movement",
  "parameters": {
    "repetition": 3,
    "axis": "y",
    "joint": "distal",
    "directions": [
      {"direction": "pos", "degrees": 30, "speed": 1.0, "hold_time": 0.5},
      {"direction": "neg", "degrees": 30, "speed": 1.0, "hold_time": 0.5}
    ]
  }
}
```

---

## Stage 1: Pose Generation (Brute-Force Search with Filtering)

**File**: `find_closest_poses.py`

Poses are generated through a brute-force search process, then filtered based on end-effector orientation criteria.

### Brute-Force Search

The system generates all possible joint angle combinations using a discrete angle set. By default, each active joint is tested at three angles: -90°, 0°, and 90° (configurable via `angle_step_deg`, `angle_min_deg`, `angle_max_deg`).

**Process**:
- For a robot with N active joints, this generates 3^N pose combinations
- Each combination is evaluated by setting the joint positions and computing the end-effector orientation
- All poses are stored with their computed orientations (roll, pitch, yaw in degrees and radians)

**Code Reference**: Lines 242-296 in `find_closest_poses.py`
```python
# Generate combinations
selected_combinations = list(product(*[range(num) for num in num_angles_per_joint]))

for combo_idx, angle_indices in tqdm(enumerate(selected_combinations), ...):
    # Set joint positions and compute orientation
    joint_pos = self.initial_joint_pos.copy()
    for i, active_joint_idx in enumerate(self.active_joint_indices):
        angle_value = joint_angle_arrays[i][angle_indices[i]]
        joint_pos[active_joint_idx] = angle_value
```

### Orientation-Based Filtering

After generating all poses, the system filters them based on target orientation criteria:

1. **Orientation Matching**: Each pose is scored by its orientation difference from target roll, pitch, and yaw angles
   - Uses circular distance calculation to handle angle wrapping (e.g., 359° and 1° are considered close)
   - Poses with orientation difference > 60° are filtered out (line 381)

2. **Top-K Selection**: The top K poses (default: 30-100) with smallest orientation differences are selected (line 402)

3. **Spatial Filtering**: Selected poses are filtered to ensure the end-effector is positioned in front of the robot root/base
   - **Forward Position Check**: Only poses where `ee_x > root_x` are kept (lines 435-452)
   - This ensures the end-effector is positioned forward relative to the robot base (positive X direction)
   - Poses where the end-effector is behind or at the same X position as the root are filtered out
   - If no front poses are found, all poses are used as fallback (with warning)

**Code Reference**: Lines 435-452 in `find_closest_poses.py`
```python
# Filter poses where EE is in front of root (x > root_x)
front_poses = []
behind_poses = []
for pose in top_poses:
    ee_x = pose["ee_position"]["x"]
    root_x = pose["root_position"]["x"]
    if ee_x > root_x:
        front_poses.append(pose)
    else:
        behind_poses.append(pose)

# If we have front poses, use them; otherwise use all poses (fallback)
if front_poses:
    top_poses = front_poses
```

4. **Distance-Based Sorting**: Filtered poses are sorted by root-to-end-effector distance (line 457)
   - Poses with smaller distances are preferred (end-effector closer to base)
   - Helps select poses that are more naturally reachable

**Code Reference**: Lines 305-373 (orientation calculation), 380-403 (filtering and top-K selection), 435-457 (spatial filtering and sorting)

**Output**: A JSONL database (`closest_poses_results.jsonl`) containing pose candidates with:
- Joint angles (degrees and radians)
- End-effector orientation (roll, pitch, yaw)
- Root-to-EE distance
- Rank and pose_id

**Pose Definitions**: Poses are defined by their target orientations:
- `Elbow_down`: roll=0°, yaw=0°
- `Elbow_up`: roll=180°, yaw=0°
- `Stretched_out`: roll=-90°, yaw=-90°

When a configuration specifies `"pose": "Elbow_down"`, the system selects from the pre-filtered poses matching that orientation, providing 30-50 candidate poses for variation.

---

## Stage 2: Movement Generation (Jacobian-Based Joint Selection)

**File**: `alphabet_jacobian.py`

Dynamic movements are generated by analyzing the Jacobian matrix to identify the best joint(s) for movement along a specific axis.

### Jacobian Computation

For each initial pose, the system computes the Jacobian matrix that relates joint velocities to end-effector velocities (both linear and angular).

**Process**:
- Uses MuJoCo's `mj_jacSite` function to compute the 6×N Jacobian matrix
  - First 3 rows: position Jacobian (relates to linear velocity)
  - Last 3 rows: rotation Jacobian (relates to angular velocity)
  - Columns correspond to each joint DOF
- The Jacobian is computed at the end-effector site location

**Code Reference**: Lines 368-378 in `alphabet_jacobian.py`
```python
# Compute Jacobian for end effector site (6xN: 3 for position, 3 for orientation)
jac_pos = np.zeros((3, mujoco_model.nv))
jac_rot = np.zeros((3, mujoco_model.nv))
mujoco.mj_jacSite(mujoco_model, mujoco_data, jac_pos, jac_rot, site_id)
jac_full = np.vstack([jac_pos, jac_rot])
```

### Joint Selection for Directional Movement

The system identifies the best joint(s) for movement along a specific axis (X, Y, or Z) by analyzing the position Jacobian.

**Scoring Function** (lines 489-493):
For each joint, the score is calculated as:
```
score = |target_axis_contribution| / (sqrt(other_axis1² + other_axis2²) + ε)
```

This metric favors joints that:
1. Have strong contribution to the target axis (numerator)
2. Have minimal contribution to the other two axes (denominator)

This ensures the selected joint moves the end-effector primarily along the desired axis while maintaining a stable plane.

**Code Reference**: Lines 447-525 in `alphabet_jacobian.py`, specifically `_find_and_sort_joints_for_axis()`

### Joint Location Preference

After sorting joints by score, the system applies a location preference:

- **Proximal** (shoulder/base): Selects the joint with the smallest DOF ID among the top 3 candidates
- **Distal** (wrist/end-effector): Selects the joint with the largest DOF ID among the top 3 candidates

**Code Reference**: Lines 218-236 in `motion_generation.py`
```python
if joint_preference == 'proximal':
    selected = min(top_3, key=lambda x: x[2])  # x[2] is joint_dof_id
elif joint_preference == 'distal':
    selected = max(top_3, key=lambda x: x[2])
```

This allows control over whether movements are driven by shoulder/base joints (larger, more stable movements) or wrist joints (finer, more precise movements).

**Execution**: When a configuration specifies `"axis": "y"` and `"joint": "distal"`, the system:
1. Computes the Jacobian at the current pose
2. Scores all joints for Y-axis movement
3. Selects the top 3 joints
4. Chooses the most distal joint from the top 3
5. Rotates that joint according to the `directions` parameters

---

## Stage 3: Augmentation and Variation

**Files**: `llm_variation.py` (semantic variation), `motion_config.json` (configuration), `motion_generation.py` (execution)

The system supports systematic variation through semantic prompts that automatically modify motion parameters and waypoints. The variation system uses rule-based mapping (with LLM integration planned for future).

### Semantic Prompt Processing

**File**: `llm_variation.py`

The system accepts semantic prompts (e.g., "make it look depressed", "make it look excited") and automatically generates motion variations by:

1. **Parsing Semantic Prompts**: `parse_semantic_prompt()` analyzes the prompt and generates variation multipliers
2. **Applying Parameter Variations**: `apply_parameter_variation()` modifies speed, degrees, hold_time, and repetition
3. **Planning Waypoint Variations**: `plan_waypoint_variation()` adjusts initial pose position via offsets
4. **Generating Modified Configurations**: `apply_variation_to_config()` combines all variations

**Code Reference**: Lines 20-121 in `llm_variation.py`

### Parameter-Based Variation

The system applies multipliers to motion parameters based on semantic interpretation:

#### 1. Speed Multiplier

Adjusts movement and transition speeds:
- **Depressed/Sad**: 0.7x (slower, more deliberate)
- **Energetic/Excited**: 1.3x (faster, more dynamic)
- **Careful/Cautious**: 0.6x (very slow, deliberate)
- **Hasty/Quick**: 1.5x (very fast, rapid)
- **Large/Exaggerated**: 1.0x (normal speed)
- **Small/Subtle**: 0.9x (slightly slower)

**Code Reference**: Lines 155-159 in `llm_variation.py`
```python
if "speed" in direction_config:
    original_speed = direction_config["speed"]
    new_speed = original_speed * variation["speed_multiplier"]
    direction_config["speed"] = max(0.1, new_speed)  # Minimum speed
```

#### 2. Degree Multiplier

Modifies movement amplitude:
- **Depressed/Sad**: 0.7x (smaller movements)
- **Energetic/Excited**: 1.2x (larger movements)
- **Careful/Cautious**: 0.8x (reduced range)
- **Large/Exaggerated**: 1.5x (much larger movements)
- **Small/Subtle**: 0.6x (minimal movements)

**Code Reference**: Lines 161-165 in `llm_variation.py`
```python
if "degrees" in direction_config:
    original_degrees = direction_config["degrees"]
    new_degrees = original_degrees * variation["degree_multiplier"]
    direction_config["degrees"] = max(5.0, new_degrees)  # Minimum degrees
```

#### 3. Hold Time Multiplier

Adjusts pause durations:
- **Depressed/Sad**: 1.5x (longer pauses)
- **Energetic/Excited**: 0.7x (shorter pauses)
- **Careful/Cautious**: 1.3x (longer pauses for deliberation)
- **Hasty/Quick**: 0.5x (very short pauses)
- **Large/Exaggerated**: 1.2x (longer pauses for emphasis)

**Code Reference**: Lines 167-171 in `llm_variation.py`

#### 4. Repetition Multiplier

Modifies movement cycle count:
- **Depressed/Sad**: 0.7x (fewer repetitions)
- **Energetic/Excited**: 1.3x (more repetitions)
- **Withdrawn/Shy**: 0.9x (slightly fewer)

**Code Reference**: Lines 146-150 in `llm_variation.py`
```python
if "repetition" in parameters:
    original_repetition = parameters["repetition"]
    new_repetition = max(1, int(original_repetition * variation["repetition_multiplier"]))
    parameters["repetition"] = new_repetition
```

### Waypoint-Based Variation

The system adjusts initial pose positions through spatial offsets:

#### 1. Initial Pose Offset

Modifies the end-effector starting position relative to the original pose:

**Offset Directions**:
- **Depressed/Sad**: `z: -0.05m` (lower position)
- **Energetic/Excited**: `z: +0.05m` (higher position)
- **Withdrawn/Shy**: `x: -0.05m` (backward position)
- **Forward/Reaching**: `x: +0.05m` (forward position)
- **Left**: `y: +0.05m` (leftward position)
- **Right**: `y: -0.05m` (rightward position)

**Code Reference**: Lines 191-214 in `llm_variation.py`
```python
def plan_waypoint_variation(config: Dict, variation: Dict) -> Dict:
    # Add initial_pose_offset to config
    initial_pose_offset = variation.get("initial_pose_offset", {"x": 0.0, "y": 0.0, "z": 0.0})
    modified_config["initial_pose_offset"] = initial_pose_offset
    return modified_config
```

The offset is applied to the first pose's end-effector position before movement execution, creating spatial variation while maintaining the same movement pattern.

#### 2. Pose Selection Variation

The system can select from 30-50 candidate poses (from the top-K filtered results) for each pose name. This provides natural variation in the starting configuration while maintaining the target orientation.

**Code Reference**: Lines 459-489 in `motion_generation.py`
- If `pose_index` is provided, that specific pose is used
- Otherwise, a random selection is made from `matching_poses`

#### 3. Quantization-Based Matching

When transitioning between poses, the system uses quantization to find smooth transitions:

**Process** (lines 239-326 in `motion_generation.py`):
1. **Quantization**: Joint angles are quantized to standard angles (default: -90°, 0°, 90°)
2. **Closest Match**: Among candidate poses matching the target orientation, the system selects the one with the smallest quantized angle difference from the current pose

This ensures smooth, natural transitions by minimizing joint angle changes between poses.

**Code Reference**:
- `_quantize_joint_angles()`: Quantizes angles to standard values
- `_find_closest_quantized_pose()`: Finds the best matching pose

#### 4. Interpolation and Frame Generation

Pose transitions use linear interpolation between joint positions:

**Code Reference**: Lines 508-534 in `motion_generation.py`
```python
# Interpolate from start to end pose
for frame_idx in range(num_transition_frames):
    t = (frame_idx + 1) / num_transition_frames  # 0 to 1
    interpolated_joint_pos = start_joint_pos * (1 - t) + end_joint_pos * t
    self._set_joint_positions(interpolated_joint_pos)
    image = self._capture_image()
    frames.append(Image.fromarray(image))
```

The number of frames is determined by `speed` and `hz` (frame rate): `num_frames = (1.0 / speed) * hz`

### Semantic Prompt Examples

The system supports various semantic prompts with automatic parameter mapping:

**Depressed/Disappointed/Sad**:
- Speed: 0.7x (slower)
- Degrees: 0.7x (smaller movements)
- Hold time: 1.5x (longer pauses)
- Repetition: 0.7x (fewer cycles)
- Position: Lower (z: -0.05m)

**Energetic/Excited/Enthusiastic**:
- Speed: 1.3x (faster)
- Degrees: 1.2x (larger movements)
- Hold time: 0.7x (shorter pauses)
- Repetition: 1.3x (more cycles)
- Position: Higher (z: +0.05m)

**Careful/Cautious/Slow**:
- Speed: 0.6x (very slow)
- Degrees: 0.8x (reduced range)
- Hold time: 1.3x (longer pauses)

**Hasty/Quick/Fast**:
- Speed: 1.5x (very fast)
- Hold time: 0.5x (very short pauses)

**Code Reference**: Lines 49-120 in `llm_variation.py` - `parse_semantic_prompt()` function

### Usage Example

**Input**:
```bash
python adhoc/robotarm/llm_variation.py \
  --robot Panda \
  --cue waving \
  --prompt "make it look depressed"
```

**Process**:
1. Loads original `waving` configuration from `motion_config.json`
2. Parses prompt "make it look depressed" → generates variation multipliers
3. Applies multipliers to all speed, degrees, hold_time, repetition parameters
4. Adds initial pose offset (z: -0.05m)
5. Creates temporary modified configuration
6. Executes motion with variations
7. Saves GIF with prompt in filename: `{timestamp}_Panda_waving_p{pose_id}_make_it_look_depressed.gif`

**Code Reference**: Lines 256-360 in `llm_variation.py` - `main()` function

### Future LLM Integration

The current implementation uses rule-based keyword matching. Future versions will integrate LLM inference to:
1. **Interpret complex prompts**: Understand nuanced semantic descriptions
2. **Generate custom variations**: Create unique parameter combinations
3. **Suggest intermediate waypoints**: Propose additional poses for smoother transitions
4. **Combine multiple cues**: Create hybrid movements from multiple cue definitions

The configuration structure separates the "what" (cue definition) from the "how" (execution parameters), enabling semantic control over motion characteristics without requiring knowledge of the underlying kinematics.

---

## Pipeline Integration

The three stages work together as follows:

1. **Pose Database Creation**: `find_closest_poses.py` generates and filters poses, storing them in `closest_poses_results.jsonl`
2. **Movement Planning**: `alphabet_jacobian.py` analyzes poses and identifies suitable joints for directional movements
3. **Motion Execution**: `motion_generation.py` reads configurations from `motion_config.json`, selects poses and joints, and generates animated GIFs

**Code Reference**: Lines 413-730 in `motion_generation.py` - `execute_cue()` method orchestrates the entire pipeline

The pipeline supports multiple robots (Panda, IIWA, Sawyer, Kinova3, Jaco, UR5e, XArm7) and can generate diverse, expressive movements through systematic variation of pose selection and motion parameters.

---

## Extension to Humanoid Robots

The pipeline can be extended to humanoid robots (e.g., GR1, Atlas, NAO) with many joints (50+ DOF). The key challenge is managing the increased complexity by focusing on specific body parts relevant to each expressive cue.

### Body Part Specification

For humanoid robots, movements should be focused on specific body parts:

- **Head movements** (nodding, shaking): Focus on head/neck joints
- **Arm movements** (waving, pointing): Focus on arm joints (shoulder, elbow, wrist)
- **Upper body** (bowing, shrugging): Focus on torso/shoulder joints
- **Lower body** (stomping, kicking): Focus on leg joints (hip, knee, ankle)

### Proposed Configuration Extension

The configuration structure can be extended to include body part specification:

```json
{
  "cue": "nodding",
  "body_part": "head",
  "end_effector_site": "head_site",
  "movements": [
    {
      "type": "movement",
      "parameters": {
        "repetition": 3,
        "axis": "z",
        "joint": "proximal",
        "directions": [
          {"direction": "pos", "degrees": 20, "speed": 1.0, "hold_time": 0.5},
          {"direction": "neg", "degrees": 20, "speed": 1.0, "hold_time": 0.5}
        ]
      }
    }
  ]
}
```

**New Parameters**:
- `body_part`: Body part name (e.g., "head", "left_arm", "right_arm", "torso", "left_leg", "right_leg")
- `end_effector_site`: MuJoCo site name for the target body part (e.g., "head_site", "gripper0_right_grip_site")

### Implementation Requirements

To extend the pipeline for humanoid robots, the following modifications would be needed:

#### 1. Body Part Joint Filtering

Filter joints to only include those relevant to the specified body part:

```python
# Example: Head joints
head_joints = ["neck_yaw", "neck_pitch", "neck_roll", "head_yaw", "head_pitch"]

# Example: Left arm joints
left_arm_joints = ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
                   "left_elbow_pitch", "left_wrist_pitch", "left_wrist_roll"]

# Filter DOF IDs to only include body part joints
body_part_dof_ids = [dof_id for dof_id, joint_name in enumerate(joint_names) 
                      if any(body_part_joint in joint_name for body_part_joint in head_joints)]
```

**Code Modification**: In `alphabet_jacobian.py`, modify `_find_and_sort_joints_for_axis()` to accept a `body_part_joint_filter` parameter that filters `dof_ids` and `joint_names_list` before analysis.

#### 2. End-Effector Site Selection

Use body part-specific end-effector sites instead of a single hand site:

- **Head**: Use head site (e.g., "head_site", "head_top")
- **Arm**: Use hand/gripper site (e.g., "gripper0_right_grip_site", "left_hand_site")
- **Leg**: Use foot site (e.g., "left_foot_site", "right_foot_site")

**Code Modification**: In `JacobianCalculator.__init__()`, accept `eef_site_name` as a parameter (currently hardcoded to gripper site).

#### 3. Pose Generation for Body Parts

For pose generation, focus brute-force search on body part joints only:

**Modification to `find_closest_poses.py`**:
- Add `body_part` parameter to limit active joints
- Filter `active_joint_indices` to only include body part joints
- Use body part-specific end-effector site for orientation calculation

**Example**: For "nodding" with `body_part="head"`:
- Only vary head/neck joints (3-5 joints instead of all 50+ joints)
- Compute orientation at head site instead of hand site
- Reduces search space from 3^50 to 3^5 (manageable)

#### 4. Jacobian Analysis for Body Parts

Restrict Jacobian computation and joint selection to body part joints:

**Modification to `_find_and_sort_joints_for_axis()`**:
- Accept `body_part_dof_ids` filter
- Compute Jacobian only for body part joints: `jac_subset = jac_full[:, body_part_dof_ids]`
- Score and rank only body part joints

**Example**: For "nodding" with `axis="z"` and `body_part="head"`:
- Compute Jacobian at head site
- Analyze only head/neck joints (3-5 joints)
- Select best joint from head/neck joints for up-down movement

### Benefits of Body Part Specification

1. **Reduced Complexity**: Focus analysis on relevant joints (5-7 joints) instead of all 50+ joints
2. **Faster Computation**: Smaller search space and Jacobian matrices
3. **Semantic Clarity**: Explicit body part specification makes cues more interpretable
4. **Scalability**: Can handle complex humanoids without exponential explosion in computation

### Example Humanoid Cue Configurations

```json
{
  "cue": "nodding",
  "body_part": "head",
  "end_effector_site": "head_site",
  "movements": [
    {
      "type": "movement",
      "parameters": {
        "repetition": 3,
        "axis": "z",
        "joint": "proximal",
        "directions": [
          {"direction": "pos", "degrees": 20, "speed": 1.0, "hold_time": 0.5},
          {"direction": "neg", "degrees": 20, "speed": 1.0, "hold_time": 0.5}
        ]
      }
    }
  ]
},
{
  "cue": "waving",
  "body_part": "right_arm",
  "end_effector_site": "gripper0_right_grip_site",
  "movements": [
    {
      "type": "pose",
      "parameters": {
        "pose": "Arm_extended"
      }
    },
    {
      "type": "movement",
      "parameters": {
        "repetition": 3,
        "axis": "y",
        "joint": "distal",
        "directions": [
          {"direction": "pos", "degrees": 30, "speed": 1.0, "hold_time": 0.5},
          {"direction": "neg", "degrees": 30, "speed": 1.0, "hold_time": 0.5}
        ]
      }
    }
  ]
}
```

### Backward Compatibility

For single-arm robots (current implementation):
- If `body_part` is not specified, use all arm joints (default behavior)
- If `end_effector_site` is not specified, use gripper site (default behavior)
- Maintains compatibility with existing configurations

This extension makes the pipeline applicable to humanoid robots while maintaining the same core principles: brute-force pose generation with filtering, Jacobian-based joint selection, and parameterized motion configuration.
