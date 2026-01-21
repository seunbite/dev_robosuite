# Brush Drawing Tasks

This folder contains scripts for robot arm brush-drawing tasks using robosuite.

## Overview

These scripts demonstrate robot control using:
1. **Inverse Kinematics (IK)** - Following a 3D line with the end effector
2. **Jacobian Analysis** - Finding optimal joints for motion along a target direction

Both scripts:
- Generate random 3D lines in the robot workspace
- **Visualize target lines and endpoints in the simulation**
- Control the robot to follow/touch the line
- Save results as animated GIFs with visual markers

## Visualization

Both scripts include real-time visualization of the target:
- **Red line**: Target trajectory to follow
- **Green sphere**: Start point of the line
- **Blue sphere**: End point of the line
- **Yellow sphere** (middle_point.py only): Midpoint target

These markers are rendered in every frame and visible in the saved GIF.

## Files

### 1. `ik.py` - IK Line Following

Uses inverse kinematics to follow a line divided into 50 segments.

**Features:**
- Generates two random points in 3D space
- Creates a target line connecting them
- **Visualizes the target line (red) and endpoints (green/blue spheres) in the simulation**
- Divides the line into 50 waypoints
- Uses IK control to move the end effector along the line
- Saves trajectory as GIF with visual markers

**Usage:**

```bash
cd /Users/sb/Downloads/workspace/dev_robosuite

# Basic usage (Panda robot)
python adhoc/brush/ik.py

# With custom robot
python adhoc/brush/ik.py --robot=IIWA

# With custom parameters
python adhoc/brush/ik.py \
  --robot=Panda \
  --num_points=50 \
  --steps_per_point=20 \
  --has_renderer=False
```

**Arguments:**
- `robot`: Robot name (default: "Panda"). Options: Panda, IIWA, Kinova3, Jaco, UR5e
- `env`: Environment name (default: "EmptySpace")
- `controller`: Controller name (default: "IK_POSE")
- `num_points`: Number of waypoints to divide the line into (default: 50)
- `steps_per_point`: Simulation steps per waypoint (default: 20)
- `has_renderer`: Show on-screen rendering (default: False)

**Output:**
- GIF saved to: `data/brush/{robot_name}/YYYYMMDD_HHMMSS_{robot_name}_ik_brush.gif`

---

### 2. `middle_point.py` - Middle Point Approach with Jacobian Analysis

Moves the end effector to the midpoint of a random line with specific orientation constraints, then analyzes which joint is best aligned with the line direction.

**Features:**
- Generates two random points in 3D space
- Computes the midpoint
- **Visualizes the target line (red), endpoints (green/blue spheres), and midpoint (yellow sphere)**
- Sets end effector orientation to r=-90°, p=90°/-90°, y=-90°
- Ensures end effector is perpendicular to the line (verifies with dot product)
- Moves to the midpoint
- Computes Jacobian matrix
- Finds joint with motion most aligned with line direction
- Saves trajectory as GIF with visual markers

**Usage:**

```bash
cd /Users/sb/Downloads/workspace/dev_robosuite

# Basic usage (Panda robot)
python adhoc/brush/middle_point.py

# With pitch = -90 instead of 90
python adhoc/brush/middle_point.py --pitch=-90

# With custom robot
python adhoc/brush/middle_point.py --robot=IIWA

# With custom parameters
python adhoc/brush/middle_point.py \
  --robot=Panda \
  --roll=-90 \
  --pitch=90 \
  --yaw=-90 \
  --steps=200 \
  --has_renderer=False
```

**Arguments:**
- `robot`: Robot name (default: "Panda")
- `env`: Environment name (default: "EmptySpace")
- `controller`: Controller name (default: "IK_POSE")
- `roll`: Roll angle in degrees (default: -90)
- `pitch`: Pitch angle in degrees (default: 90, can also use -90)
- `yaw`: Yaw angle in degrees (default: -90)
- `steps`: Number of steps to reach midpoint (default: 200)
- `has_renderer`: Show on-screen rendering (default: False)

**Output:**
- GIF saved to: `data/brush/{robot_name}/YYYYMMDD_HHMMSS_{robot_name}_middle_point.gif`
- Console output showing:
  - Line properties (start, end, midpoint, direction)
  - End effector orientation analysis
  - Perpendicularity check (angle between EE and line)
  - Jacobian analysis for each joint
  - Best aligned joint with alignment score

---

## Output Directory Structure

```
dev_robosuite/
  data/
    brush/
      Panda/
        20260106_123456_Panda_ik_brush.gif
        20260106_123457_Panda_middle_point.gif
      IIWA/
        ...
      Kinova3/
        ...
```

## Requirements

- Python 3.8+
- robosuite
- numpy
- PIL (Pillow)
- mujoco
- fire

Install with:
```bash
cd /Users/sb/Downloads/workspace/dev_robosuite
pip install -r requirements.txt
```

## Theory

### IK Line Following (`ik.py`)

The script uses differential inverse kinematics:
1. Compute position error: `dpos = target_pos - current_pos`
2. Send delta commands to IK controller
3. Controller computes joint positions to minimize error
4. Repeat for each waypoint

### Jacobian Analysis (`middle_point.py`)

The Jacobian matrix J relates joint velocities to end effector velocities:
```
v_ee = J * q_dot
```

Where:
- `v_ee`: End effector velocity (6D: 3 linear + 3 angular)
- `J`: Jacobian matrix (6 × n_joints)
- `q_dot`: Joint velocities (n_joints)

**Finding aligned joint:**
1. Extract position Jacobian (first 3 rows of J)
2. Each column represents how that joint moves the end effector
3. Normalize each column to get motion direction
4. Compute dot product with line direction
5. Highest dot product = most aligned joint

**Perpendicularity check:**
The end effector orientation is perpendicular to the line when:
```
|dot(ee_z_axis, line_direction)| ≈ 0
```

Or equivalently, the angle between them is ≈ 90°.

## Examples

### Example 1: Run both tasks for Panda robot

```bash
# IK line following
python adhoc/brush/ik.py --robot=Panda

# Middle point approach
python adhoc/brush/middle_point.py --robot=Panda
```

### Example 2: Test with different robots

```bash
# Test with IIWA
python adhoc/brush/ik.py --robot=IIWA
python adhoc/brush/middle_point.py --robot=IIWA

# Test with Kinova3
python adhoc/brush/ik.py --robot=Kinova3
python adhoc/brush/middle_point.py --robot=Kinova3

# Test with Jaco
python adhoc/brush/ik.py --robot=Jaco
python adhoc/brush/middle_point.py --robot=Jaco
```

### Example 3: Different pitch angles

```bash
# Pitch = 90°
python adhoc/brush/middle_point.py --pitch=90

# Pitch = -90°
python adhoc/brush/middle_point.py --pitch=-90
```

## Troubleshooting

### Issue: "Site not found" error
**Solution:** The robot may have a different gripper site name. Check the robot's XML definition or modify the `eef_site_name` in the code.

### Issue: Robot moves too slowly/quickly
**Solution:** Adjust `steps_per_point` (for ik.py) or `steps` (for middle_point.py).

### Issue: Target points are unreachable
**Solution:** Modify `workspace_bounds` in the `generate_random_line()` function to match your robot's reachable workspace.

### Issue: No frames captured
**Solution:** Ensure `has_offscreen_renderer=True` in the environment setup.

## Related Scripts

Other robotarm scripts in `adhoc/robotarm/`:
- `alphabet_jacobian.py` - Jacobian visualization for poses
- `motion_generation.py` - Generate motions from config
- `reconstruct_pose.py` - Reconstruct poses from filenames
- `find_closest_poses.py` - Find poses matching criteria

## Author

Created: 2026-01-06

