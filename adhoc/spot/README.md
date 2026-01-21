# Spot Robot Motion Generation

Boston Dynamics Spot quadruped robot with arm - pose exploration and expressive motion generation.

## 🤖 Two Motion Generation Pipelines

### 1. **Arm Motion Pipeline** (Existing)
Uses Spot's robotic **arm** for expressive gestures (waving, pointing, beckoning, etc.)

### 2. **Body Motion Pipeline** (New) 🆕
Uses Spot's **quadruped body** for dog-like motions (sitting, paw shake, play bow, etc.)

## 🐕 Supported Robots

1. **SpotWithArm** - Full quadruped with controllable legs (18 joints: 12 legs + 6 arm)
2. **SpotWithArmFloating** ⭐ - Floating base with 2D navigation (9 joints: 3 mobile + 6 arm) **[Recommended for arm motions]**
3. **Go2** - Generic quadruped for body motions

## 📋 Workflow

---

## 🦾 Pipeline 1: Arm Motion Generation

For Spot's robotic arm gestures (waving, pointing, etc.)

### Step 1: Export All Arm Poses

Generate all possible arm poses (729 combinations with 9 angles per joint):

```bash
# SpotWithArmFloating (recommended)
python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating

# SpotWithArm (full quadruped)
python adhoc/spot/export_spot_poses.py --robot SpotWithArm
```

**Output**: `data/poses/spot/all_{robot}_poses.jsonl`

### Step 2: Query Closest Poses (Optional but Recommended)

Pre-query closest poses for specific orientations to speed up motion generation:

```bash
python adhoc/spot/query_spot_poses.py \
    --robot SpotWithArmFloating \
    --roll 0 --pitch 90 --yaw 0

python adhoc/spot/query_spot_poses.py \
    --robot SpotWithArmFloating \
    --roll 180 --yaw 0 --height high
```

**Output**: Query results for faster motion generation

### Step 3: Generate Motions

Generate expressive motions using `motion_config.json`:

```bash
# Single motion
python adhoc/spot/motion_generation_spot.py \
    --robot SpotWithArmFloating \
    --cue waving \
    --controller OSC_POSE

# With specific pose
python adhoc/spot/motion_generation_spot.py \
    --robot SpotWithArmFloating \
    --cue beckoning \
    --pose-index 123

# Batch generation (all cues)
python adhoc/spot/meta_generate_spot_motions.py \
    --robot SpotWithArmFloating

# Batch generation (specific cues)
python adhoc/spot/meta_generate_spot_motions.py \
    --robot SpotWithArmFloating \
    --cues "waving,beckoning,pointing,nodding_substitute"
```

**Output**: `data/motions/{robot}/YYYYMMDD_HHMMSS_{robot}_{cue}_p{pose_id}.gif`

---

## 🐾 Pipeline 2: Body Motion Generation (Quadruped)

For dog-like body motions (sitting, paw shake, play bow, etc.)

### Step 1: Generate Dog Poses (240 combinations)

Smart brute force generation with symmetric legs:

```bash
# Go2 robot (240 poses in ~5 minutes)
python adhoc/spot/stack_preset_dog.py --robot Go2

# Spot robot
python adhoc/spot/stack_preset_dog.py --robot SpotWithArm
```

**Strategy**: 3 × 5 × 4 × 4 = 240 poses
- Body height: 3 (low, mid, high)
- Body tilt: 5 (neutral, front/back/left/right_high)
- Front legs: 4 (normal, lifted, extended, reached) - symmetric
- Back legs: 4 (normal, lifted, extended, reached) - symmetric

**Output**: `data/poses/quadruped/{robot}/{robot}_dog_poses.jsonl`

### Step 2: Filter Closest Dog Poses

Find poses matching `dog_pose_config.py` definitions (23 poses):

```bash
# Filter all robots
python adhoc/spot/meta_find_closest_poses_dog.py

# Filter specific robots
python adhoc/spot/meta_find_closest_poses_dog.py --robots Go2 SpotWithArm
```

**Output**: `data/poses/quadruped/closest_dog_poses.jsonl`

### Step 3: Generate Dog Motions

Generate body motions using `motion_config_quadruped.json`:

```bash
# Single motion
python adhoc/spot/motion_generation_dog.py \
    --robot Go2 \
    --cue sit_down

# With custom pose database
python adhoc/spot/motion_generation_dog.py \
    --robot SpotWithArm \
    --cue paw_shake \
    --jsonl-path data/poses/quadruped/closest_dog_poses.jsonl
```

**Available cues**: sit_down, paw_shake, play_bow, crouch_ready, tail_wag, stand_tall, body_bounce, lean_side_to_side, etc.

**Output**: `data/motions_quadruped/{robot}/YYYYMMDD_HHMMSS_{robot}_{cue}.gif`

---

## 🎯 Key Features

- **Quadruped + Arm**: Full Boston Dynamics Spot with manipulator arm
- **Floating Base Option**: SpotWithArmFloating for simpler arm-focused experiments
- **Same Motion Config**: Uses the same `motion_config.json` as other robots
- **Expressive Motions**: Waving, beckoning, pointing, nodding, and more
- **Automatic Pose Selection**: Finds optimal arm poses based on orientation
- **Joint Optimization**: Jacobian-based joint selection for each motion axis

## 🔧 Technical Details

### SpotWithArm
- **Total Joints**: 18 (12 leg + 6 arm)
- **Leg Control**: Full quadruped locomotion
- **Base**: Spot (LegBaseModel)
- **Use Case**: Quadruped + manipulation research

### SpotWithArmFloating ⭐
- **Total Joints**: 9 (3 mobile + 6 arm)
- **Leg Control**: Fixed (visual only)
- **Navigation**: 2D slide (x, y) + yaw rotation
- **Base**: SpotFloating
- **Use Case**: Arm-focused manipulation with simple navigation

## 📊 Comparison

| Feature | SpotWithArm | SpotWithArmFloating |
|---------|-------------|---------------------|
| Legs | ✅ Controllable | ❌ Fixed |
| Navigation | Quadruped gait | 2D slide + yaw |
| Complexity | High (18 joints) | Low (9 joints) |
| Setup | Complex | Simple |
| Best For | Quadruped research | Manipulation focus |

## 🚀 Quick Start

**Recommended workflow** for SpotWithArmFloating:

```bash
# 1. Export poses (once)
python adhoc/spot/export_spot_poses.py --robot SpotWithArmFloating

# 2. Generate a test motion
python adhoc/spot/motion_generation_spot.py \
    --robot SpotWithArmFloating \
    --cue waving

# 3. Generate multiple motions
python adhoc/spot/meta_generate_spot_motions.py \
    --robot SpotWithArmFloating \
    --cues "waving,beckoning,pointing"
```

## 💡 Tips

1. **Use SpotWithArmFloating** for most experiments (simpler, faster)
2. **Camera distance**: Default is 2.2 to see the full Spot body
3. **Controller**: Use OSC_POSE (IK_POSE may not be supported)
4. **Pose export**: Takes ~5-10 minutes for 729 poses
5. **Motion generation**: Each motion takes ~30 seconds

## 🎬 Example Output

Generated motions will look like:
- Spot standing with its arm waving
- Spot beckoning with a "come here" gesture
- Spot pointing in different directions
- And all other motions from `motion_config.json`!

The Spot's quadruped body adds character to the arm motions! 🐕🦾
