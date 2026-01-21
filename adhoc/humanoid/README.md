# Humanoid Pose Exploration & Motion Generation

This directory contains tools for exploring poses and generating motions for humanoid robots using a **single arm** while keeping other joints fixed.

## Overview

The system works in two phases:

1. **Phase 1: Pose Exploration** - Generate and filter poses for one arm
2. **Phase 2: Motion Generation** - Use `motion_config.json` to create expressive motions

This is the same pipeline used for manipulator arms, but adapted for humanoid robots.

## Supported Robots

All GR1 variants from robosuite:

- **GR1ArmsOnly** (14 joints) - Only arms, simplest
- **GR1FixedLowerBody** (20 joints) - Upper body with fixed legs
- **GR1FloatingBody** (20 joints) - Upper body with floating base
- **GR1** (32 joints) - Full humanoid with fixed legs

Each robot can use either **right** or **left** arm.

## Quick Start

### Step 1: Export All Poses (Once)

```bash
# Export right arm poses for GR1ArmsOnly
python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm right

# Export left arm poses
python adhoc/humanoid/export_humanoid_poses.py --robot GR1ArmsOnly --active-arm left

# Export for fixed lower body version
python adhoc/humanoid/export_humanoid_poses.py --robot GR1FixedLowerBody --active-arm right
```

This generates: `data/poses/humanoid/all_{robot}_{arm}_poses.jsonl`

### Step 2: Query Closest Poses (Fast)

```bash
# Query specific orientation
python adhoc/humanoid/query_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --roll 0 --pitch 90 --yaw 0

# Query with height filter
python adhoc/humanoid/query_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --roll 180 --yaw 0 \
    --height high

# Save tile image
python adhoc/humanoid/query_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --roll 0 --pitch 90 --yaw 0 \
    --save-tile-image
```

## Joint Configuration

### GR1ArmsOnly (14 joints)
```
Joints 0-6:  Right arm (7 DoF)
Joints 7-13: Left arm (7 DoF)
```

**Active arm = right**: Fix joints 7-13 (left arm)  
**Active arm = left**: Fix joints 0-6 (right arm)

### GR1FixedLowerBody (20 joints)
```
Joints 0-2:   Head
Joints 3-5:   Torso
Joints 6-12:  Right arm (7 DoF)
Joints 13-19: Left arm (7 DoF)
Legs: Fixed (not actuated)
```

**Active arm = right**: Fix joints 0-5, 13-19 (head, torso, left arm)  
**Active arm = left**: Fix joints 0-5, 6-12 (head, torso, right arm)

### GR1 (Full humanoid, 32 joints)
```
Joints 0-2:   Head/Base
Joints 3-5:   Torso
Joints 6-12:  Right arm (7 DoF)
Joints 13-19: Left arm (7 DoF)
Joints 20-31: Legs (fixed for pose exploration)
```

## Phase 2: Motion Generation

After finding good poses, use the same `motion_config.json` from the robotarm system:

```bash
python adhoc/humanoid/generate_humanoid_motions.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --config data/seed/motion_config.json \
    --poses-file data/poses/humanoid/closest_poses_results.jsonl
```

The motion generation uses the same declarative format:
- `beckoning`, `waving`, `pointing`, etc.
- Same movement primitives: `pose`, `movement`, `axis`, `joint`, etc.
- Personality modulation (if integrated with personality system)

## Comparison with Manipulator Arms

| Aspect | Manipulator Arm | Humanoid (One Arm) |
|--------|----------------|-------------------|
| Export script | `export_all_poses_once.py` | `export_humanoid_poses.py` |
| Query script | `query_poses_from_export.py` | `query_humanoid_poses.py` |
| Motion config | `motion_config.json` | Same file! ✓ |
| Joints | 6-7 arm joints | 7 arm joints (rest fixed) |
| Output format | JSONL (same) | JSONL (same) |

## Directory Structure

```
adhoc/humanoid/
├── export_humanoid_poses.py           # Export all poses for one arm
├── query_humanoid_poses.py            # Query closest poses
├── motion_generation_humanoid.py      # Generate motions with personality ✓
├── meta_generate_humanoid_motions.py  # Batch generation ✓
└── README.md                           # This file

data/poses/humanoid/
├── all_GR1ArmsOnly_right_poses.jsonl
├── all_GR1ArmsOnly_left_poses.jsonl
├── all_GR1FixedLowerBody_right_poses.jsonl
└── closest_poses_results.jsonl

data/motions/GR1ArmsOnly/
├── 20260113_HHMMSS_GR1ArmsOnly_right_waving_Excited.gif
├── 20260113_HHMMSS_GR1ArmsOnly_right_beckoning_Sad.gif
└── ... (generated motion GIFs)
```

## Advanced Usage

### Custom angle range
```bash
python adhoc/humanoid/export_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --angle-step 45 \
    --angle-min -90 \
    --angle-max 90
```

### Save query results to file
```bash
python adhoc/humanoid/query_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --roll 0 --pitch 90 --yaw 0 \
    --output-file my_results.json
```

### Custom tile image
```bash
python adhoc/humanoid/query_humanoid_poses.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --roll 0 --pitch 90 --yaw 0 \
    --save-tile-image \
    --tile-output my_poses.png \
    --tile-size 512
```

## Integration with Personality System

The humanoid system is designed to work with the personality-based motion generation:

1. Export poses for humanoid
2. Query poses using `motion_config.json` definitions
3. Apply personality modulation (approach/avoidance bias)
4. Generate expressive humanoid gestures

See `personality_robot_hri/` for the personality system implementation.

## Phase 2: Motion Generation (Complete!)

### Individual Motion Generation

```bash
# Generate specific motion with personality
python adhoc/humanoid/motion_generation_humanoid.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --cue waving \
    --personality Excited

# Different personality
python adhoc/humanoid/motion_generation_humanoid.py \
    --robot GR1ArmsOnly \
    --active-arm left \
    --cue beckoning \
    --personality Sad

# With specific pose
python adhoc/humanoid/motion_generation_humanoid.py \
    --robot GR1ArmsOnly \
    --active-arm right \
    --cue pointing \
    --personality Calm \
    --pose-index 42
```

### Batch Motion Generation (Meta)

```bash
# Generate all combinations (all cues × all personalities × both arms)
python adhoc/humanoid/meta_generate_humanoid_motions.py --robot GR1ArmsOnly

# Only right arm
python adhoc/humanoid/meta_generate_humanoid_motions.py \
    --robot GR1ArmsOnly \
    --arms right

# Specific cues
python adhoc/humanoid/meta_generate_humanoid_motions.py \
    --robot GR1ArmsOnly \
    --cues "waving,beckoning,pointing"

# Specific personalities
python adhoc/humanoid/meta_generate_humanoid_motions.py \
    --robot GR1ArmsOnly \
    --personalities "Excited,Sad,Calm"
```

## Supported Personalities

From `data/seed/personality_list.json`:
- **Excited**: Fast, energetic, upward bias
- **Sad**: Slow, small movements, downward bias
- **Calm**: Smooth, controlled, minimal variation
- **Anxious**: Jerky, variable, random bias
- And more...

## Next Steps

1. ✅ ~~Create motion generation scripts~~
2. Test with GR1ArmsOnly
3. Generate dataset of humanoid gestures
4. Integrate with personality-based HRI system

## Notes

- **One arm at a time**: The system focuses on single-arm gestures
- **Bimanual gestures**: For future work (both arms moving simultaneously)
- **Locomotion**: Legs are always fixed in current implementation
- **Head/torso**: Fixed to simplify pose space (can be made active if needed)
