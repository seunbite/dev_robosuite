"""
Common pose configuration for quadruped (dog) robot motion generation.
This module contains all pose definitions for 4-legged robots.

Pose features (for JSONL matching):
- body_height: 'high', 'mid', 'low' - Overall body height from ground
- body_tilt: 'neutral', 'front_high', 'back_high', 'left_high', 'right_high' - Body tilt direction
- leg_FL: Front-Left leg state ('normal', 'lifted', 'extended', 'reached')
- leg_FR: Front-Right leg state
- leg_HL: Hind-Left leg state
- leg_HR: Hind-Right leg state

Joint structure (assuming each leg has 3 joints):
- Hip (abduction/adduction): side-to-side movement
- Shoulder/Hip (flexion/extension): forward/backward movement
- Knee: bending
- Ankle: foot angle

Leg actions:
- 'normal': Default standing position
- 'lifted': Leg lifted up (knee bent ~90°, hip raised)
- 'extended': Leg stretched forward/outward (joints straightened)
- 'reached': Leg reaching forward (full extension with forward lean)
"""

# Quadruped pose set - structured for automatic pipeline
dog_pose_set = {
    # ========================================
    # 1. Standing at different heights (3 poses)
    # ========================================
    'Stand_High': {
        'body_height': 'high',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'High standing position with legs extended',
    },
    'Stand_Mid': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Normal standing position',
    },
    'Stand_Low': {
        'body_height': 'low',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Low standing/crouching position',
    },
    
    # ========================================
    # 2. Asymmetric tilts (4 poses)
    # ========================================
    'Bow_Front_High': {
        'body_height': 'mid',
        'body_tilt': 'front_high',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front legs down, back legs normal (play bow)',
    },
    'Alert_Back_High': {
        'body_height': 'mid',
        'body_tilt': 'back_high',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Back legs extended, front legs normal (alert stance)',
    },
    'Lean_Left_High': {
        'body_height': 'mid',
        'body_tilt': 'left_high',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Leaning with left side higher',
    },
    'Lean_Right_High': {
        'body_height': 'mid',
        'body_tilt': 'right_high',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Leaning with right side higher',
    },
    
    # ========================================
    # 3. Single leg actions (12 poses = 4 legs × 3 actions)
    # ========================================
    
    # --- Front-Left (FL) leg ---
    'FL_Lifted': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'lifted', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-left leg lifted',
    },
    'FL_Extended': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'extended', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-left leg extended forward',
    },
    'FL_Reached': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'reached', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-left leg reaching forward',
    },
    
    # --- Front-Right (FR) leg ---
    'FR_Lifted': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'lifted',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-right leg lifted',
    },
    'FR_Extended': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'extended',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-right leg extended forward',
    },
    'FR_Reached': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'reached',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Front-right leg reaching forward',
    },
    
    # --- Hind-Left (HL) leg ---
    'HL_Lifted': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'lifted', 'leg_HR': 'normal',
        'description': 'Hind-left leg lifted',
    },
    'HL_Extended': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'extended', 'leg_HR': 'normal',
        'description': 'Hind-left leg extended backward',
    },
    'HL_Reached': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'reached', 'leg_HR': 'normal',
        'description': 'Hind-left leg reaching backward',
    },
    
    # --- Hind-Right (HR) leg ---
    'HR_Lifted': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'lifted',
        'description': 'Hind-right leg lifted',
    },
    'HR_Extended': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'extended',
        'description': 'Hind-right leg extended backward',
    },
    'HR_Reached': {
        'body_height': 'mid',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'reached',
        'description': 'Hind-right leg reaching backward',
    },
    
    # ========================================
    # 4. Expressive compound poses (4 poses)
    # ========================================
    'Shake_Hand': {
        'body_height': 'mid',
        'body_tilt': 'right_high',
        'leg_FL': 'normal', 'leg_FR': 'lifted',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Offering right front paw (handshake)',
    },
    'Play_Bow': {
        'body_height': 'mid',
        'body_tilt': 'back_high',
        'leg_FL': 'extended', 'leg_FR': 'extended',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Play bow with front legs extended',
    },
    'Sit': {
        'body_height': 'low',
        'body_tilt': 'front_high',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Sitting position',
    },
    'Crouch': {
        'body_height': 'low',
        'body_tilt': 'neutral',
        'leg_FL': 'normal', 'leg_FR': 'normal',
        'leg_HL': 'normal', 'leg_HR': 'normal',
        'description': 'Low crouch (ready to pounce)',
    },
}

# Legacy alias for backward compatibility
pose_set = dog_pose_set

# ========================================
# Mapping definitions for automatic pipeline
# ========================================

# Body height values
body_heights = ['high', 'mid', 'low']

# Body tilt directions
body_tilts = [
    'neutral',
    'front_high',  # Front higher than back (back legs extended or front legs lowered)
    'back_high',   # Back higher than front (front legs extended or back legs lowered)
    'left_high',   # Left side higher than right
    'right_high',  # Right side higher than left
]

# Leg states
leg_states = {
    'normal': {
        'description': 'Default standing position',
        'hip_range': [0, 0],      # degrees (abduction: + = outward, - = inward)
        'shoulder_range': [0, 45], # degrees (flexion: + = forward, - = backward)
        'knee_range': [45, 90],    # degrees (+ = bent)
    },
    'lifted': {
        'description': 'Leg lifted up (knee bent)',
        'hip_range': [0, 15],      # Slightly outward for balance
        'shoulder_range': [-30, 0], # Slightly back
        'knee_range': [90, 120],    # Bent significantly
    },
    'extended': {
        'description': 'Leg stretched forward/outward (straight)',
        'hip_range': [0, 0],
        'shoulder_range': [60, 90], # Extended forward
        'knee_range': [0, 30],      # Nearly straight
    },
    'reached': {
        'description': 'Leg reaching forward (full extension)',
        'hip_range': [0, 0],
        'shoulder_range': [90, 120], # Maximum forward
        'knee_range': [0, 15],       # Very straight
    },
}

# Leg identifiers
legs = ['FL', 'FR', 'HL', 'HR']  # Front-Left, Front-Right, Hind-Left, Hind-Right

# Leg to joint name mapping (example - will vary by robot model)
# This maps leg identifiers to joint name patterns
leg_joint_patterns = {
    'FL': ['fl_hip', 'fl_shoulder', 'fl_knee'],  # Example patterns
    'FR': ['fr_hip', 'fr_shoulder', 'fr_knee'],
    'HL': ['hl_hip', 'hl_shoulder', 'hl_knee'],
    'HR': ['hr_hip', 'hr_shoulder', 'hr_knee'],
}

# Body height to joint angle mapping
# Maps body_height to target joint angles (relative to base)
body_height_joint_offset = {
    'high': {
        'all_knees': -30,  # Less bent (straighter = taller)
        'all_shoulders': -15,
    },
    'mid': {
        'all_knees': 0,    # Neutral
        'all_shoulders': 0,
    },
    'low': {
        'all_knees': 30,   # More bent (= lower)
        'all_shoulders': 15,
    },
}

# Body tilt to joint angle mapping
body_tilt_joint_offset = {
    'neutral': {},
    'front_high': {
        'front_knees': -20,  # Front legs more extended
        'back_knees': 20,    # Back legs more bent (or vice versa)
    },
    'back_high': {
        'front_knees': 20,   # Front legs more bent
        'back_knees': -20,   # Back legs more extended
    },
    'left_high': {
        'left_knees': -15,   # Left legs more extended
        'right_knees': 15,   # Right legs more bent
    },
    'right_high': {
        'left_knees': 15,    # Left legs more bent
        'right_knees': -15,  # Right legs more extended
    },
}

# Axis mapping for quadruped movements
# Unlike arms with x/y/z EE movement, quadrupeds have:
# - body_height (z-axis equivalent): controlled by all leg joints
# - body_tilt_pitch (front-back tilt): controlled by front vs back legs
# - body_tilt_roll (left-right tilt): controlled by left vs right legs
# - individual_leg: controlled by that leg's joints
movement_axes = {
    'body_z': {
        'description': 'Body height (up/down)',
        'affects': 'all_legs',
        'primary_joint': 'knee',  # Knee joint primarily controls height
    },
    'body_pitch': {
        'description': 'Body tilt (front-back)',
        'affects': 'front_vs_back',
        'primary_joint': 'knee',
    },
    'body_roll': {
        'description': 'Body tilt (left-right)',
        'affects': 'left_vs_right',
        'primary_joint': 'knee',
    },
    'leg_FL': {
        'description': 'Front-left leg movement',
        'affects': 'FL',
        'primary_joint': 'shoulder',  # Shoulder for forward/backward movement
    },
    'leg_FR': {
        'description': 'Front-right leg movement',
        'affects': 'FR',
        'primary_joint': 'shoulder',
    },
    'leg_HL': {
        'description': 'Hind-left leg movement',
        'affects': 'HL',
        'primary_joint': 'shoulder',
    },
    'leg_HR': {
        'description': 'Hind-right leg movement',
        'affects': 'HR',
        'primary_joint': 'shoulder',
    },
}

# Joint preference mapping for quadruped
# For quadrupeds, "proximal" = hip, "distal" = knee/ankle
joint_preference_map = {
    'proximal': 'hip',      # Joint closest to body
    'middle': 'shoulder',   # Middle joint
    'distal': 'knee',       # Joint farthest from body
}
