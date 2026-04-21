"""
Common pose configuration for robot motion generation.
This module contains all pose definitions and mapping dictionaries.
"""

# New direction-based pose set with 42 poses
direction_pose_set = {
    # --- FRONT (사용자 방향) ---
    'Handshake':        {'height': 'med', 'dir': 'front', 'gripper_orientation': 'vertical'},
    'Show_Hand':        {'height': 'med', 'dir': 'front', 'gripper_orientation': 'horizontal'},
    'Presenting_High':  {'height': 'high', 'dir': 'front', 'gripper_orientation': 'vertical'},
    'Covering_Face':    {'height': 'high', 'dir': 'front', 'gripper_orientation': 'horizontal'},
    'Pointing_Low':     {'height': 'low', 'dir': 'front', 'gripper_orientation': 'vertical'},
    'Offering_Floor':   {'height': 'low', 'dir': 'front', 'gripper_orientation': 'horizontal'},

    # --- UP (위 방향) ---
    'Victory_V':        {'height': 'high', 'dir': 'up', 'gripper_orientation': 'vertical'},
    'Roof_Sign':        {'height': 'high', 'dir': 'up', 'gripper_orientation': 'horizontal'},
    'Alert_High':       {'height': 'med', 'dir': 'up', 'gripper_orientation': 'vertical'},
    'Flat_Ceiling':     {'height': 'med', 'dir': 'up', 'gripper_orientation': 'horizontal'},
    'Hidden_Up':        {'height': 'low', 'dir': 'up', 'gripper_orientation': 'vertical'},
    'Low_Support':      {'height': 'low', 'dir': 'up', 'gripper_orientation': 'horizontal'},

    # --- DOWN (아래 방향) ---
    'At_Ease':          {'height': 'low', 'dir': 'down', 'gripper_orientation': 'vertical'},
    'Palm_Down_Rest':   {'height': 'low', 'dir': 'down', 'gripper_orientation': 'horizontal'},
    'Sad_Droop':        {'height': 'med', 'dir': 'down', 'gripper_orientation': 'vertical'},
    'Hovering':         {'height': 'med', 'dir': 'down', 'gripper_orientation': 'horizontal'},
    'Avoidance':        {'height': 'high', 'dir': 'down', 'gripper_orientation': 'vertical'},
    'High_Shield':      {'height': 'high', 'dir': 'down', 'gripper_orientation': 'horizontal'},

    # --- LEFT (왼쪽 방향) ---
    'Guide_Left':       {'height': 'med', 'dir': 'left', 'gripper_orientation': 'vertical'},
    'Block_Left':       {'height': 'med', 'dir': 'left', 'gripper_orientation': 'horizontal'},
    'High_Wave_L':      {'height': 'high', 'dir': 'left', 'gripper_orientation': 'vertical'},
    'Left_Shelter':     {'height': 'high', 'dir': 'left', 'gripper_orientation': 'horizontal'},
    'Low_Indicate_L':   {'height': 'low', 'dir': 'left', 'gripper_orientation': 'vertical'},
    'Lying_Left':       {'height': 'low', 'dir': 'left', 'gripper_orientation': 'horizontal'},

    # --- RIGHT (오른쪽 방향) ---
    'Guide_Right':      {'height': 'med', 'dir': 'right', 'gripper_orientation': 'vertical'},
    'Block_Right':      {'height': 'med', 'dir': 'right', 'gripper_orientation': 'horizontal'},
    'High_Wave_R':      {'height': 'high', 'dir': 'right', 'gripper_orientation': 'vertical'},
    'Right_Shelter':    {'height': 'high', 'dir': 'right', 'gripper_orientation': 'horizontal'},
    'Low_Indicate_R':   {'height': 'low', 'dir': 'right', 'gripper_orientation': 'vertical'},
    'Lying_Right':      {'height': 'low', 'dir': 'right', 'gripper_orientation': 'horizontal'},

    # --- BACK (뒤 방향) ---
    'Retreat':          {'height': 'med', 'dir': 'back', 'gripper_orientation': 'vertical'},
    'Tucked_In':        {'height': 'med', 'dir': 'back', 'gripper_orientation': 'horizontal'},
    'High_Surrender':   {'height': 'high', 'dir': 'back', 'gripper_orientation': 'vertical'},
    'Protect_Head':     {'height': 'high', 'dir': 'back', 'gripper_orientation': 'horizontal'},
    'Low_Hide':         {'height': 'low', 'dir': 'back', 'gripper_orientation': 'vertical'},
    'Base_Rest':        {'height': 'low', 'dir': 'back', 'gripper_orientation': 'horizontal'},
}

# Legacy pose_set for backward compatibility (maps to direction_pose_set)
pose_set = direction_pose_set

# Pitch values for different orientations
pitch_poses = {
    'vertical': [0, 180],
    'horizontal': [90, -90],
}

# Direction-based roll/yaw combinations
poses = {
    'up': [
        {'roll': 0, 'yaw': -180},
        {'roll': 0, 'yaw': -90},
        {'roll': 0, 'yaw': 0},
        {'roll': 0, 'yaw': 90},
        {'roll': 0, 'yaw': 180},
    ],
    'front': [
        {'roll': -90, 'yaw': -90},
        {'roll': 90, 'yaw': 90},
    ],
    'down': [
        {'roll': -180, 'yaw': -180},
        {'roll': -180, 'yaw': -90},
        {'roll': -180, 'yaw': 0},
        {'roll': -180, 'yaw': 90},
        {'roll': -180, 'yaw': 180},
        {'roll': 180, 'yaw': -180},
        {'roll': 180, 'yaw': -90},
        {'roll': 180, 'yaw': 0},
        {'roll': 180, 'yaw': 90},
        {'roll': 180, 'yaw': 180},
    ],
    'left': [
        {'roll': -90, 'yaw': -180},
        {'roll': -90, 'yaw': 180},
        {'roll': 90, 'yaw': 0},
    ],
    'back': [
        {'roll': -90, 'yaw': 90},
        {'roll': 90, 'yaw': -90},
    ],
    'right': [
        {'roll': -90, 'yaw': 0},
        {'roll': 90, 'yaw': -180},
        {'roll': 90, 'yaw': 180},
    ],
}

# Height mapping (Legacy)
height_map = {
    'low': 'low',
    'med': 'medium',
    'high': 'high'
}

# 27-grid spatial regions mapping (X, Y, Z each have low, medium, high)
region_map = {
    'low': 'low',
    'med': 'medium',
    'high': 'high'
}

# 27-grid pose set generation
grid_27_regions = []
for x in ['low', 'med', 'high']:
    for y in ['low', 'med', 'high']:
        for z in ['low', 'med', 'high']:
            grid_27_regions.append({
                'name': f"X{x.capitalize()}_Y{y.capitalize()}_Z{z.capitalize()}",
                'x': region_map[x],
                'y': region_map[y],
                'z': region_map[z]
            })
