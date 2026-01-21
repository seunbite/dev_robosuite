"""
Common pose configuration for robot motion generation.
This module contains all pose definitions and mapping dictionaries.
"""

# New direction-based pose set with 42 poses
direction_pose_set = {
    # --- FRONT (사용자 방향) ---
    'Handshake':        {'height': 'med', 'dir': 'front', 'pitch': 'vertical'},
    'Show_Hand':        {'height': 'med', 'dir': 'front', 'pitch': 'horizontal'},
    'Presenting_High':  {'height': 'high', 'dir': 'front', 'pitch': 'vertical'},
    'Covering_Face':    {'height': 'high', 'dir': 'front', 'pitch': 'horizontal'},
    'Pointing_Low':     {'height': 'low', 'dir': 'front', 'pitch': 'vertical'},
    'Offering_Floor':   {'height': 'low', 'dir': 'front', 'pitch': 'horizontal'},

    # --- UP (위 방향) ---
    'Victory_V':        {'height': 'high', 'dir': 'up', 'pitch': 'vertical'},
    'Roof_Sign':        {'height': 'high', 'dir': 'up', 'pitch': 'horizontal'},
    'Alert_High':       {'height': 'med', 'dir': 'up', 'pitch': 'vertical'},
    'Flat_Ceiling':     {'height': 'med', 'dir': 'up', 'pitch': 'horizontal'},
    'Hidden_Up':        {'height': 'low', 'dir': 'up', 'pitch': 'vertical'},
    'Low_Support':      {'height': 'low', 'dir': 'up', 'pitch': 'horizontal'},

    # --- DOWN (아래 방향) ---
    'At_Ease':          {'height': 'low', 'dir': 'down', 'pitch': 'vertical'},
    'Palm_Down_Rest':   {'height': 'low', 'dir': 'down', 'pitch': 'horizontal'},
    'Sad_Droop':        {'height': 'med', 'dir': 'down', 'pitch': 'vertical'},
    'Hovering':         {'height': 'med', 'dir': 'down', 'pitch': 'horizontal'},
    'Avoidance':        {'height': 'high', 'dir': 'down', 'pitch': 'vertical'},
    'High_Shield':      {'height': 'high', 'dir': 'down', 'pitch': 'horizontal'},

    # --- LEFT (왼쪽 방향) ---
    'Guide_Left':       {'height': 'med', 'dir': 'left', 'pitch': 'vertical'},
    'Block_Left':       {'height': 'med', 'dir': 'left', 'pitch': 'horizontal'},
    'High_Wave_L':      {'height': 'high', 'dir': 'left', 'pitch': 'vertical'},
    'Left_Shelter':     {'height': 'high', 'dir': 'left', 'pitch': 'horizontal'},
    'Low_Indicate_L':   {'height': 'low', 'dir': 'left', 'pitch': 'vertical'},
    'Lying_Left':       {'height': 'low', 'dir': 'left', 'pitch': 'horizontal'},

    # --- RIGHT (오른쪽 방향) ---
    'Guide_Right':      {'height': 'med', 'dir': 'right', 'pitch': 'vertical'},
    'Block_Right':      {'height': 'med', 'dir': 'right', 'pitch': 'horizontal'},
    'High_Wave_R':      {'height': 'high', 'dir': 'right', 'pitch': 'vertical'},
    'Right_Shelter':    {'height': 'high', 'dir': 'right', 'pitch': 'horizontal'},
    'Low_Indicate_R':   {'height': 'low', 'dir': 'right', 'pitch': 'vertical'},
    'Lying_Right':      {'height': 'low', 'dir': 'right', 'pitch': 'horizontal'},

    # --- BACK (뒤 방향) ---
    'Retreat':          {'height': 'med', 'dir': 'back', 'pitch': 'vertical'},
    'Tucked_In':        {'height': 'med', 'dir': 'back', 'pitch': 'horizontal'},
    'High_Surrender':   {'height': 'high', 'dir': 'back', 'pitch': 'vertical'},
    'Protect_Head':     {'height': 'high', 'dir': 'back', 'pitch': 'horizontal'},
    'Low_Hide':         {'height': 'low', 'dir': 'back', 'pitch': 'vertical'},
    'Base_Rest':        {'height': 'low', 'dir': 'back', 'pitch': 'horizontal'},
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

# Height mapping
height_map = {
    'low': 'low',
    'med': 'medium',
    'high': 'high'
}
