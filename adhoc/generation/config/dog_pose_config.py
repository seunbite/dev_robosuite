"""
Pose / axis metadata for ``legacy.motion_generation_quadruped``.

String poses can index ``dog_pose_set``; dict poses in cues use ``_find_matching_poses``
against a JSONL database. LC19 quadruped configs mostly use dict poses without DB rows,
so matching often falls back to the default stance — movement segments carry the motion.
"""

legs = ("FL", "FR", "HL", "HR")

body_heights = ("low", "mid", "high")
body_tilts = ("level", "pitch_forward", "pitch_back")

leg_states = ("stand", "sit", "lift")

# Optional named presets (empty = rely on dict poses + JSONL / default stance).
dog_pose_set: dict = {}
pose_set = dog_pose_set

body_height_joint_offset: dict = {}
body_tilt_joint_offset: dict = {}

# Used only by the legacy axis/joint_preference movement path (non-LC19 cues).
movement_axes = {
    "body_z": {"affects": "all_legs", "primary_joint": "shoulder"},
    "body_pitch": {"affects": "front_vs_back", "primary_joint": "hip"},
    "body_roll": {"affects": "left_vs_right", "primary_joint": "hip"},
    "leg_FL": {"affects": "FL", "primary_joint": "hip"},
    "leg_FR": {"affects": "FR", "primary_joint": "hip"},
    "leg_HL": {"affects": "HL", "primary_joint": "hip"},
    "leg_HR": {"affects": "HR", "primary_joint": "hip"},
}

joint_preference_map = {
    "proximal": "hip",
    "middle": "shoulder",
    "distal": "knee",
}
