import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch # Just in case it's needed for any robosuite internal
from find_closest_poses import ClosestPoseFinder
from arm_pose_config import poses, region_map, pitch_poses

import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch # Just in case it's needed for any robosuite internal
from find_closest_poses import ClosestPoseFinder
from arm_pose_config import poses, region_map, pitch_poses

def project_point(sim, world_pos, camera_name, width, height):
    """Project a 3D world point to 2D pixel coordinates."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    pos = sim.data.cam_xpos[cam_id]
    rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    
    # World to Camera frame
    rel_pos = world_pos - pos
    cam_pos = rot.T @ rel_pos # rot is cam-to-world, so rot.T is world-to-cam
    
    # Focal length
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    
    if cam_pos[2] >= 0: # Behind camera
        return None
    
    # Project (MuJoCo camera looks at -Z)
    dist = -cam_pos[2]
    x_pix = (width / 2) + (cam_pos[0] / dist) * focal
    y_pix = (height / 2) - (cam_pos[1] / dist) * focal
    
    return x_pix, y_pix

# Vector mapping for directions
VEC_MAP = {
    'up': [0, 0, 1], 'down': [0, 0, -1],
    'left': [0, 1, 0], 'right': [0, -1, 0],
    'front': [1, 0, 0], 'back': [-1, 0, 0]
}

def generate_random_subplots(robot_name="IIWA", num_samples=6, output_path="data/logs/random_poses_subplot.png"):
    """
    Generate 6 specific diverse subplots for a single arm manipulator.
    """
    finder = ClosestPoseFinder(robot_name=robot_name, env_name="EmptySpace", render=True)
    
    specs = [
        {"dir": "left",  "x": "high", "y": "medium", "z": "low",    "p_type": "vertical",   "caption": "Left"},
        {"dir": "right", "x": "high", "y": "medium", "z": "low",    "p_type": "vertical",   "caption": "Right"},
        {"dir": "front", "x": "high-medium", "y": "any", "z": "medium",    "p_type": "horizontal", "caption": "Horizontal"},
        {"dir": "front", "x": "high-medium", "y": "any", "z": "medium",    "p_type": "vertical",   "caption": "Vertical"},
        {"dir": "down",  "x": "high-medium", "y": "any", "z": "high",   "p_type": "any",   "caption": "z-High"},
        {"dir": "down",  "x": "high-medium", "y": "any", "z": "medium-low", "p_type": "any",   "caption": "z-Med"},
    ]
    
    selected_samples = []
    print(f"Generating 6 specific configurations for {robot_name}...")

    for i, spec in enumerate(specs):
        found = False
        attempt = 0
        while not found and attempt < 5:
            direction = spec["dir"]
            if spec["p_type"] == "any":
                pitch_val = None
            else:
                pitch_val = random.choice(pitch_poses[spec["p_type"]])
            orn_config = random.choice(poses[direction])
            
            results = finder.find_closest_poses(
                roll_deg=orn_config['roll'],
                pitch_deg=pitch_val,
                yaw_deg=orn_config['yaw'],
                x_region=spec["x"] if spec["x"] != "any" else None,
                y_region=spec["y"] if spec["y"] != "any" else None,
                z_region=spec["z"] if spec["z"] != "any" else None,
                top_k=100,
                angle_step_deg=90.0,
                stack_jsonl_path=None
            )
            
            arm_results = list(results.values())[0] if results else []
            if arm_results:
                best_pose = arm_results[0]
                selected_samples.append({
                    'caption': spec["caption"],
                    'dir': spec['dir'],
                    'joint_angles': best_pose['joint_angles_rad']
                })
                print(f"  Sample {i+1} [{spec['caption']}]: Success!")
                found = True
            else:
                attempt += 1
        
        if not found:
            selected_samples.append({
                'caption': spec["caption"],
                'dir': spec['dir'],
                'joint_angles': finder.initial_joint_pos[:len(finder.active_joint_indices)]
            })

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    for i, sample in enumerate(selected_samples):
        joint_pos = finder.initial_joint_pos.copy()
        for j, idx_val in enumerate(finder.active_joint_indices):
            joint_pos[idx_val] = sample['joint_angles'][j]
        finder._set_joint_positions(joint_pos)
        
        camera_name = "frontview"
        cam_id = finder.env.sim.model.camera_name2id(camera_name)
        if robot_name == "IIWA" and i >= 2:
            finder.env.sim.model.cam_fovy[cam_id] = 55
        else:
            finder.env.sim.model.cam_fovy[cam_id] = 40
            
        obs = finder.env.sim.render(width=640, height=640, camera_name=camera_name)
        img = np.flipud(obs)
        
        # Crop: Bottom 20%
        h, w, _ = img.shape
        m_top, m_bottom, m_side = 0.0, 0.20, 0.0
        img_cropped = img[int(h * m_top):int(h * (1 - m_bottom)), int(w * m_side):int(w * (1 - m_side)), :]
        
        axes[i].imshow(img_cropped)
        axes[i].axis('off')
        axes[i].text(0.5, -0.05, sample["caption"], transform=axes[i].transAxes, ha='center', va='top', fontsize=14)

        # Draw Arrow
        v_world = VEC_MAP.get(sample['dir'], [0,0,0])
        ee_pos = finder._get_ee_position()
        p1 = project_point(finder.env.sim, ee_pos, camera_name, 640, 640)
        p2 = project_point(finder.env.sim, ee_pos + np.array(v_world) * 0.15, camera_name, 640, 640)
        if p1 and p2:
            p1_c = (p1[0] - w * m_side, p1[1] - h * m_top)
            p2_c = (p2[0] - w * m_side, p2[1] - h * m_top)
            axes[i].annotate("", xy=p2_c, xytext=p1_c,
                             arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5, mutation_scale=15))

    plt.subplots_adjust(wspace=0.01, hspace=0, left=0.005, right=0.995, bottom=0.15, top=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.01)
    print(f"\nManipulator Subplot saved to: {output_path}")
    finder.close()

def generate_mobile_subplots(robot_name="Tiago", num_samples=6, output_path="data/logs/random_poses_mobile_subplot.png"):
    """
    Generate 6 specific subplots for mobile manipulators using provided paths.
    """
    finder = ClosestPoseFinder(robot_name=robot_name, env_name="EmptySpace", render=True)
    from stack_preset_mobile import MOBILE_ROBOT_CONFIGS
    config = MOBILE_ROBOT_CONFIGS.get(robot_name, MOBILE_ROBOT_CONFIGS['Tiago'])

    specs = [
        {"type": "simulate", "caption": "Front", "dir": "front", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_y0_htall_j3+000_j4+000_j5+090_j6+090_j7+000_j8+000.png"},
        {"type": "simulate", "caption": "Down", "dir": "down", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_y0_htall_j3+000_j4+090_j5-090_j6+000_j7-090_j8+000.png"},
        {"type": "image", "caption": "Head pan left", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_HEAD_hp+000_ht-029.png"},
        {"type": "image", "caption": "Head pan right", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_HEAD_hp+000_ht+029.png"},
        {"type": "image", "caption": "Short", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_HEAD_hp+000_ht+000.png"},
        {"type": "image", "caption": "Tall", "path": "/Users/sb/Downloads/workspace/dev_robosuite/data/poses_mobile/Tiago/Tiago_HEAD_hp+014_ht+000.png"},
    ]

    def parse_angles_from_filename(filename):
        matches = re.findall(r'j(\d+)([+-]\d+)', filename)
        angles = {int(idx): np.deg2rad(int(val)) for idx, val in matches}
        height = max(config['torso_range']) if "htall" in filename else min(config['torso_range'])
        return height, angles

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    for i, spec in enumerate(specs):
        m_top = 0.20
        if spec["type"] == "simulate":
            height, arm_angles = parse_angles_from_filename(os.path.basename(spec["path"]))
            finder.env.sim.data.qpos[config['yaw_idx']] = 0
            finder.env.sim.data.qpos[config['torso_idx']] = height
            joint_pos = finder.initial_joint_pos.copy()
            for j_idx, angle in arm_angles.items():
                if j_idx in [3, 4, 5, 6, 7, 8]:
                    joint_pos[j_idx] = angle
                    joint_pos[j_idx + 6] = angle
            finder.robot.set_robot_joint_positions(joint_pos)
            finder.env.sim.forward()
            
            camera_name = "frontview"
            cam_id = finder.env.sim.model.camera_name2id(camera_name)
            finder.env.sim.model.cam_fovy[cam_id] = 60 
            obs = finder.env.sim.render(width=640, height=640, camera_name=camera_name)
            img = np.flipud(obs)
        else:
            try:
                img = np.array(Image.open(spec["path"]))
            except:
                img = np.zeros((640, 640, 3), dtype=np.uint8)

        h, w, _ = img.shape
        img_cropped = img[int(h * m_top):, :, :]
        axes[i].imshow(img_cropped)
        axes[i].axis('off')
        axes[i].text(0.5, -0.05, spec["caption"], transform=axes[i].transAxes, ha='center', va='top', fontsize=14)

        if spec["type"] == "simulate":
            v_world = VEC_MAP.get(spec['dir'], [0,0,0])
            camera_name = "frontview"
            for side in ['right', 'left']:
                ee_pos = finder._get_ee_position(arm=side)
                p1 = project_point(finder.env.sim, ee_pos, camera_name, 640, 640)
                p2 = project_point(finder.env.sim, ee_pos + np.array(v_world) * 0.15, camera_name, 640, 640)
                if p1 and p2:
                    p1_c = (p1[0], p1[1] - h * m_top)
                    p2_c = (p2[0], p2[1] - h * m_top)
                    axes[i].annotate("", xy=p2_c, xytext=p1_c,
                                     arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5, mutation_scale=15))

    plt.subplots_adjust(wspace=0.01, hspace=0, left=0.005, right=0.995, bottom=0.15, top=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.01)
    print(f"\nMobile Subplot saved to: {output_path}")
    finder.close()

if __name__ == "__main__":
    import fire
    def run(mode=None, robot="IIWA"):
        # Auto-detect mode if not explicitly provided
        if mode is None:
            if robot in ["Tiago", "PandaOmron", "GoogleRobot"]:
                mode = "mobile"
            else:
                mode = "arm"
        
        if mode == "mobile":
            generate_mobile_subplots(robot_name=robot)
        else:
            generate_random_subplots(robot_name=robot)
    fire.Fire(run)
