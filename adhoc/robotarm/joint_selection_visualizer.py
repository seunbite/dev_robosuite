import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from PIL import Image
import mujoco
import fire

from alphabet_jacobian import JacobianCalculator
from motion_generation import MotionGenerator

def project_point(sim, world_pos, camera_name, width, height):
    """Project a 3D world point to 2D pixel coordinates."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    pos = sim.data.cam_xpos[cam_id]
    rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    rel_pos = world_pos - pos
    cam_pos = rot.T @ rel_pos
    focal = height / (2 * np.tan(np.deg2rad(fovy) / 2))
    if cam_pos[2] >= 0: return None
    dist = -cam_pos[2]
    x_pix = (width / 2) + (cam_pos[0] / dist) * focal
    y_pix = (height / 2) - (cam_pos[1] / dist) * focal
    return x_pix, y_pix

VEC_MAP = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}

def visualize_joint_selection(
    robot="IIWA", 
    output_path="data/logs/joint_selection_grid.png", 
    show_arrow=False, 
    show_circle=False, 
    show_origin=True,
    arrow_len=0.4, 
    circle_rad=15,
    origin_len=0.3,
    seed=0
    ):
    print(f"\n[Search] Starting random discrete pose search for {robot}...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        gen = MotionGenerator(robot_name=robot, has_offscreen_renderer=True)
    except Exception as e:
        print(f"Failed to init {robot}: {e}")
        return

    num_active = len(gen.jacobian_calculator.active_joint_indices)
    prox_indices = [0, 1]
    dist_indices = [num_active - 2, num_active - 1]
    
    results_by_axis = {}
    needed_axes = ["x", "y", "z"]
    axis_labels = ['x', 'y', 'z']
    
    for attempt in range(500):
        if not needed_axes: break
        
        random_angles_deg = [random.choice([-90, 0, 90]) for _ in range(num_active)]
        joint_pos = gen.initial_joint_pos.copy()
        for i, idx in enumerate(gen.jacobian_calculator.active_joint_indices):
            joint_pos[idx] = np.deg2rad(random_angles_deg[i])
        
        gen._set_joint_positions(joint_pos)
        
        # 1.5. EE Front Check (ee_x > root_x)
        sid = gen.env.sim.model.site_name2id(gen.jacobian_calculator.eef_site_name)
        ee_pos = gen.env.sim.data.site_xpos[sid]
        root_body = gen.robot.robot_model.root_body
        root_pos = gen.env.sim.data.get_body_xpos(root_body)
        
        if ee_pos[0] <= root_pos[0]:
            continue # Skip if EE is behind or at the same X as root
        
        model = gen.env.sim.model._model
        data = gen.env.sim.data._data
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, gen.jacobian_calculator.eef_site_name)
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
        
        for axis_name in list(needed_axes):
            axis_idx = axis_labels.index(axis_name)
            found_prox, found_dist = None, None
            
            for i, joint_idx in enumerate(gen.jacobian_calculator.active_joint_indices):
                column = jac_pos[:, joint_idx]
                mag = np.linalg.norm(column)
                if mag < 1e-6: continue
                score = np.abs(column[axis_idx]) / mag
                
                if score > 0.9:
                    sign = np.sign(column[axis_idx])
                    if i in prox_indices: found_prox = (joint_idx, gen.jacobian_calculator.joint_names[joint_idx], sign)
                    if i in dist_indices: found_dist = (joint_idx, gen.jacobian_calculator.joint_names[joint_idx], sign)
            
            if found_prox and found_dist:
                results_by_axis[axis_name] = {
                    "joint_pos": joint_pos.copy(),
                    "prox": (found_prox[0], found_prox[1]), "prox_sign": found_prox[2],
                    "dist": (found_dist[0], found_dist[1]), "dist_sign": found_dist[2]
                }
                needed_axes.remove(axis_name)
                print(f"  [LOCKED {axis_name.upper()}] Attempt {attempt}")

    grid_order = [("x", "prox"), ("x", "dist"), ("y", "prox"), ("y", "dist"), ("z", "prox"), ("z", "dist")]
    fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
    
    for i, (axis, pref) in enumerate(grid_order):
        if axis not in results_by_axis:
            axes[i].text(0.5, 0.5, f"MISSING\n{axis.upper()}", ha='center', va='center', color='gray')
            axes[i].axis('off'); continue
            
        data = results_by_axis[axis]
        j_id, j_name = data[pref]
        j_sign = data[f"{pref}_sign"]
        
        # 1. Render Original (50%)
        gen._set_joint_positions(data["joint_pos"])
        cam_id = gen.env.sim.model.camera_name2id("frontview")
        gen.env.sim.model.cam_fovy[cam_id] = 60
        img_orig = np.flipud(gen.env.sim.render(width=640, height=640, camera_name="frontview")).astype(float)
        
        # 2. Render Moved (Prox: 45, Dist: 90)
        moved_pos = data["joint_pos"].copy()
        rot_deg = 45 if pref == "prox" else 90
        moved_pos[j_id] += j_sign * np.deg2rad(rot_deg) # Move TOWARDS the axis arrow
        gen._set_joint_positions(moved_pos)
        img_moved = np.flipud(gen.env.sim.render(width=640, height=640, camera_name="frontview")).astype(float)
        
        # 3. Blend & Crop
        img_blended = (0.5 * img_orig + 0.5 * img_moved).astype(np.uint8)
        axes[i].imshow(img_blended[:int(img_blended.shape[0] * 0.8), :, :])
        axes[i].axis('off')
        
        axes[i].text(0.5, -0.05, f"{axis.upper()} ({'Proximal' if pref=='prox' else 'Distal'})", 
                    transform=axes[i].transAxes, ha='center', va='top', fontsize=14)
        
        # Red Circle (Original Pos)
        if show_circle:
            gen._set_joint_positions(data["joint_pos"])
            jid = gen.env.sim.model.dof_jntid[j_id]
            jnt_pos = gen.env.sim.data.xanchor[jid]
            pp = project_point(gen.env.sim, jnt_pos, "frontview", 640, 640)
            if pp and pp[1] < img_blended.shape[0] * 0.8:
                axes[i].add_patch(plt.Circle((pp[0], pp[1]), circle_rad, color='red', fill=False, lw=2))
            
        # Large White Arrow with Black Outline
        if show_arrow:
            sid = gen.env.sim.model.site_name2id(gen.jacobian_calculator.eef_site_name)
            ee_pos = gen.env.sim.data.site_xpos[sid]
            p1 = project_point(gen.env.sim, ee_pos, "frontview", 640, 640)
            p2 = project_point(gen.env.sim, ee_pos + np.array(VEC_MAP[axis])*arrow_len, "frontview", 640, 640)
            # if p1 and p2:
                # ann = axes[i].annotate("", xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                                #  arrowprops=dict(arrowstyle="->", color="white", lw=4, mutation_scale=30))
                # ann.arrow_patch.set_path_effects([PathEffects.withStroke(linewidth=6, foreground='black')])

        # World Origin Axes (RGB) - Show only the relevant axis for each subplot
        if show_origin:
            # Use root position as origin for coordinate axes
            root_body = gen.robot.robot_model.root_body
            root_pos = gen.env.sim.data.get_body_xpos(root_body)
            p_start = project_point(gen.env.sim, root_pos, "frontview", 640, 640)
            
            if p_start:
                axis_configs = [([1,0,0], '#FF3333', 'x'), ([0,1,0], '#33FF33', 'y'), ([0,0,1], '#33CCFF', 'z')]
                # Map subplot index to axis: 0,1 -> X | 2,3 -> Y | 4,5 -> Z
                target_axis_idx = i // 2
                d_vec, color, label = axis_configs[target_axis_idx]
                
                p_end = project_point(gen.env.sim, root_pos + np.array(d_vec) * origin_len, "frontview", 640, 640)
                if p_end:
                    ann_origin = axes[i].annotate("", xy=p_end, xytext=p_start,
                                     arrowprops=dict(arrowstyle="->", color=color, lw=2, mutation_scale=15))
                    ann_origin.arrow_patch.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='black')])
                    txt = axes[i].text(p_end[0], p_end[1], label, color=color, fontsize=14, fontweight='bold')
                    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])

    plt.subplots_adjust(wspace=0.01, hspace=0, left=0.005, right=0.995, bottom=0.15, top=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.01)
    print(f"\nGrid saved to: {output_path}")
    gen.close()

if __name__ == "__main__":
    fire.Fire(visualize_joint_selection)
