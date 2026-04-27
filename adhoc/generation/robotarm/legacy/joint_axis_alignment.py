import os
import random
import numpy as np
import fire
import mujoco
import robosuite as suite

# 기존 JacobianCalculator 활용
from alphabet_jacobian import JacobianCalculator

def analyze_joint_alignment(robot="IIWA"):
    """
    Sets the robot to a random discrete pose and prints the Cartesian axis 
    alignment for each joint based on its Jacobian.
    """
    # 1. Initialize environment and calculator
    # Using has_offscreen_renderer=True to run silently
    try:
        gen = JacobianCalculator(robot_name=robot, has_offscreen_renderer=True)
    except Exception as e:
        print(f"Error initializing robot {robot}: {e}")
        return

    # 2. Set Random Pose: Each active joint at -90, 0, or 90 degrees
    num_active = len(gen.active_joint_indices)
    random_angles_deg = [random.choice([-90, 0, 90]) for _ in range(num_active)]
    random_angles_rad = np.deg2rad(random_angles_deg)
    
    # Construct and apply full joint positions
    joint_pos = gen.initial_joint_pos.copy()
    for i, idx in enumerate(gen.active_joint_indices):
        joint_pos[idx] = random_angles_rad[i]
    
    gen.robot.set_robot_joint_positions(joint_pos)
    gen.env.sim.forward()
    
    print(f"\n{'='*80}")
    print(f" ROBOT: {robot}")
    print(f" RANDOM POSE (deg): {random_angles_deg}")
    print(f"{'='*80}")
    print(f"{'Joint Name':<30} | {'Idx':<4} | {'Aligned Axis':<10} | {'Dominance Score':<10}")
    print("-" * 80)
    
    # 3. Compute Jacobian at End-Effector
    model = gen.env.sim.model._model
    data = gen.env.sim.data._data
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, gen.eef_site_name)
    
    # 3xN Position Jacobian
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv)) # Not used for this summary but needed for API
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
    
    # 4. Analyze each active joint
    axis_labels = ['X', 'Y', 'Z']
    
    for i, joint_idx in enumerate(gen.active_joint_indices):
        # Extract the column corresponding to this joint from the full Jacobian
        # Note: joint_idx here corresponds to the qpos/dof index used in Jacobian
        column = jac_pos[:, joint_idx]
        
        # Determine alignment
        abs_values = np.abs(column)
        max_axis_idx = np.argmax(abs_values)
        max_val = abs_values[max_axis_idx]
        
        # Calculate dominance score: max axis magnitude / total magnitude
        magnitude = np.linalg.norm(column)
        score = max_val / magnitude if magnitude > 1e-6 else 0.0
        
        joint_name = gen.joint_names[joint_idx] if joint_idx < len(gen.joint_names) else f"DOF_{joint_idx}"
        alignment = axis_labels[max_axis_idx]
        
        # Visual indicator if score is high (very aligned)
        star = "*" if score > 0.9 else ""
        
        print(f"{joint_name:<30} | {joint_idx:<4} | {alignment:<10} | {score:.4f} {star}")

    print(f"{'='*80}\n")
    gen.close()

if __name__ == "__main__":
    fire.Fire(analyze_joint_alignment)
