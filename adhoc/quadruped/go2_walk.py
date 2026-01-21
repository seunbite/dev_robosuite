import os
import sys

# Ensure local robosuite is at the very front of sys.path
local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite
# Force assets_root to point to the local workspace
import robosuite.models
robosuite.models.assets_root = os.path.join(local_root, "robosuite", "models", "assets")

print(f"--- Debug Info ---")
print(f"Robosuite Location: {suite.__file__}")
print(f"Assets Root: {robosuite.models.assets_root}")
print(f"------------------")

import numpy as np
from robosuite.utils.ik_utils import IKSolver
import time
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER
import logging

# Set logging level to suppress info logs from IK solver
ROBOSUITE_DEFAULT_LOGGER.setLevel(logging.WARNING)

class Go2GaitController:
    def __init__(self, env):
        self.env = env
        self.robot = env.robots[0]
        self.sim = env.sim
        
        # Define leg groups
        self.legs = ['FL', 'FR', 'RL', 'RR']
        self.diagonal_pairs = [['FL', 'RR'], ['FR', 'RL']]
        
        # Define sites
        self.foot_sites = [self.robot.robot_model.correct_naming(f"{leg}_foot_site") for leg in self.legs]
        
        # Initialize IK Solvers for each leg
        self.ik_solvers = {}
        for leg in self.legs:
            # We create a config for each leg's 3-DOF IK
            joint_names = [self.robot.robot_model.correct_naming(f"{leg}_{part}_joint") for part in ['hip', 'thigh', 'calf']]
            site_name = self.robot.robot_model.correct_naming(f"{leg}_foot_site")
            
            config = {
                "joint_names": joint_names,
                "end_effector_sites": [site_name],
                "nullspace_gains": [0.1, 0.1, 0.1]
            }
            
            self.ik_solvers[leg] = IKSolver(
                model=self.sim.model,
                data=self.sim.data,
                robot_config=config,
                damping=0.01,
                integration_dt=0.01,
                max_dq=2.0,
            )

        # Gait parameters
        self.T = 0.6  # Gait cycle period
        self.duty_factor = 0.5
        self.step_height = 0.06
        self.step_length = 0.12
        self.body_height = 0.28
        
        # Initial positions of feet relative to body
        # Get initial positions after reset
        self.env.reset()
        self.initial_body_pos = self.sim.data.qpos[0:3].copy()
        self.initial_foot_pos_rel = {}
        for leg in self.legs:
            prefixed_site_name = self.robot.robot_model.correct_naming(f"{leg}_foot_site")
            site_id = self.sim.model.site_name2id(prefixed_site_name)
            # Store initial foot position RELATIVE to initial body position
            self.initial_foot_pos_rel[leg] = self.sim.data.site_xpos[site_id].copy() - self.initial_body_pos
            
    def get_foot_trajectory(self, leg, t, phase_offset=0.0):
        """Compute target foot position relative to body for a given time."""
        phase = ((t / self.T) + phase_offset) % 1.0
        
        # Start from initial relative position
        pos_rel = self.initial_foot_pos_rel[leg].copy()
        
        if phase < self.duty_factor:
            # Swing phase (Lift and move forward)
            p_swing = phase / self.duty_factor
            # Move forward in x, up/down in z relative to standing position
            pos_rel[0] += self.step_length * (np.cos(np.pi * (1 - p_swing)) * 0.5 + 0.5 - 0.5)
            pos_rel[2] += self.step_height * np.sin(np.pi * p_swing)
        else:
            # Stance phase (Push backward)
            p_stance = (phase - self.duty_factor) / (1.0 - self.duty_factor)
            pos_rel[0] += self.step_length * (0.5 - p_stance)
            
        return pos_rel

    def step(self, t):
        # Current body position (qpos[0:3]) and orientation (qpos[3:7])
        # With freejoint, the first 7 qpos are for the base
        body_pos = self.sim.data.qpos[0:3]
        
        actions = np.zeros(12)
        
        for leg_idx, leg in enumerate(self.legs):
            # Assign phases: FL/RR pair 1 (offset 0), FR/RL pair 2 (offset 0.5)
            offset = 0.0 if (leg == 'FL' or leg == 'RR') else 0.5
            
            target_rel_pos = self.get_foot_trajectory(leg, t, offset)
            # Target in world frame = current body pos + target relative pos
            target_world_pos = body_pos + target_rel_pos
            
            # Solve IK for this leg
            target_action = np.concatenate([target_world_pos, [0, 0, 0]]) 
            leg_qpos = self.ik_solvers[leg].solve(target_action)
            
            for i in range(3):
                joint_name = f"{leg}_{['hip', 'thigh', 'calf'][i]}_joint"
                prefixed_joint_name = self.robot.robot_model.correct_naming(joint_name)
                joint_id = self.sim.model.joint_name2id(prefixed_joint_name)
                jnt_range = self.sim.model.jnt_range[joint_id]
                
                # Normalize to [-1, 1]
                # Note: leg_qpos index is 0, 1, 2 for each leg's 3 joints
                val = leg_qpos[i]
                norm_val = 2 * (val - jnt_range[0]) / (jnt_range[1] - jnt_range[0]) - 1
                actions[leg_idx * 3 + i] = np.clip(norm_val, -1, 1)
                
        return actions

def run_go2_walking():
    print("Starting Unitree Go2 Gait Planner Simulation...")
    print("Inspired by: https://github.com/felixokolo/go2_gait_planner")
    
    # Load controller config and enable gravity compensation
    controller_config = suite.load_composite_controller_config(robot="Go2")
    if "legs" in controller_config:
        controller_config["legs"]["gravity_compensation"] = True
        controller_config["legs"]["kp"] = 3000 # Higher Kp for stability with freejoint
        controller_config["legs"]["kd"] = 100
    
    # Create environment
    env = suite.make(
        env_name="EmptySpace",
        robots="Go2",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=50,
        render_camera="sideview",
        controller_configs=controller_config,
    )
    
    controller = Go2GaitController(env)
    
    # Properly initialize the robot's position and pose
    # Base position (x, y, z) and orientation (w, x, y, z)
    env.sim.data.qpos[0:3] = [0, 0, 0.28] # Body height at 28cm
    env.sim.data.qpos[3:7] = [1, 0, 0, 0] # Identity rotation
    
    # Set initial leg joints to home pose (0, 0.9, -1.8)
    # The joints start after the 7 base qpos
    env.sim.data.qpos[7:19] = [0.0, 0.9, -1.8] * 4
    
    env.sim.forward()
    
    print("Simulation Running. Close window to exit.")
    
    start_time = time.time()
    try:
        # Run for a longer period (e.g., 10000 steps = 200 seconds at 50Hz)
        for i in range(10000):
            t = i * (1.0 / 50.0)
            
            # Get actions from gait planner
            actions = controller.step(t)
            
            # Step simulation
            env.step(actions)
            env.render()
            
            # Add sleep to match real-time (50Hz)
            time.sleep(0.01) # Slightly less than 0.02 to account for computation time
            
            if i % 20 == 0:
                body_pos = env.sim.data.qpos[0:3]
                print(f"Time: {t:.2f}s | Step: {i} | Body Position: x={body_pos[0]:.3f}, y={body_pos[1]:.3f}, z={body_pos[2]:.3f}", flush=True)
                
    except Exception as e:
        print(f"\nSimulation interrupted: {e}")
    finally:
        print("\nSimulation Finished. Press Enter in the terminal to close the window...")
        try:
            input() # Wait for user input before closing
        except EOFError:
            pass
        env.close()
        print("Simulation Closed.")

if __name__ == "__main__":
    run_go2_walking()
