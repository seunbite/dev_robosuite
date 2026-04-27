
import fire
import os
import json
import itertools
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Optional, List, Dict

# Import the original MotionGenerator
from motion_generation import MotionGenerator
import mujoco

class DebugMotionGenerator(MotionGenerator):
    def execute_debug(
        self,
        cue: str,
        pose_id: int,
        config_path: str = "data/results/motion_configs/manipulator/motion_configs.json",
        proximal_degree_scale: float = 0.25,
        hz: int = 10,
        cue_idx: Optional[int] = None,
    ) -> str:
        """
        Executes the cue for a specific pose_id and generates a tiled GIF 
        showing ALL sign combinations for the movement.
        """
        config = self._load_cue_config(cue, config_path, cue_idx=cue_idx)
        actual_cue = config.get('cue', cue)
        movements = config.get('movements', [])
        
        print(f"Loading poses for cue: {actual_cue}...")
        # Load poses (using the same logic as parent, but we will filter for pose_id)
        # We need to access the pose data. The parent class loads it in __init__ or via jsonl.
        # We'll assume self.pose_data is populated or we load it from jsonl.
        
        # Filter for the specific pose_id
        target_pose_data = None
        
        # 1. Try to find in loaded poses if available
        if hasattr(self, 'poses') and self.poses:
             for p in self.poses:
                 if p.get('pose_id') == pose_id:
                     target_pose_data = p
                     break
        
        # 2. If not found, try to search in the jsonl file
        if target_pose_data is None:
            print(f"Searching for pose_id {pose_id} in {self.jsonl_path}...")
            with open(self.jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('pose_id') == pose_id:
                        target_pose_data = data
                        break
        
        if target_pose_data is None:
            raise ValueError(f"Pose ID {pose_id} not found in {self.jsonl_path}")

        print(f"Found pose_id {pose_id}. Generating debug combinations...")

        # We need to replicate the run_action logic but modified to return ALL combinations
        # instead of picking the best one.
        
        # Prepare for execution
        best_p = target_pose_data
        
        # Initialize state
        self.reset_to_initial_state()
        
        # Setup metadata for tiling
        tiled_frames_list = []
        tiled_pose_ids = []
        tiled_movement_texts = []
        tiled_reversed_axes = [] # We will use this to show the SIGNS used
        
        # We will assume the cue has sequential movements. 
        # For debugging, if there are multiple movements, the branching factor explodes.
        # We will generate combinations for the FIRST valid movement found and stop,
        # or process them sequentially if possible.
        # To keep it simple and robust: We will process the FIRST 'movement' type block found
        # and generate tiles for its combinations.
        
        frames = []
        cur_p = None
        cur_p_name = None
        
        # --- Modified run_action logic ---
        def run_debug_action(item, m_idx):
            nonlocal cur_p, cur_p_name, frames
            m_type, params = item.get('type'), item.get('parameters', {})
            
            if m_type == 'pose':
                # Standard pose execution (same as original)
                p_p = params.get('pose')
                sel_p = best_p # We force the target pose
                target = self._pose_data_to_joint_positions(sel_p)
                
                step_f = []
                if cur_p is None:
                    self._set_joint_positions(target); cur_p, cur_p_name = sel_p, p_p
                    for _ in range(max(1, int(params.get('hold_time', 1.0) * hz))): 
                        step_f.append(Image.fromarray(self._capture_image()))
                else:
                    start = self._get_joint_positions(); diff = np.max(np.abs(np.rad2deg(target - start)))
                    for f in range(max(1, int((diff / (20.0 * params.get('speed', 1.0))) * hz))):
                        self._set_joint_positions(start * (1 - (f+1)/max(1, int((diff/(20.0*params.get('speed', 1.0)))*hz))) + target * ((f+1)/max(1, int((diff/(20.0*params.get('speed', 1.0)))*hz))))
                        step_f.append(Image.fromarray(self._capture_image()))
                    self._set_joint_positions(target); cur_p, cur_p_name = sel_p, p_p
                    for _ in range(max(1, int(params.get('hold_time', 1.0) * hz))): 
                        step_f.append(Image.fromarray(self._capture_image()))
                
                return [step_f], ["Pose Set"] # Return list of frame-lists (only 1 variant)

            elif m_type == 'movement':
                # This is where we branch into combinations
                
                # 1. Joint Selection (Standard)
                # We need to select joints. We'll use the standard selection logic.
                # Note: _select_joint might fail if we are not careful, but we assume it works for the valid pose.
                # We need 'm_info' equivalent. Since we skipped the pre-selection loop, we do it here on the fly.
                
                # On-the-fly joint selection
                axis_str = self._infer_axis_string(params)
                joint_str = params.get('joint')
                
                # We need Jacobian.
                mujoco.mj_forward(self.env.sim.model._model, self.env.sim.data._data)
                jac_pos = np.zeros((3, self.env.sim.model.nv))
                jac_rot = np.zeros((3, self.env.sim.model.nv))
                site_id = mujoco.mj_name2id(self.env.sim.model._model, mujoco.mjtObj.mjOBJ_SITE, self.jacobian_calculator.eef_site_name)
                mujoco.mj_jacSite(self.env.sim.model._model, self.env.sim.data._data, jac_pos, jac_rot, site_id)
                jac = np.vstack([jac_pos, jac_rot])[:, self.jacobian_calculator.ik_solver.dof_ids]
                
                # Use active joint indices for name lookup
                joint_names_list = [self.jacobian_calculator.all_joint_names[j] if j < len(self.jacobian_calculator.all_joint_names) else f"Joint_{j}" for j in self.jacobian_calculator.active_joint_indices]
                
                joints = self._select_joint_with_jac(jac, self.jacobian_calculator.ik_solver.dof_ids, joint_names_list, axis_str, joint_str)
                
                if not joints:
                    raise ValueError(f"No joints selected for axis={axis_str}")

                # Configs setup
                configs = []
                fixed_joints = getattr(self.jacobian_calculator, 'fixed_joint_indices', [])
                for j in joints:
                    if j[0] < len(self.jacobian_calculator.active_joint_indices):
                        joint_idx = self.jacobian_calculator.active_joint_indices[j[0]]
                    else:
                        joint_idx = self._find_joint_index_in_robot(j[2])
                    
                    if joint_idx is not None and joint_idx not in fixed_joints:
                        configs.append({
                            'idx': joint_idx, 
                            'active_idx': j[0],
                            'jac_sign': j[4],
                            'joint_info': j
                        })

                # Parameters
                directions_list = params.get("directions")
                if not directions_list:
                    directions_list = [{"degrees": params.get("degrees", None), "speed": params.get("speed", 1.0), "hold_time": params.get("hold_time", 0.0)}]
                
                axis_tokens = self._parse_axes(axis_str)
                if not axis_tokens: axis_tokens = ["y"]
                
                axis_joint_pairs = []
                for i_cfg, c_cfg in enumerate(configs):
                    axis_name = axis_tokens[i_cfg] if i_cfg < len(axis_tokens) else axis_tokens[-1]
                    axis_joint_pairs.append((axis_name, c_cfg["idx"]))
                movement_joint_text = ", ".join([f"{a}:{j}" for a, j in axis_joint_pairs])

                # Prepare for combinations
                # We will generate a separate timeline for EACH combination of signs.
                # Since 'repetition' and 'directions_list' can add more steps, 
                # we will simplify: We assume 1 repetition and 1 direction entry for the debug view 
                # OR we branch at the FIRST direction application.
                
                # Let's take the first direction item to generate combinations.
                dir_item = directions_list[0]
                deg_val = dir_item.get("degrees")
                
                # Normalize deg_val to dict {axis: value}
                degrees_map = {}
                if isinstance(deg_val, dict):
                    degrees_map = deg_val
                elif isinstance(deg_val, (int, float)):
                    # Legacy: apply to all axes defined in axis_str
                    for a in axis_str:
                        degrees_map[a] = float(deg_val)
                elif isinstance(deg_val, list):
                    # Legacy list: map to axis_tokens by index
                    for idx, val in enumerate(deg_val):
                        if idx < len(axis_tokens):
                            degrees_map[axis_tokens[idx]] = float(val)
                
                spd = dir_item.get("speed", params.get("speed", 1.0))
                hold_time = dir_item.get("hold_time", params.get("hold_time", 0.0))
                
                # Calculate Magnitudes
                magnitudes = []
                desired_cartesian_intents = []
                
                for i, c in enumerate(configs):
                    # Determine which axis this joint config corresponds to
                    axis_for_joint = axis_tokens[i] if i < len(axis_tokens) else axis_tokens[-1]
                    input_d = degrees_map.get(axis_for_joint, 0.0)

                    joint_idx = c['idx']
                    joint_dir_scale = 1.0
                    if self.joint_direction_scale and joint_idx in self.joint_direction_scale:
                        joint_dir_scale = self.joint_direction_scale[joint_idx]
                    
                    scale_mag = 1.0
                    joint_info = c.get('joint_info', [])
                    joint_name = joint_info[1] if isinstance(joint_info, (list, tuple)) and len(joint_info) > 1 else ""
                    
                    if hasattr(self, 'robot_name') and 'GR1' in self.robot_name:
                        scale_mag *= 0.5
                    elif params.get('joint', '').lower() == 'any' and any(p in joint_name.lower() for p in ['shoulder', 'elbow']):
                        scale_mag *= proximal_degree_scale
                    
                    abs_d = abs(float(input_d))
                    mag = abs_d * scale_mag * joint_dir_scale
                    magnitudes.append(mag)
                    
                    desired_sign = float(np.sign(input_d))
                    if desired_sign == 0: desired_sign = 1.0
                    desired_cartesian_intents.append((axis_for_joint, desired_sign))

                # Capture state before movement
                cur = self._get_joint_positions()
                eef_before = self._get_eef_position()
                
                # --- Generate Combinations ---
                variants_frames = []
                variants_logs = []
                
                axis_idx_map = {"x": 0, "y": 1, "z": 2}
                
                print(f"Generating {2**len(configs)} combinations for axes: {[x[0] for x in desired_cartesian_intents]}")
                
                for signs in itertools.product([1.0, -1.0], repeat=len(configs)):
                    # Calculate offsets for this combination
                    test_offsets = [m * s for m, s in zip(magnitudes, signs)]
                    
                    # Run attempt with frames
                    # Helper to run attempt
                    def _run_attempt_debug(offsets_deg):
                        # Aggregate offsets by joint index to handle cases where the same joint
                        # is selected for multiple axes (e.g. diagonal movement).
                        joint_offsets = {}
                        for i_cfg, c_cfg in enumerate(configs):
                            idx = c_cfg['idx']
                            off_deg = offsets_deg[i_cfg]
                            joint_offsets[idx] = joint_offsets.get(idx, 0.0) + off_deg

                        active = []
                        max_d = 0.0
                        for idx, total_off_deg in joint_offsets.items():
                            active.append({'idx': idx, 'start': cur[idx], 'offset': np.deg2rad(total_off_deg)})
                            max_d = max(max_d, abs(total_off_deg))
                        
                        # Reset to start of movement
                        self._set_joint_positions(cur, steps=1)
                        local_frames = []
                        
                        num_f = max(1, int((max_d / (20.0 * spd)) * hz))
                        for f in range(num_f):
                            p_new = cur.copy()
                            for a in active:
                                p_new[a['idx']] = a['start'] + (f + 1) / num_f * a['offset']
                            self._set_joint_positions(p_new)
                            local_frames.append(Image.fromarray(self._capture_image()))
                            
                        if hold_time > 0:
                            for _ in range(max(1, int(hold_time * hz))):
                                local_frames.append(Image.fromarray(self._capture_image()))
                                
                        after = self._get_eef_position()
                        return local_frames, after
                    
                    variant_frames, eef_after = _run_attempt_debug(test_offsets)
                    delta = eef_after - eef_before
                    
                    # Calculate Score
                    score = 0
                    score_str_parts = []
                    for i, (axis, desired_sign) in enumerate(desired_cartesian_intents):
                        comp = delta[axis_idx_map.get(axis, 2)]
                        
                        # Weight decay: 1.0, 1.0/1.5, 1.0/(1.5^2), ...
                        weight = 1.0 / (1.5 ** i)
                        
                        term = desired_sign * comp * weight
                        score += term
                        score_str_parts.append(f"{axis}(w={weight:.2f}):{term:+.3f}")
                    
                    # Log string
                    # Format: Signs: (+, -) Score: ...
                    signs_str = ",".join(["+" if s > 0 else "-" for s in signs])
                    log_line = f"Signs({signs_str}) Score={score:+.3f}"
                    
                    # Full movement log
                    full_log = (
                        f"pre({eef_before[0]:+.2f},{eef_before[1]:+.2f},{eef_before[2]:+.2f}) "
                        f"post({eef_after[0]:+.2f},{eef_after[1]:+.2f},{eef_after[2]:+.2f}) "
                        f"joints({movement_joint_text})"
                    )
                    
                    variants_frames.append(variant_frames)
                    # We pack the log info: [full_log, log_line (as the 'reversed' text)]
                    variants_logs.append((full_log, log_line))
                
                return variants_frames, variants_logs

            return [], []

        # Execute the movements
        # We assume the cue starts with a pose, then a movement.
        # We will run the pose (common to all), then branch at the movement.
        
        common_frames = []
        
        # 1. Run Pose (if exists)
        if movements and movements[0]['type'] == 'pose':
            res_frames, _ = run_debug_action(movements[0], 0)
            common_frames = res_frames[0] # Pose has only 1 variant
            
            # 2. Run Movement (if exists)
            if len(movements) > 1 and movements[1]['type'] == 'movement':
                variant_frames_list, variant_logs_list = run_debug_action(movements[1], 1)
                
                # Combine common frames with variant frames
                for i, v_frames in enumerate(variant_frames_list):
                    combined = common_frames + v_frames
                    full_log, sign_log = variant_logs_list[i]
                    
                    tiled_frames_list.append(combined)
                    tiled_pose_ids.append(pose_id)
                    tiled_movement_texts.append([full_log, (sign_log, (220, 0, 0))])
                    tiled_reversed_axes.append([])
            else:
                # Just pose
                tiled_frames_list.append(common_frames)
                tiled_pose_ids.append(pose_id)
                tiled_movement_texts.append(["Pose Only"])
                tiled_reversed_axes.append([])
        
        else:
            # Try to find first movement
            for m_idx, m in enumerate(movements):
                if m['type'] == 'movement':
                    variant_frames_list, variant_logs_list = run_debug_action(m, m_idx)
                    for i, v_frames in enumerate(variant_frames_list):
                        tiled_frames_list.append(v_frames)
                        tiled_pose_ids.append(pose_id)
                        full_log, sign_log = variant_logs_list[i]
                        tiled_movement_texts.append([full_log, (sign_log, (220, 0, 0))])
                        tiled_reversed_axes.append([])
                    break

        # Save Tiled GIF
        if tiled_frames_list:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.robot_name}_{actual_cue.replace(' ', '_')}_p{pose_id}_debug_combinations.gif"
            path = os.path.join(self.output_dir, filename)
            
            self._save_tiled_frames_as_gif(
                tiled_frames_list,
                path,
                hz,
                tiled_pose_ids,
                tiled_movement_texts,
                tiled_reversed_axes
            )
            print(f"Saved debug GIF to: {path}")
            return path
        else:
            print("No frames generated.")
            return ""

def debug_generate(
    robot="IIWA",
    cue="beckoning",
    pose_id=None,
    config_path="data/results/motion_configs/manipulator/motion_configs.json",
    cue_idx=None,
):
    if pose_id is None:
        print("Error: pose_id is required for debugging.")
        return

    generator = DebugMotionGenerator(
        robot_name=robot,
        env_name="EmptySpace",
        controller_name="IK_POSE",
        jsonl_path="data/seed/_remainder/closest_poses_results.jsonl",
    )
    
    try:
        generator.execute_debug(
            cue=cue,
            pose_id=pose_id,
            config_path=config_path,
            cue_idx=cue_idx,
        )
    finally:
        generator.close()

if __name__ == "__main__":
    fire.Fire(debug_generate)
