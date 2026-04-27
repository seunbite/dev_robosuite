import numpy as np
import os
import sys

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite

def analyze_robot(robot_name):
    print(f"\n=== {robot_name} 분석 시작 ===")
    try:
        env = suite.make(
            env_name="EmptySpace",
            robots=robot_name,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        robot = env.robots[0]
        
        print(f"1. 전체 Action Dimension: {env.action_dim}")
        
        # 컨트롤러 파트 확인
        if hasattr(robot, 'composite_controller'):
            cc = robot.composite_controller
            print("2. 컨트롤러 파트 및 인덱스:")
            for part_name, (start, end) in cc._action_split_indexes.items():
                print(f"   - {part_name:10}: {start} ~ {end} (dim: {end-start})")
        
        # 액추에이터 확인
        print("3. MuJoCo 액추에이터 목록:")
        for i in range(env.sim.model.nu):
            name = env.sim.model.actuator_id2name(i)
            print(f"   - [{i}] {name}")
            
        # 초기 qpos 확인
        print(f"4. 초기 Base qpos: {env.sim.data.qpos[0:3]}")
        
        env.close()
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    analyze_robot("Tiago")
    analyze_robot("PandaOmron")
