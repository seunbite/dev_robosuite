import numpy as np
import os
import sys
import time
import argparse

# 로컬 robosuite 경로 설정
local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite
import robosuite.models
robosuite.models.assets_root = os.path.join(local_root, "robosuite", "models", "assets")

# Stretch 2는 단일 팔과 높낮이 조절이 가능한 토르소를 가진 모바일 매니퓰레이터입니다.
# robosuite에 기본 내장된 모델 중 구조적으로 가장 유사한 'PandaOmron'을 사용합니다.
POSE_CONFIGS = {
    "stand": {
        "torso": [0.30],  # 토르소를 높게
        "arm": [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4],
        "description": "Stretched up (torso extended)"
    },
    "sit": {
        "torso": [0.05],  # 토르소를 낮게
        "arm": [0, np.pi/4, 0, -np.pi/2, 0, np.pi/4, np.pi/4],
        "description": "Sitting low (torso retracted)"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Control Stretch 2 (PandaOmron) pose in robosuite.")
    parser.add_argument(
        "--pose", 
        type=str, 
        default="stand", 
        choices=["sit", "stand"],
        help="Pose of the robot (sit or stand)"
    )
    args = parser.parse_args()

    print(f"Setting Stretch 2 (Proxy: PandaOmron) to: {args.pose} ({POSE_CONFIGS[args.pose]['description']})")

    # 환경 생성
    controller_config = suite.load_composite_controller_config(robot="PandaOmron")
    
    env = suite.make(
        env_name="EmptySpace",
        robots="PandaOmron",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=50,
        render_camera="sideview",
        controller_configs=controller_config,
    )

    # 초기 상태 설정
    config = POSE_CONFIGS[args.pose]
    env.reset()
    
    # PandaOmron qpos 구조: [base_forward, base_side, base_yaw, torso_height, arm_joints(7), gripper...]
    env.sim.data.qpos[0:3] = [0, 0, 0] # Base position
    env.sim.data.qpos[3] = config["torso"][0] # Torso height
    env.sim.data.qpos[4:11] = config["arm"] # Arm joints
    env.sim.forward()

    print("Simulation running. Close the window or press Ctrl+C to exit.")

    try:
        while True:
            # 관절 위치 유지
            env.step(np.zeros(env.action_dim))
            env.render()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
