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

# Tiago 로봇은 bimanual(양팔) 모바일 매니퓰레이터입니다.
# Google Robot (Everyday Robots)과 구조적으로 가장 유사한 모델입니다.
POSE_CONFIGS = {
    "stand": {
        "torso": [0.35],  # 토르소를 높게
        "description": "Standing tall (torso extended)"
    },
    "sit": {
        "torso": [0.05],  # 토르소를 낮게
        "description": "Sitting low (torso retracted)"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Control Google Robot (Tiago) pose in robosuite.")
    parser.add_argument(
        "--pose", 
        type=str, 
        default="stand", 
        choices=["sit", "stand"],
        help="Pose of the robot (sit or stand)"
    )
    args = parser.parse_args()

    print(f"Setting Google Robot to: {args.pose} ({POSE_CONFIGS[args.pose]['description']})")

    # 환경 생성
    controller_config = suite.load_composite_controller_config(robot="Tiago")
    
    env = suite.make(
        env_name="EmptySpace",
        robots="Tiago",
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
    
    # Tiago의 qpos 구조: [base(3), torso(1), head(2), arm_right(6), arm_left(6), gripper...]
    # 실제 인덱스는 로봇 모델 정의에 따라 다를 수 있습니다.
    # Tiago 모델의 경우 qpos[0:3]은 베이스, qpos[3]은 torso lift joint입니다.
    
    # 몸체 위치 고정
    env.sim.data.qpos[0:3] = [0, 0, 0] # Base x, y, yaw
    env.sim.data.qpos[3] = config["torso"][0] # Torso joint (0 ~ 0.35m)
    env.sim.forward()

    print("Simulation running. Close the window or press Ctrl+C to exit.")

    try:
        while True:
            # 관절 위치 유지 (Tiago는 관절이 많아 zeros(action_dim)으로 유지)
            env.step(np.zeros(env.action_dim))
            env.render()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
