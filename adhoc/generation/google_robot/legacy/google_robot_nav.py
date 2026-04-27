import numpy as np
import os
import sys
import time
from PIL import Image

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite

def save_nav_gif():
    print("Google Robot (Tiago) 정면 GIF 저장 시작...")
    
    # 환경 생성
    env = suite.make(
        env_name="EmptySpace",
        robots="Tiago",
        has_renderer=False,          # 렌더링 창 대신 데이터만 획득
        has_offscreen_renderer=True, 
        use_camera_obs=False,
        control_freq=50,
    )
    env.reset()
    
    # 포즈 최적화 (토르소를 들어 올려서 잘 보이게)
    env.sim.data.qpos[3] = 0.3
    env.sim.forward()

    robot = env.robots[0]
    cc = robot.composite_controller
    part_info = cc._action_split_indexes
    current_action = np.zeros(env.action_dim)
    # 현재 포즈 유지용 액션 설정
    if "torso" in part_info:
        s, e = part_info["torso"]
        current_action[s:e] = 0.3

    frames = []
    target_cam = "frontview" # 로봇 정면에서 보는 외부 시점

    # 네비게이션 시나리오
    moves = [("전진", [0.01, 0, 0]), ("후진", [-0.01, 0, 0])]
    
    for name, vel in moves:
        print(f"  -> {name} 이동 중 및 캡처...")
        for _ in range(40):
            env.sim.data.qpos[0] += vel[0]
            env.sim.data.qpos[1] += vel[1]
            env.step(current_action)
            
            # 프레임 캡처
            frame = env.sim.render(camera_name=target_cam, width=512, height=512, depth=False)
            frame = np.flipud(frame) # 상하 반전
            frames.append(Image.fromarray(frame))

    # GIF 저장
    save_path = "google_robot_nav.gif"
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=40, # 20fps
        loop=0
    )
    print(f"\n저장 완료: {os.path.abspath(save_path)}")
    env.close()

if __name__ == "__main__":
    save_nav_gif()
