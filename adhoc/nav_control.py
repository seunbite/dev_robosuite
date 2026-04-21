import numpy as np
import os
import sys
import argparse
from PIL import Image

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite

def run_navigation():
    parser = argparse.ArgumentParser(description="Parameterized Navigation Control for Robosuite Robots")
    parser.add_argument("--robot", type=str, default="Tiago", choices=["Tiago", "PandaOmron"], help="Robot model")
    parser.add_argument("--dir", type=float, default=0.0, help="Initial direction in degrees (0=Forward, 90=Left)")
    parser.add_argument("--dist", type=float, default=1.0, help="Distance to travel in meters")
    parser.add_argument("--speed", type=float, default=0.2, help="Travel speed in m/s")
    parser.add_argument("--curve", type=float, default=0.0, help="Turning rate in degrees/sec (positive=Left, negative=Right)")
    args = parser.parse_args()

    print(f"[{args.robot}] 네비게이션 시작: 거리={args.dist}m, 속도={args.speed}m/s, 방향={args.dir}도, 커브={args.curve}도/s")

    # 환경 생성
    env = suite.make(
        env_name="EmptySpace",
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=50,
    )
    env.reset()

    # 초기 포즈 설정 및 액션 벡터 고정
    robot = env.robots[0]
    cc = robot.composite_controller
    part_info = cc._action_split_indexes
    current_action = np.zeros(env.action_dim)

    if args.robot == "Tiago":
        height = 0.3
        env.sim.data.qpos[3] = height # qpos 강제 설정
        if "torso" in part_info:
            s, e = part_info["torso"]
            current_action[s:e] = height # 액션 명령에도 높이 고정
    else: # PandaOmron
        height = 0.1
        env.sim.data.qpos[3] = height
        if "torso" in part_info:
            s, e = part_info["torso"]
            current_action[s:e] = height
    
    # 시작 각도 설정 (Base Yaw는 qpos[2])
    env.sim.data.qpos[2] = np.deg2rad(args.dir)
    env.sim.forward()

    # 필요한 총 스텝 수 계산
    dt = 1.0 / 50.0 # 50Hz
    total_time = args.dist / args.speed
    total_steps = int(total_time / dt)
    
    frames = []
    target_cam = "frontview" # 외부 정면 시점

    # 네비게이션 루프
    for i in range(total_steps):
        # 1. 현재 방향(Yaw) 가져오기
        curr_yaw = env.sim.data.qpos[2]
        
        # 2. 이동 거리 계산 (x, y)
        vx = args.speed * np.cos(curr_yaw)
        vy = args.speed * np.sin(curr_yaw)
        
        # 3. 위치 업데이트
        env.sim.data.qpos[0] += vx * dt
        env.sim.data.qpos[1] += vy * dt
        
        # 4. 회전(Curve) 업데이트
        env.sim.data.qpos[2] += np.deg2rad(args.curve) * dt
        
        # 5. 시뮬레이션 및 캡처 (고정된 높이 액션 전달)
        env.step(current_action) 
        
        if i % 2 == 0: # 25fps로 캡처
            frame = env.sim.render(camera_name=target_cam, width=512, height=512, depth=False)
            frames.append(Image.fromarray(np.flipud(frame)))

    # GIF 저장
    save_name = f"nav_{args.robot.lower()}.gif"
    if frames:
        frames[0].save(
            save_name,
            save_all=True,
            append_images=frames[1:],
            duration=40,
            loop=0
        )
        print(f"결과 저장 완료: {os.path.abspath(save_name)}")
    else:
        print("캡처된 프레임이 없습니다.")

    env.close()

if __name__ == "__main__":
    run_navigation()
