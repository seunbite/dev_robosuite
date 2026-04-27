import os
import sys

local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if local_root not in sys.path:
    sys.path.insert(0, local_root)

import robosuite as suite

def list_cameras(robot_name):
    print(f"\n--- {robot_name} 사용 가능 카메라 목록 ---")
    try:
        env = suite.make(
            env_name="EmptySpace",
            robots=robot_name,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        robot = env.robots[0]
        # 모델에 정의된 카메라 이름 출력
        print(f"Cameras: {robot.robot_model.cameras}")
        env.close()
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    list_cameras("PandaOmron")
