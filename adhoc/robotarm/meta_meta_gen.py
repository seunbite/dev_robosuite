every = 2

import os
from tqdm import tqdm, trange

for robot in ["GR1"]:
# for robot in ["IIWA"]:
    for index in trange(0, 50, every):
        if robot == "GR1":
            cmd = f"python adhoc/humanoid/meta_generate_humanoid_motions.py --robots ['{robot}'] --start_index {index}-{index+every}"
        else:
            cmd = f"python adhoc/robotarm/meta_motion_generation.py --robots ['{robot}'] --start_index {index}-{index+every}"
        os.system(cmd)
        