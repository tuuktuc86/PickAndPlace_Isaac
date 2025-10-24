import torch
import torch.nn as nn
import cv2
import numpy as np
# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


from env import set_env_dataCollection
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)
# import os, sys
# sys.path.insert(0, os.path.dirname(__file__))  # 로컬 최우선
#from utils.visual_utils import * 
from task_utils.robot_desiredState import PickAndPlaceSm
from task_utils import visual_utils
from task_utils import robot_desiredState
#load env and device
from task_utils.setting_config import device
from task_utils.setting_config import env
from task_utils.visual_utils import robot_camera

from scipy.spatial.transform import Rotation as R

# from task_utils.visual_utils import front_camera
print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


from isaaclab.managers import SceneEntityCfg

import os, sys

# 현재 파일 기준 상위 경로 (/fail-detect_acss/PickAndPlace_forDataCollection)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 상위 디렉토리 (/fail-detect_acss)
PARENT_DIR = os.path.dirname(BASE_DIR)

# 상위 디렉토리를 sys.path에 추가
sys.path.append(PARENT_DIR)



import matplotlib.pyplot as plt
import os
# os.makedirs(save_dir, exist_ok=True)


max_steps = 25000
episodes=5

pick_and_place_sm = PickAndPlaceSm(
    dt=set_env_dataCollection.env_cfg.sim.dt * set_env_dataCollection.env_cfg.decimation, 
    num_envs=env.unwrapped.num_envs,
    device=device,
    position_threshold=0.01
)

for ep in range(episodes):
    obs, _ = env.reset()
    steps = 0
    while True:
        object_pose = torch.zeros((1, 7), device=device)

        
        if steps > 0 :
            pos = obs[0][18:21].float().to(device).unsqueeze(0)          # [1,3]
            orient = torch.tensor([[0., 1., 0., 0.]], device=device)   # [1,4]
            object_pose = torch.cat([pos, orient], dim=1)              # [1,7]
              
        prepose = object_pose.clone()
        prepose[:, 2] += 0.085
        pick_and_place_sm.grasp_pose = object_pose
        pick_and_place_sm.pregrasp_pose = prepose
        # 로봇의 End-Effector 위치와 자세를 기반으로 actions 계산

                
        robot_data = env.unwrapped.scene["robot"].data
        ee_frame_sensor = env.unwrapped.scene["ee_frame"]
        tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
        tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
        ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)

        # state machine 을 통한 action 값 출력
        actions = pick_and_place_sm.compute(
            ee_pose=ee_pose,
            grasp_pose=pick_and_place_sm.grasp_pose,
            pregrasp_pose=pick_and_place_sm.pregrasp_pose,
            robot_data=robot_data,
        )
        obs, _, terminated, truncated, info = env.step(actions)
       # print(f"state = {pick_and_place_sm.sm_state}")
        steps += 1


        done = bool(getattr(terminated, "any", lambda: terminated)()) or \
                bool(getattr(truncated, "any", lambda: truncated)())
        
        if done :
            print("*********reset****************")
            print(terminated, truncated)
            pick_and_place_sm.reset_idx(0)
            env.reset()

        if (max_steps is not None and steps >= max_steps):
            print(f"[Episode {ep+1}], steps={steps}")
            break
