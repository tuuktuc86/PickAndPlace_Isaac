import torch
import torch.nn as nn

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

#load env and device
from task_utils.setting_config import device
from task_utils.setting_config import env
from task_utils.visual_utils import robot_camera
from task_utils.visual_utils import front_camera
print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")

# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed




import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
save_dir = "/tmp/images"
os.makedirs(save_dir, exist_ok=True)


max_steps = 500
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
        pick_and_place_sm.grasp_pose = torch.tensor([[ 3.0280e-01, -5.6916e-02,  3.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10], [ 3.0280e-01, -5.6916e-02,  3.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10]], device=device)
        pick_and_place_sm.pregrasp_pose = torch.tensor([[ 3.0280e-01, -5.6916e-02,  0.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10], [ 3.0280e-01, -5.6916e-02,  0.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10]], device=device)
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
        
        image_ = robot_camera.data.output["rgb"][0]
        image = image_.permute(2, 0, 1).squeeze()          #(height, width, channels) 로 변환
        img_np = image_.squeeze().detach().cpu().numpy()
        normalized_image = (image - image.min()) / (image.max() - image.min())
        rgb_img = normalized_image.permute(1, 2, 0).detach().cpu().numpy()
        # 파일명 자동 증가

        plt.imshow(rgb_img)
        plt.title("Normalized RGB")
        fname = os.path.join(save_dir, f"wristCam_{int(steps)}.png")
        plt.savefig(fname)
        plt.close()

        image_ = front_camera.data.output["rgb"][0]
        image = image_.permute(2, 0, 1).squeeze()          #(height, width, channels) 로 변환
        img_np = image_.squeeze().detach().cpu().numpy()
        normalized_image = (image - image.min()) / (image.max() - image.min())
        rgb_img = normalized_image.permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(rgb_img)
        plt.title("Normalized RGB")
        fname = os.path.join(save_dir, f"frontCam_{int(steps)}.png")
        plt.savefig(fname)
        plt.close()

        obs, _, terminated, truncated, info = env.step(actions)
        steps += 1


        done = bool(getattr(terminated, "any", lambda: terminated)()) or \
                bool(getattr(truncated, "any", lambda: truncated)())
        if (max_steps is not None and steps >= max_steps):
            print(f"[Episode {ep+1}], steps={steps}")
            break
