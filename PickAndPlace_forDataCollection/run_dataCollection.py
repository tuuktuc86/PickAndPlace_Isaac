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
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

import matplotlib.pyplot as plt
import os
save_dir = "/tmp/images"
os.makedirs(save_dir, exist_ok=True)

DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(DIR_PATH, 'data/checkpoint/maskrcnn_ckpt/maskrcnn_trained_model_refined.pth') # <-- 사전 학습된 Weight
NUM_CLASSES = 79  # 모델 구조는 학습 때와 동일해야 함
CONFIDENCE_THRESHOLD = 0.5
# Detection 모델 로드
detection_model = visual_utils.get_model_instance_segmentation(NUM_CLASSES)
detection_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
detection_model.eval()
detection_model.to(device)
# Contact-GraspNet 모델 config를 불러오기 위한 경로 설정
grasp_model_config_path = os.path.join(DIR_PATH, 'cgnet/configs/config.yaml')
grasp_model_config = cfg_from_yaml_file(grasp_model_config_path)

# Contact-GraspNet 모델 선언 및 checkpoint 입력을 통한 모델 weight 로드
grasp_model = builder.model_builder(grasp_model_config.model)
grasp_model_path = os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
builder.load_model(grasp_model, grasp_model_path)
grasp_model.to(device)
grasp_model.eval()
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

        
        

        for num_env in range(env.num_envs):

            if pick_and_place_sm.sm_state[num_env] == robot_desiredState.PickAndPlaceSmState.PREDICT:
                # image_ = robot_camera.data.output["rgb"][num_env]
                # image = image_.permute(2, 0, 1)  # (channels, height, width)
                # img_np = image_.detach().cpu().numpy()

                # # Continue with depth processing if needed
                # normalized_image = (image - image.min()) / (image.max() - image.min())
                # depth = robot_camera.data.output["distance_to_image_plane"][num_env]
                # depth_np = depth.squeeze().detach().cpu().numpy()                    # 취득한 Depth 이미지를 통한 Point Cloud 생성
                image_ = robot_camera.data.output["rgb"][num_env]
                image = image_.permute(2, 0, 1)  # (channels, height, width)
                img_np = image_.detach().cpu().numpy()

                # RGB normalize
                normalized_image = (image - image.min()) / (image.max() - image.min())
                rgb_img = normalized_image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, C)

                # Depth
                depth = robot_camera.data.output["distance_to_image_plane"][num_env]
                depth_np = depth.squeeze().detach().cpu().numpy()

                # 두 이미지를 나란히 저장
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(rgb_img)
                plt.title(f"RGB (env {num_env})")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(depth_np, cmap="viridis")
                plt.title(f"Depth (env {num_env})")
                plt.axis("off")

                plt.tight_layout()

                save_path = os.path.join(save_dir, f"rgb_depth_env{num_env}.png")
                plt.savefig(save_path)
                plt.close()
                
                if env.num_envs > 1:
                    pc, _ = visual_utils.depth2pc(depth_np, visual_utils.robot_camera_K[num_env])
                else:
                    pc, _ = visual_utils.depth2pc(depth_np, visual_utils.robot_camera_K)
                ############################ Detection Model Inference ############################

                print("Running detection inference...")

                # Detection 모델 추론
                with torch.no_grad():
                    prediction = detection_model([normalized_image])
                
                # 결과 후처리 및 시각화
                img_np = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                
                # Bbox와 텍스트를 그릴 이미지 레이어
                img_with_boxes = img_np.copy()

                # 마스크를 그릴 투명한 이미지 레이어
                mask_overlay = img_np.copy()
                
                # prediction에서 pred_scores, pred_boxes, pred_masks, pred_labels 값을 추출
                pred_scores = prediction[0]['scores'].cpu().numpy()
                pred_boxes = prediction[0]['boxes'].cpu().numpy()
                pred_masks = prediction[0]['masks'].cpu().numpy()
                pred_labels = prediction[0]['labels'].cpu().numpy()

                print(f"Found {len(pred_scores)} objects. Visualizing valid results...")

                # Detection 결과를 rgb 이미지 위에 표시
                # 각 인스턴스에 대한 크롭된 이미지 저장용
                crop_images = []
                # 바운딩 박스 정보 저장용
                bboxes = []
                # for i in range(len(pred_scores)):
                #     score = pred_scores[i]
                #     label_id = pred_labels[i]

                #     # 신뢰도와 레이블 ID를 함께 확인하여 Background 제외
                #     if score > CONFIDENCE_THRESHOLD and label_id != 0:
                #         color = visual_utils.get_random_color()
                        
                #         # --- Bbox와 텍스트 그리기 ---
                #         box = pred_boxes[i]
                #         x1, y1, x2, y2 = map(int, box)
                #         #cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                        
                #         class_names = CLASS_NAME
                #         label_text = f"{class_names[label_id]}: {score:.2f}"
                #         cv2.putText(img_with_boxes, label_text, (x1, y1 - 10), 
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                #         # --- 마스크 그리기 ---
                #         mask = pred_masks[i, 0]
                #         binary_mask = (mask > 0.5) # Boolean mask
                #         # 마스크 영역에만 색상 적용
                #         mask_overlay[binary_mask] = color

                #         # --- 바운딩 박스와 크롭된 이미지 저장 ---
                #         bboxes.append(box)
                #         crop_image = image[:, int(y1):int(y2), int(x1):int(x2)]
                #         crop_images.append(crop_image)

                # print("pred_boxes = ", pred_boxes) 
                # #d어차피 하나가 아니다 몇개씩 나온다. clip이 최소한 그 의미는 있다. bounding box가 하나가 아니라 여러개 쳐지는데 그중에서 괜찮은거 하나 가져오게 해주니까

            ############################ Grasp Model Inference ##################################
                # targe object의 bbox 위치를 image 좌표에서 world 좌표로 변환
                if env.num_envs > 1:
                    x_min_w, y_min_w, x_max_w, y_max_w = visual_utils.get_world_bbox(depth_np, visual_utils.robot_camera_K[num_env], pred_boxes[0])
                else:
                    x_min_w, y_min_w, x_max_w, y_max_w = visual_utils.get_world_bbox(depth_np, visual_utils.robot_camera_K, pred_boxes[0])

                # Robot의 end-effector 위치 얻기
                robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                robot_entity_cfg.resolve(env.unwrapped.scene)
                hand_body_id = robot_entity_cfg.body_ids[0]
                hand_pose_w = env.unwrapped.scene["robot"].data.body_state_w[:, hand_body_id, :]  # (num_envs, 13)

                if pc is not None:
                    offset = 0.08 # 바닥이 너무 조금 나올 경우, 바닥에 파지점이 생김
                    # target object가 있는 부분의 point cloud를 world bbox 기준으로 offset을 주고 crop
                    print(f"1pc shape = {pc.shape}")

                    pc = pc[pc[:, 0] > x_min_w-offset]
                    pc = pc[pc[:, 0] < x_max_w+offset]
                    pc = pc[pc[:, 1] > y_min_w-offset]
                    pc = pc[pc[:, 1] < y_max_w+offset]
                    pc = pc[pc[:, 2] > -0.1]  # z
                    
                    # target object의 3d point cloud 시각화
                    # pc_o3d = o3d.geometry.PointCloud()
                    # pc_o3d.points = o3d.utility.Vector3dVector(pc)
                    # o3d.visualization.draw_geometries([pc_o3d])
                    print(f"2pc shape = {pc.shape}")
                    # print(f"1pc shape = {pc.shape[1]}")
                    # Contact-GraspNet 모델 추론
                    rot_ee, trans_ee, width = inference_cgnet(pc, grasp_model, device, hand_pose_w, env)
                    print(f"[INFO] Received ee coordinates from inference_cgnet")
                    print(f"[INFO] Gripper width: {width}")
                    
                    # 예측한 파지점을 Isaaclab 형식으로 변환 (rotation matrix -> quat)
                    grasp_rot = rot_ee
                    pregrasp_pos = trans_ee
                    grasp_quat = R.from_matrix(grasp_rot).as_quat()  # (x, y, z, w)
                    grasp_quat = np.array([grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2]]) # (w, x, y, z)
                    
                    # rotation matrix를 사용하여 예측한 파지점의 offset 맞추기
                    z_axis = grasp_rot[:, 2]
                    grasp_pos = pregrasp_pos + z_axis * 0.085 # change : miss put point  

                    # 예측한 파지점 pose를 torch tensor로 변환
                    pregrasp_pose = np.concatenate([pregrasp_pos, grasp_quat])
                    grasp_pose = np.concatenate([grasp_pos, grasp_quat])
                    pregrasp_pose = torch.tensor(pregrasp_pose, device=device).unsqueeze(0)
                    grasp_pose = torch.tensor(grasp_pose, device=device).unsqueeze(0)

                    # State machine 에 grasp 및 pregrasp 자세 업데이트
                    pick_and_place_sm.grasp_pose[num_env] = grasp_pose[0]
                    pick_and_place_sm.pregrasp_pose[num_env] = pregrasp_pose[0]

            # image_ = robot_camera.data.output["rgb"][num_env]
            # image = image_.permute(2, 0, 1).squeeze()          #(height, width, channels) 로 변환
            # img_np = image_.squeeze().detach().cpu().numpy()
            # normalized_image = (image - image.min()) / (image.max() - image.min())
            # rgb_img = normalized_image.permute(1, 2, 0).detach().cpu().numpy()
            # # 파일명 자동 증가

            # plt.imshow(rgb_img)
            # plt.title("Normalized RGB")
            # fname = os.path.join(save_dir, f"Epi_{int(ep)}_Env_{int(num_env)}_WristCam_{int(steps)}.png")
            # plt.savefig(fname)
            # plt.close()

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
        steps += 1


        done = bool(getattr(terminated, "any", lambda: terminated)()) or \
                bool(getattr(truncated, "any", lambda: truncated)())
        if (max_steps is not None and steps >= max_steps):
            print(f"[Episode {ep+1}], steps={steps}")
            break
