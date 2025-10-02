import sys
import os
import torch
import cv2
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
# Detection 모델 라이브러리 임포트
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task_utils.setting_config import device
from task_utils.setting_config import env

# Contact-GraspNet 모델 라이브러리 임포트
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)


DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(DIR_PATH, 'data/checkpoint/maskrcnn_ckpt/maskrcnn_trained_model_refined.pth') # <-- 사전 학습된 Weight

def depth2pc(depth, K, rgb=None):
    """ 뎁스 이미지를 포인트 클라우드로 변환하는 함수 """
    
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]
    
    if rgb is not None:
        rgb = rgb[y, x]
    
    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

def get_world_bbox(depth, K, bb): 
    """ Bounding Box의 좌표를 포인트 클라우드 기준 좌표로 변환하는 함수 """

    image_width = depth.shape[1]
    image_height = depth.shape[0]

    x_min, x_max = bb[0], bb[2]
    y_min, y_max = bb[1], bb[3]
    
    if y_min < 0:
        y_min = 0
    if y_max >= image_height:
        y_max = image_height-1
    if x_min < 0:
        x_min = 0
    if x_max >=image_width:
        x_max = image_width-1

    z_0, z_1 = depth[int(y_min), int(x_min)], depth[int(y_max), int(x_max)]
    
    def to_world(x, y, z):
        """ 뎁스 포인트를 3D 포인트로 변환하는 함수 """
        world_x = (x - K[0, 2]) * z / K[0, 0]
        world_y = (y - K[1, 2]) * z / K[1, 1]
        return world_x, world_y, z
    
    x_min_w, y_min_w, z_min_w = to_world(x_min, y_min, z_0)
    x_max_w, y_max_w, z_max_w = to_world(x_max, y_max, z_1)
    
    return x_min_w, y_min_w, x_max_w, y_max_w

def get_model_instance_segmentation(num_classes):
    """ 학습 때와 동일한 구조로 Mask R-CNN 모델을 생성합니다. """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


# Detection 모델 로드
NUM_CLASSES = 79
detection_model = get_model_instance_segmentation(NUM_CLASSES)
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


robot_camera = env.unwrapped.scene.sensors['camera']

# 카메라 인트린식(intrinsics)
K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()

