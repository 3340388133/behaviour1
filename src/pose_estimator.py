"""
头部姿态估计模块 - WHENet
基于论文: WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose
支持全角度范围 Yaw: [-180°, 180°], Pitch: [-90°, 90°], Roll: [-90°, 90°]
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoseResult:
    """姿态估计结果"""
    yaw: float      # 偏航角（左右转头）[-180, 180]
    pitch: float    # 俯仰角（抬头低头）[-90, 90]
    roll: float     # 翻滚角（歪头）[-90, 90]
    confidence: float


class WHENet(nn.Module):
    """
    WHENet网络结构
    - Backbone: EfficientNet-Lite0
    - 输出: Yaw/Pitch/Roll 分类 + 回归
    """

    def __init__(self, bins_yaw=120, bins_pitch=66, bins_roll=66):
        super().__init__()
        self.bins_yaw = bins_yaw
        self.bins_pitch = bins_pitch
        self.bins_roll = bins_roll

        # Backbone: EfficientNet-Lite0
        self.backbone = self._build_backbone()

        # 分类头
        self.fc_yaw = nn.Linear(1280, bins_yaw)
        self.fc_pitch = nn.Linear(1280, bins_pitch)
        self.fc_roll = nn.Linear(1280, bins_roll)

        # 索引张量 (用于期望值计算)
        self.idx_yaw = torch.arange(bins_yaw).float()
        self.idx_pitch = torch.arange(bins_pitch).float()
        self.idx_roll = torch.arange(bins_roll).float()

    def _build_backbone(self):
        """构建EfficientNet-Lite0 backbone"""
        try:
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            # 移除分类头，保留特征提取部分
            return nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        except ImportError:
            # 简化版backbone
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(128, 1280)
            )

    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)

        # 分类logits
        yaw_logits = self.fc_yaw(features)
        pitch_logits = self.fc_pitch(features)
        roll_logits = self.fc_roll(features)

        return yaw_logits, pitch_logits, roll_logits

    def predict(self, x):
        """预测角度值 (使用softmax期望)"""
        yaw_logits, pitch_logits, roll_logits = self.forward(x)

        # Softmax + 期望值
        yaw_prob = torch.softmax(yaw_logits, dim=1)
        pitch_prob = torch.softmax(pitch_logits, dim=1)
        roll_prob = torch.softmax(roll_logits, dim=1)

        device = x.device
        idx_yaw = self.idx_yaw.to(device)
        idx_pitch = self.idx_pitch.to(device)
        idx_roll = self.idx_roll.to(device)

        # 期望值计算
        yaw = torch.sum(yaw_prob * idx_yaw, dim=1)
        pitch = torch.sum(pitch_prob * idx_pitch, dim=1)
        roll = torch.sum(roll_prob * idx_roll, dim=1)

        # 转换为角度: WHENet bins映射
        # Yaw: 120 bins -> [-180, 177] (3度间隔)
        # Pitch/Roll: 66 bins -> [-99, 96] (3度间隔)
        yaw = yaw * 3 - 180
        pitch = pitch * 3 - 99
        roll = roll * 3 - 99

        # 置信度: 使用最大概率
        conf = torch.max(yaw_prob, dim=1)[0]

        return yaw, pitch, roll, conf


class HeadPoseEstimator:
    """头部姿态估计器 - 基于WHENet"""

    def __init__(self, model_path: str = None, device: str = None, backend: str = "whenet"):
        """
        Args:
            model_path: WHENet权重路径
            device: 计算设备
            backend: 后端类型 (whenet / 6drepnet)，目前统一使用 whenet
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.backend = backend
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载WHENet模型"""
        self.model = WHENet()
        self.model.to(self.device)

        if self.model_path and Path(self.model_path).exists():
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded WHENet weights from {self.model_path}")

        self.model.eval()

    def _preprocess(self, face_image: np.ndarray) -> torch.Tensor:
        """预处理人脸图像"""
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return img.to(self.device)

    def estimate(self, face_image: np.ndarray) -> PoseResult:
        """估计头部姿态

        Args:
            face_image: 裁剪的人脸图像 (BGR)

        Returns:
            PoseResult
        """
        if face_image.size == 0:
            return PoseResult(yaw=0, pitch=0, roll=0, confidence=0)

        img = self._preprocess(face_image)

        with torch.no_grad():
            yaw, pitch, roll, conf = self.model.predict(img)

        return PoseResult(
            yaw=float(yaw.item()),
            pitch=float(pitch.item()),
            roll=float(roll.item()),
            confidence=float(conf.item())
        )

    def estimate_from_landmarks(self, landmarks: np.ndarray,
                                 image_size: tuple) -> PoseResult:
        """基于5点关键点估计姿态（PnP回退方法）

        Args:
            landmarks: 5点关键点 [5, 2]
            image_size: (width, height)

        Returns:
            PoseResult
        """
        model_points = np.array([
            [-30, -30, -30],   # 左眼
            [30, -30, -30],    # 右眼
            [0, 0, 0],         # 鼻尖
            [-20, 30, -20],    # 左嘴角
            [20, 30, -20]      # 右嘴角
        ], dtype=np.float64)

        w, h = image_size
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            model_points, landmarks.astype(np.float64),
            camera_matrix, np.zeros((4, 1))
        )

        if not success:
            return PoseResult(yaw=0, pitch=0, roll=0, confidence=0.1)

        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat([rmat, np.zeros((3, 1))])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler.flatten()

        return PoseResult(yaw=float(yaw), pitch=float(pitch),
                         roll=float(roll), confidence=0.6)
