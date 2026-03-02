"""
头部姿态估计模块 - WHENet (ONNX)
输入: 224x224 人脸图像
输出: yaw, pitch, roll (度) + confidence
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

# 默认模型路径
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "whenet_1x3x224x224_prepost.onnx"


@dataclass
class PoseResult:
    """姿态估计结果

    角度单位: 度 (degree)
    角度范围:
        yaw:   [-180, 180] 左右转头，正值=向右
        pitch: [-90, 90]   抬头低头，正值=抬头
        roll:  [-90, 90]   歪头，正值=向右歪
    """
    yaw: float
    pitch: float
    roll: float
    confidence: float
    method: str = "whenet"  # "whenet" | "pnp"

    def to_dict(self) -> dict:
        return {
            "yaw": round(self.yaw, 2),
            "pitch": round(self.pitch, 2),
            "roll": round(self.roll, 2),
            "confidence": round(self.confidence, 3),
            "method": self.method
        }


class HeadPoseEstimator:
    """头部姿态估计器 - WHENet ONNX

    使用 ONNX Runtime 进行推理，支持 CPU 和 GPU
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(DEFAULT_MODEL_PATH)
        self.session = None
        self._load_model()

    def _load_model(self):
        """加载 ONNX 模型"""
        import onnxruntime as ort

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """预处理人脸图像 (ONNX prepost 模型已包含归一化)"""
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # 添加 batch 维度
        return img

    def estimate(self, face_image: np.ndarray,
                 landmarks: np.ndarray = None) -> PoseResult:
        """估计头部姿态

        Args:
            face_image: 裁剪的人脸图像 (BGR)
            landmarks: 5点关键点 [5, 2], 用于 PnP 回退

        Returns:
            PoseResult
        """
        if face_image.size == 0:
            return PoseResult(0, 0, 0, 0, "empty")

        # 预处理
        img = self._preprocess(face_image)

        # ONNX 推理
        outputs = self.session.run(self.output_names, {self.input_name: img})
        yaw, roll, pitch = outputs[0][0]

        result = PoseResult(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            confidence=0.9,
            method="whenet"
        )

        return result

    def _estimate_pnp(self, landmarks: np.ndarray,
                      image_size: tuple) -> PoseResult:
        """基于 5 点关键点的 PnP 姿态估计"""
        # 3D 人脸模型点 (通用模型)
        model_points = np.array([
            [-30, -30, -30],   # 左眼
            [30, -30, -30],    # 右眼
            [0, 0, 0],         # 鼻尖
            [-20, 30, -20],    # 左嘴角
            [20, 30, -20]      # 右嘴角
        ], dtype=np.float64)

        h, w = image_size
        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            model_points,
            landmarks.astype(np.float64),
            camera_matrix,
            np.zeros((4, 1))
        )

        if not success:
            return PoseResult(0, 0, 0, 0, "pnp_failed")

        rmat, _ = cv2.Rodrigues(rvec)
        pose_mat = cv2.hconcat([rmat, np.zeros((3, 1))])
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
        pitch, yaw, roll = euler.flatten()

        return PoseResult(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            confidence=0.5,
            method="pnp"
        )

    def estimate_batch(self, face_images: list) -> List[PoseResult]:
        """批量估计头部姿态

        Args:
            face_images: 人脸图像列表

        Returns:
            PoseResult 列表
        """
        if not face_images:
            return []

        # 逐个推理 (ONNX 模型为单 batch)
        results = []
        for img in face_images:
            results.append(self.estimate(img))

        return results
