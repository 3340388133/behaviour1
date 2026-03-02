"""
人脸检测模块 - RetinaFace

集成质量门控功能，支持：
1. 检测结果质量评估
2. 低质量检测过滤
3. 质量信息保存
"""
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Optional

from .face_quality import (
    FaceQualityAssessor,
    DetectionQuality,
    QualityLevel,
    QUALITY_THRESHOLDS
)


@dataclass
class FaceDetection:
    """人脸检测结果"""
    bbox: np.ndarray              # [x1, y1, x2, y2]
    confidence: float
    landmarks: np.ndarray         # 5点关键点 [5, 2]
    quality: Optional[DetectionQuality] = None  # 质量评估结果

    def to_dict(self) -> dict:
        """转为可序列化字典"""
        result = {
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            "confidence": float(self.confidence),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None
        }
        if self.quality:
            result["quality"] = self.quality.to_dict()
        return result


class RetinaFaceDetector:
    def __init__(
        self,
        device: str = None,
        conf_threshold: float = 0.5,
        enable_quality_assessment: bool = True,
        quality_thresholds: dict = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.enable_quality_assessment = enable_quality_assessment

        # 初始化质量评估器
        if enable_quality_assessment:
            self.quality_assessor = FaceQualityAssessor(quality_thresholds)
        else:
            self.quality_assessor = None

        self._load_model()

    def _load_model(self):
        """加载RetinaFace模型"""
        try:
            from retinaface import RetinaFace as RF
            self.detector = RF
            self.use_lib = True
        except ImportError:
            # 回退到insightface
            from insightface.app import FaceAnalysis
            self.app = FaceAnalysis(allowed_modules=['detection'])
            ctx_id = 0 if self.device == "cuda" else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self.use_lib = False

    def detect(
        self,
        image: np.ndarray,
        assess_quality: bool = True,
        min_quality_level: QualityLevel = None
    ) -> List[FaceDetection]:
        """检测图像中的人脸

        Args:
            image: BGR格式图像
            assess_quality: 是否评估质量（需要 enable_quality_assessment=True）
            min_quality_level: 最低质量等级，低于此等级的检测将被过滤
                               None 表示不过滤

        Returns:
            检测到的人脸列表
        """
        if self.use_lib:
            detections = self._detect_retinaface(image)
        else:
            detections = self._detect_insightface(image)

        # 质量评估
        if assess_quality and self.quality_assessor and detections:
            for det in detections:
                det.quality = self.quality_assessor.assess_detection(
                    det.bbox, det.confidence, det.landmarks
                )

        # 质量过滤
        if min_quality_level is not None and detections:
            level_order = {
                QualityLevel.HIGH: 3,
                QualityLevel.MEDIUM: 2,
                QualityLevel.LOW: 1,
                QualityLevel.REJECT: 0
            }
            min_order = level_order[min_quality_level]
            detections = [
                det for det in detections
                if det.quality is None or level_order[det.quality.level] >= min_order
            ]

        return detections

    def _detect_retinaface(self, image: np.ndarray) -> List[FaceDetection]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)

        results = []
        if not isinstance(faces, dict):
            return results

        for face_id, face_data in faces.items():
            conf = face_data['score']
            if conf < self.conf_threshold:
                continue

            bbox = np.array(face_data['facial_area'])
            landmarks = np.array([
                face_data['landmarks']['left_eye'],
                face_data['landmarks']['right_eye'],
                face_data['landmarks']['nose'],
                face_data['landmarks']['mouth_left'],
                face_data['landmarks']['mouth_right']
            ])

            results.append(FaceDetection(bbox=bbox, confidence=conf, landmarks=landmarks))

        return results

    def _detect_insightface(self, image: np.ndarray) -> List[FaceDetection]:
        faces = self.app.get(image)

        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue

            results.append(FaceDetection(
                bbox=face.bbox.astype(int),
                confidence=float(face.det_score),
                landmarks=face.kps
            ))

        return results

    def crop_face(self, image: np.ndarray, detection: FaceDetection,
                  expand_ratio: float = 1.2) -> np.ndarray:
        """裁剪人脸区域，带扩展边界"""
        x1, y1, x2, y2 = detection.bbox
        w, h = x2 - x1, y2 - y1

        # 扩展边界
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        new_w, new_h = w * expand_ratio, h * expand_ratio

        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(image.shape[1], int(cx + new_w / 2))
        y2 = min(image.shape[0], int(cy + new_h / 2))

        return image[y1:y2, x1:x2]
