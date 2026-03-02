"""
人脸检测质量门控模块

核心功能：
1. 单帧检测质量评估（confidence / bbox / landmarks）
2. Track-level 质量指标聚合
3. 低质量样本过滤与标记

质量分级：
- HIGH: 高质量，可直接用于姿态估计和训练
- MEDIUM: 中等质量，可用于姿态估计，训练时需标记
- LOW: 低质量，姿态估计可能不准，不建议用于训练
- REJECT: 拒绝，不进行后续处理
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class QualityLevel(Enum):
    """质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECT = "reject"


# ============================================================================
# 质量门控阈值配置
# ============================================================================
QUALITY_THRESHOLDS = {
    # 检测置信度阈值
    "confidence": {
        "high": 0.90,      # >= 0.90: 高质量
        "medium": 0.70,    # >= 0.70: 中等
        "low": 0.50,       # >= 0.50: 低质量
        "reject": 0.50     # < 0.50: 拒绝
    },

    # bbox 尺寸阈值（像素）
    "bbox_size": {
        "min_side": 32,           # 最小边长，低于此拒绝
        "low_side": 48,           # 低于此标记为低质量
        "high_side": 80,          # 高于此为高质量
        "min_area": 1024,         # 最小面积 (32x32)
        "high_area": 6400         # 高质量面积 (80x80)
    },

    # 高宽比阈值
    "aspect_ratio": {
        "min": 0.6,    # 最小高宽比（width/height）
        "max": 1.8,    # 最大高宽比
        "ideal_min": 0.7,   # 理想范围
        "ideal_max": 1.0
    },

    # 关键点阈值
    "landmarks": {
        "min_eye_distance": 10,       # 最小眼距（像素）
        "max_asymmetry_ratio": 0.3,   # 最大不对称比例
        "margin_ratio": 0.1           # 关键点距边界的最小距离比例
    },

    # Track-level 阈值
    "track": {
        "min_high_quality_ratio": 0.5,    # 高质量帧比例阈值
        "min_valid_ratio": 0.7,           # 有效帧（非reject）比例阈值
        "max_low_quality_ratio": 0.3      # 低质量帧比例上限
    }
}


@dataclass
class DetectionQuality:
    """单次检测的质量评估结果"""
    # 质量等级
    level: QualityLevel

    # 各维度得分 (0-1)
    confidence_score: float
    size_score: float
    aspect_ratio_score: float
    landmarks_score: float

    # 综合得分 (0-1)
    overall_score: float

    # 原始指标
    confidence: float
    bbox_width: int
    bbox_height: int
    bbox_area: int
    aspect_ratio: float
    eye_distance: Optional[float] = None
    asymmetry_ratio: Optional[float] = None

    # 问题标记
    issues: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "overall_score": round(self.overall_score, 3),
            "confidence_score": round(self.confidence_score, 3),
            "size_score": round(self.size_score, 3),
            "aspect_ratio_score": round(self.aspect_ratio_score, 3),
            "landmarks_score": round(self.landmarks_score, 3),
            "confidence": round(self.confidence, 3),
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "bbox_area": self.bbox_area,
            "aspect_ratio": round(self.aspect_ratio, 3),
            "eye_distance": round(self.eye_distance, 1) if self.eye_distance else None,
            "asymmetry_ratio": round(self.asymmetry_ratio, 3) if self.asymmetry_ratio else None,
            "issues": self.issues
        }


@dataclass
class TrackQuality:
    """Track-level 质量评估结果"""
    track_id: int
    total_detections: int
    valid_detections: int      # 非 reject 的检测数
    high_quality_count: int
    medium_quality_count: int
    low_quality_count: int
    reject_count: int

    # 质量比例
    valid_ratio: float
    high_quality_ratio: float
    low_quality_ratio: float

    # 统计指标
    mean_confidence: float
    mean_size_score: float
    mean_overall_score: float

    # 最终质量等级
    level: QualityLevel
    usable_for_training: bool

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "level": self.level.value,
            "usable_for_training": self.usable_for_training,
            "total_detections": self.total_detections,
            "valid_detections": self.valid_detections,
            "high_quality_count": self.high_quality_count,
            "medium_quality_count": self.medium_quality_count,
            "low_quality_count": self.low_quality_count,
            "reject_count": self.reject_count,
            "valid_ratio": round(self.valid_ratio, 3),
            "high_quality_ratio": round(self.high_quality_ratio, 3),
            "low_quality_ratio": round(self.low_quality_ratio, 3),
            "mean_confidence": round(self.mean_confidence, 3),
            "mean_size_score": round(self.mean_size_score, 3),
            "mean_overall_score": round(self.mean_overall_score, 3)
        }


class FaceQualityAssessor:
    """人脸检测质量评估器"""

    def __init__(self, thresholds: dict = None):
        self.thresholds = thresholds or QUALITY_THRESHOLDS

    def assess_detection(
        self,
        bbox: np.ndarray,
        confidence: float,
        landmarks: Optional[np.ndarray] = None
    ) -> DetectionQuality:
        """评估单次检测的质量

        Args:
            bbox: [x1, y1, x2, y2] 边界框
            confidence: 检测置信度
            landmarks: [5, 2] 关键点坐标，可选

        Returns:
            DetectionQuality 对象
        """
        issues = []

        # 1. 计算基础指标
        x1, y1, x2, y2 = bbox
        width = int(x2 - x1)
        height = int(y2 - y1)
        area = width * height
        aspect_ratio = width / height if height > 0 else 0

        # 2. 评估置信度
        conf_score = self._score_confidence(confidence)
        if confidence < self.thresholds["confidence"]["reject"]:
            issues.append("low_confidence")

        # 3. 评估尺寸
        size_score = self._score_size(width, height, area)
        min_side = min(width, height)
        if min_side < self.thresholds["bbox_size"]["min_side"]:
            issues.append("too_small")
        elif min_side < self.thresholds["bbox_size"]["low_side"]:
            issues.append("small_face")

        # 4. 评估高宽比
        ar_score = self._score_aspect_ratio(aspect_ratio)
        ar_thresh = self.thresholds["aspect_ratio"]
        if aspect_ratio < ar_thresh["min"] or aspect_ratio > ar_thresh["max"]:
            issues.append("abnormal_aspect_ratio")

        # 5. 评估关键点
        eye_distance = None
        asymmetry_ratio = None
        if landmarks is not None and len(landmarks) >= 5:
            lm_score, eye_distance, asymmetry_ratio, lm_issues = self._score_landmarks(
                landmarks, bbox
            )
            issues.extend(lm_issues)
        else:
            lm_score = 0.5  # 无关键点时给中等分
            if landmarks is None:
                issues.append("no_landmarks")

        # 6. 计算综合得分（加权平均）
        weights = {
            "confidence": 0.35,
            "size": 0.30,
            "aspect_ratio": 0.15,
            "landmarks": 0.20
        }
        overall_score = (
            conf_score * weights["confidence"] +
            size_score * weights["size"] +
            ar_score * weights["aspect_ratio"] +
            lm_score * weights["landmarks"]
        )

        # 7. 确定质量等级
        level = self._determine_level(
            confidence, min_side, area, aspect_ratio, overall_score, issues
        )

        return DetectionQuality(
            level=level,
            confidence_score=conf_score,
            size_score=size_score,
            aspect_ratio_score=ar_score,
            landmarks_score=lm_score,
            overall_score=overall_score,
            confidence=confidence,
            bbox_width=width,
            bbox_height=height,
            bbox_area=area,
            aspect_ratio=aspect_ratio,
            eye_distance=eye_distance,
            asymmetry_ratio=asymmetry_ratio,
            issues=issues
        )

    def _score_confidence(self, confidence: float) -> float:
        """置信度评分 (0-1)"""
        thresh = self.thresholds["confidence"]
        if confidence >= thresh["high"]:
            return 1.0
        elif confidence >= thresh["medium"]:
            return 0.7 + 0.3 * (confidence - thresh["medium"]) / (thresh["high"] - thresh["medium"])
        elif confidence >= thresh["low"]:
            return 0.4 + 0.3 * (confidence - thresh["low"]) / (thresh["medium"] - thresh["low"])
        else:
            return 0.4 * confidence / thresh["low"]

    def _score_size(self, width: int, height: int, area: int) -> float:
        """尺寸评分 (0-1)"""
        thresh = self.thresholds["bbox_size"]
        min_side = min(width, height)

        if min_side < thresh["min_side"]:
            return 0.0
        elif min_side >= thresh["high_side"]:
            return 1.0
        elif min_side >= thresh["low_side"]:
            return 0.6 + 0.4 * (min_side - thresh["low_side"]) / (thresh["high_side"] - thresh["low_side"])
        else:
            return 0.3 + 0.3 * (min_side - thresh["min_side"]) / (thresh["low_side"] - thresh["min_side"])

    def _score_aspect_ratio(self, aspect_ratio: float) -> float:
        """高宽比评分 (0-1)"""
        thresh = self.thresholds["aspect_ratio"]

        if aspect_ratio < thresh["min"] or aspect_ratio > thresh["max"]:
            return 0.0
        elif thresh["ideal_min"] <= aspect_ratio <= thresh["ideal_max"]:
            return 1.0
        elif aspect_ratio < thresh["ideal_min"]:
            return 0.5 + 0.5 * (aspect_ratio - thresh["min"]) / (thresh["ideal_min"] - thresh["min"])
        else:
            return 0.5 + 0.5 * (thresh["max"] - aspect_ratio) / (thresh["max"] - thresh["ideal_max"])

    def _score_landmarks(
        self,
        landmarks: np.ndarray,
        bbox: np.ndarray
    ) -> tuple:
        """关键点评分

        Returns:
            (score, eye_distance, asymmetry_ratio, issues)
        """
        issues = []
        thresh = self.thresholds["landmarks"]

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # 提取关键点
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]

        # 1. 检查关键点是否在 bbox 内（留边距）
        margin = min(width, height) * thresh["margin_ratio"]
        in_box_score = 1.0
        for pt in landmarks:
            if (pt[0] < x1 + margin or pt[0] > x2 - margin or
                pt[1] < y1 + margin or pt[1] > y2 - margin):
                in_box_score -= 0.1
        in_box_score = max(0, in_box_score)
        if in_box_score < 0.8:
            issues.append("landmarks_out_of_bbox")

        # 2. 计算眼距
        eye_distance = np.linalg.norm(left_eye - right_eye)
        if eye_distance < thresh["min_eye_distance"]:
            issues.append("eyes_too_close")
            eye_score = 0.3
        else:
            # 眼距占人脸宽度的比例（通常约0.3-0.5）
            eye_ratio = eye_distance / width
            if 0.25 <= eye_ratio <= 0.55:
                eye_score = 1.0
            else:
                eye_score = 0.5

        # 3. 计算不对称性
        # 左眼到鼻子 vs 右眼到鼻子
        left_dist = np.linalg.norm(left_eye - nose)
        right_dist = np.linalg.norm(right_eye - nose)
        if max(left_dist, right_dist) > 0:
            asymmetry_ratio = abs(left_dist - right_dist) / max(left_dist, right_dist)
        else:
            asymmetry_ratio = 0

        if asymmetry_ratio > thresh["max_asymmetry_ratio"]:
            issues.append("asymmetric_face")
            asym_score = 0.5
        else:
            asym_score = 1.0 - asymmetry_ratio / thresh["max_asymmetry_ratio"] * 0.5

        # 4. 综合关键点得分
        score = (in_box_score * 0.3 + eye_score * 0.4 + asym_score * 0.3)

        return score, eye_distance, asymmetry_ratio, issues

    def _determine_level(
        self,
        confidence: float,
        min_side: int,
        area: int,
        aspect_ratio: float,
        overall_score: float,
        issues: list
    ) -> QualityLevel:
        """确定最终质量等级"""
        thresh = self.thresholds

        # 硬性拒绝条件
        if confidence < thresh["confidence"]["reject"]:
            return QualityLevel.REJECT
        if min_side < thresh["bbox_size"]["min_side"]:
            return QualityLevel.REJECT
        ar = thresh["aspect_ratio"]
        if aspect_ratio < ar["min"] or aspect_ratio > ar["max"]:
            return QualityLevel.REJECT

        # 高质量条件
        if (confidence >= thresh["confidence"]["high"] and
            min_side >= thresh["bbox_size"]["high_side"] and
            overall_score >= 0.8 and
            len([i for i in issues if i not in ["small_face"]]) == 0):
            return QualityLevel.HIGH

        # 中等质量条件
        if (confidence >= thresh["confidence"]["medium"] and
            min_side >= thresh["bbox_size"]["low_side"] and
            overall_score >= 0.5):
            return QualityLevel.MEDIUM

        # 低质量
        return QualityLevel.LOW

    def assess_track(
        self,
        detection_qualities: List[DetectionQuality]
    ) -> TrackQuality:
        """评估 Track 级别的质量

        Args:
            detection_qualities: 该 track 所有检测的质量评估结果

        Returns:
            TrackQuality 对象
        """
        if not detection_qualities:
            return TrackQuality(
                track_id=-1,
                total_detections=0,
                valid_detections=0,
                high_quality_count=0,
                medium_quality_count=0,
                low_quality_count=0,
                reject_count=0,
                valid_ratio=0,
                high_quality_ratio=0,
                low_quality_ratio=0,
                mean_confidence=0,
                mean_size_score=0,
                mean_overall_score=0,
                level=QualityLevel.REJECT,
                usable_for_training=False
            )

        total = len(detection_qualities)

        # 统计各等级数量
        high_count = sum(1 for q in detection_qualities if q.level == QualityLevel.HIGH)
        medium_count = sum(1 for q in detection_qualities if q.level == QualityLevel.MEDIUM)
        low_count = sum(1 for q in detection_qualities if q.level == QualityLevel.LOW)
        reject_count = sum(1 for q in detection_qualities if q.level == QualityLevel.REJECT)

        valid_count = high_count + medium_count + low_count

        # 计算比例
        valid_ratio = valid_count / total if total > 0 else 0
        high_ratio = high_count / total if total > 0 else 0
        low_ratio = low_count / total if total > 0 else 0

        # 计算平均分
        valid_qualities = [q for q in detection_qualities if q.level != QualityLevel.REJECT]
        if valid_qualities:
            mean_conf = np.mean([q.confidence for q in valid_qualities])
            mean_size = np.mean([q.size_score for q in valid_qualities])
            mean_overall = np.mean([q.overall_score for q in valid_qualities])
        else:
            mean_conf = mean_size = mean_overall = 0

        # 确定 track 级别质量
        thresh = self.thresholds["track"]
        if (valid_ratio >= thresh["min_valid_ratio"] and
            high_ratio >= thresh["min_high_quality_ratio"]):
            level = QualityLevel.HIGH
            usable = True
        elif (valid_ratio >= thresh["min_valid_ratio"] and
              low_ratio <= thresh["max_low_quality_ratio"]):
            level = QualityLevel.MEDIUM
            usable = True
        elif valid_ratio >= 0.5:
            level = QualityLevel.LOW
            usable = False  # 不建议用于训练
        else:
            level = QualityLevel.REJECT
            usable = False

        return TrackQuality(
            track_id=-1,  # 需要外部设置
            total_detections=total,
            valid_detections=valid_count,
            high_quality_count=high_count,
            medium_quality_count=medium_count,
            low_quality_count=low_count,
            reject_count=reject_count,
            valid_ratio=valid_ratio,
            high_quality_ratio=high_ratio,
            low_quality_ratio=low_ratio,
            mean_confidence=mean_conf,
            mean_size_score=mean_size,
            mean_overall_score=mean_overall,
            level=level,
            usable_for_training=usable
        )


def filter_detections(
    detections: list,
    assessor: FaceQualityAssessor = None,
    min_level: QualityLevel = QualityLevel.LOW
) -> tuple:
    """过滤检测结果，返回合格的检测及其质量评估

    Args:
        detections: FaceDetection 对象列表
        assessor: 质量评估器，默认使用默认配置
        min_level: 最低接受的质量等级

    Returns:
        (filtered_detections, qualities): 过滤后的检测和质量评估列表
    """
    if assessor is None:
        assessor = FaceQualityAssessor()

    level_order = {
        QualityLevel.HIGH: 3,
        QualityLevel.MEDIUM: 2,
        QualityLevel.LOW: 1,
        QualityLevel.REJECT: 0
    }
    min_order = level_order[min_level]

    filtered = []
    qualities = []

    for det in detections:
        quality = assessor.assess_detection(
            det.bbox, det.confidence, det.landmarks
        )
        if level_order[quality.level] >= min_order:
            filtered.append(det)
            qualities.append(quality)

    return filtered, qualities
