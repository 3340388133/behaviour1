"""
特征窗口与语义对齐模块

核心问题：特征窗口跨越行为边界造成标签污染

场景示例：
- 时间轴：0-3秒正常，3-5秒张望，5-8秒正常
- 固定滑窗 [2-5秒]：包含"正常"和"张望"两种行为
- 标签：应该是什么？→ 标签污染

解决方案：
1. 先标注，后算特征（流程反转）
2. 以标注时间段为中心生成特征
3. 窗口长度与行为持续时间匹配
4. 边界窗口特殊处理

新的数据生成顺序：
┌─────────────────────────────────────────────────────────────┐
│              新流程（标注驱动）                               │
├─────────────────────────────────────────────────────────────┤
│  1. 视频 → 抽帧 → 检测 → 跟踪 → 姿态估计                      │
│                      ↓                                      │
│  2. 生成候选行为片段（规则预筛选）                            │
│                      ↓                                      │
│  3. 人工标注行为时间段 [t_start, t_end, label]               │
│                      ↓                                      │
│  4. 根据标注时间段提取特征（窗口对齐）                        │
│                      ↓                                      │
│  5. 生成训练样本 (feature, label)                           │
└─────────────────────────────────────────────────────────────┘

旧流程问题：
1. 固定滑窗 → 2. 算特征 → 3. 标注 → 窗口可能跨越行为边界

新流程优势：
1. 标注时间段 → 2. 对齐窗口 → 3. 算特征 → 特征与标签完全对齐
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Optional
from enum import Enum


# ============================================================================
# 窗口对齐策略
# ============================================================================
class WindowAlignmentStrategy(Enum):
    """窗口对齐策略"""
    EXACT = "exact"              # 精确匹配标注时间段
    CENTER_EXPAND = "center"     # 以标注中心扩展到标准窗口
    START_ALIGN = "start"        # 窗口起点对齐标注起点
    END_ALIGN = "end"            # 窗口终点对齐标注终点
    MULTI_WINDOW = "multi"       # 长标注段拆分为多个窗口


@dataclass
class WindowAlignmentConfig:
    """窗口对齐配置"""
    strategy: WindowAlignmentStrategy = WindowAlignmentStrategy.CENTER_EXPAND

    # 标准窗口大小（秒）
    standard_window_size: float = 3.0

    # 最小/最大窗口大小
    min_window_size: float = 1.5    # 短于此的标注不生成特征
    max_window_size: float = 10.0   # 长于此的标注拆分

    # 多窗口策略参数
    multi_window_step: float = 1.5  # 多窗口步长

    # 边界处理
    boundary_margin: float = 0.3    # 边界裁剪余量（秒）

    # 标签污染检测
    max_boundary_overlap: float = 0.2  # 最大允许的边界重叠比例


# ============================================================================
# 标注时间段
# ============================================================================
@dataclass
class AnnotationSegment:
    """标注时间段"""
    segment_id: str
    track_id: int
    person_id: int

    # 时间范围
    start_time: float          # 秒
    end_time: float
    start_frame: int
    end_frame: int

    # 标签
    label: int                 # 行为类别 ID
    label_name: str            # 行为名称
    confidence: str = "high"   # high / medium / low

    # 质量标记
    occlusion_level: int = 0   # 0-3
    quality: str = "good"      # good / blur / small

    # 边界标记
    is_boundary: bool = False  # 是否靠近行为边界
    notes: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "track_id": self.track_id,
            "person_id": self.person_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration": round(self.duration, 3),
            "label": self.label,
            "label_name": self.label_name,
            "confidence": self.confidence,
            "occlusion_level": self.occlusion_level,
            "quality": self.quality,
            "is_boundary": self.is_boundary,
            "notes": self.notes
        }


# ============================================================================
# 对齐后的特征窗口
# ============================================================================
@dataclass
class AlignedWindow:
    """对齐后的特征窗口"""
    window_id: str
    segment_id: str            # 对应的标注段

    # 窗口时间范围
    window_start: float
    window_end: float
    window_duration: float

    # 标注时间范围（用于验证对齐）
    annotation_start: float
    annotation_end: float

    # 对齐信息
    alignment_strategy: str
    overlap_ratio: float       # 窗口与标注的重叠比例
    is_valid: bool             # 是否有效（无标签污染）

    # 标签
    label: int
    label_name: str

    # 边界信息
    left_margin: float         # 左边界余量
    right_margin: float        # 右边界余量
    is_boundary_window: bool   # 是否是边界窗口

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "segment_id": self.segment_id,
            "window_start": round(self.window_start, 3),
            "window_end": round(self.window_end, 3),
            "window_duration": round(self.window_duration, 3),
            "annotation_start": round(self.annotation_start, 3),
            "annotation_end": round(self.annotation_end, 3),
            "alignment_strategy": self.alignment_strategy,
            "overlap_ratio": round(self.overlap_ratio, 3),
            "is_valid": self.is_valid,
            "label": self.label,
            "label_name": self.label_name,
            "left_margin": round(self.left_margin, 3),
            "right_margin": round(self.right_margin, 3),
            "is_boundary_window": self.is_boundary_window
        }


# ============================================================================
# 窗口对齐器
# ============================================================================
class WindowAligner:
    """特征窗口对齐器"""

    def __init__(self, config: WindowAlignmentConfig = None, fps: float = 10.0):
        self.config = config or WindowAlignmentConfig()
        self.fps = fps

    def align_windows(
        self,
        segment: AnnotationSegment,
        track_duration: float,           # track 总时长
        adjacent_segments: list = None   # 相邻的标注段（用于边界检测）
    ) -> List[AlignedWindow]:
        """为标注段生成对齐的特征窗口

        Args:
            segment: 标注时间段
            track_duration: track 总时长
            adjacent_segments: 相邻标注段

        Returns:
            对齐的窗口列表
        """
        duration = segment.duration

        # 太短的标注不处理
        if duration < self.config.min_window_size:
            return []

        strategy = self.config.strategy

        if strategy == WindowAlignmentStrategy.EXACT:
            windows = self._align_exact(segment, track_duration)
        elif strategy == WindowAlignmentStrategy.CENTER_EXPAND:
            windows = self._align_center_expand(segment, track_duration)
        elif strategy == WindowAlignmentStrategy.START_ALIGN:
            windows = self._align_start(segment, track_duration)
        elif strategy == WindowAlignmentStrategy.END_ALIGN:
            windows = self._align_end(segment, track_duration)
        elif strategy == WindowAlignmentStrategy.MULTI_WINDOW:
            windows = self._align_multi_window(segment, track_duration)
        else:
            windows = self._align_center_expand(segment, track_duration)

        # 检测边界污染
        if adjacent_segments:
            windows = self._check_boundary_pollution(windows, adjacent_segments)

        return windows

    def _align_exact(
        self,
        segment: AnnotationSegment,
        track_duration: float
    ) -> List[AlignedWindow]:
        """精确匹配标注时间段"""
        window = AlignedWindow(
            window_id=f"{segment.segment_id}_w0",
            segment_id=segment.segment_id,
            window_start=segment.start_time,
            window_end=segment.end_time,
            window_duration=segment.duration,
            annotation_start=segment.start_time,
            annotation_end=segment.end_time,
            alignment_strategy="exact",
            overlap_ratio=1.0,
            is_valid=True,
            label=segment.label,
            label_name=segment.label_name,
            left_margin=0,
            right_margin=0,
            is_boundary_window=segment.is_boundary
        )
        return [window]

    def _align_center_expand(
        self,
        segment: AnnotationSegment,
        track_duration: float
    ) -> List[AlignedWindow]:
        """以标注中心扩展到标准窗口大小"""
        center = (segment.start_time + segment.end_time) / 2
        half_window = self.config.standard_window_size / 2

        window_start = max(0, center - half_window)
        window_end = min(track_duration, center + half_window)

        # 如果标注本身就够长，使用标注时间
        if segment.duration >= self.config.standard_window_size:
            window_start = segment.start_time
            window_end = segment.start_time + self.config.standard_window_size

        # 计算重叠
        overlap_start = max(window_start, segment.start_time)
        overlap_end = min(window_end, segment.end_time)
        overlap = max(0, overlap_end - overlap_start)
        overlap_ratio = overlap / (window_end - window_start) if window_end > window_start else 0

        window = AlignedWindow(
            window_id=f"{segment.segment_id}_w0",
            segment_id=segment.segment_id,
            window_start=window_start,
            window_end=window_end,
            window_duration=window_end - window_start,
            annotation_start=segment.start_time,
            annotation_end=segment.end_time,
            alignment_strategy="center_expand",
            overlap_ratio=overlap_ratio,
            is_valid=overlap_ratio >= (1 - self.config.max_boundary_overlap),
            label=segment.label,
            label_name=segment.label_name,
            left_margin=segment.start_time - window_start,
            right_margin=window_end - segment.end_time,
            is_boundary_window=segment.is_boundary
        )
        return [window]

    def _align_start(
        self,
        segment: AnnotationSegment,
        track_duration: float
    ) -> List[AlignedWindow]:
        """窗口起点对齐标注起点"""
        window_start = segment.start_time
        window_end = min(
            segment.start_time + self.config.standard_window_size,
            track_duration
        )

        overlap_end = min(window_end, segment.end_time)
        overlap = overlap_end - window_start
        overlap_ratio = overlap / (window_end - window_start) if window_end > window_start else 0

        window = AlignedWindow(
            window_id=f"{segment.segment_id}_w0",
            segment_id=segment.segment_id,
            window_start=window_start,
            window_end=window_end,
            window_duration=window_end - window_start,
            annotation_start=segment.start_time,
            annotation_end=segment.end_time,
            alignment_strategy="start_align",
            overlap_ratio=overlap_ratio,
            is_valid=overlap_ratio >= (1 - self.config.max_boundary_overlap),
            label=segment.label,
            label_name=segment.label_name,
            left_margin=0,
            right_margin=window_end - segment.end_time if window_end > segment.end_time else 0,
            is_boundary_window=segment.is_boundary
        )
        return [window]

    def _align_end(
        self,
        segment: AnnotationSegment,
        track_duration: float
    ) -> List[AlignedWindow]:
        """窗口终点对齐标注终点"""
        window_end = segment.end_time
        window_start = max(0, segment.end_time - self.config.standard_window_size)

        overlap_start = max(window_start, segment.start_time)
        overlap = window_end - overlap_start
        overlap_ratio = overlap / (window_end - window_start) if window_end > window_start else 0

        window = AlignedWindow(
            window_id=f"{segment.segment_id}_w0",
            segment_id=segment.segment_id,
            window_start=window_start,
            window_end=window_end,
            window_duration=window_end - window_start,
            annotation_start=segment.start_time,
            annotation_end=segment.end_time,
            alignment_strategy="end_align",
            overlap_ratio=overlap_ratio,
            is_valid=overlap_ratio >= (1 - self.config.max_boundary_overlap),
            label=segment.label,
            label_name=segment.label_name,
            left_margin=segment.start_time - window_start if window_start < segment.start_time else 0,
            right_margin=0,
            is_boundary_window=segment.is_boundary
        )
        return [window]

    def _align_multi_window(
        self,
        segment: AnnotationSegment,
        track_duration: float
    ) -> List[AlignedWindow]:
        """长标注段拆分为多个窗口"""
        windows = []
        window_size = self.config.standard_window_size
        step = self.config.multi_window_step

        # 如果标注段短于标准窗口，使用 center_expand
        if segment.duration < window_size:
            return self._align_center_expand(segment, track_duration)

        # 滑动窗口
        t = segment.start_time
        window_idx = 0

        while t + window_size <= segment.end_time + step * 0.5:
            window_start = t
            window_end = min(t + window_size, segment.end_time)

            # 确保窗口完全在标注范围内
            if window_end > segment.end_time:
                window_end = segment.end_time
                window_start = max(segment.start_time, window_end - window_size)

            window = AlignedWindow(
                window_id=f"{segment.segment_id}_w{window_idx}",
                segment_id=segment.segment_id,
                window_start=window_start,
                window_end=window_end,
                window_duration=window_end - window_start,
                annotation_start=segment.start_time,
                annotation_end=segment.end_time,
                alignment_strategy="multi_window",
                overlap_ratio=1.0,  # 完全在标注范围内
                is_valid=True,
                label=segment.label,
                label_name=segment.label_name,
                left_margin=0,
                right_margin=0,
                is_boundary_window=(window_idx == 0 or window_end >= segment.end_time - step * 0.5)
            )
            windows.append(window)

            t += step
            window_idx += 1

        return windows

    def _check_boundary_pollution(
        self,
        windows: List[AlignedWindow],
        adjacent_segments: List[AnnotationSegment]
    ) -> List[AlignedWindow]:
        """检测边界污染"""
        for window in windows:
            for adj in adjacent_segments:
                # 检查窗口是否与相邻标注重叠
                overlap_start = max(window.window_start, adj.start_time)
                overlap_end = min(window.window_end, adj.end_time)

                if overlap_end > overlap_start:
                    overlap = overlap_end - overlap_start
                    overlap_ratio = overlap / window.window_duration

                    if overlap_ratio > self.config.max_boundary_overlap:
                        window.is_valid = False
                        window.is_boundary_window = True

        return windows


# ============================================================================
# 窗口与标签匹配原则
# ============================================================================
"""
窗口长度与行为持续时间的匹配原则：

1. 最小窗口原则
   - 窗口应至少包含一个完整的行为周期
   - 频繁张望：3秒内转头≥3次 → 窗口≥3秒
   - 快速回头：0.5秒完成 → 窗口≥1.5秒（包含前后上下文）

2. 行为-窗口匹配表
   | 行为类型        | 最小持续时间 | 推荐窗口大小 | 说明                |
   |----------------|-------------|-------------|---------------------|
   | quick_turn     | 0.5秒       | 1.5-2.0秒   | 需要前后上下文       |
   | glancing       | 3秒         | 3.0秒       | 正好覆盖定义         |
   | prolonged_watch| 3秒         | 3.0-5.0秒   | 可能更长            |
   | looking_down   | 5秒         | 5.0秒       | 匹配定义            |
   | normal         | 任意        | 3.0秒       | 标准窗口            |

3. 边界处理原则
   - 边界窗口标记 is_boundary_window = True
   - 边界窗口可选参与训练（提供更多上下文）
   - 严格训练时排除 is_valid = False 的窗口

4. 多窗口策略
   - 长于标准窗口的标注，拆分为多个重叠窗口
   - 步长 = 窗口大小 / 2（50%重叠）
   - 每个窗口独立标签相同

5. 标签污染检测
   - 窗口与其他标注的重叠 > 20% → 标记无效
   - 无效窗口不参与训练
"""

BEHAVIOR_WINDOW_CONFIG = {
    # 行为类型 -> (最小持续时间, 推荐窗口大小, 步长)
    0: ("normal", 1.0, 3.0, 1.5),
    1: ("glancing", 3.0, 3.0, 1.5),
    2: ("quick_turn", 0.5, 2.0, 1.0),
    3: ("prolonged_watch", 3.0, 3.0, 1.5),
    4: ("looking_down", 5.0, 5.0, 2.5),
    5: ("looking_up", 3.0, 3.0, 1.5),
}


def get_window_config_for_behavior(label: int) -> tuple:
    """获取行为对应的窗口配置"""
    if label in BEHAVIOR_WINDOW_CONFIG:
        return BEHAVIOR_WINDOW_CONFIG[label]
    return ("unknown", 1.0, 3.0, 1.5)


# ============================================================================
# 训练样本生成器
# ============================================================================
@dataclass
class TrainingSample:
    """训练样本"""
    sample_id: str
    window: AlignedWindow

    # 特征（由 BehaviorFeatureExtractor 填充）
    features: Optional[np.ndarray] = None
    feature_dict: Optional[dict] = None

    # 标签
    label: int = -1
    label_name: str = ""

    # 质量信息
    is_valid: bool = True
    quality_score: float = 1.0

    def to_dict(self) -> dict:
        result = {
            "sample_id": self.sample_id,
            "window": self.window.to_dict(),
            "label": self.label,
            "label_name": self.label_name,
            "is_valid": self.is_valid,
            "quality_score": round(self.quality_score, 3)
        }
        if self.feature_dict:
            result["features"] = self.feature_dict
        return result


class TrainingSampleGenerator:
    """训练样本生成器"""

    def __init__(self, fps: float = 10.0):
        self.fps = fps
        self.aligner = WindowAligner(fps=fps)

    def generate_samples(
        self,
        annotations: List[AnnotationSegment],
        track_duration: float,
        behavior_specific: bool = True
    ) -> List[TrainingSample]:
        """从标注生成训练样本

        Args:
            annotations: 标注列表
            track_duration: track 总时长
            behavior_specific: 是否使用行为特定的窗口配置

        Returns:
            训练样本列表
        """
        samples = []

        # 按时间排序
        sorted_annotations = sorted(annotations, key=lambda a: a.start_time)

        for i, segment in enumerate(sorted_annotations):
            # 获取相邻标注（用于边界检测）
            adjacent = []
            if i > 0:
                adjacent.append(sorted_annotations[i-1])
            if i < len(sorted_annotations) - 1:
                adjacent.append(sorted_annotations[i+1])

            # 根据行为类型调整窗口配置
            if behavior_specific:
                _, min_dur, window_size, step = get_window_config_for_behavior(segment.label)
                config = WindowAlignmentConfig(
                    strategy=WindowAlignmentStrategy.MULTI_WINDOW,
                    standard_window_size=window_size,
                    min_window_size=min_dur,
                    multi_window_step=step
                )
                aligner = WindowAligner(config, self.fps)
            else:
                aligner = self.aligner

            # 生成对齐窗口
            windows = aligner.align_windows(segment, track_duration, adjacent)

            # 生成样本
            for window in windows:
                quality_score = 1.0
                if window.is_boundary_window:
                    quality_score *= 0.8
                if segment.occlusion_level > 0:
                    quality_score *= (1 - segment.occlusion_level * 0.2)
                if segment.confidence != "high":
                    quality_score *= 0.9

                sample = TrainingSample(
                    sample_id=window.window_id,
                    window=window,
                    label=segment.label,
                    label_name=segment.label_name,
                    is_valid=window.is_valid,
                    quality_score=quality_score
                )
                samples.append(sample)

        return samples

    def filter_valid_samples(
        self,
        samples: List[TrainingSample],
        min_quality: float = 0.5
    ) -> List[TrainingSample]:
        """过滤有效样本"""
        return [
            s for s in samples
            if s.is_valid and s.quality_score >= min_quality
        ]
