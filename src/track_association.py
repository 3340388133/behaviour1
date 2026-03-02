"""
Track 关联与合并模块

解决的核心问题：
同一人的完整行为被 ByteTrack 拆成多个 track，导致：
- 标注员看不到这是同一个人
- 无法判断是否属于同一个行为实例
- 标签不一致、行为被割裂

核心策略：
1. 单视频顺序处理（所有时间索引基于统一时间轴）
2. 自动检测 Track 断裂模式
3. 规则化的 Track 关联与合并
4. 轻量级身份连续性特征（无 ReID）
5. 输出"人级时间线"供标注使用
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Optional, Tuple
from enum import Enum
import json


# ============================================================================
# Step 1: Track 断裂模式定义
# ============================================================================
"""
ByteTrack 中常见的 ID 断裂原因：

1. 转头导致 bbox 形态变化
   - 表现：正面→侧面时，bbox 高宽比突变
   - 信号：aspect_ratio 突变 > 30%

2. 短时遮挡
   - 表现：人脸被遮挡数帧后重新出现
   - 信号：时间 gap 在 [1, max_age] 范围内

3. 检测置信度下降
   - 表现：转头/遮挡导致置信度低于阈值
   - 信号：track 结束前 confidence 持续下降

4. 检测丢失（漏检）
   - 表现：检测器偶尔漏检
   - 信号：单帧或少帧缺失

5. 快速运动
   - 表现：bbox 位置跳变超出 IoU 匹配阈值
   - 信号：中心点位移 > bbox 宽度
"""

class BreakPattern(Enum):
    """断裂模式"""
    HEAD_ROTATION = "head_rotation"      # 转头导致形态变化
    SHORT_OCCLUSION = "short_occlusion"  # 短时遮挡
    CONFIDENCE_DROP = "confidence_drop"  # 置信度下降
    DETECTION_MISS = "detection_miss"    # 检测丢失
    FAST_MOTION = "fast_motion"          # 快速运动
    UNKNOWN = "unknown"                  # 未知原因


@dataclass
class TrackBreakpoint:
    """Track 断裂点信息"""
    track1_id: int               # 前一个 track
    track2_id: int               # 后一个 track
    track1_end_frame: int        # track1 结束帧
    track2_start_frame: int      # track2 开始帧
    time_gap_frames: int         # 时间间隔（帧数）
    time_gap_sec: float          # 时间间隔（秒）
    pattern: BreakPattern        # 断裂模式

    # 空间信息
    distance: float              # bbox 中心距离
    iou: float                   # bbox IoU

    # 关联置信度
    association_score: float     # 关联得分 (0-1)
    auto_merge: bool             # 是否可自动合并
    needs_review: bool           # 是否需要人工确认

    def to_dict(self) -> dict:
        return {
            "track1_id": self.track1_id,
            "track2_id": self.track2_id,
            "track1_end_frame": self.track1_end_frame,
            "track2_start_frame": self.track2_start_frame,
            "time_gap_frames": self.time_gap_frames,
            "time_gap_sec": round(self.time_gap_sec, 3),
            "pattern": self.pattern.value,
            "distance": round(self.distance, 1),
            "iou": round(self.iou, 3),
            "association_score": round(self.association_score, 3),
            "auto_merge": self.auto_merge,
            "needs_review": self.needs_review
        }


# ============================================================================
# Step 2: Track 关联规则配置
# ============================================================================
ASSOCIATION_THRESHOLDS = {
    # 时间间隔阈值（帧数，基于 10fps）
    "time_gap": {
        "auto_merge": 10,       # <= 10帧（1秒）：可自动合并
        "manual_review": 50,    # <= 50帧（5秒）：需人工确认
        "reject": 100           # > 100帧（10秒）：拒绝关联
    },

    # 空间距离阈值（相对于 bbox 尺寸）
    "distance": {
        "auto_merge": 0.5,      # 中心距离 < 0.5 * bbox_size
        "manual_review": 1.5,   # 中心距离 < 1.5 * bbox_size
        "reject": 3.0           # 中心距离 > 3.0 * bbox_size
    },

    # IoU 阈值
    "iou": {
        "auto_merge": 0.3,      # IoU > 0.3：可自动合并
        "confident": 0.5        # IoU > 0.5：高置信度
    },

    # bbox 尺寸变化阈值
    "size_change": {
        "max_ratio": 1.5        # 最大尺寸变化倍数
    },

    # 综合关联得分阈值
    "association_score": {
        "auto_merge": 0.7,      # >= 0.7：自动合并
        "manual_review": 0.4,   # >= 0.4：需人工确认
        "reject": 0.4           # < 0.4：拒绝
    }
}


@dataclass
class TrackInfo:
    """Track 基础信息（用于关联分析）"""
    track_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_sec: float

    # 起始/结束位置
    start_bbox: np.ndarray
    end_bbox: np.ndarray
    start_center: np.ndarray
    end_center: np.ndarray

    # 统计信息
    mean_bbox_size: float
    mean_confidence: float
    detection_count: int

    # 质量信息（来自 face_quality）
    quality_level: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_sec": round(self.duration_sec, 3),
            "start_bbox": self.start_bbox.tolist(),
            "end_bbox": self.end_bbox.tolist(),
            "mean_bbox_size": round(self.mean_bbox_size, 1),
            "mean_confidence": round(self.mean_confidence, 3),
            "detection_count": self.detection_count,
            "quality_level": self.quality_level
        }


# ============================================================================
# Step 3: Track 关联分析器
# ============================================================================
class TrackAssociator:
    """Track 关联分析器"""

    def __init__(self, fps: float = 10.0, thresholds: dict = None):
        self.fps = fps
        self.thresholds = thresholds or ASSOCIATION_THRESHOLDS

    def extract_track_info(self, track_data: dict) -> TrackInfo:
        """从 track 数据提取关联分析所需信息"""
        detections = track_data["detections"]
        if not detections:
            raise ValueError(f"Track {track_data['track_id']} has no detections")

        start_det = detections[0]
        end_det = detections[-1]

        start_bbox = np.array(start_det["bbox"])
        end_bbox = np.array(end_det["bbox"])

        start_center = np.array([
            (start_bbox[0] + start_bbox[2]) / 2,
            (start_bbox[1] + start_bbox[3]) / 2
        ])
        end_center = np.array([
            (end_bbox[0] + end_bbox[2]) / 2,
            (end_bbox[1] + end_bbox[3]) / 2
        ])

        # 计算平均 bbox 尺寸
        sizes = []
        confidences = []
        for det in detections:
            bbox = det["bbox"]
            sizes.append(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            confidences.append(det.get("confidence", 0.5))

        start_frame = track_data["start_frame"]
        end_frame = track_data["end_frame"]
        duration_frames = end_frame - start_frame + 1

        return TrackInfo(
            track_id=track_data["track_id"],
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=duration_frames,
            duration_sec=duration_frames / self.fps,
            start_bbox=start_bbox,
            end_bbox=end_bbox,
            start_center=start_center,
            end_center=end_center,
            mean_bbox_size=np.mean(sizes),
            mean_confidence=np.mean(confidences),
            detection_count=len(detections)
        )

    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个 bbox 的 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def analyze_breakpoint(
        self,
        track1: TrackInfo,
        track2: TrackInfo
    ) -> Optional[TrackBreakpoint]:
        """分析两个 track 之间的断裂点

        Args:
            track1: 前一个 track（时间上更早）
            track2: 后一个 track（时间上更晚）

        Returns:
            TrackBreakpoint 对象，如果不可能关联则返回 None
        """
        # 确保时间顺序正确
        if track1.end_frame >= track2.start_frame:
            return None  # 时间重叠，不是断裂关系

        time_gap = track2.start_frame - track1.end_frame
        time_gap_sec = time_gap / self.fps

        # 时间间隔过大，直接拒绝
        if time_gap > self.thresholds["time_gap"]["reject"]:
            return None

        # 计算空间距离
        distance = np.linalg.norm(track2.start_center - track1.end_center)
        avg_size = (track1.mean_bbox_size + track2.mean_bbox_size) / 2
        relative_distance = distance / avg_size if avg_size > 0 else float('inf')

        # 距离过大，直接拒绝
        if relative_distance > self.thresholds["distance"]["reject"]:
            return None

        # 计算 IoU
        iou = self.compute_iou(track1.end_bbox, track2.start_bbox)

        # 检测断裂模式
        pattern = self._detect_break_pattern(
            track1, track2, time_gap, relative_distance, iou
        )

        # 计算关联得分
        association_score = self._compute_association_score(
            time_gap, relative_distance, iou, track1, track2
        )

        # 判断是否可自动合并
        thresh = self.thresholds
        auto_merge = (
            time_gap <= thresh["time_gap"]["auto_merge"] and
            relative_distance <= thresh["distance"]["auto_merge"] and
            association_score >= thresh["association_score"]["auto_merge"]
        )

        needs_review = (
            not auto_merge and
            association_score >= thresh["association_score"]["manual_review"]
        )

        return TrackBreakpoint(
            track1_id=track1.track_id,
            track2_id=track2.track_id,
            track1_end_frame=track1.end_frame,
            track2_start_frame=track2.start_frame,
            time_gap_frames=time_gap,
            time_gap_sec=time_gap_sec,
            pattern=pattern,
            distance=distance,
            iou=iou,
            association_score=association_score,
            auto_merge=auto_merge,
            needs_review=needs_review
        )

    def _detect_break_pattern(
        self,
        track1: TrackInfo,
        track2: TrackInfo,
        time_gap: int,
        relative_distance: float,
        iou: float
    ) -> BreakPattern:
        """检测断裂模式"""
        # 短时间 gap + 位置接近 → 可能是遮挡或漏检
        if time_gap <= 5 and relative_distance < 0.3:
            return BreakPattern.DETECTION_MISS

        if time_gap <= 15 and relative_distance < 0.5:
            return BreakPattern.SHORT_OCCLUSION

        # bbox 尺寸变化大 → 可能是转头
        size_ratio = track2.mean_bbox_size / track1.mean_bbox_size
        if 0.5 < size_ratio < 2.0 and relative_distance < 1.0:
            if time_gap <= 10:
                return BreakPattern.HEAD_ROTATION

        # 位置变化大 → 可能是快速运动
        if relative_distance > 1.0 and time_gap <= 5:
            return BreakPattern.FAST_MOTION

        # 置信度信息（如果有）
        if track1.mean_confidence < 0.6 or track2.mean_confidence < 0.6:
            return BreakPattern.CONFIDENCE_DROP

        return BreakPattern.UNKNOWN

    def _compute_association_score(
        self,
        time_gap: int,
        relative_distance: float,
        iou: float,
        track1: TrackInfo,
        track2: TrackInfo
    ) -> float:
        """计算关联得分 (0-1)"""
        thresh = self.thresholds

        # 时间得分：gap 越小越好
        max_gap = thresh["time_gap"]["manual_review"]
        time_score = max(0, 1 - time_gap / max_gap)

        # 距离得分：距离越小越好
        max_dist = thresh["distance"]["manual_review"]
        dist_score = max(0, 1 - relative_distance / max_dist)

        # IoU 得分：IoU 越大越好
        iou_score = min(1, iou / thresh["iou"]["confident"])

        # 尺寸一致性得分
        size_ratio = track2.mean_bbox_size / track1.mean_bbox_size
        size_score = 1 - abs(size_ratio - 1) / 0.5
        size_score = max(0, min(1, size_score))

        # 加权综合
        weights = {
            "time": 0.30,
            "distance": 0.35,
            "iou": 0.20,
            "size": 0.15
        }

        score = (
            time_score * weights["time"] +
            dist_score * weights["distance"] +
            iou_score * weights["iou"] +
            size_score * weights["size"]
        )

        return score

    def find_all_breakpoints(
        self,
        tracks: List[dict]
    ) -> List[TrackBreakpoint]:
        """找出所有可能的 track 断裂点

        Args:
            tracks: track 数据列表

        Returns:
            所有断裂点列表
        """
        # 按开始时间排序
        sorted_tracks = sorted(tracks, key=lambda t: t["start_frame"])

        # 提取 track 信息
        track_infos = []
        for t in sorted_tracks:
            try:
                info = self.extract_track_info(t)
                track_infos.append(info)
            except ValueError:
                continue

        breakpoints = []

        # 对每对时间相邻的 track 分析
        for i, track1 in enumerate(track_infos):
            for track2 in track_infos[i+1:]:
                # 只分析时间上可能连续的 track
                if track2.start_frame - track1.end_frame > self.thresholds["time_gap"]["reject"]:
                    break  # 后续 track 时间差更大，不需要继续

                bp = self.analyze_breakpoint(track1, track2)
                if bp is not None:
                    breakpoints.append(bp)

        return breakpoints


# ============================================================================
# Step 4: 轻量级身份特征（无 ReID）
# ============================================================================
@dataclass
class IdentityFeature:
    """轻量级身份特征"""
    track_id: int

    # bbox 统计特征
    mean_width: float
    mean_height: float
    mean_aspect_ratio: float
    std_aspect_ratio: float

    # 位置特征
    mean_x: float
    mean_y: float
    position_variance: float

    # 置信度特征
    mean_confidence: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "mean_width": round(self.mean_width, 1),
            "mean_height": round(self.mean_height, 1),
            "mean_aspect_ratio": round(self.mean_aspect_ratio, 3),
            "std_aspect_ratio": round(self.std_aspect_ratio, 3),
            "mean_x": round(self.mean_x, 1),
            "mean_y": round(self.mean_y, 1),
            "position_variance": round(self.position_variance, 1),
            "mean_confidence": round(self.mean_confidence, 3)
        }


def extract_identity_features(track_data: dict) -> IdentityFeature:
    """从 track 数据提取身份特征"""
    detections = track_data["detections"]

    widths = []
    heights = []
    centers_x = []
    centers_y = []
    confidences = []

    for det in detections:
        bbox = det["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        widths.append(w)
        heights.append(h)
        centers_x.append(cx)
        centers_y.append(cy)
        confidences.append(det.get("confidence", 0.5))

    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = widths / (heights + 1e-6)

    centers = np.column_stack([centers_x, centers_y])
    position_variance = np.mean(np.var(centers, axis=0))

    return IdentityFeature(
        track_id=track_data["track_id"],
        mean_width=np.mean(widths),
        mean_height=np.mean(heights),
        mean_aspect_ratio=np.mean(aspect_ratios),
        std_aspect_ratio=np.std(aspect_ratios),
        mean_x=np.mean(centers_x),
        mean_y=np.mean(centers_y),
        position_variance=position_variance,
        mean_confidence=np.mean(confidences)
    )


def compute_identity_similarity(feat1: IdentityFeature, feat2: IdentityFeature) -> float:
    """计算两个 track 的身份相似度"""
    # 尺寸相似度
    size_ratio = (feat1.mean_width * feat1.mean_height) / (
        feat2.mean_width * feat2.mean_height + 1e-6
    )
    size_sim = 1 - abs(size_ratio - 1) / 0.5
    size_sim = max(0, min(1, size_sim))

    # 高宽比相似度
    ar_diff = abs(feat1.mean_aspect_ratio - feat2.mean_aspect_ratio)
    ar_sim = 1 - ar_diff / 0.3
    ar_sim = max(0, min(1, ar_sim))

    # 位置相似度（同一人在短时间内位置应该接近）
    pos_diff = np.sqrt(
        (feat1.mean_x - feat2.mean_x) ** 2 +
        (feat1.mean_y - feat2.mean_y) ** 2
    )
    avg_size = (feat1.mean_width + feat2.mean_width) / 2
    pos_sim = 1 - pos_diff / (avg_size * 3)
    pos_sim = max(0, min(1, pos_sim))

    # 加权综合
    similarity = size_sim * 0.3 + ar_sim * 0.3 + pos_sim * 0.4

    return similarity


# ============================================================================
# Step 5: Person-level 时间线（标注用）
# ============================================================================
@dataclass
class PersonTimeline:
    """人级时间线（用于标注）"""
    person_id: int
    merged_track_ids: list          # 合并的 track ID 列表

    # 时间范围
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    duration_sec: float

    # 断裂点信息
    breakpoints: list               # TrackBreakpoint 列表

    # 合并状态
    auto_merged: bool               # 是否全部自动合并
    needs_review: bool              # 是否需要人工确认

    # 人工确认结果
    confirmed: bool = False
    confirmed_by: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "merged_track_ids": self.merged_track_ids,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": round(self.start_time_sec, 3),
            "end_time_sec": round(self.end_time_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "breakpoints": [bp.to_dict() for bp in self.breakpoints],
            "auto_merged": self.auto_merged,
            "needs_review": self.needs_review,
            "confirmed": self.confirmed,
            "confirmed_by": self.confirmed_by
        }


class PersonTimelineBuilder:
    """构建人级时间线"""

    def __init__(self, fps: float = 10.0):
        self.fps = fps
        self.associator = TrackAssociator(fps)

    def build_timelines(
        self,
        tracks: List[dict],
        auto_merge_only: bool = False
    ) -> List[PersonTimeline]:
        """构建人级时间线

        Args:
            tracks: track 数据列表
            auto_merge_only: 是否只进行自动合并

        Returns:
            PersonTimeline 列表
        """
        if not tracks:
            return []

        # 找出所有断裂点
        breakpoints = self.associator.find_all_breakpoints(tracks)

        # 构建 track 图（并查集）
        track_ids = [t["track_id"] for t in tracks]
        parent = {tid: tid for tid in track_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # 根据断裂点合并 track
        merge_breakpoints = {}  # person_id -> breakpoints

        for bp in breakpoints:
            if bp.auto_merge or (not auto_merge_only and bp.needs_review):
                union(bp.track1_id, bp.track2_id)

        # 分组
        groups = {}
        for tid in track_ids:
            root = find(tid)
            if root not in groups:
                groups[root] = []
            groups[root].append(tid)

        # 为每个组找出相关的断裂点
        for bp in breakpoints:
            root = find(bp.track1_id)
            if root not in merge_breakpoints:
                merge_breakpoints[root] = []
            merge_breakpoints[root].append(bp)

        # 构建时间线
        timelines = []
        track_dict = {t["track_id"]: t for t in tracks}

        for person_id, (root, group_tids) in enumerate(groups.items()):
            group_tracks = [track_Dict[tid] for tid in group_tids]
            group_tracks.sort(key=lambda t: t["start_frame"])

            start_frame = min(t["start_frame"] for t in group_tracks)
            end_frame = max(t["end_frame"] for t in group_tracks)

            bps = merge_breakpoints.get(root, [])
            auto_merged = all(bp.auto_merge for bp in bps) if bps else True
            needs_review = any(bp.needs_review for bp in bps)

            timeline = PersonTimeline(
                person_id=person_id,
                merged_track_ids=sorted(group_tids),
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_sec=start_frame / self.fps,
                end_time_sec=end_frame / self.fps,
                duration_sec=(end_frame - start_frame + 1) / self.fps,
                breakpoints=bps,
                auto_merged=auto_merged,
                needs_review=needs_review
            )
            timelines.append(timeline)

        return timelines

    def export_for_annotation(
        self,
        video_id: str,
        tracks: List[dict],
        timelines: List[PersonTimeline]
    ) -> dict:
        """导出标注工具所需的数据结构

        Args:
            video_id: 视频ID
            tracks: 原始 track 数据
            timelines: 人级时间线

        Returns:
            标注工具数据结构
        """
        track_dict = {t["track_id"]: t for t in tracks}

        annotation_data = {
            "video_id": video_id,
            "fps": self.fps,
            "persons": []
        }

        for timeline in timelines:
            person_data = {
                "person_id": timeline.person_id,
                "timeline": timeline.to_dict(),
                "tracks": []
            }

            # 添加每个 track 的详细信息
            for tid in timeline.merged_track_ids:
                track = track_dict.get(tid, {})
                person_data["tracks"].append({
                    "track_id": tid,
                    "start_frame": track.get("start_frame"),
                    "end_frame": track.get("end_frame"),
                    "detection_count": len(track.get("detections", []))
                })

            # 添加断裂点高亮信息
            person_data["breakpoint_frames"] = [
                {
                    "frame": bp.track2_start_frame,
                    "pattern": bp.pattern.value,
                    "auto_merge": bp.auto_merge,
                    "needs_review": bp.needs_review
                }
                for bp in timeline.breakpoints
            ]

            annotation_data["persons"].append(person_data)

        return annotation_data


# ============================================================================
# Step 6: 人工确认数据结构
# ============================================================================
@dataclass
class ManualMergeDecision:
    """人工合并决策"""
    person_id: int
    track_ids: list                 # 确认属于同一人的 track
    decision: str                   # "merge" / "split" / "uncertain"
    annotator: str
    timestamp: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "track_ids": self.track_ids,
            "decision": self.decision,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
            "notes": self.notes
        }


def save_tracking_with_associations(
    output_path: str,
    video_id: str,
    tracks: List[dict],
    timelines: List[PersonTimeline],
    fps: float = 10.0
):
    """保存跟踪结果和关联信息

    Args:
        output_path: 输出文件路径
        video_id: 视频ID
        tracks: track 数据列表
        timelines: 人级时间线列表
        fps: 帧率
    """
    result = {
        "video_id": video_id,
        "fps": fps,
        "association_thresholds": ASSOCIATION_THRESHOLDS,
        "tracks": tracks,
        "person_timelines": [t.to_dict() for t in timelines],
        "summary": {
            "total_tracks": len(tracks),
            "total_persons": len(timelines),
            "auto_merged_persons": sum(1 for t in timelines if t.auto_merged),
            "needs_review_persons": sum(1 for t in timelines if t.needs_review)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
