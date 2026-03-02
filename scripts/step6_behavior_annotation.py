#!/usr/bin/env python3
"""
Step 6: 行为标注 - 自动生成行为标签

基于头部姿态和时序特征，自动为每个跟踪轨迹生成行为标签

行为类别定义:
    - normal: 正常排队 (目视前方，移动平稳)
    - looking_around: 东张西望 (yaw角频繁变化)
    - looking_down: 低头 (pitch角较大)
    - distracted: 分心/不专注 (姿态不稳定)
    - suspicious: 可疑行为 (综合异常)

输入:
    - dataset_root/features/pose/{video_id}/pose_opencv.json
    - dataset_root/features/temporal/{video_id}/temporal_features.json
    - dataset_root/annotations/tracking/{video_id}/

输出:
    - dataset_root/annotations/behavior/{video_id}/behavior.json
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
POSE_DIR = DATASET_ROOT / "features" / "pose"
TEMPORAL_DIR = DATASET_ROOT / "features" / "temporal"
TRACKING_DIR = DATASET_ROOT / "annotations" / "tracking"
BEHAVIOR_DIR = DATASET_ROOT / "annotations" / "behavior"

# 行为分类阈值
THRESHOLDS = {
    'yaw_variance_high': 15.0,       # yaw方差阈值 (东张西望)
    'yaw_variance_low': 5.0,         # yaw方差阈值 (正常)
    'pitch_down_threshold': -15.0,   # pitch阈值 (低头)
    'pitch_up_threshold': 15.0,      # pitch阈值 (抬头)
    'roll_variance_high': 10.0,      # roll方差阈值 (不稳定)
    'min_track_length': 5,           # 最小轨迹长度
    'attention_change_rate': 0.3,    # 注意力变化率阈值
}

# 行为类别
BEHAVIOR_LABELS = {
    'normal': '正常排队',
    'looking_around': '东张西望',
    'looking_down': '低头',
    'looking_up': '抬头观望',
    'distracted': '分心',
    'suspicious': '可疑行为',
    'unknown': '未知'
}


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TrackBehavior:
    """轨迹行为标注"""
    track_id: int
    video_id: str
    start_frame: int
    end_frame: int
    duration_frames: int

    # 行为标签
    primary_label: str
    secondary_labels: List[str]
    confidence: float

    # 统计特征
    yaw_mean: float
    yaw_std: float
    pitch_mean: float
    pitch_std: float
    roll_mean: float
    roll_std: float

    # 行为指标
    attention_score: float        # 注意力得分 (0-1, 1=专注)
    stability_score: float        # 稳定性得分 (0-1, 1=稳定)
    looking_around_ratio: float   # 东张西望时间比例

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# 行为分析器
# ============================================================================

class BehaviorAnalyzer:
    """行为分析器"""

    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or THRESHOLDS

    def analyze_track(self, track_data: Dict) -> TrackBehavior:
        """
        分析单个轨迹的行为

        Args:
            track_data: 包含 poses 列表的轨迹数据

        Returns:
            TrackBehavior 对象
        """
        track_id = track_data.get('track_id', 0)
        video_id = track_data.get('video_id', '')
        poses = track_data.get('poses', [])

        if len(poses) < self.thresholds['min_track_length']:
            return self._create_unknown_behavior(track_id, video_id, poses)

        # 提取姿态角度序列
        yaws = np.array([p.get('yaw', 0) for p in poses])
        pitches = np.array([p.get('pitch', 0) for p in poses])
        rolls = np.array([p.get('roll', 0) for p in poses])

        # 计算统计特征
        yaw_mean, yaw_std = np.mean(yaws), np.std(yaws)
        pitch_mean, pitch_std = np.mean(pitches), np.std(pitches)
        roll_mean, roll_std = np.mean(rolls), np.std(rolls)

        # 计算行为指标
        attention_score = self._calculate_attention_score(yaws, pitches)
        stability_score = self._calculate_stability_score(yaw_std, pitch_std, roll_std)
        looking_around_ratio = self._calculate_looking_around_ratio(yaws)

        # 分类行为
        primary_label, secondary_labels, confidence = self._classify_behavior(
            yaw_mean, yaw_std, pitch_mean, pitch_std, roll_std,
            attention_score, stability_score, looking_around_ratio
        )

        # 获取帧范围
        frame_indices = [p.get('frame_idx', i) for i, p in enumerate(poses)]
        start_frame = min(frame_indices) if frame_indices else 0
        end_frame = max(frame_indices) if frame_indices else 0

        return TrackBehavior(
            track_id=track_id,
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            duration_frames=len(poses),
            primary_label=primary_label,
            secondary_labels=secondary_labels,
            confidence=confidence,
            yaw_mean=round(float(yaw_mean), 2),
            yaw_std=round(float(yaw_std), 2),
            pitch_mean=round(float(pitch_mean), 2),
            pitch_std=round(float(pitch_std), 2),
            roll_mean=round(float(roll_mean), 2),
            roll_std=round(float(roll_std), 2),
            attention_score=round(float(attention_score), 3),
            stability_score=round(float(stability_score), 3),
            looking_around_ratio=round(float(looking_around_ratio), 3)
        )

    def _calculate_attention_score(self, yaws: np.ndarray, pitches: np.ndarray) -> float:
        """
        计算注意力得分
        基于 yaw 和 pitch 角度判断是否注视前方
        """
        # 假设正常注视前方时 yaw ≈ 0, pitch ≈ 0
        yaw_deviation = np.abs(yaws).mean()
        pitch_deviation = np.abs(pitches).mean()

        # 归一化得分 (0-1)
        yaw_score = max(0, 1 - yaw_deviation / 45)
        pitch_score = max(0, 1 - pitch_deviation / 30)

        return (yaw_score + pitch_score) / 2

    def _calculate_stability_score(self, yaw_std: float, pitch_std: float, roll_std: float) -> float:
        """
        计算姿态稳定性得分
        方差越小，稳定性越高
        """
        # 归一化各项方差
        yaw_stability = max(0, 1 - yaw_std / 30)
        pitch_stability = max(0, 1 - pitch_std / 20)
        roll_stability = max(0, 1 - roll_std / 15)

        return (yaw_stability + pitch_stability + roll_stability) / 3

    def _calculate_looking_around_ratio(self, yaws: np.ndarray) -> float:
        """
        计算东张西望时间比例
        基于 yaw 角度变化检测
        """
        if len(yaws) < 2:
            return 0.0

        # 计算相邻帧的 yaw 变化
        yaw_changes = np.abs(np.diff(yaws))

        # 统计变化超过阈值的帧数
        change_threshold = 3.0  # 度/帧
        looking_around_frames = np.sum(yaw_changes > change_threshold)

        return looking_around_frames / len(yaw_changes)

    def _classify_behavior(self, yaw_mean, yaw_std, pitch_mean, pitch_std, roll_std,
                          attention_score, stability_score, looking_around_ratio) -> tuple:
        """
        综合分类行为

        Returns:
            (primary_label, secondary_labels, confidence)
        """
        labels = []
        confidences = []

        # 规则1: 东张西望检测
        if yaw_std > self.thresholds['yaw_variance_high'] or looking_around_ratio > 0.3:
            labels.append('looking_around')
            conf = min(1.0, yaw_std / 30 + looking_around_ratio)
            confidences.append(conf)

        # 规则2: 低头检测
        if pitch_mean < self.thresholds['pitch_down_threshold']:
            labels.append('looking_down')
            conf = min(1.0, abs(pitch_mean) / 45)
            confidences.append(conf)

        # 规则3: 抬头检测
        if pitch_mean > self.thresholds['pitch_up_threshold']:
            labels.append('looking_up')
            conf = min(1.0, pitch_mean / 45)
            confidences.append(conf)

        # 规则4: 分心检测
        if attention_score < 0.5 and stability_score < 0.6:
            labels.append('distracted')
            conf = 1 - (attention_score + stability_score) / 2
            confidences.append(conf)

        # 规则5: 可疑行为检测 (多种异常组合)
        suspicious_score = 0
        if looking_around_ratio > 0.4:
            suspicious_score += 0.3
        if stability_score < 0.4:
            suspicious_score += 0.3
        if attention_score < 0.3:
            suspicious_score += 0.4

        if suspicious_score > 0.5:
            labels.append('suspicious')
            confidences.append(suspicious_score)

        # 如果没有异常，标记为正常
        if not labels:
            labels.append('normal')
            confidences.append(attention_score * stability_score)

        # 选择主要标签 (置信度最高的)
        if confidences:
            max_idx = np.argmax(confidences)
            primary_label = labels[max_idx]
            primary_confidence = confidences[max_idx]
            secondary_labels = [l for i, l in enumerate(labels) if i != max_idx]
        else:
            primary_label = 'normal'
            primary_confidence = 0.5
            secondary_labels = []

        return primary_label, secondary_labels, round(primary_confidence, 3)

    def _create_unknown_behavior(self, track_id: int, video_id: str, poses: List) -> TrackBehavior:
        """创建未知行为标注 (轨迹太短)"""
        return TrackBehavior(
            track_id=track_id,
            video_id=video_id,
            start_frame=0,
            end_frame=0,
            duration_frames=len(poses),
            primary_label='unknown',
            secondary_labels=[],
            confidence=0.0,
            yaw_mean=0, yaw_std=0,
            pitch_mean=0, pitch_std=0,
            roll_mean=0, roll_std=0,
            attention_score=0, stability_score=0, looking_around_ratio=0
        )


# ============================================================================
# 主处理流程
# ============================================================================

def load_pose_data(video_id: str) -> Dict:
    """加载姿态数据"""
    # 优先使用新的 OpenCV 姿态数据
    pose_file = POSE_DIR / video_id / "pose_opencv.json"
    if not pose_file.exists():
        pose_file = POSE_DIR / video_id / "pose_6drepnet.json"
    if not pose_file.exists():
        pose_file = POSE_DIR / video_id / "pose.json"

    if pose_file.exists():
        with open(pose_file, 'r') as f:
            return json.load(f)
    return {}


def reorganize_poses_by_track(pose_data: Dict, detection_data: Dict = None) -> Dict[int, List]:
    """
    将按帧组织的姿态数据重组为按轨迹组织

    由于原始数据是按帧存储的，需要根据 detection_idx 匹配轨迹
    这里简化处理：为每个帧内的检测分配临时 track_id
    """
    tracks = defaultdict(list)

    frames = pose_data.get('frames', [])

    for frame_info in frames:
        frame_idx = frame_info.get('frame_idx', 0)
        poses = frame_info.get('poses', [])

        for pose in poses:
            det_idx = pose.get('detection_idx', 0)
            # 简化: 使用 detection_idx 作为临时 track_id
            # 实际应用中应该使用跟踪结果来关联
            track_id = det_idx

            tracks[track_id].append({
                'frame_idx': frame_idx,
                'yaw': pose.get('yaw', 0),
                'pitch': pose.get('pitch', 0),
                'roll': pose.get('roll', 0),
                'confidence': pose.get('confidence', 0)
            })

    return dict(tracks)


def process_video(video_id: str, analyzer: BehaviorAnalyzer) -> Dict:
    """处理单个视频的行为标注"""

    # 加载姿态数据
    pose_data = load_pose_data(video_id)
    if not pose_data:
        print(f"  [跳过] {video_id}: 无姿态数据")
        return None

    # 重组为按轨迹组织的数据
    tracks_poses = reorganize_poses_by_track(pose_data)

    if not tracks_poses:
        print(f"  [跳过] {video_id}: 无轨迹数据")
        return None

    # 分析每个轨迹的行为
    behaviors = []
    label_counts = defaultdict(int)

    for track_id, poses in tracks_poses.items():
        track_data = {
            'track_id': track_id,
            'video_id': video_id,
            'poses': poses
        }

        behavior = analyzer.analyze_track(track_data)
        behaviors.append(behavior.to_dict())
        label_counts[behavior.primary_label] += 1

    # 构建结果
    result = {
        'video_id': video_id,
        'processed_at': datetime.now().isoformat(),
        'total_tracks': len(behaviors),
        'label_distribution': dict(label_counts),
        'behaviors': behaviors
    }

    return result


def main():
    """主函数"""
    print("=" * 60)
    print("Step 6: 行为标注")
    print("=" * 60)

    # 初始化分析器
    print("\n[1] 初始化行为分析器...")
    analyzer = BehaviorAnalyzer()

    print(f"\n行为类别:")
    for code, name in BEHAVIOR_LABELS.items():
        print(f"  - {code}: {name}")

    # 获取视频列表
    video_ids = sorted([d.name for d in POSE_DIR.iterdir() if d.is_dir()])
    print(f"\n[2] 发现 {len(video_ids)} 个视频")

    # 处理每个视频
    print("\n[3] 开始处理...")
    all_results = []
    total_label_counts = defaultdict(int)

    for video_id in tqdm(video_ids, desc="处理视频"):
        result = process_video(video_id, analyzer)

        if result:
            # 保存结果
            output_dir = BEHAVIOR_DIR / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "behavior.json"

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            all_results.append({
                'video_id': video_id,
                'total_tracks': result['total_tracks'],
                'label_distribution': result['label_distribution']
            })

            # 累计标签统计
            for label, count in result['label_distribution'].items():
                total_label_counts[label] += count

    # 生成汇总报告
    report = {
        'processed_at': datetime.now().isoformat(),
        'total_videos': len(all_results),
        'total_tracks': sum(r['total_tracks'] for r in all_results),
        'total_label_distribution': dict(total_label_counts),
        'videos': all_results
    }

    report_file = DATASET_ROOT / "step6_behavior_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("处理完成!")
    print(f"  - 视频数: {len(all_results)}")
    print(f"  - 总轨迹数: {report['total_tracks']}")
    print(f"\n标签分布:")
    for label, count in sorted(total_label_counts.items(), key=lambda x: -x[1]):
        pct = count / report['total_tracks'] * 100 if report['total_tracks'] > 0 else 0
        print(f"    {label}: {count} ({pct:.1f}%)")
    print(f"\n报告: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
