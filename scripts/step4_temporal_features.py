#!/usr/bin/env python3
"""
Step 4: 时序特征提取 (Temporal Feature Extraction)

对应论文 III-C 节: Temporal Feature Extraction

输入:
    - dataset_root/features/pose/{video_id}/pose.json

输出:
    - dataset_root/features/temporal/{video_id}/temporal_features.json
    - dataset_root/features/temporal/{video_id}/windows.json

依赖: numpy, scipy, tqdm
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

DATASET_ROOT = Path(__file__).parent.parent / "dataset_root"
POSE_DIR = DATASET_ROOT / "features" / "pose"
TEMPORAL_DIR = DATASET_ROOT / "features" / "temporal"

# 滑动窗口参数
WINDOW_SIZE_SEC = 2.0      # 窗口大小（秒）
WINDOW_STRIDE_SEC = 0.5    # 窗口步长（秒）
MIN_FRAMES_IN_WINDOW = 5   # 窗口内最少帧数

# 特征提取参数
ATTENTION_YAW_THRESHOLD = 15.0    # 注意力偏离阈值（度）
HEAD_MOTION_THRESHOLD = 5.0       # 头部运动阈值（度/秒）


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class TemporalWindow:
    """时序窗口"""
    window_id: int
    track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    num_frames: int


@dataclass
class TemporalFeatures:
    """时序特征"""
    # 基础统计特征
    yaw_mean: float
    yaw_std: float
    yaw_min: float
    yaw_max: float
    yaw_range: float

    pitch_mean: float
    pitch_std: float
    pitch_min: float
    pitch_max: float
    pitch_range: float

    roll_mean: float
    roll_std: float
    roll_min: float
    roll_max: float
    roll_range: float

    # 变化率特征
    yaw_velocity_mean: float
    yaw_velocity_std: float
    yaw_velocity_max: float

    pitch_velocity_mean: float
    pitch_velocity_std: float
    pitch_velocity_max: float

    roll_velocity_mean: float
    roll_velocity_std: float
    roll_velocity_max: float

    # 行为模式特征
    attention_ratio: float          # 注意力集中比例（yaw在阈值内的比例）
    head_motion_intensity: float    # 头部运动强度
    direction_changes: int          # 方向变化次数
    stable_ratio: float             # 稳定比例（运动小于阈值的比例）

    # 频域特征
    yaw_dominant_freq: float        # yaw主频率
    pitch_dominant_freq: float      # pitch主频率

    # 置信度
    avg_confidence: float


# ============================================================================
# 特征提取函数
# ============================================================================

def compute_velocity(values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """计算变化率（度/秒）"""
    if len(values) < 2:
        return np.array([0.0])

    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-6  # 避免除零
    dv = np.diff(values)
    return dv / dt


def compute_dominant_frequency(values: np.ndarray, fps: float) -> float:
    """计算主频率（使用FFT）"""
    if len(values) < 4:
        return 0.0

    # 去均值
    values = values - np.mean(values)

    # FFT
    fft = np.fft.fft(values)
    freqs = np.fft.fftfreq(len(values), 1.0 / fps)

    # 只取正频率部分
    positive_mask = freqs > 0
    fft_magnitude = np.abs(fft[positive_mask])
    positive_freqs = freqs[positive_mask]

    if len(fft_magnitude) == 0:
        return 0.0

    # 找主频率
    dominant_idx = np.argmax(fft_magnitude)
    return float(positive_freqs[dominant_idx])


def count_direction_changes(values: np.ndarray, threshold: float = 3.0) -> int:
    """计算方向变化次数"""
    if len(values) < 3:
        return 0

    # 计算差分
    diff = np.diff(values)

    # 过滤小变化
    diff[np.abs(diff) < threshold] = 0

    # 计算符号变化
    signs = np.sign(diff)
    signs = signs[signs != 0]  # 移除零

    if len(signs) < 2:
        return 0

    changes = np.sum(np.diff(signs) != 0)
    return int(changes)


def extract_window_features(
    poses: List[Dict],
    fps: float
) -> Optional[TemporalFeatures]:
    """从窗口内的姿态数据提取特征"""
    if len(poses) < MIN_FRAMES_IN_WINDOW:
        return None

    # 提取数组
    yaws = np.array([p["yaw"] for p in poses])
    pitches = np.array([p["pitch"] for p in poses])
    rolls = np.array([p["roll"] for p in poses])
    timestamps = np.array([p["timestamp"] for p in poses])
    confidences = np.array([p["confidence"] for p in poses])

    # 基础统计特征
    yaw_mean, yaw_std = np.mean(yaws), np.std(yaws)
    pitch_mean, pitch_std = np.mean(pitches), np.std(pitches)
    roll_mean, roll_std = np.mean(rolls), np.std(rolls)

    # 变化率
    yaw_vel = compute_velocity(yaws, timestamps)
    pitch_vel = compute_velocity(pitches, timestamps)
    roll_vel = compute_velocity(rolls, timestamps)

    # 行为模式特征
    attention_ratio = np.mean(np.abs(yaws) < ATTENTION_YAW_THRESHOLD)

    total_motion = np.sqrt(yaw_vel**2 + pitch_vel**2 + roll_vel**2)
    head_motion_intensity = np.mean(total_motion)

    direction_changes = count_direction_changes(yaws)

    stable_ratio = np.mean(total_motion < HEAD_MOTION_THRESHOLD)

    # 频域特征
    yaw_freq = compute_dominant_frequency(yaws, fps)
    pitch_freq = compute_dominant_frequency(pitches, fps)

    return TemporalFeatures(
        # 基础统计
        yaw_mean=round(yaw_mean, 3),
        yaw_std=round(yaw_std, 3),
        yaw_min=round(float(np.min(yaws)), 3),
        yaw_max=round(float(np.max(yaws)), 3),
        yaw_range=round(float(np.max(yaws) - np.min(yaws)), 3),

        pitch_mean=round(pitch_mean, 3),
        pitch_std=round(pitch_std, 3),
        pitch_min=round(float(np.min(pitches)), 3),
        pitch_max=round(float(np.max(pitches)), 3),
        pitch_range=round(float(np.max(pitches) - np.min(pitches)), 3),

        roll_mean=round(roll_mean, 3),
        roll_std=round(roll_std, 3),
        roll_min=round(float(np.min(rolls)), 3),
        roll_max=round(float(np.max(rolls)), 3),
        roll_range=round(float(np.max(rolls) - np.min(rolls)), 3),

        # 变化率
        yaw_velocity_mean=round(float(np.mean(np.abs(yaw_vel))), 3),
        yaw_velocity_std=round(float(np.std(yaw_vel)), 3),
        yaw_velocity_max=round(float(np.max(np.abs(yaw_vel))), 3),

        pitch_velocity_mean=round(float(np.mean(np.abs(pitch_vel))), 3),
        pitch_velocity_std=round(float(np.std(pitch_vel)), 3),
        pitch_velocity_max=round(float(np.max(np.abs(pitch_vel))), 3),

        roll_velocity_mean=round(float(np.mean(np.abs(roll_vel))), 3),
        roll_velocity_std=round(float(np.std(roll_vel)), 3),
        roll_velocity_max=round(float(np.max(np.abs(roll_vel))), 3),

        # 行为模式
        attention_ratio=round(float(attention_ratio), 3),
        head_motion_intensity=round(float(head_motion_intensity), 3),
        direction_changes=int(direction_changes),
        stable_ratio=round(float(stable_ratio), 3),

        # 频域
        yaw_dominant_freq=round(yaw_freq, 3),
        pitch_dominant_freq=round(pitch_freq, 3),

        # 置信度
        avg_confidence=round(float(np.mean(confidences)), 3)
    )


def create_sliding_windows(
    poses: List[Dict],
    fps: float,
    track_id: int,
    window_size_sec: float = WINDOW_SIZE_SEC,
    stride_sec: float = WINDOW_STRIDE_SEC
) -> List[tuple]:
    """创建滑动窗口"""
    if not poses:
        return []

    windows = []
    window_id = 0

    # 按时间排序
    poses = sorted(poses, key=lambda x: x["timestamp"])

    start_time = poses[0]["timestamp"]
    end_time = poses[-1]["timestamp"]

    current_start = start_time

    while current_start + window_size_sec <= end_time + stride_sec:
        window_end = current_start + window_size_sec

        # 获取窗口内的帧
        window_poses = [
            p for p in poses
            if current_start <= p["timestamp"] < window_end
        ]

        if len(window_poses) >= MIN_FRAMES_IN_WINDOW:
            window = TemporalWindow(
                window_id=window_id,
                track_id=track_id,
                start_frame=window_poses[0]["frame_idx"],
                end_frame=window_poses[-1]["frame_idx"],
                start_time=round(current_start, 3),
                end_time=round(window_end, 3),
                num_frames=len(window_poses)
            )
            windows.append((window, window_poses))
            window_id += 1

        current_start += stride_sec

    return windows


# ============================================================================
# 主处理函数
# ============================================================================

def process_video(video_id: str) -> Dict[str, Any]:
    """处理单个视频的时序特征提取"""
    pose_file = POSE_DIR / video_id / "pose.json"

    if not pose_file.exists():
        print(f"  [错误] 姿态文件不存在: {pose_file}")
        return None

    # 加载姿态数据
    with open(pose_file, 'r', encoding='utf-8') as f:
        pose_data = json.load(f)

    fps = pose_data.get("fps", 10.0)
    tracks = pose_data.get("tracks", [])

    # 创建输出目录
    output_dir = TEMPORAL_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_features = []

    for track in tqdm(tracks, desc=f"  {video_id[:20]}", leave=False):
        track_id = track["track_id"]
        poses = track.get("poses", [])

        if len(poses) < MIN_FRAMES_IN_WINDOW:
            continue

        # 创建滑动窗口
        windows = create_sliding_windows(poses, fps, track_id)

        for window, window_poses in windows:
            # 提取特征
            features = extract_window_features(window_poses, fps)

            if features:
                window_data = asdict(window)
                feature_data = asdict(features)

                all_windows.append(window_data)
                all_features.append({
                    "window_id": window.window_id,
                    "track_id": track_id,
                    "start_time": window.start_time,
                    "end_time": window.end_time,
                    "features": feature_data
                })

    # 保存窗口信息
    windows_output = {
        "video_id": video_id,
        "fps": fps,
        "window_config": {
            "window_size_sec": WINDOW_SIZE_SEC,
            "stride_sec": WINDOW_STRIDE_SEC,
            "min_frames": MIN_FRAMES_IN_WINDOW
        },
        "total_windows": len(all_windows),
        "windows": all_windows
    }

    with open(output_dir / "windows.json", 'w', encoding='utf-8') as f:
        json.dump(windows_output, f, ensure_ascii=False, indent=2)

    # 保存特征
    features_output = {
        "video_id": video_id,
        "fps": fps,
        "processed_at": datetime.now().isoformat(),
        "feature_config": {
            "attention_yaw_threshold": ATTENTION_YAW_THRESHOLD,
            "head_motion_threshold": HEAD_MOTION_THRESHOLD
        },
        "total_samples": len(all_features),
        "feature_dim": 28,  # 特征维度
        "samples": all_features
    }

    with open(output_dir / "temporal_features.json", 'w', encoding='utf-8') as f:
        json.dump(features_output, f, ensure_ascii=False, indent=2)

    return {
        "video_id": video_id,
        "total_tracks": len(tracks),
        "total_windows": len(all_windows),
        "total_samples": len(all_features)
    }


def main():
    """批量处理"""
    print("=" * 60)
    print("Step 4: 时序特征提取 (Temporal Feature Extraction)")
    print("=" * 60)

    # 获取所有视频
    video_dirs = [d for d in POSE_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not video_dirs:
        print(f"[错误] 未找到姿态数据，请先运行 Step 3")
        return

    print(f"\n找到 {len(video_dirs)} 个视频")
    print(f"窗口配置: {WINDOW_SIZE_SEC}s 窗口, {WINDOW_STRIDE_SEC}s 步长")

    all_stats = []

    for video_dir in sorted(video_dirs):
        video_id = video_dir.name

        # 检查是否已处理
        feature_file = TEMPORAL_DIR / video_id / "temporal_features.json"
        if feature_file.exists():
            print(f"\n[跳过] {video_id}: 已处理")
            with open(feature_file, 'r') as f:
                existing = json.load(f)
            all_stats.append({
                "video_id": video_id,
                "total_samples": existing.get("total_samples", 0),
                "status": "skipped"
            })
            continue

        print(f"\n处理: {video_id}")
        stats = process_video(video_id)

        if stats:
            all_stats.append(stats)
            print(f"  [完成] {stats['total_windows']} 窗口, {stats['total_samples']} 样本")

    # 总结
    print(f"\n{'='*60}")
    print("总体统计")
    print(f"{'='*60}")

    total_windows = sum(s.get("total_windows", 0) for s in all_stats)
    total_samples = sum(s.get("total_samples", 0) for s in all_stats)

    print(f"  视频数: {len(all_stats)}")
    print(f"  总窗口数: {total_windows}")
    print(f"  总样本数: {total_samples}")

    # 保存报告
    report = {
        "processed_at": datetime.now().isoformat(),
        "config": {
            "window_size_sec": WINDOW_SIZE_SEC,
            "stride_sec": WINDOW_STRIDE_SEC,
            "min_frames": MIN_FRAMES_IN_WINDOW,
            "attention_threshold": ATTENTION_YAW_THRESHOLD,
            "motion_threshold": HEAD_MOTION_THRESHOLD
        },
        "total_videos": len(all_stats),
        "total_windows": total_windows,
        "total_samples": total_samples,
        "videos": all_stats
    }

    report_path = DATASET_ROOT / "step4_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
