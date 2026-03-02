#!/usr/bin/env python3
"""
标注可视化工具 - 生成视频片段和姿态曲线图
帮助人工标注时查看具体内容
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import cv2
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "annotations" / "visualizations"


def load_pose_data(video_id):
    pose_file = DATA_DIR / "pose" / video_id / "pose.json"
    if not pose_file.exists():
        return None
    with open(pose_file, 'r') as f:
        return json.load(f)


def get_track_poses(pose_data, track_id):
    for t in pose_data['tracks']:
        if t['track_id'] == track_id:
            return t['poses']
    return None


def plot_pose_window(poses, start_frame, end_frame, output_path, sample_id):
    """绘制姿态曲线图"""
    window_poses = [p for p in poses if start_frame <= p['frame_idx'] <= end_frame]
    if not window_poses:
        return False

    frames = [p['frame_idx'] for p in window_poses]
    yaws = [p['yaw'] for p in window_poses]
    pitches = [p['pitch'] for p in window_poses]
    rolls = [p['roll'] for p in window_poses]

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    # Yaw
    axes[0].plot(frames, yaws, 'b-', linewidth=2)
    axes[0].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='±30°')
    axes[0].axhline(y=-30, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Yaw (°)')
    axes[0].set_ylim(-90, 90)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Pitch
    axes[1].plot(frames, pitches, 'g-', linewidth=2)
    axes[1].axhline(y=20, color='r', linestyle='--', alpha=0.5, label='±20°')
    axes[1].axhline(y=-20, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Pitch (°)')
    axes[1].set_ylim(-60, 60)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Roll
    axes[2].plot(frames, rolls, 'm-', linewidth=2)
    axes[2].set_ylabel('Roll (°)')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylim(-45, 45)
    axes[2].grid(True, alpha=0.3)

    # 统计信息
    yaw_std = np.std(yaws)
    yaw_mean = np.mean(yaws)
    pitch_mean = np.mean(pitches)

    fig.suptitle(f'{sample_id}\nYaw: mean={yaw_mean:.1f}°, std={yaw_std:.1f}° | Pitch mean={pitch_mean:.1f}°')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return True


def extract_video_frames(video_id, start_frame, end_frame, output_path, sample_id):
    """从抽帧图像中提取关键帧并拼接"""
    frames_dir = DATA_DIR / "frames" / video_id

    if not frames_dir.exists():
        return False

    # 选择几个关键帧
    frame_indices = np.linspace(start_frame, end_frame, 6, dtype=int)
    images = []

    for idx in frame_indices:
        frame_path = frames_dir / f"frame_{idx:06d}.jpg"
        if frame_path.exists():
            img = cv2.imread(str(frame_path))
            if img is not None:
                # 缩小图像
                h, w = img.shape[:2]
                scale = 200 / h
                img = cv2.resize(img, (int(w * scale), 200))
                images.append(img)

    if not images:
        return False

    # 水平拼接
    combined = np.hstack(images)

    # 添加标题
    cv2.putText(combined, f"{sample_id} | Frames {start_frame}-{end_frame}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(output_path), combined)
    return True


def visualize_sample(video_id, track_id, window_idx, start_frame, end_frame):
    """为单个样本生成可视化"""
    sample_id = f"{video_id}_track{track_id}_win{window_idx}"
    sample_dir = OUTPUT_DIR / video_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 加载姿态数据
    pose_data = load_pose_data(video_id)
    if not pose_data:
        return None

    poses = get_track_poses(pose_data, track_id)
    if not poses:
        return None

    # 生成姿态曲线图
    pose_plot_path = sample_dir / f"{sample_id}_pose.png"
    plot_pose_window(poses, start_frame, end_frame, pose_plot_path, sample_id)

    # 生成视频帧拼接图
    frames_path = sample_dir / f"{sample_id}_frames.png"
    extract_video_frames(video_id, start_frame, end_frame, frames_path, sample_id)

    return {
        'sample_id': sample_id,
        'pose_plot': str(pose_plot_path) if pose_plot_path.exists() else None,
        'frames': str(frames_path) if frames_path.exists() else None,
    }


def main():
    """为前 N 个待标注样本生成可视化"""
    # 加载待标注样本
    pending_file = DATA_DIR / "annotations" / "pending_annotations.json"
    if not pending_file.exists():
        print("请先运行 generate_annotation_preview.py")
        return

    with open(pending_file, 'r') as f:
        pending = json.load(f)

    samples = pending['samples']
    print(f"共 {len(samples)} 个待标注样本")

    # 只处理前 20 个样本作为示例
    num_to_visualize = min(20, len(samples))
    print(f"为前 {num_to_visualize} 个样本生成可视化...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, sample in enumerate(samples[:num_to_visualize]):
        print(f"  [{i+1}/{num_to_visualize}] {sample['sample_id']}")
        result = visualize_sample(
            sample['video_id'],
            sample['track_id'],
            sample['window_idx'],
            sample['start_frame'],
            sample['end_frame']
        )
        if result:
            result['stats'] = sample['stats']
            result['suggested_label'] = sample['suggested_label']
            result['suggested_name'] = sample['suggested_name']
            results.append(result)

    # 保存结果索引
    index_file = OUTPUT_DIR / "visualization_index.json"
    with open(index_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n可视化文件已保存到: {OUTPUT_DIR}")
    print(f"索引文件: {index_file}")


if __name__ == "__main__":
    main()
