#!/usr/bin/env python3
"""
Step 5: 构建训练数据集
将姿态序列整理为模型训练格式

数据集格式：
{
    "samples": [
        {
            "video": "视频名",
            "track_id": "track_0001",
            "pose_sequence": [[yaw, pitch, roll], ...],  # T x 3
            "timestamps": [0.0, 0.033, ...],
            "label": 0-5,  # 6类行为
        },
        ...
    ]
}

行为类别定义：
- 0: normal        正常行为      视线稳定，偶尔自然转头
- 1: glancing      频繁张望      3秒内左右转头≥3次，yaw变化>30°
- 2: quick_turn    快速回头      0.5秒内 yaw 变化>60°
- 3: prolonged_watch 长时间观察  持续>3秒注视非正前方(yaw>30°)
- 4: looking_down  持续低头      pitch<-20° 持续>5秒
- 5: looking_up    持续抬头      pitch>20° 持续>3秒
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime


# ============== 配置 ==============
DATA_ROOT = Path("data")
POSE_OUTPUT_DIR = DATA_ROOT / "pose_output"
DATASET_DIR = DATA_ROOT / "dataset"


def load_pose_data(pose_file: Path) -> Dict:
    """加载姿态数据"""
    with open(pose_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_sequences(
    pose_data: Dict,
    seq_length: int = 32,
    stride: int = 16,
    min_valid_ratio: float = 0.8,
) -> List[Dict]:
    """
    从姿态数据中提取固定长度的序列

    Args:
        pose_data: 姿态数据
        seq_length: 序列长度
        stride: 滑动步长
        min_valid_ratio: 最小有效帧比例

    Returns:
        序列列表
    """
    sequences = []
    video_name = pose_data.get("video_name", "unknown")

    for track_id, track_data in pose_data.get("tracks", {}).items():
        poses = track_data.get("poses", [])

        if len(poses) < seq_length * min_valid_ratio:
            continue

        # 按帧号排序
        poses = sorted(poses, key=lambda x: x["frame"])

        # 滑动窗口提取序列
        for start_idx in range(0, len(poses) - seq_length + 1, stride):
            seq_poses = poses[start_idx:start_idx + seq_length]

            # 提取姿态值
            pose_sequence = [
                [p["yaw"], p["pitch"], p["roll"]]
                for p in seq_poses
            ]
            timestamps = [p["frame"] for p in seq_poses]

            # 使用规则生成伪标签
            label = generate_pseudo_label(pose_sequence)

            sequences.append({
                "video": video_name,
                "track_id": track_id,
                "start_frame": timestamps[0],
                "end_frame": timestamps[-1],
                "pose_sequence": pose_sequence,
                "timestamps": timestamps,
                "label": label,
            })

    return sequences


"""
行为类别常量
"""
BEHAVIOR_CLASSES = {
    0: 'normal',          # 正常行为
    1: 'glancing',        # 频繁张望
    2: 'quick_turn',      # 快速回头
    3: 'prolonged_watch', # 长时间观察
    4: 'looking_down',    # 持续低头
    5: 'looking_up',      # 持续抬头
}

BEHAVIOR_NAMES_CN = {
    0: '正常行为',
    1: '频繁张望',
    2: '快速回头',
    3: '长时间观察',
    4: '持续低头',
    5: '持续抬头',
}


def smooth_sequence(seq: np.ndarray, window: int = 5) -> np.ndarray:
    """对序列进行滑动平均平滑，减少噪声"""
    if len(seq) < window:
        return seq
    kernel = np.ones(window) / window
    # 边缘处理
    padded = np.pad(seq, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(seq)]


def generate_pseudo_label(pose_sequence: List[List[float]], fps: float = 30.0) -> int:
    """
    基于规则生成6类行为伪标签（使用平滑后的序列减少噪声影响）

    类别定义：
    - 0: normal        正常行为      视线稳定，yaw变化小
    - 1: glancing      频繁张望      多次方向切换，yaw振幅较大
    - 2: quick_turn    快速回头      从正面快速转向侧面或背面
    - 3: prolonged_watch 长时间观察  持续注视非正前方，方向稳定
    - 4: looking_down  持续低头      pitch明显低于平均
    - 5: looking_up    持续抬头      pitch明显高于平均

    Args:
        pose_sequence: [[yaw, pitch, roll], ...] 姿态序列
        fps: 视频帧率

    Returns:
        label: 0-5 行为类别
    """
    seq_len = len(pose_sequence)
    yaw_raw = np.array([p[0] for p in pose_sequence])
    pitch_raw = np.array([p[1] for p in pose_sequence])

    # 平滑处理减少噪声
    yaw_seq = smooth_sequence(yaw_raw, window=5)
    pitch_seq = smooth_sequence(pitch_raw, window=5)

    # 面向摄像头的帧（|yaw| < 60° 为正面范围）
    front_mask = np.abs(yaw_seq) < 60
    front_facing_yaw = yaw_seq[front_mask]
    front_ratio = len(front_facing_yaw) / seq_len

    # 背对摄像头太多，归为正常
    if front_ratio < 0.2:
        return 0

    # 计算平滑后的方向切换次数
    direction_changes = 0
    prev_direction = 0
    smoothed_diffs = np.diff(yaw_seq)
    for diff in smoothed_diffs:
        if abs(diff) < 2:  # 忽略微小变化
            continue
        curr_direction = 1 if diff > 0 else -1
        if prev_direction != 0 and curr_direction != prev_direction:
            direction_changes += 1
        prev_direction = curr_direction

    # yaw统计
    yaw_mean = np.mean(yaw_seq)
    yaw_std = np.std(yaw_seq)
    yaw_amplitude = np.max(yaw_seq) - np.min(yaw_seq)

    # pitch统计
    pitch_mean = np.mean(pitch_seq)

    # ============ 类别2: quick_turn 快速回头 ============
    # 更严格的定义：前半段保持正面，然后快速转头
    first_half = yaw_seq[:seq_len//2]
    second_half = yaw_seq[seq_len//2:]

    # 前半段大部分时间正面朝向（|yaw|<35°的帧占70%以上）
    first_half_frontal = np.sum(np.abs(first_half) < 35) / len(first_half)
    # 后半段有明显转向（|yaw|>50°的帧占50%以上）
    second_half_turned = np.sum(np.abs(second_half) > 50) / len(second_half)

    # 转向幅度大
    first_half_mean = np.mean(np.abs(first_half))
    second_half_mean = np.mean(np.abs(second_half))

    if first_half_frontal > 0.7 and second_half_turned > 0.5 and (second_half_mean - first_half_mean) > 30:
        return 2

    # ============ 类别5: looking_up 持续抬头 ============
    # pitch明显偏高（相对于数据整体均值-13°）
    looking_up_ratio = np.sum(pitch_seq > 5) / seq_len
    if pitch_mean > 5 or looking_up_ratio > 0.5:
        return 5

    # ============ 类别4: looking_down 持续低头 ============
    # pitch明显偏低（相对于数据整体均值-13°，需要更低）
    looking_down_ratio = np.sum(pitch_seq < -18) / seq_len
    if pitch_mean < -20 or looking_down_ratio > 0.6:
        return 4

    # ============ 类别3: prolonged_watch 长时间观察 ============
    # 持续看向一侧（|yaw|>35°），且方向稳定（std小，切换少）
    side_watch_ratio = np.sum(np.abs(yaw_seq) > 35) / seq_len
    if side_watch_ratio > 0.6 and yaw_std < 25 and direction_changes < 3:
        return 3

    # ============ 类别1: glancing 频繁张望（可疑级别）============
    # 数据统计: 中位数 dir=14, std=57.6
    # 使用 Top 20-25% 阈值来识别真正异常的张望行为

    # 条件A: 超高频方向切换（>=18次，Top 25%）
    if direction_changes >= 18:
        return 1

    # 条件B: 极高标准差（>70°，Top 30%）+ 较多切换
    if yaw_std > 70 and direction_changes >= 15:
        return 1

    # 条件C: 超大振幅（>100°）+ 高频切换
    if yaw_amplitude > 100 and direction_changes >= 12:
        return 1

    # ============ 类别0: normal 正常行为 ============
    return 0


def analyze_dataset(samples: List[Dict]) -> Dict:
    """分析数据集统计信息（6类）"""
    labels = [s["label"] for s in samples]

    # 统计各类别数量
    class_counts = {}
    for i in range(6):
        class_counts[i] = labels.count(i)

    # 姿态统计
    all_yaw = []
    all_pitch = []
    for s in samples:
        for pose in s["pose_sequence"]:
            all_yaw.append(pose[0])
            all_pitch.append(pose[1])

    total = len(samples) if samples else 1

    return {
        "total_samples": len(samples),
        "num_classes": 6,
        "class_distribution": {
            BEHAVIOR_CLASSES[i]: {
                "count": class_counts[i],
                "ratio": round(class_counts[i] / total, 4),
                "name_cn": BEHAVIOR_NAMES_CN[i],
            }
            for i in range(6)
        },
        # 保持向后兼容
        "normal_samples": class_counts[0],
        "suspicious_samples": sum(class_counts[i] for i in range(1, 6)),
        "yaw_stats": {
            "min": round(min(all_yaw), 2) if all_yaw else 0,
            "max": round(max(all_yaw), 2) if all_yaw else 0,
            "mean": round(np.mean(all_yaw), 2) if all_yaw else 0,
            "std": round(np.std(all_yaw), 2) if all_yaw else 0,
        },
        "pitch_stats": {
            "min": round(min(all_pitch), 2) if all_pitch else 0,
            "max": round(max(all_pitch), 2) if all_pitch else 0,
            "mean": round(np.mean(all_pitch), 2) if all_pitch else 0,
            "std": round(np.std(all_pitch), 2) if all_pitch else 0,
        },
    }


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    划分数据集

    如果视频数量足够，按视频划分；否则按样本随机划分
    """
    np.random.seed(seed)

    # 按视频分组
    video_samples = {}
    for s in samples:
        video = s["video"]
        if video not in video_samples:
            video_samples[video] = []
        video_samples[video].append(s)

    videos = list(video_samples.keys())

    # 如果视频数量少于3个，改为按样本随机划分
    if len(videos) < 3:
        print("  视频数量不足，改为按样本随机划分")
        indices = np.arange(len(samples))
        np.random.shuffle(indices)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return {
            "train": [samples[i] for i in train_idx],
            "val": [samples[i] for i in val_idx],
            "test": [samples[i] for i in test_idx],
        }

    # 按视频划分
    np.random.shuffle(videos)

    n_videos = len(videos)
    train_end = int(n_videos * train_ratio)
    val_end = int(n_videos * (train_ratio + val_ratio))

    train_videos = videos[:train_end]
    val_videos = videos[train_end:val_end]
    test_videos = videos[val_end:]

    train_samples = []
    val_samples = []
    test_samples = []

    for v in train_videos:
        train_samples.extend(video_samples[v])
    for v in val_videos:
        val_samples.extend(video_samples[v])
    for v in test_videos:
        test_samples.extend(video_samples[v])

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 5: 构建训练数据集")
    parser.add_argument("--input", "-i", type=str, default=str(POSE_OUTPUT_DIR))
    parser.add_argument("--output", "-o", type=str, default=str(DATASET_DIR))
    parser.add_argument("--seq-length", type=int, default=32,
                        help="序列长度（帧数）")
    parser.add_argument("--stride", type=int, default=16,
                        help="滑动窗口步长")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    args = parser.parse_args()

    print("=" * 60)
    print("Step 5: 构建训练数据集")
    print("=" * 60)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有姿态文件
    pose_files = list(input_dir.glob("*_poses.json"))

    if not pose_files:
        print(f"未找到姿态文件: {input_dir}")
        return

    print(f"找到 {len(pose_files)} 个姿态文件")

    # 提取所有序列
    all_samples = []
    for pose_file in pose_files:
        print(f"处理: {pose_file.name}")
        pose_data = load_pose_data(pose_file)
        sequences = extract_sequences(
            pose_data,
            seq_length=args.seq_length,
            stride=args.stride,
        )
        all_samples.extend(sequences)
        print(f"  提取了 {len(sequences)} 个序列")

    print(f"\n总共提取 {len(all_samples)} 个序列")

    # 分析数据集
    stats = analyze_dataset(all_samples)
    print(f"\n数据集统计 (6类):")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  各类别分布:")
    for i in range(6):
        class_name = BEHAVIOR_CLASSES[i]
        class_info = stats['class_distribution'][class_name]
        print(f"    [{i}] {class_info['name_cn']:8s} ({class_name:15s}): "
              f"{class_info['count']:6d} ({class_info['ratio']:.1%})")
    print(f"  Yaw 范围: [{stats['yaw_stats']['min']}, {stats['yaw_stats']['max']}]")

    # 划分数据集
    test_ratio = 1 - args.train_ratio - args.val_ratio
    splits = split_dataset(
        all_samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )

    print(f"\n数据集划分:")
    print(f"  训练集: {len(splits['train'])} 样本")
    print(f"  验证集: {len(splits['val'])} 样本")
    print(f"  测试集: {len(splits['test'])} 样本")

    # 保存数据集
    for split_name, samples in splits.items():
        output_path = output_dir / f"{split_name}.json"
        dataset = {
            "split": split_name,
            "seq_length": args.seq_length,
            "num_samples": len(samples),
            "statistics": analyze_dataset(samples),
            "samples": samples,
            "created_at": datetime.now().isoformat(),
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"  保存: {output_path}")

    # 保存元数据
    metadata = {
        "seq_length": args.seq_length,
        "stride": args.stride,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": test_ratio,
        "total_samples": len(all_samples),
        "statistics": stats,
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("数据集构建完成!")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
