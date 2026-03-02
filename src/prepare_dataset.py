"""
数据准备脚本
从原始视频中按标注提取片段，构建动作识别数据集
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict

import cv2
from tqdm import tqdm


def load_labels(csv_path):
    """加载标注文件"""
    samples = defaultdict(list)

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row['video_name']
            samples[video_name].append({
                'start': float(row['start_time']),
                'end': float(row['end_time']),
                'label': int(row['label']),
                'track_id': row['track_id']
            })

    return samples


def find_video_file(video_name, video_dir):
    """查找对应的视频文件"""
    video_dir = Path(video_dir)
    extensions = ['.mp4', '.MP4', '.avi', '.mov']

    for ext in extensions:
        # 尝试直接匹配
        path = video_dir / f"{video_name}{ext}"
        if path.exists():
            return path

    # 模糊匹配
    for f in video_dir.iterdir():
        if video_name in f.stem:
            return f

    return None


def extract_clip(video_path, start_time, end_time, output_path):
    """从视频中提取片段"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


def prepare_dataset(
    label_csv,
    video_dir,
    output_dir,
    train_ratio=0.8,
    max_samples_per_class=None
):
    """准备数据集"""
    output_dir = Path(output_dir)
    video_dir = Path(video_dir)

    # 创建目录结构
    for split in ['train', 'val']:
        for label in ['0', '1']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # 加载标注
    samples = load_labels(label_csv)
    print(f"加载了 {len(samples)} 个视频的标注")

    # 收集所有样本
    all_samples = {'0': [], '1': []}

    for video_name, annotations in samples.items():
        video_path = find_video_file(video_name, video_dir)
        if video_path is None:
            print(f"警告: 找不到视频 {video_name}")
            continue

        for ann in annotations:
            label = str(ann['label'])
            all_samples[label].append({
                'video_path': video_path,
                'video_name': video_name,
                'start': ann['start'],
                'end': ann['end'],
                'track_id': ann['track_id']
            })

    # 打印统计
    for label, items in all_samples.items():
        print(f"类别 {label}: {len(items)} 个样本")

    return all_samples, output_dir, train_ratio


def extract_and_save(all_samples, output_dir, train_ratio=0.8, max_per_class=None):
    """提取并保存视频片段"""
    clip_count = 0

    for label, items in all_samples.items():
        random.shuffle(items)

        # 限制样本数量
        if max_per_class and len(items) > max_per_class:
            items = items[:max_per_class]

        split_idx = int(len(items) * train_ratio)

        train_items = items[:split_idx]
        val_items = items[split_idx:]

        for split, split_items in [('train', train_items), ('val', val_items)]:
            print(f"\n处理 {split}/{label}: {len(split_items)} 个片段")

            for item in tqdm(split_items):
                clip_name = f"{item['video_name']}_{item['track_id']}_{item['start']:.1f}_{item['end']:.1f}.mp4"
                output_path = output_dir / split / label / clip_name

                if output_path.exists():
                    continue

                try:
                    extract_clip(
                        item['video_path'],
                        item['start'],
                        item['end'],
                        output_path
                    )
                    clip_count += 1
                except Exception as e:
                    print(f"提取失败: {clip_name}, 错误: {e}")

    print(f"\n完成! 共提取 {clip_count} 个片段")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='准备动作识别数据集')
    parser.add_argument('--label_csv', type=str,
                        default='../data/behavior_labels.csv')
    parser.add_argument('--video_dir', type=str,
                        default='../data/raw_videos')
    parser.add_argument('--output_dir', type=str,
                        default='../data/action_dataset')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--max_per_class', type=int, default=None,
                        help='每类最大样本数')

    args = parser.parse_args()

    all_samples, output_dir, train_ratio = prepare_dataset(
        args.label_csv,
        args.video_dir,
        args.output_dir,
        args.train_ratio
    )

    extract_and_save(all_samples, output_dir, train_ratio, args.max_per_class)
