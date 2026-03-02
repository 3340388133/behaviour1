"""
标注数据转换工具
将人工标注结果转换为模型训练所需的格式
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import json


def load_manual_annotations(annotations_dir: str) -> pd.DataFrame:
    """加载所有人工标注文件"""
    ann_dir = Path(annotations_dir)
    csv_files = list(ann_dir.glob('*_gt.csv'))

    if not csv_files:
        print(f"未找到标注文件: {annotations_dir}")
        return pd.DataFrame()

    all_annotations = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # 过滤掉未标注的行
        df = df[df['label'].notna()]
        df['label'] = df['label'].astype(int)
        all_annotations.append(df)
        print(f"加载 {csv_file.name}: {len(df)} 条标注")

    combined = pd.concat(all_annotations, ignore_index=True)
    print(f"\n总计: {len(combined)} 条有效标注")
    return combined


def convert_to_training_format(
    annotations_df: pd.DataFrame,
    pose_results_dir: str,
    output_path: str
) -> pd.DataFrame:
    """转换为训练数据格式

    输出格式与 behavior_label_generator.py 一致
    """
    pose_dir = Path(pose_results_dir)
    training_data = []

    for _, row in annotations_df.iterrows():
        video_name = row['video_name']
        start_time = row['start_time']
        end_time = row['end_time']
        label = row['label']

        # 加载对应的 pose 数据
        pose_file = pose_dir / f"{video_name}.csv"
        if not pose_file.exists():
            print(f"警告: 未找到 pose 文件 {pose_file}")
            continue

        pose_df = pd.read_csv(pose_file)

        # 获取时间窗口内的数据
        mask = (pose_df['time_sec'] >= start_time) & \
               (pose_df['time_sec'] < end_time)
        window_df = pose_df[mask]

        if len(window_df) < 3:
            continue

        # 计算特征
        yaws = window_df['yaw'].values
        pitches = window_df['pitch'].values if 'pitch' in window_df else np.zeros_like(yaws)
        rolls = window_df['roll'].values if 'roll' in window_df else np.zeros_like(yaws)

        training_data.append({
            'video_name': video_name,
            'track_id': row.get('track_id', 0),
            'start_time': start_time,
            'end_time': end_time,
            'label': label,
            'yaw_mean': round(np.mean(yaws), 2),
            'yaw_std': round(np.std(yaws), 2),
            'yaw_range': round(np.max(yaws) - np.min(yaws), 2),
            'pitch_mean': round(np.mean(pitches), 2),
            'pitch_std': round(np.std(pitches), 2),
            'roll_mean': round(np.mean(rolls), 2),
            'roll_std': round(np.std(rolls), 2),
            'sample_count': len(window_df),
            'source': 'manual'
        })

    result_df = pd.DataFrame(training_data)

    if output_path and len(result_df) > 0:
        result_df.to_csv(output_path, index=False)
        print(f"\n已保存训练数据: {output_path}")

    return result_df


def print_statistics(df: pd.DataFrame):
    """打印标注统计"""
    if len(df) == 0:
        print("无数据")
        return

    print("\n" + "="*50)
    print("标注统计")
    print("="*50)
    print(f"总样本数: {len(df)}")
    print(f"正常样本 (label=0): {(df['label']==0).sum()}")
    print(f"可疑样本 (label=1): {(df['label']==1).sum()}")
    print(f"可疑比例: {df['label'].mean()*100:.1f}%")
    print("\n各视频分布:")
    print(df.groupby(['video_name', 'label']).size().unstack(fill_value=0))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='标注数据转换工具')
    parser.add_argument('--annotations-dir', default='data/manual_annotations')
    parser.add_argument('--pose-dir', default='data/pose_results')
    parser.add_argument('--output', default='data/training_labels.csv')

    args = parser.parse_args()

    # 加载标注
    annotations = load_manual_annotations(args.annotations_dir)

    if len(annotations) > 0:
        # 转换格式
        training_df = convert_to_training_format(
            annotations, args.pose_dir, args.output
        )
        # 打印统计
        print_statistics(training_df)
