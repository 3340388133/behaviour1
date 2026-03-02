#!/usr/bin/env python
"""
使用 WHENet 生成伪标注
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from head_pose import HeadPoseEstimator


def generate_pseudo_labels(
    data_dir: str,
    input_csv: str,
    output_csv: str
):
    """使用 WHENet 生成伪标注"""
    data_dir = Path(data_dir)
    df = pd.read_csv(input_csv)

    estimator = HeadPoseEstimator()
    print(f"加载 WHENet 模型完成")
    print(f"处理 {len(df)} 张图像...")

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        face_path = data_dir / row['face_path']

        if not face_path.exists():
            results.append({'yaw': 0, 'pitch': 0, 'roll': 0})
            continue

        img = cv2.imread(str(face_path))
        if img is None:
            results.append({'yaw': 0, 'pitch': 0, 'roll': 0})
            continue

        pose = estimator.estimate(img)
        results.append({
            'yaw': round(pose.yaw, 2),
            'pitch': round(pose.pitch, 2),
            'roll': round(pose.roll, 2)
        })

    # 更新 DataFrame
    df['yaw'] = [r['yaw'] for r in results]
    df['pitch'] = [r['pitch'] for r in results]
    df['roll'] = [r['roll'] for r in results]

    df.to_csv(output_csv, index=False)
    print(f"伪标注已保存到: {output_csv}")

    # 统计信息
    print(f"\n统计信息:")
    print(f"  Yaw:   mean={df['yaw'].mean():.1f}, std={df['yaw'].std():.1f}, range=[{df['yaw'].min():.1f}, {df['yaw'].max():.1f}]")
    print(f"  Pitch: mean={df['pitch'].mean():.1f}, std={df['pitch'].std():.1f}, range=[{df['pitch'].min():.1f}, {df['pitch'].max():.1f}]")
    print(f"  Roll:  mean={df['roll'].mean():.1f}, std={df['roll'].std():.1f}, range=[{df['roll'].min():.1f}, {df['roll'].max():.1f}]")


if __name__ == "__main__":
    generate_pseudo_labels(
        data_dir="../data",
        input_csv="../data/sample_gt.csv",
        output_csv="../data/sample_gt_whenet.csv"
    )
