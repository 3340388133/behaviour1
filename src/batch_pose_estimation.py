"""
批量头部姿态估计
输入: data/annotations/*.csv
输出: data/pose_results/<video_name>.csv
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

from torch.utils.data import Dataset, DataLoader
from head_pose import HeadPoseEstimator


class FaceDataset(Dataset):
    """人脸数据集"""

    def __init__(self, csv_path: str, data_root: str):
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        face_path = self.data_root / row['face_path']

        # 读取图像
        img = cv2.imread(str(face_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        return {
            'image': img,
            'frame_id': row['frame_id'],
            'time_sec': row['time_sec'],
            'face_path': row['face_path']
        }


def collate_fn(batch):
    """自定义 collate 函数"""
    images = [item['image'] for item in batch]
    frame_ids = [item['frame_id'] for item in batch]
    time_secs = [item['time_sec'] for item in batch]
    face_paths = [item['face_path'] for item in batch]
    return images, frame_ids, time_secs, face_paths


def process_video(csv_path: str, data_root: str, output_dir: str,
                  batch_size: int = 32, num_workers: int = 4):
    """处理单个视频的所有人脸"""
    csv_path = Path(csv_path)
    video_name = csv_path.stem
    output_path = Path(output_dir) / f"{video_name}.csv"

    # 创建数据集和加载器
    dataset = FaceDataset(csv_path, data_root)
    if len(dataset) == 0:
        print(f"  跳过 {video_name}: 无数据")
        return 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # 初始化估计器
    estimator = HeadPoseEstimator()

    # 批量推理
    results = []
    for images, frame_ids, time_secs, face_paths in tqdm(
        loader, desc=f"  {video_name}", leave=False
    ):
        for img, fid, ts, fp in zip(images, frame_ids, time_secs, face_paths):
            pose = estimator.estimate(img)
            results.append({
                'frame_id': fid,
                'time_sec': round(ts, 3),
                'face_path': fp,
                'yaw': round(pose.yaw, 2),
                'pitch': round(pose.pitch, 2),
                'roll': round(pose.roll, 2),
                'pose_confidence': round(pose.confidence, 3)
            })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    return len(results)


def main():
    parser = argparse.ArgumentParser(description='批量头部姿态估计')
    parser.add_argument('--data-root', default='../data', help='数据根目录')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotations_dir = data_root / 'annotations'
    output_dir = data_root / 'pose_results'
    output_dir.mkdir(exist_ok=True)

    # 获取所有 CSV 文件
    csv_files = sorted(annotations_dir.glob('*.csv'))
    print(f"找到 {len(csv_files)} 个标注文件")

    total = 0
    for csv_path in csv_files:
        print(f"\n处理: {csv_path.name}")
        count = process_video(
            csv_path, data_root, output_dir,
            args.batch_size, args.num_workers
        )
        total += count
        print(f"  完成: {count} 张")

    print(f"\n总计处理: {total} 张人脸")


if __name__ == '__main__':
    main()
