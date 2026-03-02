"""
视频数据集加载器
用于加载和预处理视频数据进行动作识别训练
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class VideoDataset(Dataset):
    """
    视频数据集类

    数据目录结构:
    data/
    ├── train/
    │   ├── action1/
    │   │   ├── video1.mp4
    │   │   └── video2.mp4
    │   └── action2/
    │       └── video3.mp4
    └── val/
        └── ...
    """

    def __init__(
        self,
        data_path: str,
        num_frames: int = 16,
        frame_size: tuple = (224, 224),
        transform=None
    ):
        """
        Args:
            data_path: 数据目录路径
            num_frames: 每个视频采样的帧数
            frame_size: 帧的目标尺寸 (H, W)
            transform: 数据增强变换
        """
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

        # 获取类别列表
        self.classes = sorted([
            d.name for d in self.data_path.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集所有视频文件
        self.samples = self._collect_samples()

    def _collect_samples(self):
        """收集所有视频样本"""
        samples = []
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

        for class_name in self.classes:
            class_path = self.data_path / class_name
            for video_file in class_path.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    samples.append({
                        'path': str(video_file),
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
        return samples

    def _load_video(self, video_path: str) -> np.ndarray:
        """加载视频并均匀采样帧"""
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise ValueError(f"无法读取视频: {video_path}")

        # 均匀采样帧索引
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 调整尺寸
                frame = cv2.resize(frame, self.frame_size)
                frames.append(frame)
            else:
                # 如果读取失败，用黑帧填充
                frames.append(np.zeros((*self.frame_size, 3), dtype=np.uint8))

        cap.release()
        return np.array(frames)  # (T, H, W, C)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载视频帧
        frames = self._load_video(sample['path'])  # (T, H, W, C)

        # 应用数据增强
        if self.transform:
            frames = self.transform(frames)

        # 转换为 tensor: (T, H, W, C) -> (C, T, H, W)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # (C, T, H, W)

        # 归一化到 [0, 1]
        frames = frames / 255.0

        label = torch.tensor(sample['label'], dtype=torch.long)

        return frames, label


def get_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 4
):
    """创建训练和验证数据加载器"""
    from torch.utils.data import DataLoader

    train_dataset = VideoDataset(
        data_path=os.path.join(data_root, 'train'),
        num_frames=num_frames
    )

    val_dataset = VideoDataset(
        data_path=os.path.join(data_root, 'val'),
        num_frames=num_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes
