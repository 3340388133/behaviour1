#!/usr/bin/env python3
"""
行为识别数据集 - 从已构建的数据集加载训练数据
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional


class BehaviorRecognitionDataset(Dataset):
    """
    行为识别数据集

    从 data/dataset/ 目录加载已构建的训练数据
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        seq_len: int = 32,
        normalize: bool = True,
    ):
        """
        Args:
            data_path: 数据目录路径 (包含 train.json, val.json, test.json)
            split: 'train', 'val', 或 'test'
            seq_len: 序列长度 (会截断或填充到这个长度)
            normalize: 是否归一化姿态角度
        """
        self.data_path = Path(data_path)
        self.split = split
        self.seq_len = seq_len
        self.normalize = normalize

        # 加载数据
        self._load_data()

        # 类别名称
        self.class_names = ['normal', 'suspicious']

    def _load_data(self):
        """加载数据"""
        json_path = self.data_path / f'{self.split}.json'

        if not json_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = data['samples']
        self.statistics = data.get('statistics', {})

        # 获取归一化参数
        if self.normalize:
            yaw_stats = self.statistics.get('yaw_stats', {})
            pitch_stats = self.statistics.get('pitch_stats', {})

            self.yaw_mean = yaw_stats.get('mean', 0)
            self.yaw_std = yaw_stats.get('std', 30)
            self.pitch_mean = pitch_stats.get('mean', 0)
            self.pitch_std = pitch_stats.get('std', 15)

        print(f"加载 {self.split} 集: {len(self.samples)} 样本")
        print(f"  - Normal: {self.statistics.get('normal_samples', 'N/A')}")
        print(f"  - Suspicious: {self.statistics.get('suspicious_samples', 'N/A')}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # 获取姿态序列
        pose_seq = np.array(sample['pose_sequence'], dtype=np.float32)

        # 处理序列长度
        if len(pose_seq) > self.seq_len:
            # 截断 - 取中间部分
            start = (len(pose_seq) - self.seq_len) // 2
            pose_seq = pose_seq[start:start + self.seq_len]
        elif len(pose_seq) < self.seq_len:
            # 填充 - 重复最后一帧
            pad_len = self.seq_len - len(pose_seq)
            padding = np.tile(pose_seq[-1:], (pad_len, 1))
            pose_seq = np.concatenate([pose_seq, padding], axis=0)

        # 归一化
        if self.normalize:
            pose_seq[:, 0] = (pose_seq[:, 0] - self.yaw_mean) / (self.yaw_std + 1e-6)
            pose_seq[:, 1] = (pose_seq[:, 1] - self.pitch_mean) / (self.pitch_std + 1e-6)
            pose_seq[:, 2] = pose_seq[:, 2] / 30.0  # roll 通常较小

        # 标签
        label = sample['label']

        return torch.from_numpy(pose_seq), torch.tensor(label, dtype=torch.long)

    def get_labels(self) -> List[int]:
        """返回所有标签 (用于采样器)"""
        return [s['label'] for s in self.samples]

    def get_class_counts(self) -> List[int]:
        """返回各类别样本数"""
        labels = self.get_labels()
        return [labels.count(c) for c in range(len(self.class_names))]

    def get_class_weights(self) -> torch.Tensor:
        """返回类别权重 (用于不平衡处理)"""
        counts = self.get_class_counts()
        total = sum(counts)
        weights = [total / (len(counts) * c) for c in counts]
        return torch.tensor(weights, dtype=torch.float32)


def load_datasets(
    data_path: str = 'data/dataset',
    seq_len: int = 32,
) -> Tuple[BehaviorRecognitionDataset, BehaviorRecognitionDataset, BehaviorRecognitionDataset]:
    """
    加载训练/验证/测试数据集

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = BehaviorRecognitionDataset(data_path, 'train', seq_len)
    val_dataset = BehaviorRecognitionDataset(data_path, 'val', seq_len)
    test_dataset = BehaviorRecognitionDataset(data_path, 'test', seq_len)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    # 测试数据集加载
    print("测试数据集加载...")

    dataset = BehaviorRecognitionDataset('data/dataset', 'train')

    print(f"\n样本数: {len(dataset)}")
    print(f"类别分布: {dataset.get_class_counts()}")
    print(f"类别权重: {dataset.get_class_weights()}")

    # 测试获取样本
    pose, label = dataset[0]
    print(f"\n样本形状: {pose.shape}")
    print(f"标签: {label}")
    print(f"姿态范围: yaw=[{pose[:, 0].min():.2f}, {pose[:, 0].max():.2f}]")
