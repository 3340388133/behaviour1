#!/usr/bin/env python3
"""
渐进式平衡采样器

支持:
1. 标准过采样/欠采样
2. 渐进式平衡 (训练初期正常采样，后期逐步平衡)
3. 类别感知采样
4. 少数类Mixup增强
"""

import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
from typing import List, Optional, Iterator, Tuple, Dict
from collections import Counter


class ProgressiveBalancedSampler(Sampler):
    """
    渐进式平衡采样器

    训练初期: 按原始分布采样 (学习多数类模式)
    训练后期: 逐步平衡采样 (关注少数类)

    Args:
        labels: 所有样本的标签
        num_samples: 每个epoch采样的总样本数
        total_epochs: 总训练epoch数
        current_epoch: 当前epoch
        warmup_epochs: 线性过渡的epoch数
        final_balance_ratio: 最终平衡程度 (0=原始分布, 1=完全平衡)
    """

    def __init__(
        self,
        labels: List[int],
        num_samples: Optional[int] = None,
        total_epochs: int = 50,
        warmup_epochs: int = 10,
        final_balance_ratio: float = 0.8,
    ):
        self.labels = np.array(labels)
        self.num_samples = num_samples if num_samples else len(labels)
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.final_balance_ratio = final_balance_ratio

        self.current_epoch = 0

        # 分析类别分布
        self.num_classes = len(np.unique(self.labels))
        self.class_indices = {
            c: np.where(self.labels == c)[0]
            for c in range(self.num_classes)
        }
        self.class_counts = {
            c: len(indices) for c, indices in self.class_indices.items()
        }

        # 原始采样权重
        total = len(labels)
        self.original_weights = np.array([
            1.0 / self.class_counts[self.labels[i]]
            for i in range(total)
        ])
        self.original_weights = self.original_weights / self.original_weights.sum()

        # 平衡采样权重
        self.balanced_weights = np.array([
            1.0 / (self.num_classes * self.class_counts[self.labels[i]])
            for i in range(total)
        ])
        self.balanced_weights = self.balanced_weights / self.balanced_weights.sum()

    def set_epoch(self, epoch: int):
        """设置当前epoch"""
        self.current_epoch = epoch

    def _get_balance_ratio(self) -> float:
        """
        获取当前的平衡比例

        0 = 原始分布
        1 = 完全平衡
        """
        if self.current_epoch < self.warmup_epochs:
            # 线性增长
            ratio = self.final_balance_ratio * (self.current_epoch / self.warmup_epochs)
        else:
            ratio = self.final_balance_ratio
        return ratio

    def _get_sample_weights(self) -> np.ndarray:
        """获取当前的采样权重"""
        ratio = self._get_balance_ratio()

        # 混合原始和平衡权重
        weights = (1 - ratio) * self.original_weights + ratio * self.balanced_weights

        return weights

    def __iter__(self) -> Iterator[int]:
        """生成采样索引"""
        weights = self._get_sample_weights()

        # 按权重采样
        indices = np.random.choice(
            len(self.labels),
            size=self.num_samples,
            replace=True,
            p=weights
        )

        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def get_class_sample_counts(self) -> Dict[int, int]:
        """
        获取预期的各类别采样数量（用于调试）
        """
        weights = self._get_sample_weights()

        expected_counts = {}
        for c in range(self.num_classes):
            class_mask = self.labels == c
            class_weight_sum = weights[class_mask].sum()
            expected_counts[c] = int(class_weight_sum * self.num_samples)

        return expected_counts


class ClassAwareSampler(Sampler):
    """
    类别感知采样器

    确保每个batch包含所有类别的样本
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        samples_per_class: Optional[int] = None,
    ):
        self.labels = np.array(labels)
        self.batch_size = batch_size

        self.num_classes = len(np.unique(self.labels))

        # 每个类别的采样数
        if samples_per_class:
            self.samples_per_class = samples_per_class
        else:
            self.samples_per_class = batch_size // self.num_classes

        # 类别索引
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in range(self.num_classes)
        }

        # 计算总样本数
        self.num_batches = len(labels) // batch_size

    def __iter__(self) -> Iterator[int]:
        indices = []

        for _ in range(self.num_batches):
            batch_indices = []

            for c in range(self.num_classes):
                class_indices = self.class_indices[c]

                # 随机采样
                sampled = np.random.choice(
                    class_indices,
                    size=min(self.samples_per_class, len(class_indices)),
                    replace=len(class_indices) < self.samples_per_class
                )
                batch_indices.extend(sampled.tolist())

            # 如果不够batch_size，随机补充
            while len(batch_indices) < self.batch_size:
                random_idx = np.random.randint(len(self.labels))
                batch_indices.append(random_idx)

            # 打乱batch内顺序
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices[:self.batch_size])

        return iter(indices)

    def __len__(self) -> int:
        return self.num_batches * self.batch_size


class MixupCollator:
    """
    Mixup数据增强Collator

    对少数类进行Mixup增强
    """

    def __init__(
        self,
        minority_classes: List[int],
        alpha: float = 0.2,
        mixup_prob: float = 0.5,
    ):
        """
        Args:
            minority_classes: 少数类ID列表
            alpha: Beta分布参数
            mixup_prob: Mixup的概率
        """
        self.minority_classes = set(minority_classes)
        self.alpha = alpha
        self.mixup_prob = mixup_prob

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        处理batch

        Args:
            batch: [(features, label), ...]

        Returns:
            (mixed_features, labels_a, labels_b, lam)
            如果没有mixup: labels_b = None, lam = 1.0
        """
        features = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        batch_size = len(batch)

        # 判断是否进行mixup
        if np.random.random() > self.mixup_prob:
            return features, labels, None, 1.0

        # 找到少数类样本
        minority_mask = torch.tensor([
            label.item() in self.minority_classes
            for label in labels
        ])

        if minority_mask.sum() == 0:
            return features, labels, None, 1.0

        # 对少数类样本进行mixup
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # 确保lam >= 0.5

        # 随机打乱
        perm = torch.randperm(batch_size)

        mixed_features = lam * features + (1 - lam) * features[perm]
        labels_b = labels[perm]

        return mixed_features, labels, labels_b, lam


class BalancedBatchSampler(Sampler):
    """
    平衡Batch采样器

    每个batch尽量保持类别平衡
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = True,
    ):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.num_classes = len(np.unique(self.labels))

        # 类别索引
        self.class_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in range(self.num_classes)
        }

        # 每个类别每batch的样本数
        self.samples_per_class = batch_size // self.num_classes

    def __iter__(self) -> Iterator[List[int]]:
        # 复制并打乱每个类的索引
        class_iterators = {
            c: iter(np.random.permutation(indices).tolist())
            for c, indices in self.class_indices.items()
        }

        # 追踪剩余样本
        remaining = {c: list(indices) for c, indices in self.class_indices.items()}
        for c in remaining:
            np.random.shuffle(remaining[c])

        while True:
            batch = []

            for c in range(self.num_classes):
                for _ in range(self.samples_per_class):
                    if len(remaining[c]) == 0:
                        # 重新填充
                        remaining[c] = list(self.class_indices[c])
                        np.random.shuffle(remaining[c])

                    idx = remaining[c].pop()
                    batch.append(idx)

            # 补充到batch_size
            while len(batch) < self.batch_size:
                c = np.random.randint(self.num_classes)
                if len(remaining[c]) == 0:
                    remaining[c] = list(self.class_indices[c])
                    np.random.shuffle(remaining[c])
                batch.append(remaining[c].pop())

            np.random.shuffle(batch)
            yield batch

            # 检查是否完成一个epoch
            total_remaining = sum(len(v) for v in remaining.values())
            if total_remaining < self.batch_size and self.drop_last:
                break

    def __len__(self) -> int:
        total = len(self.labels)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    # 测试采样器
    print("Testing Balanced Samplers...")

    # 创建不平衡标签
    labels = [1] * 78 + [2] * 17 + [0] * 5  # normal:5%, looking_around:78%, unknown:17%

    # 测试渐进式平衡采样器
    print("\n1. Progressive Balanced Sampler:")
    sampler = ProgressiveBalancedSampler(
        labels=labels,
        num_samples=100,
        warmup_epochs=10,
        final_balance_ratio=0.8,
    )

    for epoch in [0, 5, 10]:
        sampler.set_epoch(epoch)
        indices = list(sampler)
        counts = Counter([labels[i] for i in indices])
        print(f"   Epoch {epoch}: balance={sampler._get_balance_ratio():.2f}, counts={dict(counts)}")

    # 测试类别感知采样器
    print("\n2. Class Aware Sampler:")
    class_sampler = ClassAwareSampler(labels=labels, batch_size=12)
    batch_indices = list(class_sampler)[:24]  # 取两个batch
    batch1_labels = [labels[i] for i in batch_indices[:12]]
    batch2_labels = [labels[i] for i in batch_indices[12:24]]
    print(f"   Batch 1: {Counter(batch1_labels)}")
    print(f"   Batch 2: {Counter(batch2_labels)}")

    # 测试Mixup Collator
    print("\n3. Mixup Collator:")
    collator = MixupCollator(minority_classes=[0], alpha=0.2, mixup_prob=1.0)

    # 模拟batch
    batch = [
        (torch.randn(10), torch.tensor(0)),  # 少数类
        (torch.randn(10), torch.tensor(1)),
        (torch.randn(10), torch.tensor(1)),
        (torch.randn(10), torch.tensor(2)),
    ]
    mixed, labels_a, labels_b, lam = collator(batch)
    print(f"   Lambda: {lam:.3f}")
    print(f"   Labels A: {labels_a}")
    print(f"   Labels B: {labels_b}")

    # 测试平衡Batch采样器
    print("\n4. Balanced Batch Sampler:")
    batch_sampler = BalancedBatchSampler(labels=labels, batch_size=12)
    batches = [next(iter(batch_sampler)) for _ in range(3)]
    for i, batch in enumerate(batches):
        counts = Counter([labels[idx] for idx in batch])
        print(f"   Batch {i+1}: {dict(counts)}")

    print("\nAll Sampler tests passed!")
