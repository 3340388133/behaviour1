#!/usr/bin/env python3
"""
自适应Focal Loss

支持:
1. 标准Focal Loss
2. 自适应gamma (随训练进度调整)
3. 类别加权
4. Label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter, higher = more focus on hard examples
        alpha: class weights, [num_classes] or scalar
        reduction: 'none', 'mean', 'sum'
        label_smoothing: label smoothing epsilon
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes] 未经softmax的logits
            targets: [batch] 类别标签

        Returns:
            loss: scalar or [batch] depending on reduction
        """
        num_classes = logits.shape[-1]
        batch_size = logits.shape[0]
        device = logits.device

        # Label smoothing
        if self.label_smoothing > 0:
            # 将hard labels转换为soft labels
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = (
                targets_one_hot * (1 - self.label_smoothing) +
                self.label_smoothing / num_classes
            )
        else:
            targets_one_hot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_one_hot

        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类别的概率 p_t
        p_t = (probs * targets_one_hot).sum(dim=-1)  # [batch]

        # Focal权重: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # 计算CE loss (使用log_softmax提高数值稳定性)
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(targets_smooth * log_probs).sum(dim=-1)  # [batch]

        # 应用focal权重
        focal_loss = focal_weight * ce_loss

        # 应用类别权重
        if self.alpha is not None:
            alpha = self.alpha.to(device)
            alpha_t = alpha[targets]  # [batch]
            focal_loss = alpha_t * focal_loss

        # Reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class AdaptiveFocalLoss(nn.Module):
    """
    自适应Focal Loss

    gamma随训练进度从0渐增到max_gamma
    - 训练初期: gamma小，关注所有样本
    - 训练后期: gamma大，关注困难样本

    Args:
        max_gamma: 最大gamma值
        warmup_epochs: gamma线性增长的epoch数
        alpha: 类别权重
        label_smoothing: label smoothing epsilon
    """

    def __init__(
        self,
        max_gamma: float = 2.0,
        warmup_epochs: int = 10,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        reduction: str = 'mean',
    ):
        super().__init__()

        self.max_gamma = max_gamma
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

        # 当前gamma (由外部训练循环更新)
        self.register_buffer('current_gamma', torch.tensor(0.0))
        self.register_buffer('current_epoch', torch.tensor(0))

    def update_gamma(self, epoch: int):
        """
        更新gamma值

        Args:
            epoch: 当前epoch (0-indexed)
        """
        self.current_epoch = torch.tensor(epoch, device=self.current_gamma.device)

        if epoch < self.warmup_epochs:
            # 线性增长
            gamma = self.max_gamma * (epoch / self.warmup_epochs)
        else:
            gamma = self.max_gamma

        self.current_gamma = torch.tensor(gamma, device=self.current_gamma.device)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
        """
        gamma = self.current_gamma.item()
        focal_loss = FocalLoss(
            gamma=gamma,
            alpha=self.alpha,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        return focal_loss(logits, targets)

    def get_current_gamma(self) -> float:
        """获取当前gamma值"""
        return self.current_gamma.item()


class ClassBalancedFocalLoss(nn.Module):
    """
    基于类别频率的平衡Focal Loss

    自动根据类别分布计算权重:
    alpha_c = (1 - beta) / (1 - beta^n_c)
    where n_c is the number of samples in class c

    Args:
        beta: 平衡因子, 建议 0.9999 for highly imbalanced data
        gamma: focal loss gamma
    """

    def __init__(
        self,
        class_counts: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.gamma = gamma
        self.label_smoothing = label_smoothing

        # 计算有效样本数
        effective_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / effective_num

        # 归一化
        weights = weights / weights.sum() * len(class_counts)

        self.register_buffer('alpha', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, num_classes]
            targets: [batch]
        """
        focal_loss = FocalLoss(
            gamma=self.gamma,
            alpha=self.alpha,
            reduction='mean',
            label_smoothing=self.label_smoothing,
        )
        return focal_loss(logits, targets)


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    method: str = 'inverse',
) -> torch.Tensor:
    """
    计算类别权重

    Args:
        labels: [N] 所有标签
        num_classes: 类别数
        method: 'inverse', 'sqrt_inverse', 'effective_number'

    Returns:
        [num_classes] 权重
    """
    # 统计每个类别的样本数
    counts = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    # 避免除零
    counts = counts.clamp(min=1)

    if method == 'inverse':
        # 反比例权重
        weights = 1.0 / counts
    elif method == 'sqrt_inverse':
        # 平方根反比例 (更温和)
        weights = 1.0 / torch.sqrt(counts)
    elif method == 'effective_number':
        # 有效样本数
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # 归一化
    weights = weights / weights.sum() * num_classes

    return weights


if __name__ == '__main__':
    # 测试Focal Loss
    print("Testing Focal Loss...")

    batch_size = 16
    num_classes = 3

    # 创建测试数据
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # 模拟不平衡分布
    targets[:12] = 1  # 多数类
    targets[12:15] = 2
    targets[15:] = 0  # 少数类

    # 测试标准Focal Loss
    print("\n1. Standard Focal Loss:")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    # 测试带权重的Focal Loss
    print("\n2. Weighted Focal Loss:")
    alpha = torch.tensor([3.0, 0.5, 1.0])  # 少数类更高权重
    focal_loss_weighted = FocalLoss(gamma=2.0, alpha=alpha)
    loss = focal_loss_weighted(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    # 测试自适应Focal Loss
    print("\n3. Adaptive Focal Loss:")
    adaptive_focal = AdaptiveFocalLoss(max_gamma=2.0, warmup_epochs=10)

    for epoch in [0, 5, 10, 15]:
        adaptive_focal.update_gamma(epoch)
        loss = adaptive_focal(logits, targets)
        print(f"   Epoch {epoch}: gamma={adaptive_focal.get_current_gamma():.2f}, loss={loss.item():.4f}")

    # 测试类别平衡Focal Loss
    print("\n4. Class Balanced Focal Loss:")
    class_counts = torch.tensor([5.0, 78.0, 17.0])  # 模拟不平衡
    cb_focal = ClassBalancedFocalLoss(class_counts, gamma=2.0)
    loss = cb_focal(logits, targets)
    print(f"   Class weights: {cb_focal.alpha}")
    print(f"   Loss: {loss.item():.4f}")

    # 测试权重计算
    print("\n5. Class weight computation:")
    all_labels = torch.tensor([1]*78 + [2]*17 + [0]*5)  # 模拟分布
    for method in ['inverse', 'sqrt_inverse', 'effective_number']:
        weights = compute_class_weights(all_labels, num_classes, method)
        print(f"   {method}: {weights}")

    print("\nAll Focal Loss tests passed!")
