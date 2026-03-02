#!/usr/bin/env python3
"""
创新点4: 类别不平衡自适应训练 (Class Imbalance Adaptive Training, CIAT)

整合:
1. 渐进式平衡采样
2. 自适应Focal Loss
3. 少数类Mixup增强
4. 动态学习率调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple, Any, Callable
import numpy as np
from collections import Counter
from dataclasses import dataclass
from tqdm import tqdm

from .focal_loss import AdaptiveFocalLoss, compute_class_weights
from .balanced_sampler import ProgressiveBalancedSampler, MixupCollator


@dataclass
class CIATConfig:
    """CIAT训练配置"""
    # 基础训练参数
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Focal Loss参数
    focal_gamma_max: float = 2.0
    focal_warmup_epochs: int = 10
    label_smoothing: float = 0.1

    # 采样参数
    sampler_warmup_epochs: int = 10
    final_balance_ratio: float = 0.8

    # Mixup参数
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    mixup_prob: float = 0.5

    # 学习率调度
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'plateau'
    lr_warmup_epochs: int = 5
    lr_min_ratio: float = 0.01

    # 早停
    early_stopping_patience: int = 10

    # 对比学习 (BPCL)
    use_contrastive: bool = True
    contrastive_weight: float = 0.3

    # 设备
    device: str = 'cuda'


class ClassImbalanceAdaptiveTrainer:
    """
    类别不平衡自适应训练器 (CIAT)

    Args:
        model: PyTorch模型
        config: CIAT配置
        class_counts: 各类别样本数
        contrastive_module: BPCL模块 (可选)
    """

    def __init__(
        self,
        model: nn.Module,
        config: CIATConfig,
        class_counts: List[int],
        contrastive_module: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.class_counts = torch.tensor(class_counts, dtype=torch.float)
        self.num_classes = len(class_counts)
        self.contrastive_module = contrastive_module

        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        if contrastive_module:
            self.contrastive_module = contrastive_module.to(self.device)

        # 识别少数类
        total = sum(class_counts)
        class_ratios = [c / total for c in class_counts]
        self.minority_classes = [
            i for i, ratio in enumerate(class_ratios) if ratio < 0.2
        ]

        # 初始化损失函数
        self._init_loss_functions()

        # 初始化优化器
        self._init_optimizer()

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1_macro': [],
            'val_f1_per_class': [],
            'gamma': [],
            'balance_ratio': [],
            'lr': [],
        }

        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _init_loss_functions(self):
        """初始化损失函数"""
        # 计算类别权重
        self.class_weights = compute_class_weights(
            torch.arange(self.num_classes).repeat_interleave(
                self.class_counts.long()
            ),
            self.num_classes,
            method='effective_number'
        ).to(self.device)

        # 自适应Focal Loss
        self.criterion = AdaptiveFocalLoss(
            max_gamma=self.config.focal_gamma_max,
            warmup_epochs=self.config.focal_warmup_epochs,
            alpha=self.class_weights,
            label_smoothing=self.config.label_smoothing,
        ).to(self.device)

    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs - self.config.lr_warmup_epochs,
                eta_min=self.config.learning_rate * self.config.lr_min_ratio,
            )
        elif self.config.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
            )

    def _get_lr(self, epoch: int) -> float:
        """获取当前学习率（含warmup）"""
        if epoch < self.config.lr_warmup_epochs:
            return self.config.learning_rate * (epoch + 1) / self.config.lr_warmup_epochs
        return self.optimizer.param_groups[0]['lr']

    def _warmup_lr(self, epoch: int):
        """学习率warmup"""
        if epoch < self.config.lr_warmup_epochs:
            lr = self.config.learning_rate * (epoch + 1) / self.config.lr_warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def create_dataloader(
        self,
        dataset: Dataset,
        labels: List[int],
        is_train: bool = True,
    ) -> DataLoader:
        """
        创建DataLoader

        Args:
            dataset: 数据集
            labels: 标签列表
            is_train: 是否训练集
        """
        if is_train:
            sampler = ProgressiveBalancedSampler(
                labels=labels,
                num_samples=len(labels),
                warmup_epochs=self.config.sampler_warmup_epochs,
                final_balance_ratio=self.config.final_balance_ratio,
            )
            self.train_sampler = sampler

            if self.config.use_mixup:
                collator = MixupCollator(
                    minority_classes=self.minority_classes,
                    alpha=self.config.mixup_alpha,
                    mixup_prob=self.config.mixup_prob,
                )
            else:
                collator = None

            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=collator,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        if hasattr(self, 'train_sampler'):
            self.train_sampler.set_epoch(epoch)

        # 更新自适应参数
        self.criterion.update_gamma(epoch)
        self._warmup_lr(epoch)

        total_loss = 0.0
        correct = 0
        total = 0

        contrastive_loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)

        for batch in pbar:
            # 处理Mixup输出
            if len(batch) == 4:
                features, labels_a, labels_b, lam = batch
                features = features.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device) if labels_b is not None else None
                use_mixup = labels_b is not None
            else:
                features, labels = batch[0].to(self.device), batch[1].to(self.device)
                labels_a = labels
                labels_b = None
                lam = 1.0
                use_mixup = False

            self.optimizer.zero_grad()

            # 前向传播
            if hasattr(self.model, 'forward_features'):
                # 如果模型有分离的特征提取
                feat = self.model.forward_features(features)
                logits = self.model.classifier(feat)
            else:
                outputs = self.model(features)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                    feat = outputs.get('features', None)
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                    feat = outputs[1] if len(outputs) > 1 else None
                else:
                    logits = outputs
                    feat = None

            # 计算分类损失
            if use_mixup:
                loss_a = self.criterion(logits, labels_a)
                loss_b = self.criterion(logits, labels_b)
                cls_loss = lam * loss_a + (1 - lam) * loss_b
            else:
                cls_loss = self.criterion(logits, labels_a)

            # 对比学习损失
            if self.config.use_contrastive and self.contrastive_module and feat is not None:
                cont_loss, _ = self.contrastive_module.compute_contrastive_loss(
                    feat, labels_a
                )
                loss = cls_loss + self.config.contrastive_weight * cont_loss
                contrastive_loss_sum += cont_loss.item()
            else:
                loss = cls_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 更新BPCL原型
            if self.contrastive_module and hasattr(self.contrastive_module, 'update_prototypes_momentum'):
                self.contrastive_module.update_prototypes_momentum()

            # 统计
            total_loss += loss.item()
            _, predicted = logits.max(dim=1)
            correct += (predicted == labels_a).sum().item()
            total += labels_a.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
            })

        # 更新学习率
        if epoch >= self.config.lr_warmup_epochs:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass  # 在验证后更新
            else:
                self.scheduler.step()

        return {
            'loss': total_loss / len(train_loader),
            'acc': correct / total,
            'contrastive_loss': contrastive_loss_sum / len(train_loader) if contrastive_loss_sum > 0 else 0,
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            features, labels = batch[0].to(self.device), batch[1].to(self.device)

            # 前向传播
            outputs = self.model(features)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()

            _, predicted = logits.max(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = (all_preds == all_labels).mean()

        # 计算F1分数
        f1_per_class = []
        for c in range(self.num_classes):
            tp = ((all_preds == c) & (all_labels == c)).sum()
            fp = ((all_preds == c) & (all_labels != c)).sum()
            fn = ((all_preds != c) & (all_labels == c)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_per_class.append(f1)

        f1_macro = np.mean(f1_per_class)

        return {
            'loss': total_loss / len(val_loader),
            'acc': acc,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List]:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            callbacks: 回调函数列表

        Returns:
            训练历史
        """
        print(f"Starting CIAT training on {self.device}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Class counts: {self.class_counts.tolist()}")
        print(f"  - Minority classes: {self.minority_classes}")
        print(f"  - Focal gamma: 0 -> {self.config.focal_gamma_max}")
        print(f"  - Balance ratio: 0 -> {self.config.final_balance_ratio}")

        for epoch in range(self.config.epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            if val_loader:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {'loss': 0, 'acc': 0, 'f1_macro': 0, 'f1_per_class': []}

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.history['val_f1_per_class'].append(val_metrics['f1_per_class'])
            self.history['gamma'].append(self.criterion.get_current_gamma())
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if hasattr(self, 'train_sampler'):
                self.history['balance_ratio'].append(
                    self.train_sampler._get_balance_ratio()
                )

            # 打印进度
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
            if val_loader:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
                      f"F1: {val_metrics['f1_macro']:.4f}")
                print(f"  F1 per class: {[f'{f:.3f}' for f in val_metrics['f1_per_class']]}")
            print(f"  Gamma: {self.criterion.get_current_gamma():.2f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 更新学习率 (plateau)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['f1_macro'])

            # 检查最佳模型
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_epoch = epoch
                self.patience_counter = 0
                # 保存最佳模型
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

            # 回调
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_metrics, val_metrics)

        # 恢复最佳模型
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model from epoch {self.best_epoch+1} "
                  f"(F1: {self.best_val_f1:.4f})")

        return self.history

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch,
        }, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = checkpoint['history']
        self.best_val_f1 = checkpoint['best_val_f1']
        self.best_epoch = checkpoint['best_epoch']


# 别名
CIAT = ClassImbalanceAdaptiveTrainer


if __name__ == '__main__':
    # 测试CIAT
    print("Testing Class Imbalance Adaptive Trainer (CIAT)...")

    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, num_classes=3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # 创建模拟数据集
    class DummyDataset(Dataset):
        def __init__(self, size=100, input_dim=10, imbalanced=True):
            self.size = size
            self.data = torch.randn(size, input_dim)

            if imbalanced:
                # 模拟不平衡分布
                self.labels = torch.zeros(size, dtype=torch.long)
                self.labels[:int(0.78*size)] = 1  # looking_around
                self.labels[int(0.78*size):int(0.95*size)] = 2  # unknown
                # 剩余为normal (5%)
            else:
                self.labels = torch.randint(0, 3, (size,))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # 创建数据集
    train_dataset = DummyDataset(size=200, imbalanced=True)
    val_dataset = DummyDataset(size=50, imbalanced=True)

    # 获取标签
    train_labels = [train_dataset[i][1].item() for i in range(len(train_dataset))]

    # 统计类别
    class_counts = [
        sum(1 for l in train_labels if l == c)
        for c in range(3)
    ]
    print(f"\nClass distribution: {class_counts}")

    # 创建模型和训练器
    model = SimpleModel()
    config = CIATConfig(
        epochs=5,
        batch_size=16,
        focal_warmup_epochs=2,
        sampler_warmup_epochs=2,
        use_contrastive=False,  # 简化测试
        use_mixup=False,
    )

    trainer = CIAT(model, config, class_counts)

    # 创建DataLoader
    train_loader = trainer.create_dataloader(train_dataset, train_labels, is_train=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 训练
    history = trainer.fit(train_loader, val_loader)

    print("\n训练完成!")
    print(f"最佳验证F1: {trainer.best_val_f1:.4f} (epoch {trainer.best_epoch+1})")

    print("\nAll CIAT tests passed!")
