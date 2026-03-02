#!/usr/bin/env python3
"""
6类行为识别训练脚本 - 使用类别平衡技术

解决类别不平衡问题:
- Focal Loss (关注难样本)
- 过采样少数类
- 类别权重
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))

import json
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse

from temporal_transformer import create_model


# 6类行为
CLASS_NAMES = ['normal', 'glancing', 'quick_turn', 'prolonged_watch', 'looking_down', 'looking_up']


class FocalLoss(nn.Module):
    """Focal Loss - 关注难分类样本"""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # 类别权重
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BalancedBehaviorDataset(Dataset):
    """支持过采样的平衡数据集"""

    def __init__(self, data_path: str, split: str = 'train', seq_len: int = 90, oversample: bool = False):
        self.seq_len = seq_len

        with open(f'{data_path}/{split}.json', 'r') as f:
            data = json.load(f)

        self.samples = data['samples']

        if oversample and split == 'train':
            self._oversample()

        print(f"[{split}] 加载 {len(self.samples)} 样本")
        labels = [s['label'] for s in self.samples]
        print(f"  类别分布: {dict(Counter(sorted(labels)))}")

    def _oversample(self):
        """对少数类过采样到最大类的50%"""
        labels = [s['label'] for s in self.samples]
        counter = Counter(labels)
        max_count = max(counter.values())
        target_count = max_count // 2  # 至少达到最大类的50%

        new_samples = []
        for label in range(6):
            class_samples = [s for s in self.samples if s['label'] == label]
            if len(class_samples) == 0:
                continue

            if len(class_samples) < target_count:
                # 过采样
                ratio = target_count // len(class_samples)
                remainder = target_count % len(class_samples)
                oversampled = class_samples * ratio + class_samples[:remainder]
                new_samples.extend(oversampled)
            else:
                new_samples.extend(class_samples)

        np.random.shuffle(new_samples)
        self.samples = new_samples
        print(f"  过采样后: {len(self.samples)} 样本")
        labels = [s['label'] for s in self.samples]
        print(f"  新分布: {dict(Counter(sorted(labels)))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pose_seq = np.array(sample['pose_sequence'], dtype=np.float32)

        # 处理序列长度
        if len(pose_seq) > self.seq_len:
            start = (len(pose_seq) - self.seq_len) // 2
            pose_seq = pose_seq[start:start + self.seq_len]
        elif len(pose_seq) < self.seq_len:
            pad_len = self.seq_len - len(pose_seq)
            padding = np.tile(pose_seq[-1:], (pad_len, 1))
            pose_seq = np.concatenate([pose_seq, padding], axis=0)

        label = sample['label']
        return torch.from_numpy(pose_seq), torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s['label'] for s in self.samples]


def compute_class_weights(labels, num_classes=6):
    """计算类别权重 (反比例)"""
    counter = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)
        # 反比例权重
        weight = total / (num_classes * count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device, num_classes=6):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for pose_seq, labels in loader:
            pose_seq, labels = pose_seq.to(device), labels.to(device)
            logits, _ = model(pose_seq)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=range(num_classes))

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_f1': per_class_f1.tolist()
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for pose_seq, labels in pbar:
        pose_seq, labels = pose_seq.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(pose_seq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=f'{100*correct/total:.1f}%')

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description='6类行为识别训练(平衡版)')
    parser.add_argument('--data-path', default='data/dataset', help='数据路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--seq-length', type=int, default=90, help='序列长度')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal Loss gamma')
    parser.add_argument('--oversample', action='store_true', help='过采样少数类')
    parser.add_argument('--weighted-sampler', action='store_true', help='使用加权采样器')
    parser.add_argument('--output', default='checkpoints/transformer_balanced.pt', help='输出路径')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载数据
    print("\n=== 加载数据 ===")
    train_dataset = BalancedBehaviorDataset(args.data_path, 'train', args.seq_length, oversample=args.oversample)
    val_dataset = BalancedBehaviorDataset(args.data_path, 'val', args.seq_length, oversample=False)
    test_dataset = BalancedBehaviorDataset(args.data_path, 'test', args.seq_length, oversample=False)

    # 计算类别权重
    train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(train_labels, num_classes=6).to(device)
    print(f"\n类别权重: {class_weights.tolist()}")

    # 创建数据加载器
    if args.weighted_sampler:
        # 加权采样器 - 让少数类更容易被采样
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    print("\n=== 创建模型 ===")
    model = create_model(
        model_type='transformer',
        pose_input_dim=3,
        pose_d_model=64,
        pose_nhead=4,
        pose_num_layers=2,
        use_multimodal=False,
        hidden_dim=128,
        num_classes=6,
        dropout=0.1,
        uncertainty_weighting=True,
    ).to(device)

    # Focal Loss
    criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    print("\n=== 开始训练 ===")
    best_f1 = 0
    best_state = None
    patience = 0
    max_patience = 15

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Train: loss={train_loss:.4f}, acc={train_acc*100:.1f}%")
        print(f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']*100:.1f}%, "
              f"P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f}, F1={val_metrics['f1']:.3f}")
        print(f"Per-class F1: {[f'{f:.2f}' for f in val_metrics['per_class_f1']]}")

        # 保存最佳模型
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = model.state_dict().copy()
            patience = 0
            print(f"  *** 新最佳 F1: {best_f1:.3f}")
        else:
            patience += 1

        # 早停
        if patience >= max_patience:
            print(f"\n早停于 epoch {epoch+1}")
            break

    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)

    # 测试
    print("\n=== 测试集评估 ===")
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test: acc={test_metrics['accuracy']*100:.1f}%, "
          f"P={test_metrics['precision']:.3f}, R={test_metrics['recall']:.3f}, F1={test_metrics['f1']:.3f}")
    print(f"Per-class F1: {[f'{f:.2f}' for f in test_metrics['per_class_f1']]}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: F1={test_metrics['per_class_f1'][i]:.3f}")

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'config': vars(args),
    }, args.output)
    print(f"\n模型已保存: {args.output}")


if __name__ == '__main__':
    main()
