#!/usr/bin/env python3
"""
使用真实数据训练SBRN模型

使用方法:
    python experiments/scripts/train_real_data.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm

from recognition import create_sbrn, BPCL
from recognition.dataset import BehaviorRecognitionDataset
from recognition.training import AdaptiveFocalLoss, ProgressiveBalancedSampler


def train():
    """训练SBRN模型"""
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 1. 加载数据 ==========
    print("\n" + "=" * 50)
    print("加载数据集...")
    print("=" * 50)

    data_path = project_root / 'data' / 'dataset'

    train_dataset = BehaviorRecognitionDataset(data_path, 'train', seq_len=32)
    val_dataset = BehaviorRecognitionDataset(data_path, 'val', seq_len=32)

    class_counts = train_dataset.get_class_counts()
    class_weights = train_dataset.get_class_weights()
    print(f"\n类别分布: {class_counts}")
    print(f"类别权重: {class_weights}")

    # 创建采样器 (渐进式平衡)
    train_labels = train_dataset.get_labels()
    sampler = ProgressiveBalancedSampler(
        labels=train_labels,
        num_samples=len(train_labels),
        warmup_epochs=5,
        final_balance_ratio=0.6,
    )

    # DataLoader
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ========== 2. 创建模型 ==========
    print("\n" + "=" * 50)
    print("创建SBRN模型...")
    print("=" * 50)

    # 二分类任务
    model = create_sbrn(
        num_classes=2,
        use_multimodal=False,  # 只用姿态
        use_contrastive=True,  # 启用BPCL
        d_model=128,
        num_layers=4,
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params:,}")

    # ========== 3. 损失函数和优化器 ==========
    # 自适应Focal Loss
    criterion = AdaptiveFocalLoss(
        max_gamma=2.0,
        warmup_epochs=5,
        alpha=class_weights.to(device),
        label_smoothing=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=30,
        eta_min=1e-5,
    )

    # ========== 4. 训练循环 ==========
    print("\n" + "=" * 50)
    print("开始训练...")
    print("=" * 50)

    epochs = 30
    best_val_f1 = 0
    best_epoch = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_recall_suspicious': [],
    }

    for epoch in range(epochs):
        # 更新自适应参数
        sampler.set_epoch(epoch)
        criterion.update_gamma(epoch)

        # ---- 训练 ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for pose, labels in pbar:
            pose = pose.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(pose, return_features=True)
            logits = outputs['logits']

            # 分类损失
            cls_loss = criterion(logits, labels)

            # 对比学习损失
            if hasattr(model, 'bpcl') and 'features' in outputs:
                cont_loss, _ = model.bpcl.compute_contrastive_loss(
                    outputs['features'], labels
                )
                loss = cls_loss + 0.3 * cont_loss
            else:
                loss = cls_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # 更新BPCL原型
            if hasattr(model, 'bpcl'):
                model.bpcl.update_prototypes_momentum()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.1f}%',
            })

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # ---- 验证 ----
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for pose, labels in val_loader:
                pose = pose.to(device)
                labels = labels.to(device)

                outputs = model(pose)
                logits = outputs['logits']
                loss = F.cross_entropy(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        val_acc = (all_preds == all_labels).mean()

        # 计算F1和Recall
        tp = ((all_preds == 1) & (all_labels == 1)).sum()
        fp = ((all_preds == 1) & (all_labels == 0)).sum()
        fn = ((all_preds == 0) & (all_labels == 1)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall_suspicious = tp / (tp + fn + 1e-8)
        f1_suspicious = 2 * precision * recall_suspicious / (precision + recall_suspicious + 1e-8)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(f1_suspicious)
        history['val_recall_suspicious'].append(recall_suspicious)

        # 打印结果
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Suspicious - F1: {f1_suspicious:.4f}, Recall: {recall_suspicious:.4f}, Precision: {precision:.4f}")
        print(f"  Gamma: {criterion.get_current_gamma():.2f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if f1_suspicious > best_val_f1:
            best_val_f1 = f1_suspicious
            best_epoch = epoch + 1
            save_path = project_root / 'models' / 'sbrn_best.pt'
            save_path.parent.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': f1_suspicious,
                'history': history,
            }, save_path)
            print(f"  >> 保存最佳模型 (F1: {f1_suspicious:.4f})")

    # ========== 5. 训练完成 ==========
    print("\n" + "=" * 50)
    print("训练完成!")
    print("=" * 50)
    print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    print(f"模型保存至: {save_path}")

    return history


if __name__ == '__main__':
    train()
