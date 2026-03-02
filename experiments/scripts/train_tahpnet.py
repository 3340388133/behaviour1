#!/usr/bin/env python3
"""
TAHPNet 训练脚本

训练流程：
1. 从跟踪裁剪图像中提取头部区域
2. 使用 WHENet 生成伪标签（或真实标签）
3. 训练 TAHPNet 模型

创新点验证实验：
- A1: 基线 (无时序模块)
- A2: 完整 TAHPNet (有时序模块)
- A3: 消融实验 (不同时序模块配置)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.recognition.models.tahpnet import TAHPNet, TAHPNetLoss, create_tahpnet


# ============== 数据集 ==============
class HeadPoseSequenceDataset(Dataset):
    """头部姿态序列数据集

    从跟踪裁剪图像中加载头部图像序列
    """

    def __init__(
        self,
        data_root: str,
        pose_labels_path: str,
        seq_len: int = 16,
        stride: int = 8,
        head_ratio: float = 0.4,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        mode: str = 'train',
    ):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.head_ratio = head_ratio
        self.image_size = image_size
        self.augment = augment and mode == 'train'
        self.mode = mode

        # 加载姿态标签
        with open(pose_labels_path, 'r') as f:
            self.pose_data = json.load(f)

        # 构建样本列表
        self.samples = self._build_samples()
        print(f"[{mode}] 加载 {len(self.samples)} 个序列样本")

    def _build_samples(self) -> List[Dict]:
        """构建序列样本"""
        samples = []

        for video_name, video_data in self.pose_data.items():
            # 跳过 summary 键
            if video_name == 'summary':
                continue
            tracks = video_data.get('tracks', {})

            for track_id, track_data in tracks.items():
                poses = track_data.get('poses', [])
                if len(poses) < self.seq_len:
                    continue

                # 滑动窗口生成样本
                for start_idx in range(0, len(poses) - self.seq_len + 1, self.stride):
                    end_idx = start_idx + self.seq_len
                    sample = {
                        'video': video_name,
                        'track_id': track_id,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'poses': poses[start_idx:end_idx],
                    }
                    samples.append(sample)

        return samples

    def _load_image(self, video: str, track_id: str, frame: int) -> Optional[np.ndarray]:
        """加载单帧图像"""
        # 构建路径
        img_path = self.data_root / video / "crops" / track_id / f"frame_{frame:06d}.jpg"

        if not img_path.exists():
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        # 裁剪头部区域
        h, w = img.shape[:2]
        head_h = int(h * self.head_ratio)
        head_img = img[:head_h, :]

        # Resize
        head_img = cv2.resize(head_img, self.image_size)

        return head_img

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """数据增强"""
        # 随机水平翻转
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)

        # 随机亮度/对比度
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # 对比度
            beta = np.random.randint(-20, 20)  # 亮度
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 随机噪声
        if np.random.random() > 0.7:
            noise = np.random.randn(*img.shape) * 10
            img = np.clip(img + noise, 0, 255).astype(np.uint8)

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        images = []
        poses = []
        valid_mask = []

        for pose_item in sample['poses']:
            frame = pose_item['frame']
            img = self._load_image(sample['video'], sample['track_id'], frame)

            if img is not None:
                if self.augment:
                    img = self._augment_image(img)
                # BGR -> RGB, HWC -> CHW, normalize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                images.append(img)
                poses.append([pose_item['yaw'], pose_item['pitch'], pose_item['roll']])
                valid_mask.append(1.0)
            else:
                # 填充零
                images.append(np.zeros((3, *self.image_size), dtype=np.float32))
                poses.append([0.0, 0.0, 0.0])
                valid_mask.append(0.0)

        images = torch.tensor(np.stack(images), dtype=torch.float32)  # [T, 3, H, W]
        poses = torch.tensor(poses, dtype=torch.float32)  # [T, 3]
        valid_mask = torch.tensor(valid_mask, dtype=torch.float32)  # [T]

        return {
            'images': images,
            'poses': poses,
            'valid_mask': valid_mask,
            'video': sample['video'],
            'track_id': sample['track_id'],
        }


# ============== 训练器 ==============
class TAHPNetTrainer:
    """TAHPNet 训练器"""

    def __init__(
        self,
        model: TAHPNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: str = 'cuda:0',
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        save_dir: str = 'checkpoints',
        use_temporal: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_temporal = use_temporal

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )

        # 损失函数
        self.criterion = TAHPNetLoss(
            pose_loss_weight=1.0,
            smoothness_weight=0.1 if use_temporal else 0.0,
            velocity_weight=0.05 if use_temporal else 0.0,
            raw_loss_weight=0.3 if use_temporal else 0.0,
        )

        self.best_val_loss = float('inf')
        self.history = {'train': [], 'val': []}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        total_pose_loss = 0
        total_smooth_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images = batch['images'].to(self.device)  # [B, T, 3, H, W]
            poses = batch['poses'].to(self.device)  # [B, T, 3]
            valid_mask = batch['valid_mask'].to(self.device)  # [B, T]

            # 前向传播
            outputs = self.model(images, use_temporal=self.use_temporal)

            # 计算损失
            losses = self.criterion(outputs, poses)

            # 应用 mask
            if valid_mask.sum() > 0:
                loss = losses['total']
            else:
                continue

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_pose_loss += losses['pose_loss'].item()
            if 'smoothness_loss' in losses:
                total_smooth_loss += losses['smoothness_loss'].item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pose': f"{losses['pose_loss'].item():.4f}",
            })

        return {
            'loss': total_loss / max(n_batches, 1),
            'pose_loss': total_pose_loss / max(n_batches, 1),
            'smooth_loss': total_smooth_loss / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_jitter = 0
        n_batches = 0

        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            poses = batch['poses'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)

            outputs = self.model(images, use_temporal=self.use_temporal)
            losses = self.criterion(outputs, poses)

            total_loss += losses['total'].item()

            # MAE
            pred_pose = outputs['pose']
            mae = (pred_pose - poses).abs().mean().item()
            total_mae += mae

            # Jitter (帧间变化的标准差)
            if pred_pose.dim() == 3 and pred_pose.size(1) > 1:
                diff = pred_pose[:, 1:, :] - pred_pose[:, :-1, :]
                jitter = diff.std().item()
                total_jitter += jitter

            n_batches += 1

        return {
            'loss': total_loss / max(n_batches, 1),
            'mae': total_mae / max(n_batches, 1),
            'jitter': total_jitter / max(n_batches, 1),
        }

    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"开始训练 TAHPNet")
        print(f"时序模块: {'启用' if self.use_temporal else '禁用'}")
        print(f"训练样本: {len(self.train_loader.dataset)}")
        print(f"验证样本: {len(self.val_loader.dataset) if self.val_loader else 0}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.epochs + 1):
            # 训练
            train_metrics = self.train_epoch(epoch)
            self.history['train'].append(train_metrics)

            # 验证
            val_metrics = self.validate()
            self.history['val'].append(val_metrics)

            # 学习率调整
            self.scheduler.step()

            # 打印
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Pose: {train_metrics['pose_loss']:.4f}, "
                  f"Smooth: {train_metrics['smooth_loss']:.4f}")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"MAE: {val_metrics['mae']:.2f}°, "
                      f"Jitter: {val_metrics['jitter']:.2f}°")

            # 保存最佳模型
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best.pt', epoch, val_metrics)
                print(f"  >> 保存最佳模型")

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt', epoch, val_metrics)

        # 保存最终模型
        self.save_checkpoint('final.pt', self.epochs, val_metrics)

        # 保存训练历史
        history_path = self.save_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n训练完成! 最佳验证损失: {self.best_val_loss:.4f}")

    def save_checkpoint(self, name: str, epoch: int, metrics: Dict):
        """保存检查点"""
        path = self.save_dir / name
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'use_temporal': self.use_temporal,
        }, path)


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description='TAHPNet 训练')
    parser.add_argument('--data-root', type=str, default='data/tracked_output',
                        help='跟踪输出目录')
    parser.add_argument('--pose-labels', type=str, default='data/pose_output/all_poses.json',
                        help='姿态标签文件')
    parser.add_argument('--save-dir', type=str, default='checkpoints/tahpnet',
                        help='保存目录')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--no-temporal', action='store_true',
                        help='禁用时序模块（基线对比）')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['baseline', 'no_bidirectional', 'shallow_gru'],
                        help='消融实验类型')

    args = parser.parse_args()

    # 创建模型
    use_temporal = not args.no_temporal
    model_kwargs = {
        'hidden_dim': 64,
        'temporal_layers': 2,
        'bidirectional': True,
    }

    # 消融实验配置
    if args.ablation == 'baseline':
        use_temporal = False
        args.save_dir = args.save_dir + '_baseline'
    elif args.ablation == 'no_bidirectional':
        model_kwargs['bidirectional'] = False
        args.save_dir = args.save_dir + '_no_bidir'
    elif args.ablation == 'shallow_gru':
        model_kwargs['temporal_layers'] = 1
        args.save_dir = args.save_dir + '_shallow'

    model = TAHPNet(**model_kwargs)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 检查数据是否存在
    if not Path(args.pose_labels).exists():
        print(f"姿态标签文件不存在: {args.pose_labels}")
        print("请先运行 step4_head_pose.py 生成姿态标签")
        return

    # 创建数据集
    train_dataset = HeadPoseSequenceDataset(
        data_root=args.data_root,
        pose_labels_path=args.pose_labels,
        seq_len=args.seq_len,
        stride=args.seq_len // 2,
        mode='train',
    )

    val_dataset = HeadPoseSequenceDataset(
        data_root=args.data_root,
        pose_labels_path=args.pose_labels,
        seq_len=args.seq_len,
        stride=args.seq_len,
        mode='val',
        augment=False,
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 训练
    trainer = TAHPNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        save_dir=args.save_dir,
        use_temporal=use_temporal,
    )

    trainer.train()


if __name__ == '__main__':
    main()
