#!/usr/bin/env python3
"""
SBRN训练脚本

使用方法:
    python experiments/scripts/train_sbrn.py --config experiments/configs/full_model.yaml
    python experiments/scripts/train_sbrn.py --config experiments/configs/baseline.yaml
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition import (
    SBRN, SBRNConfig, create_sbrn,
    BPCL, CIAT, CIATConfig,
    AdaptiveFocalLoss, ProgressiveBalancedSampler,
)


class BehaviorDataset(Dataset):
    """行为识别数据集"""

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 32,
        split: str = 'train',
        use_appearance: bool = False,
        use_motion: bool = False,
    ):
        """
        Args:
            data_dir: 数据目录
            seq_len: 序列长度
            split: 'train' or 'val'
            use_appearance: 是否加载外观特征
            use_motion: 是否加载运动特征
        """
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.use_appearance = use_appearance
        self.use_motion = use_motion

        # 加载数据
        self.samples = self._load_data()

        # 类别名称
        self.class_names = ['normal', 'looking_around', 'unknown']

    def _load_data(self) -> List[Dict]:
        """加载数据"""
        samples = []

        # 遍历所有跟踪结果
        tracked_dir = self.data_dir

        if not tracked_dir.exists():
            print(f"Warning: Data directory {tracked_dir} not found. Using dummy data.")
            return self._create_dummy_data()

        for video_dir in tracked_dir.iterdir():
            if not video_dir.is_dir():
                continue

            tracking_json = video_dir / 'tracking_result.json'
            if not tracking_json.exists():
                continue

            with open(tracking_json, 'r') as f:
                tracking_data = json.load(f)

            for track in tracking_data.get('tracks', []):
                track_id = track['track_id']
                poses = track.get('head_poses', [])

                if len(poses) < self.seq_len:
                    continue

                # 创建序列样本
                for i in range(0, len(poses) - self.seq_len + 1, self.seq_len // 2):
                    seq = poses[i:i + self.seq_len]
                    pose_array = np.array([
                        [p.get('yaw', 0), p.get('pitch', 0), p.get('roll', 0)]
                        for p in seq
                    ], dtype=np.float32)

                    # 获取标签
                    label = track.get('behavior_label', 1)  # 默认looking_around

                    samples.append({
                        'video': video_dir.name,
                        'track_id': track_id,
                        'pose': pose_array,
                        'label': label,
                    })

        if len(samples) == 0:
            print("Warning: No data found. Using dummy data.")
            return self._create_dummy_data()

        return samples

    def _create_dummy_data(self, n_samples: int = 200) -> List[Dict]:
        """创建模拟数据用于测试"""
        samples = []
        np.random.seed(42)

        for i in range(n_samples):
            # 模拟不平衡分布
            if i < int(0.05 * n_samples):
                label = 0  # normal (5%)
            elif i < int(0.83 * n_samples):
                label = 1  # looking_around (78%)
            else:
                label = 2  # unknown (17%)

            # 根据标签生成不同模式的姿态
            if label == 0:  # normal - 稳定
                pose = np.random.randn(self.seq_len, 3) * 5
            elif label == 1:  # looking_around - 大幅度变化
                pose = np.cumsum(np.random.randn(self.seq_len, 3) * 10, axis=0)
                pose[:, 0] += np.sin(np.linspace(0, 4 * np.pi, self.seq_len)) * 30
            else:  # unknown - 随机
                pose = np.random.randn(self.seq_len, 3) * 20

            samples.append({
                'video': 'dummy',
                'track_id': i,
                'pose': pose.astype(np.float32),
                'label': label,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        pose = torch.from_numpy(sample['pose'])
        label = torch.tensor(sample['label'], dtype=torch.long)

        return pose, label

    def get_labels(self) -> List[int]:
        """返回所有标签"""
        return [s['label'] for s in self.samples]

    def get_class_counts(self) -> List[int]:
        """返回各类别样本数"""
        labels = self.get_labels()
        return [labels.count(c) for c in range(3)]


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict) -> SBRN:
    """根据配置创建模型"""
    model_config = config.get('model', {})

    sbrn_config = SBRNConfig(
        pose_input_dim=model_config.get('pose_input_dim', 3),
        appearance_dim=model_config.get('appearance_dim', 512),
        motion_dim=model_config.get('motion_dim', 64),
        d_model=model_config.get('d_model', 128),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 4),
        dim_feedforward=model_config.get('dim_feedforward', 512),
        dropout=model_config.get('dropout', 0.1),
        num_classes=model_config.get('num_classes', 3),
        hidden_dim=model_config.get('hidden_dim', 128),
        max_seq_len=model_config.get('max_seq_len', 512),
        periods=model_config.get('periods'),
        use_relative_bias=model_config.get('use_relative_bias', True),
        use_multimodal=model_config.get('use_multimodal', False),
        use_quality_estimation=model_config.get('use_quality_estimation', True),
        use_contrastive=model_config.get('use_contrastive', True),
        num_prototypes_per_class=model_config.get('num_prototypes_per_class', 3),
        temperature=model_config.get('temperature', 0.07),
        contrastive_margin=model_config.get('contrastive_margin', 0.5),
        uncertainty_weighting=model_config.get('uncertainty_weighting', True),
    )

    return SBRN(sbrn_config)


def train(config: Dict, args: argparse.Namespace):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    log_dir = Path(config.get('logging', {}).get('log_dir', 'experiments/logs'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = log_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # 创建TensorBoard writer
    writer = SummaryWriter(run_dir / 'tensorboard')

    # 加载数据
    data_config = config.get('data', {})
    train_dataset = BehaviorDataset(
        data_dir=data_config.get('train_path', 'data/tracked_output'),
        seq_len=data_config.get('seq_len', 32),
        split='train',
        use_appearance=data_config.get('use_appearance', False),
        use_motion=data_config.get('use_motion', False),
    )

    # 划分训练/验证集
    val_split = data_config.get('val_split', 0.2)
    n_val = int(len(train_dataset) * val_split)
    n_train = len(train_dataset) - n_val
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # 获取标签
    train_labels = [train_dataset.samples[i]['label'] for i in train_subset.indices]
    class_counts = train_dataset.get_class_counts()

    print(f"\nDataset info:")
    print(f"  Total samples: {len(train_dataset)}")
    print(f"  Train samples: {n_train}")
    print(f"  Val samples: {n_val}")
    print(f"  Class distribution: {class_counts}")

    # 创建模型
    model = create_model_from_config(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    # 训练配置
    train_config = config.get('training', {})

    # 使用CIAT训练器
    if train_config.get('use_balanced_sampler', False) or train_config.get('use_focal_loss', False):
        # 使用CIAT
        ciat_config = CIATConfig(
            epochs=train_config.get('epochs', 50),
            batch_size=train_config.get('batch_size', 32),
            learning_rate=train_config.get('learning_rate', 1e-3),
            weight_decay=train_config.get('weight_decay', 1e-4),
            focal_gamma_max=train_config.get('focal_gamma_max', 2.0),
            focal_warmup_epochs=train_config.get('focal_warmup_epochs', 10),
            label_smoothing=train_config.get('label_smoothing', 0.1),
            sampler_warmup_epochs=train_config.get('sampler_warmup_epochs', 10),
            final_balance_ratio=train_config.get('final_balance_ratio', 0.8),
            use_mixup=train_config.get('use_mixup', False),
            mixup_alpha=train_config.get('mixup_alpha', 0.2),
            mixup_prob=train_config.get('mixup_prob', 0.5),
            lr_scheduler=train_config.get('lr_scheduler', 'cosine'),
            lr_warmup_epochs=train_config.get('lr_warmup_epochs', 5),
            lr_min_ratio=train_config.get('lr_min_ratio', 0.01),
            early_stopping_patience=train_config.get('early_stopping_patience', 10),
            use_contrastive=config.get('model', {}).get('use_contrastive', True),
            contrastive_weight=train_config.get('contrastive_weight', 0.3),
            device=str(device),
        )

        # 获取BPCL模块
        bpcl_module = model.bpcl if hasattr(model, 'bpcl') else None

        trainer = CIAT(model, ciat_config, class_counts, bpcl_module)

        # 创建DataLoader
        train_loader = trainer.create_dataloader(train_subset, train_labels, is_train=True)
        val_loader = DataLoader(val_subset, batch_size=ciat_config.batch_size, shuffle=False)

        # 训练
        history = trainer.fit(train_loader, val_loader)

    else:
        # 标准训练循环
        train_loader = DataLoader(
            train_subset,
            batch_size=train_config.get('batch_size', 32),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=train_config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.get('learning_rate', 1e-3),
            weight_decay=train_config.get('weight_decay', 1e-4),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_config.get('epochs', 50),
        )

        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        best_val_f1 = 0

        for epoch in range(train_config.get('epochs', 50)):
            # 训练
            model.train()
            train_loss = 0

            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
                pose, labels = batch
                pose = pose.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(pose, return_features=True)
                loss, _ = model.compute_loss(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    pose, labels = batch
                    pose = pose.to(device)
                    labels = labels.to(device)

                    outputs = model(pose, return_features=True)
                    loss, _ = model.compute_loss(outputs, labels)
                    val_loss += loss.item()

                    preds = outputs['logits'].argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

            # 计算F1
            from sklearn.metrics import f1_score
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            scheduler.step()

            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('F1/val', val_f1, epoch)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), run_dir / 'best_model.pt')

    # 保存最终模型
    torch.save(model.state_dict(), run_dir / 'final_model.pt')

    # 保存训练历史
    with open(run_dir / 'history.json', 'w') as f:
        # 转换numpy数组为列表
        history_serializable = {}
        for k, v in history.items():
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], np.ndarray):
                    history_serializable[k] = [x.tolist() for x in v]
                elif isinstance(v[0], (np.float32, np.float64)):
                    history_serializable[k] = [float(x) for x in v]
                else:
                    history_serializable[k] = v
            else:
                history_serializable[k] = v
        json.dump(history_serializable, f, indent=2)

    writer.close()
    print(f"\nTraining complete! Results saved to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train SBRN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args)


if __name__ == '__main__':
    main()
