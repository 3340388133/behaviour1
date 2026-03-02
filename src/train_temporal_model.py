"""
时序模型完整训练流程
- 视频级数据划分（避免数据泄露）
- 类别不平衡处理（加权损失 + 过采样）
- 完整评估指标（Precision / Recall / F1 / AUC）
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import json

from temporal_model import (
    BehaviorGRU, BehaviorDataset, ModelConfig, TemporalModelTrainer
)


class BalancedBehaviorDataset(BehaviorDataset):
    """支持过采样的平衡数据集"""

    def __init__(self, *args, oversample: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample = oversample
        if oversample:
            self._oversample()

    def _oversample(self):
        """对少数类进行过采样"""
        pos_samples = [s for s in self.samples if s['label'] == 1]
        neg_samples = [s for s in self.samples if s['label'] == 0]

        # 过采样少数类
        if len(pos_samples) < len(neg_samples):
            minority, majority = pos_samples, neg_samples
        else:
            minority, majority = neg_samples, pos_samples

        # 重复少数类样本
        ratio = len(majority) // len(minority)
        remainder = len(majority) % len(minority)
        oversampled = minority * ratio + minority[:remainder]

        self.samples = majority + oversampled
        np.random.shuffle(self.samples)


def video_split(
    labels_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """视频级别数据划分（避免数据泄露）

    Args:
        labels_df: 行为标签 DataFrame
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子

    Returns:
        train_df, val_df, test_df
    """
    videos = labels_df['video_name'].unique()
    np.random.seed(random_state)
    np.random.shuffle(videos)

    n_test = max(1, int(len(videos) * test_size))
    n_val = max(1, int(len(videos) * val_size))

    test_videos = videos[:n_test]
    val_videos = videos[n_test:n_test + n_val]
    train_videos = videos[n_test + n_val:]

    train_df = labels_df[labels_df['video_name'].isin(train_videos)]
    val_df = labels_df[labels_df['video_name'].isin(val_videos)]
    test_df = labels_df[labels_df['video_name'].isin(test_videos)]

    return train_df, val_df, test_df


def compute_class_weights(labels_df: pd.DataFrame) -> torch.Tensor:
    """计算类别权重（用于加权损失）"""
    counts = labels_df['label'].value_counts()
    total = len(labels_df)
    # 权重 = 总数 / (类别数 * 类别样本数)
    weights = torch.tensor([
        total / (2 * counts.get(0, 1)),
        total / (2 * counts.get(1, 1))
    ], dtype=torch.float32)
    return weights


class WeightedBCELoss(nn.Module):
    """加权二元交叉熵损失"""

    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCE with pos_weight
        loss = -self.pos_weight * target * torch.log(pred + 1e-7) \
               - (1 - target) * torch.log(1 - pred + 1e-7)
        return loss.mean()


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    threshold: float = 0.5
) -> Dict:
    """完整模型评估

    Returns:
        包含 precision, recall, f1, auc, confusion_matrix 的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            probs = model(x)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'accuracy': (all_preds == all_labels).mean(),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'threshold': threshold,
        'n_samples': len(all_labels),
        'n_positive': int(all_labels.sum()),
        'n_negative': int((1 - all_labels).sum())
    }

    return metrics


def find_optimal_threshold(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str
) -> Tuple[float, Dict]:
    """寻找最优阈值（最大化 F1）"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            probs = model(x)
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}

    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision_score(all_labels, preds, zero_division=0),
                'recall': recall_score(all_labels, preds, zero_division=0),
                'f1': f1,
                'threshold': threshold
            }

    return best_threshold, best_metrics


class AdvancedTrainer:
    """高级训练器 - 支持类别平衡和完整评估"""

    def __init__(
        self,
        config: ModelConfig = None,
        device: str = None,
        pos_weight: float = None
    ):
        self.config = config or ModelConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BehaviorGRU(self.config).to(self.device)
        self.pos_weight = pos_weight

    def train(
        self,
        train_dataset,
        val_dataset,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        early_stopping: int = 10
    ) -> Dict:
        """训练模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            early_stopping: 早停轮数

        Returns:
            训练历史
        """
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # 损失函数
        if self.pos_weight:
            criterion = WeightedBCELoss(pos_weight=self.pos_weight)
        else:
            criterion = nn.BCELoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        history = {
            'train_loss': [], 'val_loss': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
        }

        best_f1 = 0
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证
            val_metrics = evaluate_model(self.model, val_loader, self.device)
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auc'].append(val_metrics['auc'])

            scheduler.step(val_metrics['f1'])

            # 早停
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"P: {val_metrics['precision']:.4f} | "
                      f"R: {val_metrics['recall']:.4f} | "
                      f"F1: {val_metrics['f1']:.4f} | "
                      f"AUC: {val_metrics['auc']:.4f}")

            if patience_counter >= early_stopping:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 恢复最佳模型
        if best_state:
            self.model.load_state_dict(best_state)

        return history

    def save(self, path: str):
        """保存模型"""
        config_dict = {
            'input_dim': self.config.input_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'model_type': self.config.model_type,
            'seq_len': self.config.seq_len
        }
        torch.save({
            'config': config_dict,
            'model_state': self.model.state_dict()
        }, path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='时序模型完整训练流程')
    parser.add_argument('--labels', default='../data/behavior_labels.csv')
    parser.add_argument('--pose-dir', default='../data/pose_results')
    parser.add_argument('--output', default='../models/temporal_gru_v2.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--oversample', action='store_true', help='对少数类过采样')
    parser.add_argument('--pos-weight', type=float, default=None, help='正样本权重')
    args = parser.parse_args()

    print("=" * 60)
    print("时序模型训练流程")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    labels_df = pd.read_csv(args.labels)
    print(f"总样本数: {len(labels_df)}")
    print(f"视频数: {labels_df['video_name'].nunique()}")
    print(f"正负比例: {labels_df['label'].mean()*100:.1f}% 可疑")

    # 2. 视频级数据划分
    print("\n[2/5] 视频级数据划分...")
    train_df, val_df, test_df = video_split(labels_df, test_size=0.2, val_size=0.1)

    print(f"训练集: {len(train_df)} 样本, {train_df['video_name'].nunique()} 视频")
    print(f"验证集: {len(val_df)} 样本, {val_df['video_name'].nunique()} 视频")
    print(f"测试集: {len(test_df)} 样本, {test_df['video_name'].nunique()} 视频")

    print("\n各集合类别分布:")
    for name, df in [('训练', train_df), ('验证', val_df), ('测试', test_df)]:
        pos_rate = df['label'].mean() * 100
        print(f"  {name}: {pos_rate:.1f}% 可疑")

    # 3. 计算类别权重
    if args.pos_weight is None:
        class_weights = compute_class_weights(train_df)
        pos_weight = class_weights[1].item() / class_weights[0].item()
        print(f"\n自动计算正样本权重: {pos_weight:.3f}")
    else:
        pos_weight = args.pos_weight
        print(f"\n使用指定正样本权重: {pos_weight:.3f}")

    # 4. 创建数据集
    print("\n[3/5] 创建数据集...")
    config = ModelConfig(seq_len=args.seq_len, hidden_dim=args.hidden_dim)

    train_dataset = BalancedBehaviorDataset(
        train_df, args.pose_dir, seq_len=args.seq_len, oversample=args.oversample
    )
    val_dataset = BalancedBehaviorDataset(
        val_df, args.pose_dir, seq_len=args.seq_len, oversample=False
    )
    test_dataset = BalancedBehaviorDataset(
        test_df, args.pose_dir, seq_len=args.seq_len, oversample=False
    )

    print(f"训练序列: {len(train_dataset)}")
    print(f"验证序列: {len(val_dataset)}")
    print(f"测试序列: {len(test_dataset)}")

    # 5. 训练
    print("\n[4/5] 开始训练...")
    trainer = AdvancedTrainer(config, pos_weight=pos_weight)
    history = trainer.train(
        train_dataset, val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping=10
    )

    # 6. 测试集评估
    print("\n[5/5] 测试集评估...")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 寻找最优阈值
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    best_threshold, _ = find_optimal_threshold(trainer.model, val_loader, trainer.device)
    print(f"最优阈值: {best_threshold:.2f}")

    # 测试集评估
    test_metrics = evaluate_model(trainer.model, test_loader, trainer.device, threshold=best_threshold)

    print("\n" + "=" * 60)
    print("测试集结果")
    print("=" * 60)
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"AUC:       {test_metrics['auc']:.4f}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"\n混淆矩阵:")
    cm = test_metrics['confusion_matrix']
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

    # 保存模型
    trainer.save(args.output)
    print(f"\n模型已保存: {args.output}")

    # 保存评估结果
    results_path = args.output.replace('.pt', '_results.json')
    results = {
        'config': {
            'seq_len': args.seq_len,
            'hidden_dim': args.hidden_dim,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'pos_weight': pos_weight,
            'oversample': args.oversample
        },
        'data_split': {
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_videos': train_df['video_name'].unique().tolist(),
            'val_videos': val_df['video_name'].unique().tolist(),
            'test_videos': test_df['video_name'].unique().tolist()
        },
        'test_metrics': test_metrics,
        'best_threshold': best_threshold
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存: {results_path}")


if __name__ == '__main__':
    main()
