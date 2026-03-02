"""
轻量级 GRU/LSTM 时序模型 - 用于可疑行为识别
输入: 窗口级时序特征 (yaw_mean, yaw_std, yaw_range, yaw_speed_mean, yaw_switch_count)
输出: p_model ∈ [0,1] 可疑概率
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from fusion.base import BaseTemporalModel, ScorerResult


# 特征列定义
FEATURE_COLS = ['yaw_mean', 'yaw_std', 'yaw_range', 'yaw_speed_mean', 'yaw_switch_count']


@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 5          # 输入特征维度
    hidden_dim: int = 32        # 隐藏层维度
    num_layers: int = 1         # RNN 层数
    dropout: float = 0.1        # Dropout 比例
    model_type: str = 'gru'     # 'gru' or 'lstm'
    seq_len: int = 4            # 序列长度（连续窗口数）


class BehaviorGRU(nn.Module):
    """轻量级 GRU 模型"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 特征归一化层
        self.input_norm = nn.BatchNorm1d(config.input_dim)

        # RNN 层
        rnn_cls = nn.GRU if config.model_type == 'gru' else nn.LSTM
        self.rnn = rnn_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            [batch, 1] 可疑概率
        """
        batch_size, seq_len, _ = x.shape

        # 归一化: [batch * seq_len, input_dim]
        x_flat = x.view(-1, self.config.input_dim)
        x_norm = self.input_norm(x_flat)
        x = x_norm.view(batch_size, seq_len, -1)

        # RNN
        if self.config.model_type == 'lstm':
            _, (h, _) = self.rnn(x)
        else:
            _, h = self.rnn(x)

        # 取最后一层隐藏状态
        h = h[-1]  # [batch, hidden_dim]

        # 分类
        out = self.classifier(h)
        return out


class BehaviorDataset(torch.utils.data.Dataset):
    """行为数据集 - 从 pose CSV 和 behavior_labels 构建序列"""

    def __init__(
        self,
        labels_df: pd.DataFrame,
        pose_dir: str,
        seq_len: int = 4,
        feature_extractor=None
    ):
        """
        Args:
            labels_df: behavior_labels.csv 的 DataFrame
            pose_dir: pose CSV 目录
            seq_len: 序列长度
            feature_extractor: TemporalFeatureExtractor 实例
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.pose_dir = Path(pose_dir)
        self.seq_len = seq_len

        if feature_extractor is None:
            from temporal_features import TemporalFeatureExtractor
            feature_extractor = TemporalFeatureExtractor()
        self.feature_extractor = feature_extractor

        # 预计算所有特征
        self._precompute_features()

    def _precompute_features(self):
        """预计算所有视频的时序特征"""
        self.features_cache = {}
        self.samples = []

        for video_name in self.labels_df['video_name'].unique():
            pose_csv = self.pose_dir / f"{video_name}.csv"
            if not pose_csv.exists():
                continue

            # 加载 pose 数据
            pose_df = pd.read_csv(pose_csv)
            pose_df['track_id'] = pose_df.groupby('frame_id').cumcount()

            # 获取该视频的标签
            video_labels = self.labels_df[self.labels_df['video_name'] == video_name]

            for track_id in video_labels['track_id'].unique():
                track_labels = video_labels[video_labels['track_id'] == track_id].sort_values('start_time')
                track_pose = pose_df[pose_df['track_id'] == track_id].sort_values('time_sec')

                if len(track_pose) < 5:
                    continue

                # 提取时序特征
                times = track_pose['time_sec'].values
                yaws = track_pose['yaw'].values
                features = self.feature_extractor.extract_from_track(times, yaws, track_id)

                # 构建特征字典 {(start_time, end_time): feature_vector}
                feat_dict = {}
                for feat in features:
                    key = (round(feat.window_start, 3), round(feat.window_end, 3))
                    feat_dict[key] = np.array([
                        feat.yaw_mean / 180.0,      # 归一化到 [-1, 1]
                        feat.yaw_std / 90.0,        # 归一化
                        feat.yaw_range / 180.0,     # 归一化
                        feat.yaw_speed_mean / 100.0,  # 归一化
                        feat.yaw_switch_count / 5.0   # 归一化
                    ], dtype=np.float32)

                self.features_cache[(video_name, track_id)] = feat_dict

                # 构建序列样本
                for i in range(len(track_labels) - self.seq_len + 1):
                    window_labels = track_labels.iloc[i:i + self.seq_len]
                    keys = [
                        (round(row['start_time'], 3), round(row['end_time'], 3))
                        for _, row in window_labels.iterrows()
                    ]

                    # 检查所有窗口特征是否存在
                    if all(k in feat_dict for k in keys):
                        # 使用最后一个窗口的标签
                        label = window_labels.iloc[-1]['label']
                        self.samples.append({
                            'video_name': video_name,
                            'track_id': track_id,
                            'keys': keys,
                            'label': label
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feat_dict = self.features_cache[(sample['video_name'], sample['track_id'])]

        # 构建序列特征
        seq_features = np.stack([feat_dict[k] for k in sample['keys']])
        label = np.array([sample['label']], dtype=np.float32)

        return torch.from_numpy(seq_features), torch.from_numpy(label)


class TemporalModelTrainer:
    """时序模型训练器"""

    def __init__(
        self,
        config: ModelConfig = None,
        device: str = None
    ):
        self.config = config or ModelConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BehaviorGRU(self.config).to(self.device)

    def train(
        self,
        train_dataset: BehaviorDataset,
        val_dataset: BehaviorDataset = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        pos_weight: float = 1.0
    ) -> Dict[str, List[float]]:
        """训练模型

        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            epochs: 训练轮数
            batch_size: 批大小
            lr: 学习率
            pos_weight: 正样本权重（处理类别不平衡）

        Returns:
            训练历史
        """
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = None
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

        # 损失函数: 带权重的 BCE
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

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
            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                scheduler.step(val_loss)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")

        return history

    def _evaluate(self, loader, criterion) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()

                pred_label = (pred > 0.5).float()
                correct += (pred_label == y).sum().item()
                total += y.size(0)

        return total_loss / len(loader), correct / total

    def save(self, path: str):
        """保存模型"""
        # 保存 config 为字典，避免序列化问题
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
        print(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        # 兼容旧格式（dataclass）和新格式（dict）
        if isinstance(config_dict, dict):
            self.config = ModelConfig(**config_dict)
        else:
            self.config = config_dict
        self.model = BehaviorGRU(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        print(f"Model loaded from {path}")


class GRUModelScorer(BaseTemporalModel):
    """GRU 模型评分器 - 可接入 fusion 模块"""

    def __init__(self, model_path: str = None, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.feature_buffer = []  # 特征缓冲区

        if model_path:
            self.load_weights(model_path)

    def get_name(self) -> str:
        return "GRUModelScorer"

    def load_weights(self, path: str) -> None:
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config_dict = checkpoint['config']
        # 兼容旧格式（dataclass）和新格式（dict）
        if isinstance(config_dict, dict):
            self.config = ModelConfig(**config_dict)
        else:
            self.config = config_dict
        self.model = BehaviorGRU(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def train(self, data: Any, labels: Any) -> None:
        """训练接口（通过 TemporalModelTrainer 实现）"""
        pass

    def score(self, features: Dict[str, Any]) -> ScorerResult:
        """计算可疑分数

        Args:
            features: 时序特征字典，需包含:
                - yaw_mean, yaw_std, yaw_range, yaw_speed_mean, yaw_switch_count
                或
                - sequence: 预构建的特征序列 [seq_len, input_dim]

        Returns:
            ScorerResult
        """
        if self.model is None:
            return ScorerResult(
                score=0.0,
                confidence=0.0,
                details={'error': 'model not loaded'}
            )

        # 构建特征向量
        if 'sequence' in features:
            seq = features['sequence']
        else:
            # 从单窗口特征构建
            feat_vec = np.array([
                features.get('yaw_mean', 0) / 180.0,
                features.get('yaw_std', 0) / 90.0,
                features.get('yaw_range', 0) / 180.0,
                features.get('yaw_speed_mean', 0) / 100.0,
                features.get('yaw_switch_count', 0) / 5.0
            ], dtype=np.float32)

            # 更新缓冲区
            self.feature_buffer.append(feat_vec)
            if len(self.feature_buffer) > self.config.seq_len:
                self.feature_buffer.pop(0)

            # 如果缓冲区不足，返回低置信度
            if len(self.feature_buffer) < self.config.seq_len:
                return ScorerResult(
                    score=0.0,
                    confidence=0.3,
                    details={'status': 'buffering', 'buffer_size': len(self.feature_buffer)}
                )

            seq = np.stack(self.feature_buffer)

        # 推理
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            score = self.model(x).item()

        return ScorerResult(
            score=score,
            confidence=0.8,
            details={'model': 'gru', 'seq_len': self.config.seq_len}
        )

    def reset_buffer(self):
        """重置特征缓冲区（切换 track 时调用）"""
        self.feature_buffer = []


def train_model(
    labels_csv: str,
    pose_dir: str,
    output_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 4,
    val_split: float = 0.2
):
    """训练模型的便捷函数"""
    print("Loading data...")
    labels_df = pd.read_csv(labels_csv)

    # 按视频划分训练/验证集
    videos = labels_df['video_name'].unique()
    np.random.shuffle(videos)
    split_idx = int(len(videos) * (1 - val_split))
    train_videos = videos[:split_idx]
    val_videos = videos[split_idx:]

    train_df = labels_df[labels_df['video_name'].isin(train_videos)]
    val_df = labels_df[labels_df['video_name'].isin(val_videos)]

    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # 创建数据集
    config = ModelConfig(seq_len=seq_len)
    train_dataset = BehaviorDataset(train_df, pose_dir, seq_len=seq_len)
    val_dataset = BehaviorDataset(val_df, pose_dir, seq_len=seq_len)

    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # 训练
    trainer = TemporalModelTrainer(config)
    history = trainer.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size
    )

    # 保存
    trainer.save(output_path)

    return history


def main():
    import argparse

    parser = argparse.ArgumentParser(description='训练时序行为识别模型')
    parser.add_argument('--labels', default='../data/behavior_labels.csv', help='行为标签 CSV')
    parser.add_argument('--pose-dir', default='../data/pose_results', help='pose CSV 目录')
    parser.add_argument('--output', default='../models/temporal_gru.pt', help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--seq-len', type=int, default=4, help='序列长度')
    args = parser.parse_args()

    train_model(
        labels_csv=args.labels,
        pose_dir=args.pose_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )


if __name__ == '__main__':
    main()
