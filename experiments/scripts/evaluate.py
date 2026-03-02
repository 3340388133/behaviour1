#!/usr/bin/env python3
"""
SBRN评估脚本

使用方法:
    python experiments/scripts/evaluate.py --model experiments/logs/xxx/best_model.pt --config experiments/configs/full_model.yaml
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, accuracy_score, roc_auc_score, roc_curve, auc,
)
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recognition import SBRN, SBRNConfig


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


class DummyTestDataset(Dataset):
    """模拟测试数据集"""

    def __init__(self, n_samples: int = 100, seq_len: int = 32):
        self.seq_len = seq_len
        self.samples = self._create_data(n_samples)
        self.class_names = ['normal', 'looking_around', 'unknown']

    def _create_data(self, n_samples: int) -> List[Dict]:
        samples = []
        np.random.seed(123)

        for i in range(n_samples):
            if i < int(0.05 * n_samples):
                label = 0
            elif i < int(0.83 * n_samples):
                label = 1
            else:
                label = 2

            if label == 0:
                pose = np.random.randn(self.seq_len, 3) * 5
            elif label == 1:
                pose = np.cumsum(np.random.randn(self.seq_len, 3) * 10, axis=0)
                pose[:, 0] += np.sin(np.linspace(0, 4 * np.pi, self.seq_len)) * 30
            else:
                pose = np.random.randn(self.seq_len, 3) * 20

            samples.append({
                'pose': pose.astype(np.float32),
                'label': label,
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pose = torch.from_numpy(sample['pose'])
        label = torch.tensor(sample['label'], dtype=torch.long)
        return pose, label


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = 'Confusion Matrix',
):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_path: str,
):
    """绘制ROC曲线"""
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    # 为每个类别计算ROC
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_prob_class = y_probs[:, i]

        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_class)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_prototype_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    prototypes: np.ndarray,
    class_names: List[str],
    save_path: str,
):
    """绘制特征和原型的t-SNE可视化"""
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("Warning: sklearn not installed, skipping t-SNE visualization")
        return

    # 合并特征和原型
    n_samples = features.shape[0]
    n_prototypes = prototypes.shape[0] * prototypes.shape[1]
    all_features = np.vstack([
        features,
        prototypes.reshape(-1, prototypes.shape[-1])
    ])

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_features)

    sample_embedded = embedded[:n_samples]
    proto_embedded = embedded[n_samples:]

    # 绘图
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 绘制样本
    for i, name in enumerate(class_names):
        mask = labels == i
        plt.scatter(
            sample_embedded[mask, 0],
            sample_embedded[mask, 1],
            c=colors[i],
            alpha=0.5,
            label=f'{name} samples',
            s=20,
        )

    # 绘制原型
    n_proto_per_class = prototypes.shape[1]
    for i, name in enumerate(class_names):
        start_idx = i * n_proto_per_class
        end_idx = start_idx + n_proto_per_class
        plt.scatter(
            proto_embedded[start_idx:end_idx, 0],
            proto_embedded[start_idx:end_idx, 1],
            c=colors[i],
            marker='*',
            s=200,
            edgecolors='black',
            linewidths=1.5,
            label=f'{name} prototypes',
        )

    plt.legend()
    plt.title('Feature Space Visualization (t-SNE)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate(args):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载配置和模型
    config = load_config(args.config)
    model = create_model_from_config(config)

    # 加载权重
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {args.model}")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据
    test_dataset = DummyTestDataset(n_samples=200, seq_len=32)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = test_dataset.class_names

    # 推理
    all_preds = []
    all_labels = []
    all_probs = []
    all_confs = []
    all_features = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            pose, labels = batch
            pose = pose.to(device)

            outputs = model(pose, return_features=True)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            confs = outputs['confidence'].squeeze()
            features = outputs.get('features', None)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            if features is not None:
                all_features.append(features.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    all_confs = np.array(all_confs)
    if all_features:
        all_features = np.vstack(all_features)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 per class:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {f1_per_class[i]:.4f}")

    # 分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 计算AUC-ROC
    try:
        auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        print(f"AUC-ROC Macro: {auc_macro:.4f}")
    except Exception as e:
        print(f"Could not compute AUC-ROC: {e}")
        auc_macro = None

    # 保存结果
    results = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_per_class': {name: float(f1) for name, f1 in zip(class_names, f1_per_class)},
        'auc_macro': float(auc_macro) if auc_macro else None,
        'class_names': class_names,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, output_dir / 'confusion_matrix.png')
    print(f"\nSaved confusion matrix to {output_dir / 'confusion_matrix.png'}")

    # 绘制ROC曲线
    plot_roc_curves(all_labels, all_probs, class_names, output_dir / 'roc_curves.png')
    print(f"Saved ROC curves to {output_dir / 'roc_curves.png'}")

    # 如果有BPCL，绘制原型可视化
    if hasattr(model, 'bpcl') and len(all_features) > 0:
        prototypes = model.bpcl.prototypes.detach().cpu().numpy()
        plot_prototype_tsne(
            all_features, all_labels, prototypes, class_names,
            output_dir / 'prototype_tsne.png'
        )
        print(f"Saved prototype t-SNE to {output_dir / 'prototype_tsne.png'}")

    print(f"\nAll results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SBRN model')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output', type=str, default='experiments/eval_results',
                        help='Output directory for results')
    args = parser.parse_args()

    evaluate(args)


if __name__ == '__main__':
    main()
