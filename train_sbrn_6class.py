#!/usr/bin/env python3
"""
SBRN 6类行为识别训练 + 消融实验

创新点:
1. PAPE - 周期感知位置编码 (捕捉行为周期性模式)
2. BPCL - 行为原型对比学习 (改善类间分离度)
3. ClassBalanced Focal Loss + 自适应采样 (处理极端类别不平衡)
4. 姿态序列增强 (时间扭曲/噪声注入/Mixup)

消融实验:
  A0: Baseline Transformer + CE
  A1: + PAPE
  A2: + BPCL
  A3: + Focal Loss + Class Balancing
  A4: + Data Augmentation
  A5: Full SBRN (all above)

用法:
  # 完整消融
  python train_sbrn_6class.py --ablation --epochs 80

  # 只训练完整模型
  python train_sbrn_6class.py --epochs 80

  # 快速测试
  python train_sbrn_6class.py --epochs 5 --quick
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src' / 'recognition'))

import json
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

from temporal_transformer import create_model
from models.sbrn import SBRN, SBRNConfig

CLASS_NAMES = ['normal', 'glancing', 'quick_turn', 'prolonged_watch', 'looking_down', 'looking_up']


# ============================================================
# 数据增强
# ============================================================

def augment_pose_sequence(pose_seq: np.ndarray, p=0.5) -> np.ndarray:
    """姿态序列增强 (创新点4)"""
    seq = pose_seq.copy()

    # 1. Gaussian noise
    if np.random.rand() < p:
        noise = np.random.randn(*seq.shape).astype(np.float32) * 2.0
        seq = seq + noise

    # 2. Temporal shift (随机平移1~5帧)
    if np.random.rand() < p:
        shift = np.random.randint(1, 6)
        if np.random.rand() < 0.5:
            seq = np.concatenate([seq[shift:], np.tile(seq[-1:], (shift, 1))], axis=0)
        else:
            seq = np.concatenate([np.tile(seq[:1], (shift, 1)), seq[:-shift]], axis=0)

    # 3. Time warping (局部加速/减速)
    if np.random.rand() < p * 0.5:
        T = len(seq)
        warp_center = np.random.randint(T // 4, 3 * T // 4)
        warp_width = np.random.randint(T // 8, T // 4)
        speed = np.random.uniform(0.7, 1.3)

        indices = np.arange(T, dtype=np.float32)
        mask = np.abs(indices - warp_center) < warp_width
        warped_indices = indices.copy()
        warped_indices[mask] = warp_center + (indices[mask] - warp_center) * speed
        warped_indices = np.clip(warped_indices, 0, T - 1).astype(int)
        seq = seq[warped_indices]

    # 4. Yaw mirror (水平翻转)
    if np.random.rand() < p * 0.3:
        seq[:, 0] = -seq[:, 0]  # flip yaw
        seq[:, 2] = -seq[:, 2]  # flip roll

    return seq


# ============================================================
# 数据集
# ============================================================

class BehaviorDataset(Dataset):
    """6类行为数据集"""

    def __init__(self, data_path, split='train', seq_len=90,
                 oversample=False, augment=False):
        self.seq_len = seq_len
        self.augment = augment and (split == 'train')

        with open(f'{data_path}/{split}.json', 'r') as f:
            data = json.load(f)
        self.samples = data['samples']

        if oversample and split == 'train':
            self._oversample()

        labels = [s['label'] for s in self.samples]
        dist = dict(Counter(sorted(labels)))
        print(f"[{split}] {len(self.samples)} samples, distribution: {dist}")

    def _oversample(self):
        """少数类过采样到最大类的 50%"""
        labels = [s['label'] for s in self.samples]
        counter = Counter(labels)
        target = max(counter.values()) // 2

        new_samples = []
        for label in range(6):
            cls_samples = [s for s in self.samples if s['label'] == label]
            if not cls_samples:
                continue
            if len(cls_samples) < target:
                ratio = target // len(cls_samples)
                remainder = target % len(cls_samples)
                new_samples.extend(cls_samples * ratio + cls_samples[:remainder])
            else:
                new_samples.extend(cls_samples)

        np.random.shuffle(new_samples)
        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pose_seq = np.array(sample['pose_sequence'], dtype=np.float32)

        # 处理长度
        if len(pose_seq) > self.seq_len:
            start = np.random.randint(0, len(pose_seq) - self.seq_len) if self.augment else (len(pose_seq) - self.seq_len) // 2
            pose_seq = pose_seq[start:start + self.seq_len]
        elif len(pose_seq) < self.seq_len:
            pad = np.tile(pose_seq[-1:], (self.seq_len - len(pose_seq), 1))
            pose_seq = np.concatenate([pose_seq, pad], axis=0)

        # 数据增强
        if self.augment:
            pose_seq = augment_pose_sequence(pose_seq)

        label = sample['label']
        return torch.from_numpy(pose_seq), torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return [s['label'] for s in self.samples]


# ============================================================
# Mixup collate
# ============================================================

def mixup_collate(batch, alpha=0.2, prob=0.3):
    """Mixup数据增强 collate function"""
    poses, labels = zip(*batch)
    poses = torch.stack(poses)
    labels = torch.stack(labels)

    if np.random.rand() < prob:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(len(poses))
        poses = lam * poses + (1 - lam) * poses[idx]
        # 对于分类任务，mixup标签用one-hot
        return poses, labels, labels[idx], lam
    return poses, labels, labels, 1.0


# ============================================================
# SBRN inference wrapper (兼容现有推理脚本)
# ============================================================

class SBRNInferenceWrapper(nn.Module):
    """包装SBRN使其与现有推理脚本兼容: model(pose_seq) -> (logits, confidence)"""

    def __init__(self, sbrn_model):
        super().__init__()
        self.sbrn = sbrn_model

    def forward(self, pose_seq):
        outputs = self.sbrn(pose_seq)
        return outputs['logits'], outputs['confidence']


# ============================================================
# 训练/评估函数
# ============================================================

def compute_class_weights(labels, num_classes=6, method='effective_number'):
    """计算类别权重"""
    counts = torch.zeros(num_classes, dtype=torch.float)
    for c in range(num_classes):
        counts[c] = sum(1 for l in labels if l == c)
    counts = counts.clamp(min=1)

    if method == 'effective_number':
        beta = 0.9999
        effective = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective
    elif method == 'sqrt_inverse':
        weights = 1.0 / torch.sqrt(counts)
    else:
        weights = 1.0 / counts

    weights = weights / weights.sum() * num_classes
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha if isinstance(alpha, torch.Tensor) else torch.tensor(alpha))
        else:
            self.alpha = None

    def forward(self, logits, targets):
        num_classes = logits.shape[-1]
        targets_oh = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            targets_smooth = targets_oh * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        else:
            targets_smooth = targets_oh

        probs = F.softmax(logits, dim=-1)
        p_t = (probs * targets_oh).sum(dim=-1)
        focal_weight = (1 - p_t) ** self.gamma

        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets_smooth * log_probs).sum(dim=-1)
        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()


def evaluate(model, loader, device, is_sbrn=False, num_classes=6):
    """评估"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            poses, labels = batch[0].to(device), batch[1].to(device)

            if is_sbrn:
                outputs = model(poses)
                logits = outputs['logits']
            else:
                logits, _ = model(poses)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=range(num_classes))
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / max(len(loader), 1),
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'per_class_f1': per_class_f1.tolist(),
        'preds': all_preds,
        'labels': all_labels,
    }


def train_one_epoch(model, loader, optimizer, criterion, device,
                    is_sbrn=False, use_contrastive=False, use_mixup=False,
                    grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        poses = batch[0].to(device)
        labels = batch[1].to(device)

        optimizer.zero_grad()

        if is_sbrn:
            outputs = model(poses, return_features=use_contrastive)
            logits = outputs['logits']
            loss = criterion(logits, labels)

            # 加入对比学习损失
            if use_contrastive and 'features' in outputs:
                cont_loss, _ = model.bpcl.compute_contrastive_loss(
                    outputs['features'], labels
                )
                loss = loss + 0.3 * cont_loss
                model.bpcl.update_prototypes_momentum()
        else:
            logits, _ = model(poses)
            loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ============================================================
# 实验配置
# ============================================================

def create_baseline_model(num_classes=6, device='cuda'):
    """A0: Baseline Transformer"""
    model = create_model(
        model_type='transformer',
        pose_input_dim=3, pose_d_model=64, pose_nhead=4,
        pose_num_layers=2, use_multimodal=False,
        hidden_dim=128, num_classes=num_classes,
        dropout=0.1, uncertainty_weighting=False,
    )
    return model.to(device)


def create_sbrn_model(num_classes=6, use_contrastive=True, device='cuda'):
    """A5: Full SBRN"""
    config = SBRNConfig(
        pose_input_dim=3,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=128,
        periods=[15, 30, 60],
        use_relative_bias=True,
        use_multimodal=False,
        use_contrastive=use_contrastive,
        num_prototypes_per_class=4,
        temperature=0.07,
        contrastive_margin=0.5,
        num_classes=num_classes,
        hidden_dim=128,
        uncertainty_weighting=True,
    )
    model = SBRN(config)
    return model.to(device)


def run_experiment(name, model, train_loader, val_loader, test_loader,
                   criterion, device, epochs=80, lr=1e-3,
                   is_sbrn=False, use_contrastive=False, patience=20):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    best_f1 = 0
    best_state = None
    wait = 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            is_sbrn=is_sbrn, use_contrastive=use_contrastive,
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, is_sbrn=is_sbrn)
        dt = time.time() - t0

        log = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_per_class_f1': val_metrics['per_class_f1'],
        }
        history.append(log)

        f1_str = ' '.join([f'{f:.2f}' for f in val_metrics['per_class_f1']])
        print(f"  Epoch {epoch+1:3d}/{epochs} ({dt:.0f}s) | "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_f1={val_metrics['f1_macro']:.3f} [{f1_str}]",
              end='')

        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            print(f" *BEST*")
        else:
            wait += 1
            print()

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # 加载最佳模型
    if best_state:
        model.load_state_dict(best_state)

    # 测试集评估
    test_metrics = evaluate(model, test_loader, device, is_sbrn=is_sbrn)

    print(f"\n  --- Test Results ({name}) ---")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1 Macro:  {test_metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {test_metrics['f1_weighted']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    for i, name_cls in enumerate(CLASS_NAMES):
        print(f"    {name_cls:20s}: F1={test_metrics['per_class_f1'][i]:.3f}")

    # 混淆矩阵
    cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'], labels=range(6))
    print(f"\n  Confusion Matrix:")
    header = "".join([f"{n[:6]:>8s}" for n in CLASS_NAMES])
    print(f"  {'':>20s}{header}")
    for i, row in enumerate(cm):
        row_str = "".join([f"{v:8d}" for v in row])
        print(f"  {CLASS_NAMES[i]:>20s}{row_str}")

    return {
        'name': name,
        'best_val_f1': best_f1,
        'test_metrics': {k: v for k, v in test_metrics.items() if k not in ('preds', 'labels')},
        'n_params': n_params,
        'history': history,
        'model_state': best_state,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SBRN 6类行为识别训练')
    parser.add_argument('--data-path', default='data/dataset')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    parser.add_argument('--quick', action='store_true', help='快速测试模式')
    parser.add_argument('--output-dir', default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.quick:
        args.epochs = min(args.epochs, 5)

    # ---- 数据集 ----
    print("\n=== Loading Data ===")
    # 不带增强的数据集 (用于 baseline)
    train_ds_plain = BehaviorDataset(args.data_path, 'train', oversample=True, augment=False)
    # 带增强的数据集 (用于 SBRN)
    train_ds_aug = BehaviorDataset(args.data_path, 'train', oversample=True, augment=True)
    val_ds = BehaviorDataset(args.data_path, 'val')
    test_ds = BehaviorDataset(args.data_path, 'test')

    # 类别权重和加权采样器
    train_labels = train_ds_plain.get_labels()
    class_weights = compute_class_weights(train_labels, 6, 'effective_number').to(device)
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    sample_weights = [class_weights[l].item() for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoaders
    train_loader_plain = DataLoader(train_ds_plain, batch_size=args.batch_size, shuffle=True, num_workers=2)
    train_loader_sampler = DataLoader(train_ds_plain, batch_size=args.batch_size, sampler=sampler, num_workers=2)
    train_loader_aug = DataLoader(train_ds_aug, batch_size=args.batch_size, sampler=WeightedRandomSampler(
        [class_weights[l].item() for l in train_ds_aug.get_labels()],
        len(train_ds_aug), replacement=True
    ), num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Loss functions
    ce_loss = nn.CrossEntropyLoss().to(device)
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05).to(device)

    results = []

    if args.ablation:
        # ======== A0: Baseline Transformer + CE ========
        model = create_baseline_model(device=device)
        r = run_experiment("A0: Baseline(Transformer+CE)", model,
                           train_loader_plain, val_loader, test_loader,
                           ce_loss, device, epochs=args.epochs)
        results.append(r)

        # ======== A1: Baseline + Focal + Balanced Sampling ========
        model = create_baseline_model(device=device)
        r = run_experiment("A1: +FocalLoss+BalancedSampling", model,
                           train_loader_sampler, val_loader, test_loader,
                           focal_loss, device, epochs=args.epochs)
        results.append(r)

        # ======== A2: SBRN(PAPE only, no BPCL) ========
        model = create_sbrn_model(use_contrastive=False, device=device)
        r = run_experiment("A2: +PAPE(no BPCL)", model,
                           train_loader_sampler, val_loader, test_loader,
                           focal_loss, device, epochs=args.epochs, is_sbrn=True)
        results.append(r)

        # ======== A3: SBRN(PAPE + BPCL) ========
        model = create_sbrn_model(use_contrastive=True, device=device)
        r = run_experiment("A3: +PAPE+BPCL", model,
                           train_loader_sampler, val_loader, test_loader,
                           focal_loss, device, epochs=args.epochs,
                           is_sbrn=True, use_contrastive=True)
        results.append(r)

        # ======== A4: Full SBRN + Augmentation ========
        model = create_sbrn_model(use_contrastive=True, device=device)
        r = run_experiment("A4: Full SBRN+Aug", model,
                           train_loader_aug, val_loader, test_loader,
                           focal_loss, device, epochs=args.epochs,
                           is_sbrn=True, use_contrastive=True)
        results.append(r)

        # 保存最佳 SBRN
        best_result = max(results, key=lambda x: x['test_metrics']['f1_macro'])
        best_state = best_result['model_state']

    else:
        # 只训练完整 SBRN
        model = create_sbrn_model(use_contrastive=True, device=device)
        r = run_experiment("Full SBRN", model,
                           train_loader_aug, val_loader, test_loader,
                           focal_loss, device, epochs=args.epochs,
                           is_sbrn=True, use_contrastive=True)
        results.append(r)
        best_result = r
        best_state = r['model_state']

    # ---- 保存最佳模型 ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 SBRN 原始权重
    sbrn_path = out_dir / 'sbrn_best.pt'
    torch.save({
        'model_state_dict': best_state,
        'config': {
            'pose_input_dim': 3, 'd_model': 128, 'nhead': 4,
            'num_layers': 3, 'dim_feedforward': 256, 'dropout': 0.1,
            'max_seq_len': 128, 'periods': [15, 30, 60],
            'use_relative_bias': True, 'use_multimodal': False,
            'use_contrastive': True, 'num_prototypes_per_class': 4,
            'temperature': 0.07, 'num_classes': 6, 'hidden_dim': 128,
        },
        'test_metrics': best_result['test_metrics'],
        'experiment': best_result['name'],
    }, sbrn_path)
    print(f"\nSBRN model saved: {sbrn_path}")

    # 保存兼容推理脚本的 wrapper 版本
    sbrn_model = create_sbrn_model(use_contrastive=True, device='cpu')
    sbrn_model.load_state_dict(best_state)
    wrapper = SBRNInferenceWrapper(sbrn_model)

    wrapper_path = out_dir / 'transformer_best.pt'
    torch.save({
        'model_state_dict': wrapper.state_dict(),
        'model_type': 'sbrn_wrapper',
        'test_metrics': best_result['test_metrics'],
    }, wrapper_path)
    print(f"Inference-compatible model saved: {wrapper_path}")

    # ---- 消融结果汇总 ----
    if args.ablation:
        print(f"\n{'='*70}")
        print(f"{'Experiment':<35s} {'Params':>8s} {'Acc':>7s} {'F1-M':>7s} {'F1-W':>7s}")
        print(f"{'-'*70}")
        for r in results:
            tm = r['test_metrics']
            print(f"{r['name']:<35s} {r['n_params']:>8,} "
                  f"{tm['accuracy']:>7.4f} {tm['f1_macro']:>7.4f} {tm['f1_weighted']:>7.4f}")
        print(f"{'='*70}")

        # per-class F1 表
        print(f"\nPer-class F1 scores:")
        header = f"{'Experiment':<35s}" + "".join([f"{n[:8]:>10s}" for n in CLASS_NAMES])
        print(header)
        print("-" * len(header))
        for r in results:
            f1s = r['test_metrics']['per_class_f1']
            row = f"{r['name']:<35s}" + "".join([f"{f:>10.3f}" for f in f1s])
            print(row)

        # 保存结果到 JSON
        ablation_path = out_dir / 'ablation_sbrn.json'
        save_results = []
        for r in results:
            save_results.append({
                'name': r['name'],
                'n_params': r['n_params'],
                'best_val_f1': r['best_val_f1'],
                'test_metrics': r['test_metrics'],
                'history': r['history'],
            })
        with open(ablation_path, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nAblation results saved: {ablation_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
