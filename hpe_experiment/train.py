"""
Complete training pipeline for improved WHENet (WHENet+)
Trains on 300W-LP (or synthetic), tests on AFLW2000 and BIWI
Reports MAE and generates comparison tables
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

# Add project paths
sys.path.insert(0, '/root/autodl-tmp/behaviour')
sys.path.insert(0, '/root/autodl-tmp/behaviour/SuspiciousGazeDetection')

from hpe_experiment.dataset import create_datasets
from src.models.head_pose.whenet import WHENet, GeodesicLoss
from src.models.head_pose.whenet_plus import WHENetPlus, WHENetPlusLoss


class WHENetPlusSimple(nn.Module):
    """
    Simplified WHENet+ that doesn't require external attention/fusion imports.
    Self-contained implementation for reliable training.
    """

    def __init__(self, pretrained=True, num_bins=66, attention_type='cbam'):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base = efficientnet_b0(weights=weights)
        else:
            base = efficientnet_b0(weights=None)

        self.features = base.features  # All 9 stages

        # Multi-scale channels: stage 2=24, stage 4=80, stage 6=192
        self.scale_channels = [24, 80, 192]
        self.out_channels = 128

        # Channel alignment for multi-scale fusion
        self.lateral_2 = nn.Sequential(
            nn.Conv2d(24, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.lateral_4 = nn.Sequential(
            nn.Conv2d(80, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )
        self.lateral_6 = nn.Sequential(
            nn.Conv2d(192, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

        # Adaptive fusion weights (learnable)
        self.fusion_weight = nn.Parameter(torch.ones(3) / 3)

        # CBAM-style attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.out_channels, self.out_channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 8, self.out_channels),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Pose regression heads
        self.num_bins = num_bins
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.fc_yaw = nn.Linear(256, num_bins)
        self.fc_pitch = nn.Linear(256, num_bins)
        self.fc_roll = nn.Linear(256, num_bins)

        self.fc_yaw_reg = nn.Linear(256, 1)
        self.fc_pitch_reg = nn.Linear(256, 1)
        self.fc_roll_reg = nn.Linear(256, 1)

        idx_tensor = torch.arange(num_bins, dtype=torch.float32)
        self.register_buffer('idx_tensor', idx_tensor)

        self.angle_range = (-99, 99)

    def forward(self, x):
        # Extract multi-scale features
        feat_2 = feat_4 = feat_6 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 2:
                feat_2 = x
            elif i == 4:
                feat_4 = x
            elif i == 6:
                feat_6 = x

        # Multi-scale lateral connections
        lat_2 = self.lateral_2(feat_2)
        lat_4 = self.lateral_4(feat_4)
        lat_6 = self.lateral_6(feat_6)

        # Resize all to same spatial size
        target_size = lat_2.shape[2:]
        lat_4_up = nn.functional.interpolate(lat_4, size=target_size, mode='bilinear', align_corners=False)
        lat_6_up = nn.functional.interpolate(lat_6, size=target_size, mode='bilinear', align_corners=False)

        # Adaptive weighted fusion
        w = torch.softmax(self.fusion_weight, dim=0)
        fused = w[0] * lat_2 + w[1] * lat_4_up + w[2] * lat_6_up

        # CBAM attention
        B, C, H, W = fused.shape
        # Channel attention
        ca = self.channel_attn(fused).view(B, C, 1, 1)
        fused = fused * ca
        # Spatial attention
        sa_avg = torch.mean(fused, dim=1, keepdim=True)
        sa_max, _ = torch.max(fused, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([sa_avg, sa_max], dim=1))
        fused = fused * sa

        # Global pooling
        pooled = self.gap(fused).flatten(1)
        feat = self.fc(pooled)

        # Classification + regression
        yaw_cls = self.fc_yaw(feat)
        pitch_cls = self.fc_pitch(feat)
        roll_cls = self.fc_roll(feat)

        yaw_soft = torch.softmax(yaw_cls, dim=1)
        pitch_soft = torch.softmax(pitch_cls, dim=1)
        roll_soft = torch.softmax(roll_cls, dim=1)

        yaw_exp = (yaw_soft * self.idx_tensor).sum(dim=1)
        pitch_exp = (pitch_soft * self.idx_tensor).sum(dim=1)
        roll_exp = (roll_soft * self.idx_tensor).sum(dim=1)

        bin_width = (self.angle_range[1] - self.angle_range[0]) / self.num_bins
        yaw = yaw_exp * bin_width + self.angle_range[0] + self.fc_yaw_reg(feat).squeeze(-1)
        pitch = pitch_exp * bin_width + self.angle_range[0] + self.fc_pitch_reg(feat).squeeze(-1)
        roll = roll_exp * bin_width + self.angle_range[0] + self.fc_roll_reg(feat).squeeze(-1)

        return {
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll,
            'cls_logits': (yaw_cls, pitch_cls, roll_cls),
            'features': feat,
        }


class CombinedLoss(nn.Module):
    """Combined classification + regression + geodesic loss."""

    def __init__(self, num_bins=66, angle_range=(-99, 99),
                 cls_weight=1.0, reg_weight=1.5, geo_weight=0.5):
        super().__init__()
        self.num_bins = num_bins
        self.angle_range = angle_range
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.geo_weight = geo_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.geo_loss = GeodesicLoss()

    def angle_to_bin(self, angle):
        """Convert continuous angle to bin index."""
        bin_width = (self.angle_range[1] - self.angle_range[0]) / self.num_bins
        bin_idx = ((angle - self.angle_range[0]) / bin_width).long()
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        return bin_idx

    def forward(self, output, target_angles):
        yaw_gt, pitch_gt, roll_gt = target_angles[:, 0], target_angles[:, 1], target_angles[:, 2]

        # Clamp targets to valid range for classification
        yaw_gt_c = torch.clamp(yaw_gt, self.angle_range[0], self.angle_range[1])
        pitch_gt_c = torch.clamp(pitch_gt, self.angle_range[0], self.angle_range[1])
        roll_gt_c = torch.clamp(roll_gt, self.angle_range[0], self.angle_range[1])

        # Classification loss
        yaw_bin = self.angle_to_bin(yaw_gt_c)
        pitch_bin = self.angle_to_bin(pitch_gt_c)
        roll_bin = self.angle_to_bin(roll_gt_c)

        yaw_cls, pitch_cls, roll_cls = output['cls_logits']
        cls_loss = (self.ce_loss(yaw_cls, yaw_bin) +
                    self.ce_loss(pitch_cls, pitch_bin) +
                    self.ce_loss(roll_cls, roll_bin)) / 3

        # Regression loss (Smooth L1)
        reg_loss = (self.smooth_l1(output['yaw'], yaw_gt) +
                    self.smooth_l1(output['pitch'], pitch_gt) +
                    self.smooth_l1(output['roll'], roll_gt)) / 3

        # Geodesic loss
        geo_loss = self.geo_loss(
            (output['yaw'], output['pitch'], output['roll']),
            (yaw_gt, pitch_gt, roll_gt),
        )

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.geo_weight * geo_loss

        return {
            'total': total,
            'cls': cls_loss.item(),
            'reg': reg_loss.item(),
            'geo': geo_loss.item(),
        }


def compute_mae(model, dataloader, device):
    """Compute MAE on a dataset."""
    model.eval()
    yaw_errors = []
    pitch_errors = []
    roll_errors = []

    with torch.no_grad():
        for images, angles in dataloader:
            images = images.to(device)
            angles = angles.numpy()

            output = model(images)
            yaw_pred = output['yaw'].cpu().numpy()
            pitch_pred = output['pitch'].cpu().numpy()
            roll_pred = output['roll'].cpu().numpy()

            yaw_errors.extend(np.abs(yaw_pred - angles[:, 0]))
            pitch_errors.extend(np.abs(pitch_pred - angles[:, 1]))
            roll_errors.extend(np.abs(roll_pred - angles[:, 2]))

    yaw_mae = np.mean(yaw_errors)
    pitch_mae = np.mean(pitch_errors)
    roll_mae = np.mean(roll_errors)
    mean_mae = (yaw_mae + pitch_mae + roll_mae) / 3

    return {
        'yaw_mae': float(yaw_mae),
        'pitch_mae': float(pitch_mae),
        'roll_mae': float(roll_mae),
        'mean_mae': float(mean_mae),
    }


def compute_mae_by_range(model, dataloader, device):
    """Compute MAE broken down by yaw angle ranges."""
    model.eval()
    ranges = {
        '[0,30]': {'yaw': [], 'pitch': [], 'roll': []},
        '[30,60]': {'yaw': [], 'pitch': [], 'roll': []},
        '[60,90]': {'yaw': [], 'pitch': [], 'roll': []},
        '[90,180]': {'yaw': [], 'pitch': [], 'roll': []},
    }

    with torch.no_grad():
        for images, angles in dataloader:
            images = images.to(device)
            angles_np = angles.numpy()

            output = model(images)
            yaw_pred = output['yaw'].cpu().numpy()
            pitch_pred = output['pitch'].cpu().numpy()
            roll_pred = output['roll'].cpu().numpy()

            for i in range(len(angles_np)):
                abs_yaw = abs(angles_np[i, 0])
                yaw_err = abs(yaw_pred[i] - angles_np[i, 0])
                pitch_err = abs(pitch_pred[i] - angles_np[i, 1])
                roll_err = abs(roll_pred[i] - angles_np[i, 2])

                if abs_yaw <= 30:
                    key = '[0,30]'
                elif abs_yaw <= 60:
                    key = '[30,60]'
                elif abs_yaw <= 90:
                    key = '[60,90]'
                else:
                    key = '[90,180]'

                ranges[key]['yaw'].append(yaw_err)
                ranges[key]['pitch'].append(pitch_err)
                ranges[key]['roll'].append(roll_err)

    result = {}
    for key, data in ranges.items():
        if data['yaw']:
            result[key] = {
                'yaw_mae': float(np.mean(data['yaw'])),
                'pitch_mae': float(np.mean(data['pitch'])),
                'roll_mae': float(np.mean(data['roll'])),
                'mean_mae': float((np.mean(data['yaw']) + np.mean(data['pitch']) + np.mean(data['roll'])) / 3),
                'count': len(data['yaw']),
            }
    return result


def train_baseline(train_loader, val_loader, device, epochs=80, lr=1e-4):
    """Train baseline WHENet model."""
    print("\n" + "=" * 60)
    print("Training BASELINE WHENet (EfficientNet-B0, no improvements)")
    print("=" * 60)

    # Build baseline with correct architecture
    class BaselineWrapper(nn.Module):
        def __init__(self, pretrained=True, num_bins=66):
            super().__init__()
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            if pretrained:
                base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                base = efficientnet_b0(weights=None)
            self.features = base.features
            self.gap = nn.AdaptiveAvgPool2d(1)
            # EfficientNet-B0 final features output = 1280 (stage 8)
            self.num_bins = num_bins
            self.fc = nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            )
            self.fc_yaw = nn.Linear(256, num_bins)
            self.fc_pitch = nn.Linear(256, num_bins)
            self.fc_roll = nn.Linear(256, num_bins)
            self.fc_yaw_reg = nn.Linear(256, 1)
            self.fc_pitch_reg = nn.Linear(256, 1)
            self.fc_roll_reg = nn.Linear(256, 1)
            idx_tensor = torch.arange(num_bins, dtype=torch.float32)
            self.register_buffer('idx_tensor', idx_tensor)
            self.angle_range = (-99, 99)

        def forward(self, x):
            x = self.features(x)
            pooled = self.gap(x).flatten(1)
            feat = self.fc(pooled)
            yaw_cls = self.fc_yaw(feat)
            pitch_cls = self.fc_pitch(feat)
            roll_cls = self.fc_roll(feat)
            yaw_soft = torch.softmax(yaw_cls, dim=1)
            pitch_soft = torch.softmax(pitch_cls, dim=1)
            roll_soft = torch.softmax(roll_cls, dim=1)
            yaw_exp = (yaw_soft * self.idx_tensor).sum(dim=1)
            pitch_exp = (pitch_soft * self.idx_tensor).sum(dim=1)
            roll_exp = (roll_soft * self.idx_tensor).sum(dim=1)
            bw = (self.angle_range[1] - self.angle_range[0]) / self.num_bins
            yaw = yaw_exp * bw + self.angle_range[0] + self.fc_yaw_reg(feat).squeeze(-1)
            pitch = pitch_exp * bw + self.angle_range[0] + self.fc_pitch_reg(feat).squeeze(-1)
            roll = roll_exp * bw + self.angle_range[0] + self.fc_roll_reg(feat).squeeze(-1)
            return {'yaw': yaw, 'pitch': pitch, 'roll': roll,
                    'cls_logits': (yaw_cls, pitch_cls, roll_cls), 'features': feat}

    wrapped = BaselineWrapper(pretrained=True).to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(wrapped.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_mae = float('inf')
    best_state = None

    for epoch in range(epochs):
        wrapped.train()
        total_loss = 0
        n_batches = 0

        for images, angles in train_loader:
            images, angles = images.to(device), angles.to(device)
            optimizer.zero_grad()
            output = wrapped(images)
            losses = criterion(output, angles)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += losses['total'].item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Validation
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_mae = compute_mae(wrapped, val_loader, device)
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val MAE: Y={val_mae['yaw_mae']:.2f} P={val_mae['pitch_mae']:.2f} "
                  f"R={val_mae['roll_mae']:.2f} Mean={val_mae['mean_mae']:.2f}")

            if val_mae['mean_mae'] < best_val_mae:
                best_val_mae = val_mae['mean_mae']
                best_state = {k: v.cpu().clone() for k, v in wrapped.state_dict().items()}

    if best_state:
        wrapped.load_state_dict(best_state)
    return wrapped


def train_improved(train_loader, val_loader, device, epochs=100, lr=1e-4):
    """Train improved WHENet+ model."""
    print("\n" + "=" * 60)
    print("Training IMPROVED WHENet+ (Multi-scale + CBAM + Geodesic)")
    print("=" * 60)

    model = WHENetPlusSimple(pretrained=True, attention_type='cbam').to(device)
    criterion = CombinedLoss(cls_weight=1.0, reg_weight=1.5, geo_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine schedule
    warmup_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=lr * 0.01
    )

    best_val_mae = float('inf')
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        model.train()

        # Warmup
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        total_loss = 0
        n_batches = 0

        for images, angles in train_loader:
            images, angles = images.to(device), angles.to(device)
            optimizer.zero_grad()
            output = model(images)
            losses = criterion(output, angles)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += losses['total'].item()
            n_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_mae = compute_mae(model, val_loader, device)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | "
                  f"Val MAE: Y={val_mae['yaw_mae']:.2f} P={val_mae['pitch_mae']:.2f} "
                  f"R={val_mae['roll_mae']:.2f} Mean={val_mae['mean_mae']:.2f}")

            if val_mae['mean_mae'] < best_val_mae:
                best_val_mae = val_mae['mean_mae']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_and_report(baseline_model, improved_model, test_aflw_loader, test_biwi_loader,
                       device, output_dir):
    """Run full evaluation and generate report."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = {}

    # AFLW2000
    print("\n--- AFLW2000 ---")
    if test_aflw_loader is not None:
        baseline_aflw = compute_mae(baseline_model, test_aflw_loader, device)
        improved_aflw = compute_mae(improved_model, test_aflw_loader, device)
        improved_aflw_range = compute_mae_by_range(improved_model, test_aflw_loader, device)

        print(f"  Baseline  WHENet:  Yaw={baseline_aflw['yaw_mae']:.2f}  Pitch={baseline_aflw['pitch_mae']:.2f}  "
              f"Roll={baseline_aflw['roll_mae']:.2f}  Mean={baseline_aflw['mean_mae']:.2f}")
        print(f"  Improved WHENet+:  Yaw={improved_aflw['yaw_mae']:.2f}  Pitch={improved_aflw['pitch_mae']:.2f}  "
              f"Roll={improved_aflw['roll_mae']:.2f}  Mean={improved_aflw['mean_mae']:.2f}")

        improvement = baseline_aflw['mean_mae'] - improved_aflw['mean_mae']
        pct = (improvement / baseline_aflw['mean_mae'] * 100) if baseline_aflw['mean_mae'] > 0 else 0
        print(f"  Improvement: {improvement:.2f}° ({pct:.1f}%)")

        results['aflw2000'] = {
            'baseline': baseline_aflw,
            'improved': improved_aflw,
            'by_range': improved_aflw_range,
        }

    # BIWI
    print("\n--- BIWI ---")
    if test_biwi_loader is not None:
        baseline_biwi = compute_mae(baseline_model, test_biwi_loader, device)
        improved_biwi = compute_mae(improved_model, test_biwi_loader, device)
        improved_biwi_range = compute_mae_by_range(improved_model, test_biwi_loader, device)

        print(f"  Baseline  WHENet:  Yaw={baseline_biwi['yaw_mae']:.2f}  Pitch={baseline_biwi['pitch_mae']:.2f}  "
              f"Roll={baseline_biwi['roll_mae']:.2f}  Mean={baseline_biwi['mean_mae']:.2f}")
        print(f"  Improved WHENet+:  Yaw={improved_biwi['yaw_mae']:.2f}  Pitch={improved_biwi['pitch_mae']:.2f}  "
              f"Roll={improved_biwi['roll_mae']:.2f}  Mean={improved_biwi['mean_mae']:.2f}")

        improvement = baseline_biwi['mean_mae'] - improved_biwi['mean_mae']
        pct = (improvement / baseline_biwi['mean_mae'] * 100) if baseline_biwi['mean_mae'] > 0 else 0
        print(f"  Improvement: {improvement:.2f}° ({pct:.1f}%)")

        results['biwi'] = {
            'baseline': baseline_biwi,
            'improved': improved_biwi,
            'by_range': improved_biwi_range,
        }

    # Generate comparison table
    generate_report(results, output_dir)
    return results


def generate_report(results, output_dir):
    """Generate markdown report with comparison tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    with open(os.path.join(output_dir, 'hpe_results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Generate markdown report
    report = []
    report.append("# 头部姿态估计（HPE）公开数据集验证报告\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")

    # Literature values
    literature = {
        'aflw2000': {
            'HopeNet (2018)': {'yaw': 6.47, 'pitch': 6.56, 'roll': 5.44, 'mean': 6.16},
            'FSA-Net (2019)': {'yaw': 4.50, 'pitch': 6.08, 'roll': 4.64, 'mean': 5.07},
            'WHENet (2020)': {'yaw': 4.44, 'pitch': 5.75, 'roll': 4.31, 'mean': 4.83},
            '6DRepNet (2022)': {'yaw': 3.63, 'pitch': 4.91, 'roll': 3.37, 'mean': 3.97},
        },
        'biwi': {
            'HopeNet (2018)': {'yaw': 5.17, 'pitch': 6.98, 'roll': 3.39, 'mean': 5.18},
            'FSA-Net (2019)': {'yaw': 4.27, 'pitch': 4.96, 'roll': 2.76, 'mean': 4.00},
            'WHENet (2020)': {'yaw': 3.60, 'pitch': 4.10, 'roll': 2.73, 'mean': 3.48},
            '6DRepNet (2022)': {'yaw': 3.24, 'pitch': 4.48, 'roll': 2.68, 'mean': 3.47},
        },
    }

    for dataset_name, lit_values in literature.items():
        if dataset_name not in results:
            continue

        ds_results = results[dataset_name]
        ds_label = 'AFLW2000' if dataset_name == 'aflw2000' else 'BIWI'

        report.append(f"\n## {ds_label} 数据集上的MAE对比 (°)\n")
        report.append("| 方法 | Yaw | Pitch | Roll | Mean |")
        report.append("|------|:---:|:-----:|:----:|:----:|")

        for method, vals in lit_values.items():
            report.append(f"| {method} | {vals['yaw']:.2f} | {vals['pitch']:.2f} | {vals['roll']:.2f} | {vals['mean']:.2f} |")

        bl = ds_results['baseline']
        report.append(f"| WHENet (本文复现) | {bl['yaw_mae']:.2f} | {bl['pitch_mae']:.2f} | {bl['roll_mae']:.2f} | {bl['mean_mae']:.2f} |")

        imp = ds_results['improved']
        report.append(f"| **WHENet+ (本文改进)** | **{imp['yaw_mae']:.2f}** | **{imp['pitch_mae']:.2f}** | **{imp['roll_mae']:.2f}** | **{imp['mean_mae']:.2f}** |")

        # Improvement note
        improvement = bl['mean_mae'] - imp['mean_mae']
        pct = (improvement / bl['mean_mae'] * 100) if bl['mean_mae'] > 0 else 0
        report.append(f"\n改进效果：平均MAE降低 **{improvement:.2f}°** (相对降低 **{pct:.1f}%**)\n")

        # By range breakdown
        if 'by_range' in ds_results and ds_results['by_range']:
            report.append(f"\n### {ds_label} 按角度范围分析\n")
            report.append("| Yaw范围 | 样本数 | Yaw MAE | Pitch MAE | Roll MAE | Mean MAE |")
            report.append("|---------|:------:|:-------:|:---------:|:--------:|:--------:|")
            for rng, vals in sorted(ds_results['by_range'].items()):
                report.append(f"| {rng} | {vals['count']} | {vals['yaw_mae']:.2f} | {vals['pitch_mae']:.2f} | {vals['roll_mae']:.2f} | {vals['mean_mae']:.2f} |")

    # Model comparison summary
    report.append("\n---\n")
    report.append("## 改进方法说明\n")
    report.append("| 改进点 | 具体方法 | 作用 |")
    report.append("|-------|---------|------|")
    report.append("| 多尺度特征融合 | EfficientNet-B0 Stage 2/4/6 三层特征自适应加权融合 | 增强小目标和远距离人头的检测能力 |")
    report.append("| CBAM注意力机制 | 通道注意力 + 空间注意力串联 | 引导模型关注头部关键区域 |")
    report.append("| 测地距离损失 | 分类CE + 回归SmoothL1 + 测地距离联合优化 | 处理角度周期性，提升极端姿态精度 |")
    report.append("| 遮挡数据增强 | CoarseDropout + Random Erasing | 增强对面部遮挡的鲁棒性 |")

    report_text = '\n'.join(report)
    report_path = os.path.join(output_dir, 'hpe_experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {os.path.join(output_dir, 'hpe_results.json')}")


def main():
    parser = argparse.ArgumentParser(description='WHENet+ Training Pipeline')
    parser.add_argument('--data_dir', default='/root/autodl-tmp/behaviour/data/hpe_datasets',
                       help='Dataset root directory')
    parser.add_argument('--output_dir', default='/root/autodl-tmp/behaviour/hpe_experiment/results',
                       help='Output directory')
    parser.add_argument('--baseline_epochs', type=int, default=60,
                       help='Training epochs for baseline')
    parser.add_argument('--improved_epochs', type=int, default=80,
                       help='Training epochs for improved model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset, val_dataset, test_aflw, test_biwi, use_synthetic = create_datasets(
        args.data_dir, args.image_size
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_aflw_loader = DataLoader(test_aflw, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_biwi_loader = DataLoader(test_biwi, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Train baseline
    print("\n=== Phase 1: Training Baseline ===")
    baseline_model = train_baseline(train_loader, val_loader, device,
                                   epochs=args.baseline_epochs, lr=args.lr)

    # Train improved model
    print("\n=== Phase 2: Training Improved WHENet+ ===")
    improved_model = train_improved(train_loader, val_loader, device,
                                   epochs=args.improved_epochs, lr=args.lr)

    # Evaluate
    print("\n=== Phase 3: Evaluation ===")
    results = evaluate_and_report(
        baseline_model, improved_model,
        test_aflw_loader, test_biwi_loader,
        device, args.output_dir
    )

    # Save models
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(baseline_model.state_dict(),
               os.path.join(args.output_dir, 'whenet_baseline.pth'))
    torch.save(improved_model.state_dict(),
               os.path.join(args.output_dir, 'whenet_plus.pth'))
    print(f"\nModels saved to {args.output_dir}")

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
