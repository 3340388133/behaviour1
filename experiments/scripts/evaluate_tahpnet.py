#!/usr/bin/env python3
"""TAHPNet 评估脚本

评估指标：
- MAE: 平均绝对误差 (度)
- Jitter: 帧间抖动 (度/帧)
- Smoothness: 加速度平滑度
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, 'src/recognition/models')
from tahpnet import TAHPNet


def evaluate_jitter(poses):
    """计算姿态序列的抖动度"""
    if poses.shape[1] < 2:
        return 0.0
    diff = poses[:, 1:, :] - poses[:, :-1, :]  # [B, T-1, 3]
    jitter = diff.abs().mean().item()
    return jitter


def evaluate_smoothness(poses):
    """计算姿态序列的平滑度（加速度）"""
    if poses.shape[1] < 3:
        return 0.0
    diff = poses[:, 1:, :] - poses[:, :-1, :]
    accel = diff[:, 1:, :] - diff[:, :-1, :]
    smoothness = accel.abs().mean().item()
    return smoothness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, default='data/tracked_output')
    parser.add_argument('--pose-labels', type=str, default='data/pose_output/all_poses.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model = TAHPNet()
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(args.device)
    model.eval()

    use_temporal = ckpt.get('use_temporal', True)
    print(f"时序模块: {'启用' if use_temporal else '禁用'}")

    # 加载数据
    from train_tahpnet import HeadPoseSequenceDataset
    from torch.utils.data import DataLoader

    dataset = HeadPoseSequenceDataset(
        data_root=args.data_root,
        pose_labels_path=args.pose_labels,
        seq_len=16,
        stride=16,
        mode='test',
        augment=False
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # 评估
    total_mae = []
    total_jitter = []
    total_smoothness = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估中"):
            images = batch['images'].to(args.device)
            gt_poses = batch['poses'].to(args.device)

            outputs = model(images, use_temporal=use_temporal)
            pred_poses = outputs['pose']

            # MAE
            mae = (pred_poses - gt_poses).abs().mean(dim=[1, 2])
            total_mae.extend(mae.cpu().numpy())

            # Jitter
            jitter = evaluate_jitter(pred_poses)
            total_jitter.append(jitter)

            # Smoothness
            smooth = evaluate_smoothness(pred_poses)
            total_smoothness.append(smooth)

    # 结果
    results = {
        'checkpoint': args.checkpoint,
        'use_temporal': use_temporal,
        'MAE': float(np.mean(total_mae)),
        'Jitter': float(np.mean(total_jitter)),
        'Smoothness': float(np.mean(total_smoothness)),
        'num_samples': len(dataset),
    }

    print("\n" + "=" * 50)
    print("评估结果:")
    print(f"  MAE: {results['MAE']:.2f}°")
    print(f"  Jitter: {results['Jitter']:.2f}°/帧")
    print(f"  Smoothness: {results['Smoothness']:.4f}")
    print("=" * 50)

    # 保存结果
    output_path = Path(args.checkpoint).parent / 'eval_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"结果保存到: {output_path}")


if __name__ == '__main__':
    main()
