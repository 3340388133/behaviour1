#!/usr/bin/env python
"""
使用人工标注数据进行评估
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import torch


def evaluate_whenet(data_dir, df):
    """评估 WHENet"""
    from head_pose import HeadPoseEstimator
    estimator = HeadPoseEstimator()

    preds = []
    times = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="WHENet"):
        img = cv2.imread(str(data_dir / row['face_path']))
        if img is None:
            preds.append((0, 0, 0))
            times.append(0)
            continue

        start = time.time()
        pose = estimator.estimate(img)
        times.append(time.time() - start)
        preds.append((pose.yaw, pose.pitch, pose.roll))

    return preds, np.mean(times) * 1000


def evaluate_6drepnet(data_dir, df):
    """评估 6DRepNet"""
    from sixdrepnet.model import SixDRepNet
    from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices
    from torchvision import transforms
    from PIL import Image

    device = torch.device('cpu')
    model = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
    checkpoint_path = Path.home() / '.cache/torch/hub/checkpoints/6DRepNet_300W_LP_AFLW2000.pth'
    if checkpoint_path.exists():
        state_dict = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    preds = []
    times = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="6DRepNet"):
        img_path = str(data_dir / row['face_path'])
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            start = time.time()
            with torch.no_grad():
                R_pred = model(img_tensor)
                euler = compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
                pitch = euler[:, 0].cpu().numpy()[0]
                yaw = euler[:, 1].cpu().numpy()[0]
                roll = euler[:, 2].cpu().numpy()[0]
            times.append(time.time() - start)
            preds.append((float(yaw), float(pitch), float(roll)))
        except:
            preds.append((0, 0, 0))
            times.append(0)

    return preds, np.mean(times) * 1000


def compute_metrics(preds, gt_yaw, gt_pitch, gt_roll):
    """计算评估指标"""
    pred_yaw = np.array([p[0] for p in preds])
    pred_pitch = np.array([p[1] for p in preds])
    pred_roll = np.array([p[2] for p in preds])

    yaw_mae = np.mean(np.abs(pred_yaw - gt_yaw))
    pitch_mae = np.mean(np.abs(pred_pitch - gt_pitch))
    roll_mae = np.mean(np.abs(pred_roll - gt_roll))

    yaw_acc5 = np.mean(np.abs(pred_yaw - gt_yaw) <= 5) * 100
    pitch_acc5 = np.mean(np.abs(pred_pitch - gt_pitch) <= 5) * 100
    roll_acc5 = np.mean(np.abs(pred_roll - gt_roll) <= 5) * 100

    return {
        'yaw_mae': yaw_mae,
        'pitch_mae': pitch_mae,
        'roll_mae': roll_mae,
        'yaw_acc5': yaw_acc5,
        'pitch_acc5': pitch_acc5,
        'roll_acc5': roll_acc5,
        'avg_mae': (yaw_mae + pitch_mae + roll_mae) / 3,
        'avg_acc5': (yaw_acc5 + pitch_acc5 + roll_acc5) / 3
    }


def main():
    data_dir = Path('data')
    df = pd.read_csv('data/manual_gt.csv')

    print(f"使用 {len(df)} 张人工标注图像进行评估\n")

    gt_yaw = df['yaw'].values
    gt_pitch = df['pitch'].values
    gt_roll = df['roll'].values

    results = []

    # WHENet
    print("[1/2] 评估 WHENet...")
    preds, avg_time = evaluate_whenet(data_dir, df)
    metrics = compute_metrics(preds, gt_yaw, gt_pitch, gt_roll)
    metrics['method'] = 'WHENet'
    metrics['time_ms'] = avg_time
    results.append(metrics)

    # 6DRepNet
    print("\n[2/2] 评估 6DRepNet...")
    preds, avg_time = evaluate_6drepnet(data_dir, df)
    metrics = compute_metrics(preds, gt_yaw, gt_pitch, gt_roll)
    metrics['method'] = '6DRepNet'
    metrics['time_ms'] = avg_time
    results.append(metrics)

    # 打印结果
    print("\n" + "=" * 95)
    print("头部姿态估计方法性能对比 (基于人工标注)")
    print("=" * 95)
    print(f"{'Method':<12} {'Yaw MAE':>10} {'Pitch MAE':>10} {'Roll MAE':>10} "
          f"{'Yaw@5°':>10} {'Pitch@5°':>10} {'Roll@5°':>10} {'Time(ms)':>10}")
    print("-" * 95)

    for r in results:
        print(f"{r['method']:<12} {r['yaw_mae']:>10.2f} {r['pitch_mae']:>10.2f} {r['roll_mae']:>10.2f} "
              f"{r['yaw_acc5']:>9.1f}% {r['pitch_acc5']:>9.1f}% {r['roll_acc5']:>9.1f}% "
              f"{r['time_ms']:>10.1f}")
    print("=" * 95)

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/benchmark_manual_gt.csv', index=False)
    print(f"\n结果已保存到: data/benchmark_manual_gt.csv")


if __name__ == "__main__":
    main()
