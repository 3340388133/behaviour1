#!/usr/bin/env python
"""
多方法姿态估计对比评估 - 支持 CPU
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


def evaluate_fsanet(data_dir, df):
    """评估 FSA-Net (Keras)"""
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    model_path = Path(__file__).parent.parent.parent / "models" / "fsanet.h5"
    if not model_path.exists():
        raise FileNotFoundError(f"FSA-Net model not found: {model_path}")

    model = tf.keras.models.load_model(str(model_path), compile=False)

    preds = []
    times = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FSA-Net"):
        img = cv2.imread(str(data_dir / row['face_path']))
        if img is None:
            preds.append((0, 0, 0))
            times.append(0)
            continue

        # 预处理: resize to 64x64, normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (64, 64))
        img_norm = (img_resized.astype(np.float32) - 127.5) / 128.0
        img_input = np.expand_dims(img_norm, axis=0)

        start = time.time()
        output = model.predict(img_input, verbose=0)
        times.append(time.time() - start)

        # FSA-Net 输出: [yaw, pitch, roll]
        yaw, pitch, roll = output[0]
        preds.append((float(yaw), float(pitch), float(roll)))

    return preds, np.mean(times) * 1000


def evaluate_6drepnet_cpu(data_dir, df):
    """评估 6DRepNet (CPU 版本)"""
    from sixdrepnet.model import SixDRepNet_Detector
    from torchvision import transforms
    from PIL import Image

    # 手动加载模型到 CPU
    model = SixDRepNet_Detector(gpu_id=-1)  # -1 表示 CPU

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
        img = cv2.imread(img_path)
        if img is None:
            preds.append((0, 0, 0))
            times.append(0)
            continue

        start = time.time()
        pitch, yaw, roll = model.predict(img_path)
        times.append(time.time() - start)
        preds.append((float(yaw), float(pitch), float(roll)))

    return preds, np.mean(times) * 1000


def evaluate_6drepnet_manual(data_dir, df):
    """评估 6DRepNet - 手动加载模型"""
    from sixdrepnet.model import SixDRepNet
    from sixdrepnet.utils import compute_euler_angles_from_rotation_matrices
    from torchvision import transforms
    from PIL import Image

    device = torch.device('cpu')

    # 加载模型
    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    # 加载权重
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
        except Exception as e:
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
    df = pd.read_csv('data/sample_gt_labeled.csv')

    gt_yaw = df['yaw'].values
    gt_pitch = df['pitch'].values
    gt_roll = df['roll'].values

    results = []

    # WHENet
    print("\n[1/3] 评估 WHENet...")
    preds, avg_time = evaluate_whenet(data_dir, df)
    metrics = compute_metrics(preds, gt_yaw, gt_pitch, gt_roll)
    metrics['method'] = 'WHENet'
    metrics['time_ms'] = avg_time
    results.append(metrics)

    # FSA-Net
    print("\n[2/3] 评估 FSA-Net...")
    try:
        preds, avg_time = evaluate_fsanet(data_dir, df)
        metrics = compute_metrics(preds, gt_yaw, gt_pitch, gt_roll)
        metrics['method'] = 'FSA-Net'
        metrics['time_ms'] = avg_time
        results.append(metrics)
    except Exception as e:
        print(f"FSA-Net 评估失败: {e}")

    # 6DRepNet
    print("\n[3/3] 评估 6DRepNet (CPU)...")
    try:
        preds, avg_time = evaluate_6drepnet_manual(data_dir, df)
        metrics = compute_metrics(preds, gt_yaw, gt_pitch, gt_roll)
        metrics['method'] = '6DRepNet'
        metrics['time_ms'] = avg_time
        results.append(metrics)
    except Exception as e:
        print(f"6DRepNet 评估失败: {e}")

    # 打印结果
    print("\n" + "=" * 95)
    print("头部姿态估计方法性能对比 (基于 WHENet 伪标注)")
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
    results_df.to_csv('data/benchmark_results.csv', index=False)
    print(f"\n结果已保存到: data/benchmark_results.csv")


if __name__ == "__main__":
    main()
