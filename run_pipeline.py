#!/usr/bin/env python3
"""
完整流水线：可疑人员识别系统

一键运行所有步骤：
1. 视频跟踪（YOLOv8 + BoT-SORT）
2. 头部姿态估计
3. 构建数据集
4. 训练识别层模型
5. 消融实验
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


def run_command(cmd: str, desc: str, timeout: int = None) -> bool:
    """运行命令"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {desc}")
    print(f"{'='*60}")
    print(f"命令: {cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            print(f"警告: 命令返回码 {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"超时: {timeout}秒")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="可疑人员识别系统完整流水线")
    parser.add_argument("--skip-tracking", action="store_true", help="跳过跟踪步骤")
    parser.add_argument("--skip-pose", action="store_true", help="跳过姿态估计")
    parser.add_argument("--skip-dataset", action="store_true", help="跳过数据集构建")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    print("=" * 60)
    print("可疑人员识别系统 - 完整流水线")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # Step 1: 跟踪
    if not args.skip_tracking:
        success = run_command(
            f"python3 step3_person_tracking.py "
            f"--input data/raw_videos "
            f"--output data/tracked_output "
            f"--tracker botsort.yaml "
            f"--device {args.device.replace('cuda:', '')} "
            f"--conf 0.5",
            "Step 1: 视频跟踪 (YOLOv8 + BoT-SORT)",
            timeout=7200,  # 2小时
        )
        results["tracking"] = "完成" if success else "失败"
    else:
        print("\n跳过跟踪步骤")
        results["tracking"] = "跳过"

    # Step 2: 姿态估计
    if not args.skip_pose:
        success = run_command(
            f"python3 step4_head_pose.py "
            f"--input data/tracked_output "
            f"--output data/pose_output "
            f"--estimator opencv "
            f"--sample-rate 3",
            "Step 2: 头部姿态估计",
            timeout=3600,  # 1小时
        )
        results["pose"] = "完成" if success else "失败"
    else:
        print("\n跳过姿态估计步骤")
        results["pose"] = "跳过"

    # Step 3: 构建数据集
    if not args.skip_dataset:
        success = run_command(
            "python3 step5_build_dataset.py "
            "--input data/pose_output "
            "--output data/dataset "
            "--seq-length 32 "
            "--stride 16",
            "Step 3: 构建训练数据集",
            timeout=600,
        )
        results["dataset"] = "完成" if success else "失败"
    else:
        print("\n跳过数据集构建步骤")
        results["dataset"] = "跳过"

    # Step 4: 训练
    if not args.skip_training:
        if args.ablation:
            success = run_command(
                f"python3 step6_train_recognition.py "
                f"--data-dir data/dataset "
                f"--device {args.device} "
                f"--epochs {args.epochs} "
                f"--ablation",
                "Step 4: 消融实验",
                timeout=7200,
            )
            results["training"] = "消融实验完成" if success else "失败"
        else:
            success = run_command(
                f"python3 step6_train_recognition.py "
                f"--data-dir data/dataset "
                f"--model transformer "
                f"--device {args.device} "
                f"--epochs {args.epochs}",
                "Step 4: 训练 Transformer 模型",
                timeout=3600,
            )
            results["training"] = "完成" if success else "失败"
    else:
        print("\n跳过训练步骤")
        results["training"] = "跳过"

    # 总结
    print("\n" + "=" * 60)
    print("流水线完成!")
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n步骤状态:")
    for step, status in results.items():
        print(f"  {step}: {status}")

    # 保存结果
    results["timestamp"] = datetime.now().isoformat()
    with open("pipeline_results.json", 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
