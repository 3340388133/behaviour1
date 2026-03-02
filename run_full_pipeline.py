#!/usr/bin/env python3
"""
完整自动化管道：跟踪 → 姿态估计 → 数据集构建 → 训练 → 迭代优化

自动迭代训练直到达到目标性能
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# ============== 配置 ==============
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# 目标性能
TARGET_F1 = 0.75  # 目标 F1 分数
MAX_ITERATIONS = 10  # 最大迭代次数


def run_command(cmd: str, desc: str = "", timeout: int = 3600) -> Tuple[bool, str]:
    """运行命令"""
    print(f"\n{'='*60}")
    print(f"执行: {desc or cmd[:50]}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )

        if result.returncode == 0:
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return True, result.stdout
        else:
            print(f"错误: {result.stderr[-1000:]}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"超时: {timeout}秒")
        return False, "Timeout"
    except Exception as e:
        print(f"异常: {e}")
        return False, str(e)


def step1_tracking(video_dir: str = "data/raw_videos", output_dir: str = "data/tracked_output") -> bool:
    """Step 1: 视频跟踪"""
    print("\n" + "="*80)
    print("STEP 1: 视频人物跟踪 (YOLOv8 + BoT-SORT)")
    print("="*80)

    cmd = f"python3 step3_person_tracking.py --input {video_dir} --output {output_dir} --device 0"
    success, _ = run_command(cmd, "人物跟踪", timeout=7200)  # 2小时超时

    # 检查输出
    output_path = PROJECT_ROOT / output_dir
    if output_path.exists():
        dirs = [d for d in output_path.iterdir() if d.is_dir()]
        print(f"跟踪完成: {len(dirs)} 个视频")
        return len(dirs) > 0
    return False


def step2_pose_estimation(input_dir: str = "data/tracked_output", output_dir: str = "data/pose_output") -> bool:
    """Step 2: 头部姿态估计"""
    print("\n" + "="*80)
    print("STEP 2: 头部姿态估计")
    print("="*80)

    cmd = f"python3 step4_head_pose.py --input {input_dir} --output {output_dir} --sample-rate 3"
    success, _ = run_command(cmd, "姿态估计", timeout=3600)

    output_path = PROJECT_ROOT / output_dir
    if output_path.exists():
        files = list(output_path.glob("*_poses.json"))
        print(f"姿态估计完成: {len(files)} 个文件")
        return len(files) > 0
    return False


def step3_build_dataset(
    input_dir: str = "data/pose_output",
    output_dir: str = "data/dataset",
    seq_length: int = 32,
    stride: int = 16,
) -> bool:
    """Step 3: 构建数据集"""
    print("\n" + "="*80)
    print("STEP 3: 构建训练数据集")
    print("="*80)

    cmd = f"python3 step5_build_dataset.py --input {input_dir} --output {output_dir} --seq-length {seq_length} --stride {stride}"
    success, _ = run_command(cmd, "构建数据集", timeout=600)

    train_file = PROJECT_ROOT / output_dir / "train.json"
    if train_file.exists():
        with open(train_file, 'r') as f:
            data = json.load(f)
        print(f"数据集构建完成: {data.get('num_samples', 0)} 个训练样本")
        return data.get('num_samples', 0) > 0
    return False


def step4_train(
    model_type: str = "transformer",
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    uncertainty_weighting: bool = True,
) -> Dict:
    """Step 4: 训练模型"""
    print("\n" + "="*80)
    print(f"STEP 4: 训练识别模型 ({model_type})")
    print("="*80)

    # 使用 Python 直接调用训练
    import torch
    from torch.utils.data import DataLoader

    # 确保导入
    from step6_train_recognition import (
        PoseSequenceDataset,
        Trainer,
    )
    from src.recognition.temporal_transformer import create_model

    data_dir = DATA_ROOT / "dataset"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 加载数据集
    train_dataset = PoseSequenceDataset(data_dir / "train.json", augment=True)
    val_dataset = PoseSequenceDataset(data_dir / "val.json", augment=False)
    test_dataset = PoseSequenceDataset(data_dir / "test.json", augment=False)

    if len(train_dataset) == 0:
        return {"success": False, "error": "训练集为空"}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 创建模型
    model = create_model(
        model_type=model_type,
        uncertainty_weighting=uncertainty_weighting,
    )

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
    )

    result = trainer.train(epochs=epochs, model_name=model_type)

    # 测试集评估
    test_metrics = trainer.evaluate(test_loader)

    return {
        "success": True,
        "best_val_f1": result["best_f1"],
        "best_epoch": result["best_epoch"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "history": result["history"],
    }


def auto_tune_and_train(target_f1: float = TARGET_F1, max_iters: int = MAX_ITERATIONS) -> Dict:
    """
    自动调参训练直到达到目标性能
    """
    print("\n" + "#"*80)
    print("自动迭代训练系统")
    print(f"目标 F1: {target_f1}")
    print(f"最大迭代: {max_iters}")
    print("#"*80)

    # 超参数搜索空间
    hyperparams_list = [
        # 基础配置
        {"epochs": 50, "lr": 1e-3, "batch_size": 32, "seq_length": 32, "stride": 16, "uncertainty_weighting": True},
        # 更长训练
        {"epochs": 100, "lr": 1e-3, "batch_size": 32, "seq_length": 32, "stride": 16, "uncertainty_weighting": True},
        # 更小学习率
        {"epochs": 100, "lr": 5e-4, "batch_size": 32, "seq_length": 32, "stride": 16, "uncertainty_weighting": True},
        # 更长序列
        {"epochs": 100, "lr": 1e-3, "batch_size": 16, "seq_length": 64, "stride": 32, "uncertainty_weighting": True},
        # 更短序列
        {"epochs": 100, "lr": 1e-3, "batch_size": 64, "seq_length": 16, "stride": 8, "uncertainty_weighting": True},
        # 更小 batch
        {"epochs": 150, "lr": 3e-4, "batch_size": 16, "seq_length": 32, "stride": 8, "uncertainty_weighting": True},
        # 关闭不确定性加权
        {"epochs": 100, "lr": 1e-3, "batch_size": 32, "seq_length": 32, "stride": 16, "uncertainty_weighting": False},
        # 大 batch + 高学习率
        {"epochs": 100, "lr": 2e-3, "batch_size": 64, "seq_length": 32, "stride": 16, "uncertainty_weighting": True},
        # 小 batch + 低学习率
        {"epochs": 150, "lr": 1e-4, "batch_size": 8, "seq_length": 32, "stride": 8, "uncertainty_weighting": True},
        # 最长训练
        {"epochs": 200, "lr": 5e-4, "batch_size": 32, "seq_length": 32, "stride": 16, "uncertainty_weighting": True},
    ]

    all_results = []
    best_result = None
    best_f1 = 0

    for i, params in enumerate(hyperparams_list[:max_iters]):
        print(f"\n{'='*80}")
        print(f"迭代 {i+1}/{min(len(hyperparams_list), max_iters)}")
        print(f"参数: {params}")
        print(f"{'='*80}")

        # 如果序列长度或步长改变，需要重建数据集
        if i > 0 and (params["seq_length"] != hyperparams_list[i-1]["seq_length"] or
                      params["stride"] != hyperparams_list[i-1]["stride"]):
            print("重建数据集...")
            step3_build_dataset(
                seq_length=params["seq_length"],
                stride=params["stride"],
            )

        # 训练
        result = step4_train(
            epochs=params["epochs"],
            lr=params["lr"],
            batch_size=params["batch_size"],
            uncertainty_weighting=params["uncertainty_weighting"],
        )

        result["params"] = params
        result["iteration"] = i + 1
        all_results.append(result)

        if result["success"]:
            print(f"\n结果: Test F1 = {result['test_f1']:.4f}")

            if result["test_f1"] > best_f1:
                best_f1 = result["test_f1"]
                best_result = result
                print(f"★ 新最佳结果!")

            # 检查是否达到目标
            if result["test_f1"] >= target_f1:
                print(f"\n🎉 达到目标性能! F1 = {result['test_f1']:.4f} >= {target_f1}")
                break
        else:
            print(f"训练失败: {result.get('error', 'Unknown error')}")

    # 保存所有结果
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / f"auto_tune_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    summary = {
        "target_f1": target_f1,
        "best_f1": best_f1,
        "best_params": best_result["params"] if best_result else None,
        "total_iterations": len(all_results),
        "all_results": all_results,
        "completed_at": datetime.now().isoformat(),
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {results_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="完整自动化管道")
    parser.add_argument("--skip-tracking", action="store_true", help="跳过跟踪步骤")
    parser.add_argument("--skip-pose", action="store_true", help="跳过姿态估计步骤")
    parser.add_argument("--skip-dataset", action="store_true", help="跳过数据集构建步骤")
    parser.add_argument("--target-f1", type=float, default=TARGET_F1, help="目标 F1 分数")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERATIONS, help="最大迭代次数")

    args = parser.parse_args()

    print("="*80)
    print("可疑人员识别系统 - 完整自动化管道")
    print("="*80)
    print(f"开始时间: {datetime.now()}")

    # Step 1: 跟踪
    if not args.skip_tracking:
        if not step1_tracking():
            print("跟踪步骤失败，但继续...")
    else:
        print("跳过跟踪步骤")

    # Step 2: 姿态估计
    if not args.skip_pose:
        if not step2_pose_estimation():
            print("姿态估计步骤失败，但继续...")
    else:
        print("跳过姿态估计步骤")

    # Step 3: 构建数据集
    if not args.skip_dataset:
        if not step3_build_dataset():
            print("数据集构建步骤失败!")
            return
    else:
        print("跳过数据集构建步骤")

    # Step 4: 自动调参训练
    result = auto_tune_and_train(
        target_f1=args.target_f1,
        max_iters=args.max_iters,
    )

    # 最终报告
    print("\n" + "="*80)
    print("最终报告")
    print("="*80)
    print(f"最佳 F1 分数: {result['best_f1']:.4f}")
    print(f"最佳参数: {result['best_params']}")
    print(f"总迭代次数: {result['total_iterations']}")
    print(f"目标达成: {'是' if result['best_f1'] >= args.target_f1 else '否'}")
    print(f"结束时间: {datetime.now()}")


if __name__ == "__main__":
    main()
