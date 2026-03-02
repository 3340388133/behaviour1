#!/usr/bin/env python3
"""
Step 6: 训练识别层模型

支持模型：
- transformer: 时序 Transformer（创新点）
- lstm: LSTM 基线
- rule: 规则基线

6类行为分类：
- 0: normal        正常行为
- 1: glancing      频繁张望
- 2: quick_turn    快速回头
- 3: prolonged_watch 长时间观察
- 4: looking_down  持续低头
- 5: looking_up    持续抬头

消融实验设计：
- A1: Transformer vs 规则
- A2: Transformer vs LSTM
- A3: 不确定性加权 vs 固定权重
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
from tqdm import tqdm
import sys

# 直接导入，避免加载整个 src 包
sys.path.insert(0, str(Path(__file__).parent / "src" / "recognition"))
from temporal_transformer import (
    SuspiciousBehaviorClassifier,
    LSTMBaseline,
    RuleBaseline,
    create_model,
)

# ============== 配置 ==============
DATA_ROOT = Path("data")
DATASET_DIR = DATA_ROOT / "dataset"
CHECKPOINT_DIR = Path("checkpoints")


class PoseSequenceDataset(Dataset):
    """姿态序列数据集"""

    def __init__(self, data_file: Path, augment: bool = False):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = data["samples"]
        self.seq_length = data.get("seq_length", 32)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 姿态序列 [T, 3]
        pose_seq = torch.tensor(sample["pose_sequence"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)

        # 数据增强
        if self.augment:
            pose_seq = self._augment(pose_seq)

        return {
            "pose_seq": pose_seq,
            "label": label,
            "video": sample["video"],
            "track_id": sample["track_id"],
        }

    def _augment(self, pose_seq: torch.Tensor) -> torch.Tensor:
        """数据增强"""
        # 随机噪声
        if np.random.random() < 0.5:
            noise = torch.randn_like(pose_seq) * 2.0  # 2度噪声
            pose_seq = pose_seq + noise

        # 随机时间翻转
        if np.random.random() < 0.3:
            pose_seq = torch.flip(pose_seq, dims=[0])

        # 随机左右翻转（Yaw 取反）
        if np.random.random() < 0.5:
            pose_seq[:, 0] = -pose_seq[:, 0]  # Yaw 取反

        return pose_seq


class Trainer:
    """训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda:0",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        # 记录
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            pose_seq = batch["pose_seq"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            logits, confidence = self.model(pose_seq)

            # 计算损失
            if hasattr(self.model, 'compute_loss'):
                loss, _ = self.model.compute_loss(logits, confidence, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """评估"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_confs = []

        for batch in loader:
            pose_seq = batch["pose_seq"].to(self.device)
            labels = batch["label"].to(self.device)

            logits, confidence = self.model(pose_seq)

            if hasattr(self.model, 'compute_loss'):
                loss, _ = self.model.compute_loss(logits, confidence, labels)
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confidence.squeeze().cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算指标
        accuracy = (all_preds == all_labels).mean()

        # 多分类 Macro F1 Score
        num_classes = max(all_labels.max(), all_preds.max()) + 1
        precisions = []
        recalls = []
        f1s = []

        for c in range(num_classes):
            tp = ((all_preds == c) & (all_labels == c)).sum()
            fp = ((all_preds == c) & (all_labels != c)).sum()
            fn = ((all_preds != c) & (all_labels == c)).sum()

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        # Macro 平均
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return {
            "loss": total_loss / len(loader),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_f1": f1s,
        }

    def train(
        self,
        epochs: int = 100,
        save_dir: Path = CHECKPOINT_DIR,
        model_name: str = "model",
    ) -> Dict:
        """完整训练流程"""
        save_dir.mkdir(parents=True, exist_ok=True)
        best_f1 = 0
        best_epoch = 0

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_metrics = self.evaluate(self.val_loader)

            # 更新学习率
            self.scheduler.step()

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])

            # 打印
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")

            # 保存最佳模型
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch + 1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                }, save_dir / f"{model_name}_best.pt")
                print(f"  ★ 保存最佳模型 (F1: {best_f1:.4f})")

        # 保存最终模型
        torch.save({
            "epoch": epochs,
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }, save_dir / f"{model_name}_final.pt")

        return {
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "history": self.history,
        }


def run_ablation_experiments(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str = "cuda:0",
    epochs: int = 50,
) -> Dict:
    """运行消融实验"""
    results = {}

    experiments = [
        ("rule", "规则基线", {"model_type": "rule", "num_classes": 6}),
        ("lstm", "LSTM 基线", {"model_type": "lstm", "num_classes": 6}),
        ("transformer", "Transformer（创新点）", {"model_type": "transformer", "num_classes": 6, "uncertainty_weighting": False}),
        ("transformer_uw", "Transformer + 不确定性加权", {"model_type": "transformer", "num_classes": 6, "uncertainty_weighting": True}),
    ]

    for exp_name, exp_desc, kwargs in experiments:
        print(f"\n{'='*60}")
        print(f"实验: {exp_desc}")
        print(f"{'='*60}")

        model = create_model(**kwargs)

        if exp_name == "rule":
            # 规则基线不需要训练
            model = model.to(device)
            model.eval()

            # 直接评估
            all_preds = []
            all_labels = []

            for batch in test_loader:
                pose_seq = batch["pose_seq"].to(device)
                labels = batch["label"]

                with torch.no_grad():
                    logits, _ = model(pose_seq)
                    pred = logits.argmax(dim=-1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            accuracy = (all_preds == all_labels).mean()

            # 多分类 Macro F1
            num_classes = 6
            precisions = []
            recalls = []
            f1s = []
            for c in range(num_classes):
                tp = ((all_preds == c) & (all_labels == c)).sum()
                fp = ((all_preds == c) & (all_labels != c)).sum()
                fn = ((all_preds != c) & (all_labels == c)).sum()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                precisions.append(p)
                recalls.append(r)
                f1s.append(f)

            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)

            results[exp_name] = {
                "description": exp_desc,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
            }
            print(f"Test Acc: {accuracy:.4f}, F1: {f1:.4f}")

        else:
            # 训练模型
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
            )

            train_result = trainer.train(
                epochs=epochs,
                model_name=exp_name,
            )

            # 测试集评估
            test_metrics = trainer.evaluate(test_loader)

            results[exp_name] = {
                "description": exp_desc,
                "best_val_f1": train_result["best_f1"],
                "best_epoch": train_result["best_epoch"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Step 6: 训练识别层模型")
    parser.add_argument("--data-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "lstm", "rule"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ablation", action="store_true",
                        help="运行消融实验")

    args = parser.parse_args()

    print("=" * 60)
    print("Step 6: 训练识别层模型")
    print("=" * 60)

    data_dir = Path(args.data_dir)

    # 检查数据集
    train_file = data_dir / "train.json"
    val_file = data_dir / "val.json"
    test_file = data_dir / "test.json"

    if not train_file.exists():
        print(f"数据集不存在: {train_file}")
        print("请先运行 step5_build_dataset.py")
        return

    # 加载数据集
    print("\n加载数据集...")
    train_dataset = PoseSequenceDataset(train_file, augment=True)
    val_dataset = PoseSequenceDataset(val_file, augment=False)
    test_dataset = PoseSequenceDataset(test_file, augment=False)

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    if args.ablation:
        # 消融实验
        print("\n运行消融实验...")
        results = run_ablation_experiments(
            train_loader, val_loader, test_loader,
            device=args.device,
            epochs=args.epochs,
        )

        # 打印结果表格
        print("\n" + "=" * 80)
        print("消融实验结果")
        print("=" * 80)
        print(f"{'模型':<30} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 80)
        for name, r in results.items():
            print(f"{r['description']:<30} {r['test_accuracy']:.4f}       "
                  f"{r['test_precision']:.4f}       {r['test_recall']:.4f}       {r['test_f1']:.4f}")

        # 保存结果
        results_path = CHECKPOINT_DIR / "ablation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {results_path}")

    else:
        # 单模型训练
        print(f"\n训练模型: {args.model}")

        model = create_model(
            model_type=args.model,
            num_classes=6,  # 6类行为分类
            uncertainty_weighting=True,
        )

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            lr=args.lr,
        )

        result = trainer.train(
            epochs=args.epochs,
            model_name=args.model,
        )

        # 测试集评估
        test_metrics = trainer.evaluate(test_loader)

        print("\n" + "=" * 60)
        print("训练完成!")
        print(f"  最佳验证 F1: {result['best_f1']:.4f} (Epoch {result['best_epoch']})")
        print(f"  测试集准确率: {test_metrics['accuracy']:.4f}")
        print(f"  测试集 F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
