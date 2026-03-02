"""
动作识别模型训练脚本
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from video_dataset import get_dataloaders
from action_model import R3DNet


def set_seed(seed=42):
    """固定随机种子"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training")
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for frames, labels in tqdm(loader, desc="Validating"):
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main(args):
    """主训练函数"""
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, val_loader, classes = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_workers=args.num_workers
    )
    print(f"类别: {classes}")
    print(f"训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")

    # 创建模型
    model = R3DNet(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classes': classes
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"保存最佳模型, 准确率: {best_acc:.2f}%")

    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动作识别训练')
    parser.add_argument('--data_root', type=str, required=True, help='数据根目录')
    parser.add_argument('--save_dir', type=str, default='../models', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=16, help='每个视频采样帧数')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)