"""
训练视频Transformer模型

用法:
  python train_video_transformer.py --config config.json
  python train_video_transformer.py --quick_test  # 快速测试
"""

import torch
import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
from video_transformer_pipeline import (
    BehaviorVideoDataset,
    BehaviorVideoROIDataset,
    VideoTransformerClassifier,
    Trainer
)


def create_dataloaders_from_splits(
    split_dir: str,
    dataset_root: str,
    use_roi: bool = False,
    num_frames: int = 16,
    batch_size: int = 4,
    num_workers: int = 4
):
    """
    从split文件创建数据加载器
    """
    split_dir = Path(split_dir)
    dataset_root = Path(dataset_root)

    # 读取划分配置
    with open(split_dir / 'split_config.json') as f:
        split_config = json.load(f)

    print(f"数据集信息:")
    print(f"  总视频数: {split_config['total_videos']}")
    print(f"  总轨迹数: {split_config['total_tracks']}")
    print(f"  标签分布: {split_config['label_distribution']}")

    # 加载训练/验证/测试split
    with open(split_dir / 'train.json') as f:
        train_split = json.load(f)
    with open(split_dir / 'val.json') as f:
        val_split = json.load(f)
    with open(split_dir / 'test.json') as f:
        test_split = json.load(f)

    # 创建数据集
    train_datasets = []
    val_datasets = []
    test_datasets = []

    DatasetClass = BehaviorVideoROIDataset if use_roi else BehaviorVideoDataset

    # 训练集
    for video_id in train_split['video_ids']:
        behavior_json = dataset_root / 'annotations' / 'behavior' / video_id / 'behavior.json'
        frames_dir = dataset_root / 'frames' / video_id

        if use_roi:
            detections_dir = dataset_root / 'annotations' / 'detection'
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                detections_dir=str(detections_dir),
                num_frames=num_frames,
                mode='train'
            )
        else:
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                num_frames=num_frames,
                mode='train'
            )
        train_datasets.append(dataset)

    # 验证集
    for video_id in val_split['video_ids']:
        behavior_json = dataset_root / 'annotations' / 'behavior' / video_id / 'behavior.json'
        frames_dir = dataset_root / 'frames' / video_id

        if use_roi:
            detections_dir = dataset_root / 'annotations' / 'detection'
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                detections_dir=str(detections_dir),
                num_frames=num_frames,
                mode='val'
            )
        else:
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                num_frames=num_frames,
                mode='val'
            )
        val_datasets.append(dataset)

    # 测试集
    for video_id in test_split['video_ids']:
        behavior_json = dataset_root / 'annotations' / 'behavior' / video_id / 'behavior.json'
        frames_dir = dataset_root / 'frames' / video_id

        if use_roi:
            detections_dir = dataset_root / 'annotations' / 'detection'
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                detections_dir=str(detections_dir),
                num_frames=num_frames,
                mode='test'
            )
        else:
            dataset = DatasetClass(
                behavior_json_path=str(behavior_json),
                frames_dir=str(frames_dir),
                num_frames=num_frames,
                mode='test'
            )
        test_datasets.append(dataset)

    # 合并数据集
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def main(args):
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 创建数据加载器
    print("\n准备数据...")
    train_loader, val_loader, test_loader = create_dataloaders_from_splits(
        split_dir=args.split_dir,
        dataset_root=args.dataset_root,
        use_roi=args.use_roi,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    print("\n创建模型...")
    model = VideoTransformerClassifier(
        num_classes=3,  # normal, looking_around, unknown
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=4.0,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )

    # 开始训练
    trainer.train()

    # 在测试集上评估
    print("\n在测试集上评估...")
    test_loss, test_acc = trainer.validate()
    print(f"测试集 Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

    # 保存最终结果
    results = {
        'best_val_acc': trainer.best_acc,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'config': vars(args)
    }

    with open(Path(args.save_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到: {args.save_dir}/results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练视频Transformer')

    # 数据相关
    parser.add_argument('--dataset_root', type=str, default='dataset_root',
                        help='数据集根目录')
    parser.add_argument('--split_dir', type=str, default='dataset_root/splits',
                        help='数据集划分目录')
    parser.add_argument('--use_roi', action='store_true',
                        help='使用ROI裁剪（多人场景）')

    # 模型参数
    parser.add_argument('--num_frames', type=int, default=16,
                        help='每个clip的帧数')
    parser.add_argument('--img_size', type=int, default=224,
                        help='图像大小')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch大小')
    parser.add_argument('--tubelet_size', type=int, default=2,
                        help='时间维度的tubelet大小')
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='嵌入维度')
    parser.add_argument('--depth', type=int, default=12,
                        help='Transformer层数')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='注意力头数')
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout率')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1,
                        help='注意力Dropout率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 其他
    parser.add_argument('--save_dir', type=str, default='checkpoints_video_transformer',
                        help='模型保存目录')
    parser.add_argument('--quick_test', action='store_true',
                        help='快速测试模式（少量epoch）')

    args = parser.parse_args()

    # 快速测试模式
    if args.quick_test:
        args.num_epochs = 5
        args.batch_size = 2
        args.depth = 4  # 更小的模型
        args.embed_dim = 384
        args.num_heads = 6
        print("⚠️ 快速测试模式")

    main(args)
