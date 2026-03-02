"""
测试视频Transformer Pipeline

验证所有组件是否正常工作
"""

import torch
import numpy as np
from pathlib import Path
from video_transformer_pipeline import (
    BehaviorVideoDataset,
    BehaviorVideoROIDataset,
    VideoTransformerClassifier
)


def test_dataset():
    """测试数据集加载"""
    print("=" * 60)
    print("测试 1: 数据集加载")
    print("=" * 60)

    try:
        dataset = BehaviorVideoDataset(
            behavior_json_path='dataset_root/annotations/behavior/MQ_S_IN_001/behavior.json',
            frames_dir='dataset_root/frames/MQ_S_IN_001',
            num_frames=16,
            image_size=224,
            mode='train'
        )

        print(f"✓ 数据集创建成功")
        print(f"  样本数: {len(dataset)}")

        # 加载一个样本
        clip, label = dataset[0]
        print(f"✓ 样本加载成功")
        print(f"  视频clip shape: {clip.shape}")  # 应该是 [16, 3, 224, 224]
        print(f"  标签: {label}")

        assert clip.shape == (16, 3, 224, 224), f"形状错误: {clip.shape}"
        print("✓ 数据集测试通过\n")
        return True

    except Exception as e:
        print(f"✗ 数据集测试失败: {e}\n")
        return False


def test_roi_dataset():
    """测试ROI数据集"""
    print("=" * 60)
    print("测试 2: ROI数据集")
    print("=" * 60)

    try:
        dataset = BehaviorVideoROIDataset(
            behavior_json_path='dataset_root/annotations/behavior/MQ_S_IN_001/behavior.json',
            frames_dir='dataset_root/frames/MQ_S_IN_001',
            detections_dir='dataset_root/annotations/detection',
            num_frames=16,
            mode='train'
        )

        print(f"✓ ROI数据集创建成功")
        print(f"  样本数: {len(dataset)}")

        clip, label = dataset[0]
        print(f"✓ ROI样本加载成功")
        print(f"  视频clip shape: {clip.shape}")
        print(f"  标签: {label}")

        print("✓ ROI数据集测试通过\n")
        return True

    except Exception as e:
        print(f"✗ ROI数据集测试失败: {e}\n")
        return False


def test_model():
    """测试模型"""
    print("=" * 60)
    print("测试 3: 模型前向传播")
    print("=" * 60)

    try:
        # 创建小模型用于测试
        model = VideoTransformerClassifier(
            num_classes=3,
            img_size=224,
            patch_size=16,
            num_frames=16,
            tubelet_size=2,
            embed_dim=384,  # 小一些
            depth=4,        # 少一些层
            num_heads=6,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功")
        print(f"  参数量: {total_params / 1e6:.2f}M")

        # 测试前向传播
        batch_size = 2
        dummy_input = torch.randn(batch_size, 16, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ 前向传播成功")
        print(f"  输入shape: {dummy_input.shape}")
        print(f"  输出shape: {output.shape}")

        assert output.shape == (batch_size, 3), f"输出形状错误: {output.shape}"
        print("✓ 模型测试通过\n")
        return True

    except Exception as e:
        print(f"✗ 模型测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试完整pipeline"""
    print("=" * 60)
    print("测试 4: 完整Pipeline")
    print("=" * 60)

    try:
        # 数据集
        dataset = BehaviorVideoDataset(
            behavior_json_path='dataset_root/annotations/behavior/MQ_S_IN_001/behavior.json',
            frames_dir='dataset_root/frames/MQ_S_IN_001',
            num_frames=16,
            mode='train'
        )

        # DataLoader
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # 模型
        model = VideoTransformerClassifier(
            num_classes=3,
            embed_dim=384,
            depth=4,
            num_heads=6
        )

        print(f"✓ 数据集和模型创建成功")

        # 获取一个batch
        videos, labels = next(iter(loader))
        print(f"✓ 数据加载成功")
        print(f"  videos shape: {videos.shape}")
        print(f"  labels shape: {labels.shape}")

        # 前向传播
        with torch.no_grad():
            outputs = model(videos)

        print(f"✓ 推理成功")
        print(f"  outputs shape: {outputs.shape}")

        # 计算损失
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        print(f"✓ 损失计算成功: {loss.item():.4f}")

        print("✓ 完整Pipeline测试通过\n")
        return True

    except Exception as e:
        print(f"✗ Pipeline测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试内存使用"""
    print("=" * 60)
    print("测试 5: 内存使用")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU内存测试\n")
        return True

    try:
        device = 'cuda'
        model = VideoTransformerClassifier(
            num_classes=3,
            embed_dim=384,
            depth=4,
            num_heads=6
        ).to(device)

        # 测试不同batch size
        for batch_size in [1, 2, 4]:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(batch_size, 16, 3, 224, 224).to(device)

            with torch.no_grad():
                output = model(dummy_input)

            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Batch size {batch_size}: {mem_used:.2f} GB")

            del dummy_input, output

        print("✓ 内存测试通过\n")
        return True

    except Exception as e:
        print(f"✗ 内存测试失败: {e}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("视频Transformer Pipeline 测试套件")
    print("=" * 60 + "\n")

    results = []

    # 运行所有测试
    results.append(("数据集加载", test_dataset()))
    results.append(("ROI数据集", test_roi_dataset()))
    results.append(("模型前向传播", test_model()))
    results.append(("完整Pipeline", test_full_pipeline()))
    results.append(("内存使用", test_memory_usage()))

    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！可以开始训练了。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    exit(main())
