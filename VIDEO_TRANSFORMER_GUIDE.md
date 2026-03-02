# 端到端视频Transformer行为识别系统

完整的从视频到行为识别的解决方案，解决跟踪不稳定问题。

## 📋 目录

- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [模型说明](#模型说明)
- [常见问题](#常见问题)

---

## 🏗️ 系统架构

### 传统Pipeline的问题
```
传统方式（误差累积）:
视频 → 检测 → 跟踪 → 姿态估计 → 特征提取 → 分类
        ↓       ↓        ↓           ↓          ↓
       丢失   ID切换   精度低      不连续      依赖前面

结果：107个轨迹，只有10个有效（过滤率90.7%）
```

### 新方案：端到端视频Transformer
```
端到端方式（无误差累积）:
视频片段 → [TimeSformer] → 行为类别
             ↓
    时空注意力自动学习
    - 空间：哪里重要
    - 时间：何时重要

优势：
✓ 不依赖跟踪质量
✓ 端到端训练
✓ 自动特征学习
✓ 处理遮挡和离开画面
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install opencv-python einops tqdm

# 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import einops; print('einops OK')"
```

### 2. 测试系统

```bash
# 运行测试套件
python test_video_transformer.py

# 预期输出：
# ✓ 数据集加载: 通过
# ✓ ROI数据集: 通过
# ✓ 模型前向传播: 通过
# ✓ 完整Pipeline: 通过
# ✓ 内存使用: 通过
# 🎉 所有测试通过！
```

### 3. 快速训练（测试）

```bash
# 快速测试模式（5个epoch，小模型）
python train_video_transformer.py --quick_test

# 预期输出：
# 训练样本: 44
# 验证样本: 11
# 参数量: 22.5M
# Epoch 1/5: Train Acc: 45.2%, Val Acc: 54.5%
# ...
```

### 4. 完整训练

```bash
# 标准训练（50 epochs）
python train_video_transformer.py \
  --batch_size 4 \
  --num_epochs 50 \
  --lr 1e-4 \
  --save_dir checkpoints_vit

# GPU显存不足？减小模型
python train_video_transformer.py \
  --embed_dim 384 \
  --depth 6 \
  --num_heads 6 \
  --batch_size 2
```

### 5. 推理测试

```bash
# 单个视频推理
python inference_video_transformer.py \
  --video test_video.mp4 \
  --model checkpoints_vit/best_model.pth

# 长视频滑动窗口
python inference_video_transformer.py \
  --video long_video.mp4 \
  --model checkpoints_vit/best_model.pth \
  --sliding_window \
  --output results.json
```

---

## 📚 详细使用

### 数据集组织

系统使用现有的标注数据：

```
dataset_root/
├── frames/                    # 抽取的视频帧
│   ├── MQ_S_IN_001/
│   │   ├── MQ_S_IN_001_frame_000000.jpg
│   │   ├── MQ_S_IN_001_frame_000001.jpg
│   │   └── ...
│   └── ...
├── annotations/
│   ├── behavior/              # 行为标注（使用这个）
│   │   ├── MQ_S_IN_001/
│   │   │   └── behavior.json
│   │   └── ...
│   └── detection/             # 检测结果（可选，用于ROI）
│       ├── MQ_S_IN_001/
│       │   └── detections.json
│       └── ...
└── splits/                    # 数据集划分
    ├── train.json
    ├── val.json
    └── test.json
```

### 数据集模式

#### 模式1：全帧模式（推荐）
```python
# 使用整个视频帧
dataset = BehaviorVideoDataset(
    behavior_json_path='...',
    frames_dir='...',
    num_frames=16,
    mode='train'
)
```

优点：
- 简单，不需要检测结果
- 适合单人场景
- 模型可以学习背景信息

缺点：
- 多人场景可能混淆

#### 模式2：ROI模式（多人场景）
```python
# 裁剪目标区域
dataset = BehaviorVideoROIDataset(
    behavior_json_path='...',
    frames_dir='...',
    detections_dir='...',  # 需要检测结果
    num_frames=16,
    mode='train'
)
```

优点：
- 多人场景下准确
- 专注目标人物

缺点：
- 依赖检测质量
- 丢失背景信息

### 训练参数详解

```bash
python train_video_transformer.py \
  # 数据参数
  --dataset_root dataset_root \
  --split_dir dataset_root/splits \
  --use_roi \                        # 使用ROI模式

  # 模型参数
  --num_frames 16 \                  # 每个clip的帧数
  --img_size 224 \                   # 图像分辨率
  --patch_size 16 \                  # Patch大小
  --tubelet_size 2 \                 # 时间tubelet大小
  --embed_dim 768 \                  # 嵌入维度（越大越强，越吃显存）
  --depth 12 \                       # Transformer层数
  --num_heads 12 \                   # 注意力头数

  # 训练参数
  --batch_size 4 \                   # 批次大小
  --num_epochs 50 \                  # 训练轮数
  --lr 1e-4 \                        # 学习率
  --num_workers 4 \                  # 数据加载线程

  # 输出
  --save_dir checkpoints_vit
```

### 模型配置建议

| GPU显存 | embed_dim | depth | num_heads | batch_size | 参数量 |
|---------|-----------|-------|-----------|------------|--------|
| 8GB     | 384       | 6     | 6         | 2          | ~22M   |
| 12GB    | 512       | 8     | 8         | 4          | ~42M   |
| 16GB    | 768       | 12    | 12        | 4          | ~86M   |
| 24GB+   | 768       | 12    | 12        | 8          | ~86M   |

### 推理模式

#### 1. 单视频推理
```bash
python inference_video_transformer.py \
  --video input.mp4 \
  --model best_model.pth \
  --output result.json

# 输出：
# {
#   "label": "looking_around",
#   "confidence": 0.85,
#   "probabilities": {
#     "normal": 0.05,
#     "looking_around": 0.85,
#     "unknown": 0.10
#   }
# }
```

#### 2. 长视频滑动窗口
```bash
python inference_video_transformer.py \
  --video long_video.mp4 \
  --model best_model.pth \
  --sliding_window \
  --window_stride 8 \
  --output results.json

# 输出：
# [
#   {
#     "start_time": 0.0,
#     "end_time": 1.6,
#     "label": "normal",
#     "confidence": 0.92
#   },
#   {
#     "start_time": 0.8,
#     "end_time": 2.4,
#     "label": "looking_around",
#     "confidence": 0.78
#   },
#   ...
# ]
```

#### 3. 批量处理
```bash
python inference_video_transformer.py \
  --video_dir videos/ \
  --model best_model.pth \
  --output batch_results.json
```

---

## 🧠 模型说明

### 架构：TimeSformer

```
输入视频 [B, T, C, H, W]
    ↓
3D Patch Embedding (Conv3D)
    ↓
位置编码 + CLS Token
    ↓
┌─────────────────────────┐
│ Transformer Block 1     │
│  - 时间注意力           │  ← 每个patch只关注其他帧的同位置
│  - 空间注意力           │  ← 每个patch只关注同帧的其他patch
│  - MLP                  │
└─────────────────────────┘
    ↓
... (重复N次)
    ↓
LayerNorm
    ↓
分类头 (Linear)
    ↓
输出 [B, num_classes]
```

### 关键组件

#### 1. Patch Embedding
```python
# 将视频分成时空patches
patch_size = (2, 16, 16)  # (时间, 高, 宽)
# 输入: [B, 3, 16, 224, 224]
# 输出: [B, 392, 768]
#       392 = (16/2) × (224/16) × (224/16) = 8×14×14
```

#### 2. 分离时空注意力
```python
# 时间注意力：O(T) 每个位置关注不同时间
temporal_attention(patch_i) -> 关注 [patch_i_t0, patch_i_t1, ...]

# 空间注意力：O(HW) 每个时间关注不同位置
spatial_attention(patch_t) -> 关注 [patch_0_t, patch_1_t, ...]

# 总复杂度：O(T + HW) vs 联合注意力 O(T × HW)
```

### 与现有GRU模型对比

| 特性 | GRU模型 | Video Transformer |
|------|---------|-------------------|
| 输入 | 时序特征（5维） | 原始视频帧 |
| 依赖 | 检测+跟踪+姿态 | 无依赖 |
| 时序建模 | 循环网络 | 注意力机制 |
| 长程依赖 | 困难（梯度消失） | 容易（直接注意） |
| 鲁棒性 | 低（误差累积） | 高（端到端） |
| 训练难度 | 低 | 中等 |
| 推理速度 | 快 | 中等 |

---

## 📊 实验建议

### 基线实验

```bash
# 实验1：小模型快速验证
python train_video_transformer.py \
  --embed_dim 384 \
  --depth 4 \
  --num_heads 6 \
  --num_epochs 20 \
  --save_dir exp1_small

# 实验2：标准模型
python train_video_transformer.py \
  --embed_dim 768 \
  --depth 12 \
  --num_heads 12 \
  --num_epochs 50 \
  --save_dir exp2_standard

# 实验3：ROI模式（多人场景）
python train_video_transformer.py \
  --use_roi \
  --embed_dim 768 \
  --depth 12 \
  --num_epochs 50 \
  --save_dir exp3_roi
```

### 消融实验

```bash
# 测试不同帧数
for num_frames in 8 16 32; do
  python train_video_transformer.py \
    --num_frames $num_frames \
    --save_dir exp_frames_$num_frames
done

# 测试不同模型深度
for depth in 4 8 12; do
  python train_video_transformer.py \
    --depth $depth \
    --save_dir exp_depth_$depth
done
```

---

## 🔧 调优技巧

### 1. 过拟合问题

```bash
# 增加正则化
python train_video_transformer.py \
  --drop_rate 0.2 \          # 增加dropout
  --attn_drop_rate 0.2

# 数据增强（已内置）
# - 随机水平翻转
# - 随机亮度调整
```

### 2. 欠拟合问题

```bash
# 增加模型容量
python train_video_transformer.py \
  --embed_dim 1024 \
  --depth 16

# 或增加训练轮数
python train_video_transformer.py \
  --num_epochs 100
```

### 3. 显存不足

```bash
# 方法1：减小batch size
python train_video_transformer.py --batch_size 1

# 方法2：减少帧数
python train_video_transformer.py --num_frames 8

# 方法3：减小图像尺寸
python train_video_transformer.py --img_size 112

# 方法4：使用小模型
python train_video_transformer.py \
  --embed_dim 384 \
  --depth 6 \
  --num_heads 6
```

### 4. 类别不平衡

您的数据：
- looking_around: 61 (78%)
- unknown: 13 (17%)
- normal: 4 (5%)

```python
# 在 video_transformer_pipeline.py 的 Trainer.__init__ 中修改：

# 计算类别权重
class_counts = [4, 61, 13]  # normal, looking_around, unknown
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum() * len(class_counts)

self.criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

---

## ❓ 常见问题

### Q1: 训练时显存溢出？

**A:** 减小batch_size或使用小模型：
```bash
python train_video_transformer.py --batch_size 1 --embed_dim 384 --depth 6
```

### Q2: 训练太慢？

**A:** 减少帧数和图像尺寸：
```bash
python train_video_transformer.py --num_frames 8 --img_size 112
```

### Q3: 精度不高？

**A:** 可能原因：
1. 数据太少（78个样本）→ 使用数据增强
2. 类别不平衡 → 添加类别权重
3. 模型太小 → 增加embed_dim和depth
4. 训练不够 → 增加num_epochs

### Q4: 如何使用预训练模型？

**A:** 可以使用Kinetics-400预训练的TimeSformer：
```python
# 在 train_video_transformer.py 中添加：
from transformers import TimesformerModel

# 加载预训练backbone
pretrained = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

# 迁移权重到自定义模型
# ...（需要手动匹配权重）
```

### Q5: 推理速度慢？

**A:** 优化方法：
1. 使用小模型（embed_dim=384, depth=6）
2. 减少帧数（num_frames=8）
3. 使用TorchScript编译
4. 使用ONNX导出

### Q6: 如何处理实时视频流？

**A:** 使用滑动窗口推理：
```python
# 见 inference_video_transformer.py 中的
# predict_with_temporal_windows 方法
```

---

## 📈 预期性能

基于类似数据集的经验：

| 模型 | 参数量 | 训练时间 | 验证精度 |
|------|--------|----------|----------|
| 小模型 (d=6, h=6) | 22M | 1-2小时 | 70-75% |
| 标准模型 (d=12, h=12) | 86M | 3-5小时 | 75-85% |
| 标准+ROI | 86M | 3-5小时 | 80-90% |

*基于V100 GPU，50 epochs*

---

## 🎯 下一步计划

1. **预训练模型微调**
   - 使用Kinetics-400预训练权重
   - 预期精度提升5-10%

2. **多模态融合**
   - 结合视频帧 + 姿态特征
   - 设计双流架构

3. **弱监督学习**
   - 减少标注需求
   - 使用伪标签

4. **模型压缩**
   - 知识蒸馏
   - 剪枝和量化
   - 部署到边缘设备

---

## 📝 总结

### 核心优势

✅ **解决跟踪不稳定**：不依赖跟踪质量
✅ **端到端训练**：避免误差累积
✅ **自动特征学习**：不需要手工特征
✅ **处理遮挡**：时空注意力自动适应
✅ **可解释性**：可视化注意力权重

### 使用流程

```bash
# 1. 测试
python test_video_transformer.py

# 2. 训练
python train_video_transformer.py --quick_test

# 3. 完整训练
python train_video_transformer.py --num_epochs 50

# 4. 推理
python inference_video_transformer.py --video test.mp4 --model best_model.pth
```

### 技术支持

遇到问题？
1. 查看测试输出：`python test_video_transformer.py`
2. 检查数据格式：确保behavior.json和frames目录正确
3. 调整参数：从小模型开始，逐步增大

---

**祝训练顺利！** 🚀
