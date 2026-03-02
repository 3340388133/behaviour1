# SuspiciousGazeDetection

可疑张望行为识别系统 - 基于头部姿态估计与时序特征分析

## 项目概述

本项目旨在开发一套智能监控系统，能够自动识别视频中的"可疑张望"行为。系统通过人头检测、头部姿态估计、目标追踪，获取头部角度变化，并结合时序特征进行分析，最终判断是否为可疑张望行为。

## 技术架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SuspiciousGazeDetection Pipeline                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Stage 1    │    │   Stage 2    │    │   Stage 3    │               │
│  │  Detection   │───▶│   Tracking   │───▶│  Head Pose   │               │
│  │   YOLOv8     │    │  StrongSORT  │    │   WHENet+    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                  │                       │
│                                                  ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Stage 6    │    │   Stage 5    │    │   Stage 4    │               │
│  │   Output     │◀───│  Classifier  │◀───│  Temporal    │               │
│  │   Alerts     │    │    3D-CNN    │    │  LSTM/GRU    │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 创新点

### 1. 头部姿态估计模块 (WHENet+)
- 在WHENet基础上引入**自注意力机制 (Self-Attention)**
- **多尺度特征融合** 增强小目标检测能力
- 支持正/侧机位的**坐标系自适应转换**

### 2. 时序特征建模
- **双向LSTM/GRU** 捕获前后文时序依赖
- **时间注意力机制** 关注关键时间节点
- **滑动窗口** 策略处理长视频

### 3. 联合追踪-姿态模型
- 头部姿态估计与目标追踪**深度融合**
- 基于时间连续性检测**频繁异常张望行为**
- **Re-ID特征** 辅助跨帧身份关联

## 项目结构

```
SuspiciousGazeDetection/
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认配置
│   ├── train.yaml             # 训练配置
│   └── inference.yaml         # 推理配置
├── src/                        # 源代码
│   ├── models/                # 模型定义
│   │   ├── head_pose/         # 头部姿态估计
│   │   │   ├── whenet.py      # WHENet基础模型
│   │   │   ├── attention.py   # 注意力模块
│   │   │   └── whenet_plus.py # WHENet+ (创新)
│   │   ├── tracker/           # 目标追踪
│   │   │   ├── strong_sort.py # StrongSORT追踪器
│   │   │   └── joint_model.py # 联合追踪-姿态模型
│   │   ├── temporal/          # 时序模型
│   │   │   ├── lstm.py        # LSTM模型
│   │   │   ├── gru.py         # GRU模型
│   │   │   └── temporal_attention.py # 时间注意力
│   │   └── classifier/        # 行为分类器
│   │       ├── cnn3d.py       # 3D-CNN分类器
│   │       └── fusion.py      # 多模态融合
│   ├── data/                  # 数据处理
│   │   ├── datasets/          # 数据集定义
│   │   ├── transforms/        # 数据增强
│   │   └── loaders/           # 数据加载器
│   ├── utils/                 # 工具函数
│   │   ├── metrics.py         # 评估指标
│   │   ├── visualization.py   # 可视化
│   │   └── coordinate.py      # 坐标系转换
│   ├── visualization/         # 可视化模块
│   └── inference/             # 推理模块
├── scripts/                   # 运行脚本
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   └── demo.py               # 演示脚本
├── checkpoints/               # 模型权重
│   ├── pretrained/           # 预训练权重
│   └── trained/              # 训练后权重
├── logs/                      # 日志文件
├── docs/                      # 文档
├── tests/                     # 单元测试
└── notebooks/                 # Jupyter notebooks
```

## 环境配置

```bash
# 创建虚拟环境
conda create -n suspicious_gaze python=3.10
conda activate suspicious_gaze

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备
```bash
# 数据位于 ../data/raw_videos/
# 支持正机位和侧机位视频
```

### 2. 训练模型
```bash
python scripts/train.py --config configs/train.yaml
```

### 3. 推理检测
```bash
python scripts/demo.py --input path/to/video.mp4 --output results/
```

## 数据集

- **正机位数据**: 模拟口岸正面监控视频
- **侧机位数据**: 模拟口岸侧面监控视频
- **标注格式**: JSON (包含时间戳、边界框、行为标签)

## 性能指标

| 模块 | 指标 | 目标值 |
|------|------|--------|
| 头部检测 | mAP@0.5 | > 90% |
| 姿态估计 | MAE (Yaw/Pitch/Roll) | < 5° |
| 行为识别 | Accuracy | > 85% |
| 整体流水线 | FPS | > 15 |

## 参考文献

1. WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose
2. StrongSORT: Make DeepSORT Great Again
3. 3D Convolutional Neural Networks for Human Action Recognition

## License

MIT License
