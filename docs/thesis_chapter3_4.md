# 毕业论文 - 第三章 & 第四章 内容框架

## 第三章：数据集构建与多模态特征提取

### 3.1 视频数据采集
- 摄像机设置：正机位、侧机位
- 场景描述：监控场景
- 数据规模：9个视频

### 3.2 人物检测与跟踪
- **检测器**：YOLOv8m
- **跟踪器**：BoT-SORT（内置 Re-ID）
- **对比**：ByteTrack vs BoT-SORT
  - ByteTrack：仅 IoU 匹配，遮挡后丢失
  - BoT-SORT：融合 Re-ID，遮挡后可重识别

### 3.3 头部姿态估计
- **基线模型**：WHENet (ONNX)
- **输出**：yaw, pitch, roll（度）
- **改进点**：见第四章

### 3.4 数据集统计
- 总轨迹数：1,504
- 总姿态帧：479,622
- 训练/验证样本：58,044 / 29,305 序列

---

## 第四章：时序感知头部姿态识别网络 (TAHPNet)

### 4.1 问题分析
单帧姿态估计存在的问题：
1. 帧间抖动：相邻帧估计结果不连续
2. 噪声敏感：单帧估计易受图像质量影响
3. 时序信息丢失：无法捕捉姿态变化模式

### 4.2 网络架构
```
TAHPNet
├── Backbone: RepVGG-A0 (轻量高效，可重参数化)
├── Pose Head: MLP (单帧姿态估计)
└── Temporal Module: Bidirectional GRU (时序平滑)
    ├── 双向 GRU 捕捉前后文
    ├── 残差连接保留原始估计
    └── 可学习平滑权重
```

### 4.3 创新点
1. **时序平滑模块**：双向 GRU 解决帧间抖动
2. **多任务学习**：同时预测姿态 + 姿态变化率
3. **时序一致性损失**：约束加速度平滑
4. **RepVGG Backbone**：推理时融合为单卷积，轻量高效

### 4.4 损失函数
```
L_total = L_pose + λ1 * L_raw + λ2 * L_smooth + λ3 * L_velocity

- L_pose: 平滑后姿态的 SmoothL1 损失
- L_raw: 原始估计的辅助监督
- L_smooth: 加速度最小化（时序一致性）
- L_velocity: 速度预测监督
```

### 4.5 消融实验设计

| 实验 | 配置 | 对比目的 |
|-----|------|---------|
| Baseline | 无时序模块 | 证明时序模块的必要性 |
| Full | 完整 TAHPNet | 最佳效果 |
| No BiDir | 单向 GRU | 证明双向的优势 |
| Shallow | 单层 GRU | 证明深度的影响 |

### 4.6 评估指标
1. **MAE**：姿态估计平均绝对误差（度）
2. **Jitter**：帧间变化标准差（度/帧）
3. **Smoothness**：加速度平滑度

### 4.7 实验结果（待填写）

| 方法 | MAE↓ | Jitter↓ | Smoothness↓ |
|-----|------|---------|-------------|
| WHENet (单帧) | - | - | - |
| TAHPNet (Baseline) | - | - | - |
| TAHPNet (Full) | - | - | - |

---

## 模型参数量
- TAHPNet: 8,659,335 参数 (~33MB)
- RepVGG 推理模式：更少参数（融合后）

## 文件位置
- 模型代码：`src/recognition/models/tahpnet.py`
- 训练脚本：`experiments/scripts/train_tahpnet.py`
- 评估脚本：`experiments/scripts/evaluate_tahpnet.py`
- 消融实验：`experiments/scripts/run_ablation.sh`
