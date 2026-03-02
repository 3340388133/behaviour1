# 基于头部姿态估计的口岸人员可疑张望行为识别系统

## 系统概述

本系统采用 **"检测-跟踪-估计-识别"** 四阶段级联架构，从原始监控视频出发，自动完成行人检测、多目标跟踪、头部姿态估计与时序行为分类，最终输出带有实时行为标注的可视化结果。

**实验规模**：8个真实口岸场景、171,212 帧视频、3,026 条人物轨迹、885,412 次姿态估计

### 系统架构

```
原始监控视频
    │
    ▼
┌──────────────────────────────────────────────────┐
│  阶段一：目标检测与多目标跟踪                        │
│  YOLOv8m + StrongSORT (Re-ID身份一致性保证)         │
│  → 输出: 带唯一 track_id 的人物轨迹                  │
│  核心代码: src/tracker.py, step3_person_tracking.py  │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  阶段二：双路径头部姿态估计 (Fallback容错)           │
│  主路径: SSD人脸检测 → 头部框扩展 → WHENet           │
│  备路径: 人体比例先验 → 头部区域估计 → WHENet         │
│  → 覆盖率: 77.9% → 100%                            │
│  核心代码: src/head_pose.py, step4_head_pose.py      │
└──────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────┐
│  阶段三：三级混合行为识别  ★ 核心创新 ★              │
│  Level 1: 姿态门控 (实时响应)                        │
│  Level 2: Transformer时序模型 (SBRN, 90帧窗口)       │
│  Level 3: 规则检测器 (瞬态极端行为)                   │
│  → 输出: 6类行为分类 + 可疑度评分                     │
│  核心代码: src/recognition/models/sbrn.py             │
└──────────────────────────────────────────────────┘
    │
    ▼
  6类行为标注结果:
  Normal / Glancing / QuickTurn / Prolonged / LookDown / LookUp
```

---

## 核心代码说明

### 1. 创新模型 — SBRN (Suspicious Behavior Recognition Network)

> 文件: `src/recognition/models/sbrn.py` (561行)

集成四个核心创新点的主模型，架构：输入 → 特征投影 → PAPE → Transformer Encoder → DGCMF → BPCL/分类头 → 输出

```python
# 模型配置
SBRNConfig(
    pose_input_dim=3,          # yaw, pitch, roll
    d_model=128,               # Transformer 维度
    nhead=8,                   # 多头注意力
    num_layers=4,              # Transformer 层数
    periods=[15, 30, 60],      # PAPE 周期 (0.5s/1.0s/2.0s @30fps)
    num_classes=6,             # 6类行为分类
    num_prototypes_per_class=3 # BPCL 每类原型数
)
```

**关键实验结果**:
- 二分类 F1=0.882 (Transformer+UW)，显著优于规则基线 0.746 (+18.2%)
- 六分类 F1-Macro=0.717，相比 Baseline 提升 12.7%

---

### 2. 创新点一 — PAPE (周期感知位置编码)

> 文件: `src/recognition/position_encoding/periodic_aware_pe.py`

**问题**: 标准正弦位置编码仅表示绝对时间位置，无法显式捕捉可疑行为的周期性特征（如"频繁张望"表现为 1.5~3 秒周期的方向切换）。

**方案**: 融合三种位置信息：
1. **标准正弦PE** — 绝对时间位置
2. **多尺度周期编码** — 0.5s / 1.0s / 2.0s 三个尺度的正弦函数，配备可学习相位偏移和幅度权重
3. **相对位置偏置** — 可学习偏置表 `[2L-1, num_heads]`，直接作用于自注意力权重矩阵

**效果**: QuickTurn F1 从 0.200 提升至 0.519 (+160%)

---

### 3. 创新点二 — BPCL (行为原型对比学习)

> 文件: `src/recognition/contrastive/behavior_prototype.py`

**问题**: 数据极度不平衡，少数类样本不足导致决策边界模糊。

**方案**:
- 每个行为类别维护多个可学习原型向量，捕捉类内变化
- InfoNCE 损失拉近样本与同类原型、推远异类原型
- EMA 动量更新原型，防止训练震荡
- 困难负样本挖掘 + 边界损失显式拉大类间距离

---

### 4. 创新点三 — DGCMF (动态门控跨模态融合)

> 文件: `src/recognition/fusion/dynamic_gated_fusion.py`

**问题**: 不同模态（姿态/外观/运动）在不同场景下可靠性不同（遮挡→外观不可靠，静止→运动不可靠）。

**方案**:
- **质量评估网络**: 预测每个模态的可靠性分数 [0,1]
- **跨模态注意力**: 建模模态间交互关系
- **残差门控**: 保留原始单模态信息，防止信息丢失

---

### 5. 创新点四 — CIAT (类别不平衡自适应训练)

> 文件: `src/recognition/training/focal_loss.py`, `src/recognition/training/balanced_sampler.py`

- 自适应 Focal Loss（gamma 随训练进度调整）
- 类别平衡采样器
- 不确定性加权 (Uncertainty Weighting) 自动平衡分类损失与对比损失

---

### 6. 端到端推理管道

> 文件: `src/e2e_pipeline.py`, `step7_head_detection_inference.py`

完整推理流程：
```
人体跟踪(预计算) → 人脸检测(SSD) → 头部框扩展 → BBox平滑 →
姿态估计(WHENet) → 时序特征提取(滑动窗口) →
行为分类(Transformer+规则混合) → 轨迹级投票 → 视频标注输出
```

### 7. 规则引擎

> 文件: `src/rule_engine.py`

4条固定规则（经敏感性分析优化）：
| 规则 | 条件 | 含义 |
|------|------|------|
| 持续侧向 | \|yaw\| > 35°, 比例 > 70% | 长时间侧向注视 |
| 频繁扫视 | 切换 ≥ 2次, 速度 > 15°/s | 快速左右张望 |
| 高变异性 | yaw_std > 50° | 头部运动不稳定 |
| 大范围转头 | yaw_range > 120° | 大角度转头搜索 |

### 8. 融合决策

> 文件: `src/fusion/fusion.py`

三级优先级级联：姿态门控(即时) → Transformer时序模型(全局) → 规则检测(兜底)

消融实验表明：三级融合的 Shannon 熵 (1.545) 显著高于单一方法 (0.855/0.857)，证明框架能生成最均衡的行为分类结果。

---

## 目录结构

```
├── src/                            # 核心源码
│   ├── recognition/                # ★ 行为识别模块（核心创新）
│   │   ├── models/
│   │   │   └── sbrn.py            # SBRN 主模型（4个创新点集成）
│   │   ├── position_encoding/
│   │   │   └── periodic_aware_pe.py  # 创新点1: PAPE
│   │   ├── contrastive/
│   │   │   └── behavior_prototype.py # 创新点2: BPCL
│   │   ├── fusion/
│   │   │   └── dynamic_gated_fusion.py # 创新点3: DGCMF
│   │   ├── training/
│   │   │   ├── focal_loss.py      # 创新点4: 自适应Focal Loss
│   │   │   ├── balanced_sampler.py # 类别平衡采样
│   │   │   └── adaptive_trainer.py # 自适应训练器
│   │   ├── dataset.py             # 数据集加载
│   │   └── temporal_transformer.py # 时序Transformer基础模块
│   ├── tracker.py                 # 目标跟踪 (ByteTrack/Kalman)
│   ├── head_pose.py               # 头部姿态估计 (WHENet ONNX)
│   ├── face_detector.py           # 人脸检测 (RetinaFace)
│   ├── temporal_features.py       # 时序特征提取 (滑动窗口2.0s)
│   ├── rule_engine.py             # 规则引擎 (4条规则)
│   ├── e2e_pipeline.py            # 端到端推理管道
│   ├── fusion/                    # 融合决策模块
│   │   ├── fusion.py              # 规则+模型加权融合
│   │   ├── rule_scorer.py         # 规则评分器
│   │   └── model_scorer.py        # 模型评分器
│   ├── alert_generator.py         # 告警生成
│   └── benchmark/                 # 基准测试与对比实验
│
├── step1~step8*.py                # 完整处理管道脚本
│   ├── step1_build_dataset_structure.py  # 数据集结构构建
│   ├── step2_frame_extraction.py         # 视频帧提取
│   ├── step3_person_tracking.py          # YOLOv8+StrongSORT 跟踪
│   ├── step4_head_pose.py                # 头部姿态估计
│   ├── step5_build_dataset.py            # 训练数据集构建
│   ├── step6_train_recognition.py        # 模型训练
│   ├── step7_head_detection_inference.py # 推理（核心推理入口）
│   ├── step7_visualize_results.py        # 结果可视化
│   ├── step8_ablation_baseline.py        # 消融实验
│   └── step8b_sensitivity.py             # 敏感性分析
│
├── train_sbrn_6class.py           # SBRN 训练
├── train_6class_balanced.py       # 平衡训练
├── run_inference.py               # 推理入口
├── run_full_pipeline.py           # 完整管道
├── generate_*_figures.py          # 论文图表生成（7个）
│
├── experiments/                   # 实验配置与结果
│   ├── configs/                   # 消融实验配置 (baseline/PAPE/BPCL/DGCMF)
│   ├── scripts/                   # 训练/评估/推理脚本
│   └── *.json                     # 实验结果数据
│
├── thesis_figures/                # 论文图表与实验报告
│   ├── experiment_report.md       # 完整实验报告
│   ├── arch_fig*.png              # 系统架构图
│   ├── sbrn_fig*.png              # SBRN模型实验结果
│   ├── ablation_fig*.png          # 消融实验图表
│   ├── composite_fig*.png         # 综合统计图
│   ├── whenet_fig*.png            # WHENet姿态分析图
│   ├── sensitivity_fig*.png       # 参数敏感性分析
│   └── table*.tex                 # LaTeX 实验结果表
│
├── scripts/                       # 辅助批处理脚本
├── configs/                       # 配置文件
│
├── tests/test_recognition/        # 单元测试
│   ├── test_sbrn.py               # SBRN 模型测试
│   ├── test_pape.py               # PAPE 测试
│   ├── test_bpcl.py               # BPCL 测试
│   └── test_dgcmf.py             # DGCMF 测试
│
└── docs/                          # 项目文档
    ├── 项目方案.md                  # 系统总体方案
    ├── thesis_chapter3_4.md        # 论文第三四章框架
    └── 客户反馈回复与论文撰写指南.md  # 完整论文写作指南
```

---

## 关键实验结果

### 二分类对比 (Normal vs Suspicious)

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Rule Baseline | 1.000 | 0.595 | 0.746 |
| LSTM | 0.971 | 0.790 | 0.871 |
| Transformer | 0.680 | 0.969 | 0.799 |
| **Transformer+UW (Ours)** | **0.862** | **0.903** | **0.882** |

### 六分类消融实验 (SBRN)

| 配置 | Accuracy | F1-Macro | QuickTurn F1 | 参数量 |
|------|----------|----------|-------------|--------|
| A0: Baseline (Transformer+CE) | 77.4% | 0.636 | 0.200 | 114K |
| A2: +PAPE | 77.4% | 0.658 (+3.5%) | 0.519 (+160%) | 455K |
| A5: Full SBRN | **79.1%** | **0.717 (+12.7%)** | — | 500K |

### 双路径 Fallback 容错

| 指标 | 无 Fallback | 有 Fallback |
|------|------------|------------|
| 姿态估计覆盖率 | 77.9% | **100%** |
| 有效轨迹比例 | ~78% | **100%** |
| 时序连续性 | 频繁中断 | **完全连续** |

---

## 运行方式

```bash
# 完整推理管道
python step7_head_detection_inference.py \
    --video data/raw_videos/sample.mp4 \
    --output output/

# 训练 SBRN 模型
python train_sbrn_6class.py \
    --config experiments/configs/full_model.yaml

# 消融实验
python step8_ablation_baseline.py
python step8b_sensitivity.py
```

---

## 技术栈

- **目标检测**: YOLOv8m (Ultralytics)
- **多目标跟踪**: StrongSORT (Re-ID)
- **头部姿态估计**: WHENet (ONNX Runtime)
- **时序建模**: PyTorch Transformer / LSTM
- **人脸检测**: SSD (OpenCV DNN)
