"""
识别层模块 - 可疑行为识别

四个核心创新点:
1. PAPE - 周期感知位置编码
2. BPCL - 行为原型对比学习
3. DGCMF - 动态门控跨模态融合
4. CIAT - 类别不平衡自适应训练
"""

from .models import SuspiciousBehaviorRecognitionNetwork, SBRN, create_sbrn
from .models.sbrn import SBRNConfig
from .position_encoding import PeriodicAwarePositionalEncoding, PAPE
from .contrastive import BehaviorPrototypeContrastiveLearning, BPCL
from .fusion import DynamicGatedCrossModalFusion, DGCMF
from .training import (
    FocalLoss,
    AdaptiveFocalLoss,
    ProgressiveBalancedSampler,
    ClassAwareSampler,
    ClassImbalanceAdaptiveTrainer,
    CIAT,
)

__all__ = [
    # 主模型
    'SuspiciousBehaviorRecognitionNetwork',
    'SBRN',
    'SBRNConfig',
    'create_sbrn',
    # 创新点1: 位置编码
    'PeriodicAwarePositionalEncoding',
    'PAPE',
    # 创新点2: 对比学习
    'BehaviorPrototypeContrastiveLearning',
    'BPCL',
    # 创新点3: 跨模态融合
    'DynamicGatedCrossModalFusion',
    'DGCMF',
    # 创新点4: 训练策略
    'FocalLoss',
    'AdaptiveFocalLoss',
    'ProgressiveBalancedSampler',
    'ClassAwareSampler',
    'ClassImbalanceAdaptiveTrainer',
    'CIAT',
]
