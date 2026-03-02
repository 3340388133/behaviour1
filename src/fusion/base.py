"""
行为检测融合框架 - 基础接口定义
支持规则引擎和时序模型的灵活替换
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ScorerResult:
    """评分器输出结果"""
    score: float                          # 0-1分数
    confidence: float                     # 置信度
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息
    triggered_rules: List[str] = field(default_factory=list)  # 触发的规则


@dataclass
class FusionResult:
    """融合决策结果"""
    final_score: float                    # 最终分数
    rule_score: float                     # 规则分数
    model_score: float                    # 模型分数
    alpha: float                          # 融合权重
    is_suspicious: bool                   # 是否可疑
    rule_details: ScorerResult            # 规则详情
    model_details: ScorerResult           # 模型详情


class BaseScorer(ABC):
    """评分器基类 - 所有评分器必须继承此类"""

    @abstractmethod
    def score(self, features: Dict[str, Any]) -> ScorerResult:
        """
        计算分数
        Args:
            features: 特征字典，包含时序特征
        Returns:
            ScorerResult
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回评分器名称"""
        pass


class BaseTemporalModel(BaseScorer):
    """时序模型基类 - LSTM/Transformer等继承此类"""

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """加载模型权重"""
        pass

    @abstractmethod
    def train(self, data: Any, labels: Any) -> None:
        """训练模型"""
        pass
