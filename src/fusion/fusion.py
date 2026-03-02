"""
融合决策模块 - 规则+模型加权融合
"""
from typing import Dict, Any, Optional
from .base import BaseScorer, ScorerResult, FusionResult
from .rule_scorer import RuleEngine
from .model_scorer import PlaceholderModel, LSTMModel, TransformerModel


class FusionDecider:
    """融合决策器 - 支持规则和模型的加权融合"""

    def __init__(
        self,
        rule_scorer: BaseScorer = None,
        model_scorer: BaseScorer = None,
        alpha: float = 0.6,
        threshold: float = 0.3
    ):
        """
        Args:
            rule_scorer: 规则评分器
            model_scorer: 模型评分器
            alpha: 规则权重 (0-1), 模型权重为 1-alpha
            threshold: 可疑判定阈值
        """
        self.rule_scorer = rule_scorer or RuleEngine()
        self.model_scorer = model_scorer or PlaceholderModel()
        self.alpha = alpha
        self.threshold = threshold

    def decide(self, features: Dict[str, Any]) -> FusionResult:
        """
        融合决策
        Args:
            features: 特征字典
        Returns:
            FusionResult
        """
        # 规则评分
        rule_result = self.rule_scorer.score(features)
        rule_score = rule_result.score

        # 模型评分
        model_result = self.model_scorer.score(features)
        model_score = model_result.score

        # 加权融合: score = α*rule + (1-α)*model
        final_score = self.alpha * rule_score + (1 - self.alpha) * model_score

        return FusionResult(
            final_score=round(final_score, 4),
            rule_score=round(rule_score, 4),
            model_score=round(model_score, 4),
            alpha=self.alpha,
            is_suspicious=final_score >= self.threshold,
            rule_details=rule_result,
            model_details=model_result
        )

    def set_alpha(self, alpha: float) -> None:
        """动态调整融合权重"""
        self.alpha = max(0.0, min(1.0, alpha))

    def set_model(self, model_scorer: BaseScorer) -> None:
        """替换模型评分器"""
        self.model_scorer = model_scorer

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'rule_scorer': self.rule_scorer.get_name(),
            'model_scorer': self.model_scorer.get_name(),
            'alpha': self.alpha,
            'threshold': self.threshold
        }


def create_fusion_decider(
    model_type: str = 'placeholder',
    alpha: float = 0.6,
    threshold: float = 0.3,
    model_path: str = None
) -> FusionDecider:
    """
    工厂函数 - 创建融合决策器

    Args:
        model_type: 'placeholder', 'lstm', 'transformer'
        alpha: 规则权重
        threshold: 可疑阈值
        model_path: 模型权重路径
    """
    rule_scorer = RuleEngine()

    if model_type == 'lstm':
        model_scorer = LSTMModel()
        if model_path:
            model_scorer.load_weights(model_path)
    elif model_type == 'transformer':
        model_scorer = TransformerModel()
        if model_path:
            model_scorer.load_weights(model_path)
    else:
        model_scorer = PlaceholderModel()

    return FusionDecider(
        rule_scorer=rule_scorer,
        model_scorer=model_scorer,
        alpha=alpha,
        threshold=threshold
    )
