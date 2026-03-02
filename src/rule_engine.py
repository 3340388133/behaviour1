"""
可疑张望行为规则引擎
规则 (固定4条):
1. 持续侧向: |yaw| > 35°, 比例 > 0.7
2. 频繁扫视: 切换 >= 2 且 速度 > 15°/s
3. 高变异性: yaw_std > 50°
4. 大范围转头: yaw_range > 120°

阈值经过敏感性分析优化，使触发率稳定在 20-30%
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class RuleResult:
    """单条规则结果"""
    rule_name: str
    triggered: bool
    score: float
    explanation: str
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'rule_name': self.rule_name,
            'triggered': self.triggered,
            'score': round(self.score, 3),
            'explanation': self.explanation,
            'details': self.details
        }


@dataclass
class EvaluationResult:
    """评估结果"""
    rules: List[RuleResult]
    weights: Dict[str, float]

    @property
    def triggered_count(self) -> int:
        return sum(1 for r in self.rules if r.triggered)

    @property
    def weighted_score(self) -> float:
        score = 0.0
        for r in self.rules:
            w = self.weights.get(r.rule_name, 0.25)
            score += r.score * w
        return score

    @property
    def is_suspicious(self) -> bool:
        return self.weighted_score >= 0.5

    def to_dict(self) -> dict:
        return {
            'is_suspicious': self.is_suspicious,
            'weighted_score': round(self.weighted_score, 3),
            'triggered_count': self.triggered_count,
            'rules': [r.to_dict() for r in self.rules]
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class RuleEngine:
    """基于规则的行为判定引擎"""

    DEFAULT_WEIGHTS = {
        'sustained_side_gaze': 0.3,
        'frequent_scanning': 0.3,
        'high_variability': 0.2,
        'wide_range_turn': 0.2
    }

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def evaluate(self, features: Dict) -> EvaluationResult:
        """评估时序特征

        Args:
            features: 时序特征字典，需包含:
                - yaw_mean, yaw_std, yaw_range
                - yaw_speed_mean, yaw_switch_count
                - yaws (可选): 原始 yaw 序列

        Returns:
            EvaluationResult
        """
        results = [
            self._rule_sustained_side_gaze(features),
            self._rule_frequent_scanning(features),
            self._rule_high_variability(features),
            self._rule_wide_range_turn(features)
        ]
        return EvaluationResult(results, self.weights)

    def _rule_sustained_side_gaze(self, f: Dict) -> RuleResult:
        """规则1: 持续侧向 - |yaw| > 35°, 比例 > 0.7"""
        yaws = f.get('yaws')
        if yaws is not None and len(yaws) > 0:
            side_ratio = np.mean(np.abs(yaws) > 35)
        else:
            side_ratio = 1.0 if abs(f.get('yaw_mean', 0)) > 35 else 0.0

        triggered = side_ratio > 0.7
        score = min(1.0, side_ratio / 0.7) if side_ratio > 0.5 else 0.0

        return RuleResult(
            rule_name='sustained_side_gaze',
            triggered=triggered,
            score=score,
            explanation=f'侧向比例 {side_ratio:.1%} {">" if triggered else "<="} 70%',
            details={'side_ratio': round(side_ratio, 3)}
        )

    def _rule_frequent_scanning(self, f: Dict) -> RuleResult:
        """规则2: 频繁扫视 - 切换 >= 2 且 速度 > 15°/s"""
        switch_count = f.get('yaw_switch_count', 0)
        speed = f.get('yaw_speed_mean', 0)

        triggered = switch_count >= 2 and speed > 15
        score = 0.0
        if switch_count >= 2:
            score = min(1.0, speed / 30) if speed > 15 else 0.3

        return RuleResult(
            rule_name='frequent_scanning',
            triggered=triggered,
            score=score,
            explanation=f'切换 {switch_count} 次, 速度 {speed:.1f}°/s',
            details={'switch_count': switch_count, 'speed': round(speed, 2)}
        )

    def _rule_high_variability(self, f: Dict) -> RuleResult:
        """规则3: 高变异性 - yaw_std > 50°"""
        yaw_std = f.get('yaw_std', 0)

        triggered = yaw_std > 50
        score = min(1.0, yaw_std / 100) if yaw_std > 30 else 0.0

        return RuleResult(
            rule_name='high_variability',
            triggered=triggered,
            score=score,
            explanation=f'yaw 标准差 {yaw_std:.1f}° {">" if triggered else "<="} 50°',
            details={'yaw_std': round(yaw_std, 2)}
        )

    def _rule_wide_range_turn(self, f: Dict) -> RuleResult:
        """规则4: 大范围转头 - yaw_range > 120°"""
        yaw_range = f.get('yaw_range', 0)

        triggered = yaw_range > 120
        score = min(1.0, yaw_range / 240) if yaw_range > 80 else 0.0

        return RuleResult(
            rule_name='wide_range_turn',
            triggered=triggered,
            score=score,
            explanation=f'yaw 范围 {yaw_range:.1f}° {">" if triggered else "<="} 120°',
            details={'yaw_range': round(yaw_range, 2)}
        )


class BehaviorClassifier:
    """行为分类器 - 基于规则引擎的封装"""

    def __init__(self, threshold: float = 0.5, weights: Dict[str, float] = None):
        """
        Args:
            threshold: 判定为可疑行为的阈值
            weights: 规则权重
        """
        self.threshold = threshold
        self.engine = RuleEngine(weights=weights)

    def classify(self, features) -> tuple:
        """分类行为

        Args:
            features: TemporalFeatures 对象

        Returns:
            (is_suspicious, score, triggered_rules)
        """
        # 将 TemporalFeatures 转换为字典
        feature_dict = {
            'yaw_mean': features.yaw_mean,
            'yaw_std': features.yaw_std,
            'yaw_range': features.yaw_max - features.yaw_min,
            'yaw_speed_mean': features.angular_velocity_mean,
            'yaw_switch_count': features.switch_count,
            'pitch_mean': features.pitch_mean,
            'pitch_std': features.pitch_std,
        }

        # 如果有原始 yaw 序列
        if hasattr(features, 'yaws'):
            feature_dict['yaws'] = features.yaws

        # 使用规则引擎评估
        result = self.engine.evaluate(feature_dict)

        is_suspicious = result.weighted_score >= self.threshold
        triggered_rules = [r for r in result.rules if r.triggered]

        return is_suspicious, result.weighted_score, result.rules
