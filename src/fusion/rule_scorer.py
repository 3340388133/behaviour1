"""
规则引擎模块 - 实现BaseScorer接口
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from .base import BaseScorer, ScorerResult


class RuleEngine(BaseScorer):
    """基于规则的评分器"""

    def __init__(self, thresholds: Dict[str, float] = None):
        # 默认阈值
        self.thresholds = thresholds or {
            'side_gaze_angle': 30,
            'side_gaze_ratio': 0.8,
            'switch_count': 3,
            'speed_threshold': 20,
            'range_threshold': 60,
            'std_threshold': 15,
        }

        # 规则权重
        self.weights = {
            'sustained_side': 0.35,
            'frequent_scan': 0.30,
            'high_variability': 0.20,
            'large_range': 0.15,
        }

    def get_name(self) -> str:
        return "RuleEngine"

    def score(self, features: Dict[str, Any]) -> ScorerResult:
        """计算规则分数"""
        score = 0.0
        triggered = []
        details = {}

        # 获取特征
        yaws = features.get('yaws', np.array([]))
        yaw_std = features.get('yaw_std', 0)
        yaw_range = features.get('yaw_range', 0)
        yaw_speed_mean = features.get('yaw_speed_mean', 0)
        switch_count = features.get('yaw_switch_count', 0)

        th = self.thresholds

        # 规则1: 持续侧向
        if len(yaws) > 0:
            side_ratio = np.mean(np.abs(yaws) > th['side_gaze_angle'])
            details['side_ratio'] = round(side_ratio, 3)
            if side_ratio >= th['side_gaze_ratio']:
                score += self.weights['sustained_side']
                triggered.append(f"sustained_side({side_ratio:.0%})")

        # 规则2: 频繁扫视
        details['switch_count'] = switch_count
        details['speed_mean'] = round(yaw_speed_mean, 2)
        if switch_count >= th['switch_count'] and yaw_speed_mean > th['speed_threshold']:
            score += self.weights['frequent_scan']
            triggered.append(f"frequent_scan(n={switch_count})")

        # 规则3: 高变异性
        details['yaw_std'] = round(yaw_std, 2)
        if yaw_std > th['std_threshold']:
            score += self.weights['high_variability']
            triggered.append(f"high_var(std={yaw_std:.1f})")

        # 规则4: 大范围转头
        details['yaw_range'] = round(yaw_range, 2)
        if yaw_range > th['range_threshold']:
            score += self.weights['large_range']
            triggered.append(f"large_range({yaw_range:.1f}°)")

        return ScorerResult(
            score=min(1.0, score),
            confidence=0.9,
            details=details,
            triggered_rules=triggered
        )
