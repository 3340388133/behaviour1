"""
人工标注一致性控制模块

核心目标：降低主观标注偏差对模型上限的影响

主要内容：
1. 行为标签判定准则（清晰的正/负样本定义）
2. 标准样例库建设方案
3. 标注一致性验证流程
4. 质量评估工具
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import datetime
import json


# ============================================================================
# 行为标签判定准则
# ============================================================================
"""
行为类别定义与判定准则

设计原则：
1. 客观可测量：尽量使用可量化的指标
2. 边界明确：明确定义阈值，减少主观判断
3. 优先级清晰：多种行为同时存在时，有明确的判定顺序
"""

class BehaviorLabel(Enum):
    """行为类别"""
    NORMAL = 0           # 正常行为
    GLANCING = 1         # 频繁左右张望
    QUICK_TURN = 2       # 快速回头
    PROLONGED_WATCH = 3  # 长时间观察周围
    LOOKING_DOWN = 4     # 持续低头
    LOOKING_UP = 5       # 持续抬头
    UNCERTAIN = -1       # 无法判断


@dataclass
class BehaviorCriteria:
    """行为判定准则"""
    label: BehaviorLabel
    name: str
    definition: str

    # 量化指标阈值
    primary_metric: str          # 主要判定指标
    primary_threshold: str       # 主要阈值
    secondary_metrics: list      # 辅助指标

    # 时间要求
    min_duration_sec: float      # 最小持续时间
    observation_window_sec: float  # 观察窗口

    # 正样本特征
    positive_characteristics: list
    # 负样本特征（不应出现）
    negative_characteristics: list

    # 边界情况处理
    boundary_handling: str

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "name": self.name,
            "definition": self.definition,
            "primary_metric": self.primary_metric,
            "primary_threshold": self.primary_threshold,
            "secondary_metrics": self.secondary_metrics,
            "min_duration_sec": self.min_duration_sec,
            "observation_window_sec": self.observation_window_sec,
            "positive_characteristics": self.positive_characteristics,
            "negative_characteristics": self.negative_characteristics,
            "boundary_handling": self.boundary_handling
        }


# 行为判定准则库
BEHAVIOR_CRITERIA = {
    BehaviorLabel.NORMAL: BehaviorCriteria(
        label=BehaviorLabel.NORMAL,
        name="正常行为",
        definition="视线稳定，偶尔自然转头，无明显异常动作模式",
        primary_metric="yaw_std",
        primary_threshold="< 20°",
        secondary_metrics=[
            "yaw_switch_count < 2 (在3秒窗口内)",
            "yaw_speed_mean < 30°/s",
            "|yaw_rel_mean| < 30°"
        ],
        min_duration_sec=1.5,
        observation_window_sec=3.0,
        positive_characteristics=[
            "视线主要朝前",
            "头部移动缓慢平稳",
            "偶尔的自然转头（查看手机、与人交谈等）"
        ],
        negative_characteristics=[
            "频繁快速转头",
            "长时间侧向注视",
            "突然的大幅度回头"
        ],
        boundary_handling="当指标接近阈值但未超过时，标记为 normal"
    ),

    BehaviorLabel.GLANCING: BehaviorCriteria(
        label=BehaviorLabel.GLANCING,
        name="频繁左右张望",
        definition="3秒内左右转头≥3次，每次yaw变化>30°",
        primary_metric="yaw_switch_count",
        primary_threshold="≥ 3 (在3秒窗口内)",
        secondary_metrics=[
            "每次转头 yaw 变化 > 30°",
            "yaw_speed_mean > 15°/s",
            "yaw_range > 60°"
        ],
        min_duration_sec=3.0,
        observation_window_sec=3.0,
        positive_characteristics=[
            "头部左右快速摆动",
            "每次转头幅度明显（>30°）",
            "转头频率高（≥1次/秒）"
        ],
        negative_characteristics=[
            "单向持续注视",
            "缓慢平稳的头部移动",
            "只有1-2次转头"
        ],
        boundary_handling="2次转头且幅度>45°时，可标记为 glancing"
    ),

    BehaviorLabel.QUICK_TURN: BehaviorCriteria(
        label=BehaviorLabel.QUICK_TURN,
        name="快速回头",
        definition="0.5秒内yaw变化>60°的突然转头动作",
        primary_metric="yaw_speed_max",
        primary_threshold="> 120°/s (0.5秒内变化>60°)",
        secondary_metrics=[
            "单次转头动作",
            "转头前相对稳定",
            "转头后可能维持或回转"
        ],
        min_duration_sec=0.5,
        observation_window_sec=2.0,
        positive_characteristics=[
            "突然的大幅度转头",
            "转头速度快（>120°/s）",
            "通常伴随肩部或身体转动"
        ],
        negative_characteristics=[
            "缓慢的转头",
            "小幅度的头部移动",
            "连续的左右摆动（应标为 glancing）"
        ],
        boundary_handling="速度在100-120°/s之间，标记为 uncertain"
    ),

    BehaviorLabel.PROLONGED_WATCH: BehaviorCriteria(
        label=BehaviorLabel.PROLONGED_WATCH,
        name="长时间观察周围",
        definition="持续>3秒注视非正前方（|yaw|>30°）",
        primary_metric="prolonged_side_ratio",
        primary_threshold="> 0.7 (70%时间 |yaw|>30°)",
        secondary_metrics=[
            "|yaw_rel_mean| > 30°",
            "yaw_std < 15° (相对稳定的侧向注视)",
            "持续时间 > 3秒"
        ],
        min_duration_sec=3.0,
        observation_window_sec=5.0,
        positive_characteristics=[
            "持续的侧向注视",
            "头部位置相对稳定",
            "注视方向明确（非频繁切换）"
        ],
        negative_characteristics=[
            "频繁转头（应标为 glancing）",
            "视线主要朝前",
            "持续时间<3秒"
        ],
        boundary_handling="持续2.5-3秒时，标记为 uncertain"
    ),

    BehaviorLabel.LOOKING_DOWN: BehaviorCriteria(
        label=BehaviorLabel.LOOKING_DOWN,
        name="持续低头",
        definition="持续>5秒 pitch<-20°",
        primary_metric="pitch_mean",
        primary_threshold="< -20° 持续>5秒",
        secondary_metrics=[
            "pitch_std < 10° (稳定低头)",
            "低头比例 > 80%"
        ],
        min_duration_sec=5.0,
        observation_window_sec=5.0,
        positive_characteristics=[
            "持续的低头姿态",
            "头部位置稳定",
            "可能在看手机/读书等"
        ],
        negative_characteristics=[
            "偶尔的低头（<3秒）",
            "频繁抬头低头交替"
        ],
        boundary_handling="持续3-5秒时，标记为 uncertain"
    ),

    BehaviorLabel.LOOKING_UP: BehaviorCriteria(
        label=BehaviorLabel.LOOKING_UP,
        name="持续抬头",
        definition="持续>3秒 pitch>20°",
        primary_metric="pitch_mean",
        primary_threshold="> 20° 持续>3秒",
        secondary_metrics=[
            "pitch_std < 10° (稳定抬头)",
            "抬头比例 > 70%"
        ],
        min_duration_sec=3.0,
        observation_window_sec=3.0,
        positive_characteristics=[
            "持续的抬头姿态",
            "可能在看天花板/指示牌等"
        ],
        negative_characteristics=[
            "偶尔的抬头（<2秒）",
            "频繁抬头低头交替"
        ],
        boundary_handling="持续2-3秒时，标记为 uncertain"
    ),

    BehaviorLabel.UNCERTAIN: BehaviorCriteria(
        label=BehaviorLabel.UNCERTAIN,
        name="无法判断",
        definition="因遮挡/质量差/边界情况无法准确判断",
        primary_metric="N/A",
        primary_threshold="N/A",
        secondary_metrics=[],
        min_duration_sec=0,
        observation_window_sec=0,
        positive_characteristics=[
            "人脸被严重遮挡（>50%）",
            "图像模糊/质量差",
            "行为指标处于边界值",
            "多种行为特征混合"
        ],
        negative_characteristics=[],
        boundary_handling="uncertain 样本不参与训练，仅用于分析"
    )
}


# ============================================================================
# 标注优先级规则
# ============================================================================
"""
当多种行为特征同时存在时的判定优先级：

1. 先检查 uncertain 条件（质量问题优先）
2. 检查瞬时行为：quick_turn（0.5秒尺度）
3. 检查频繁行为：glancing（3秒尺度，优先于持续行为）
4. 检查持续行为：prolonged_watch, looking_down, looking_up
5. 默认：normal

示例：
- 3秒内有1次快速回头 + 2次普通转头 → quick_turn（瞬时行为优先）
- 频繁张望 + 侧向注视时间>70% → glancing（频繁行为优先）
- pitch<-20° 持续5秒 + yaw 小幅变化 → looking_down
"""

LABEL_PRIORITY = [
    BehaviorLabel.UNCERTAIN,      # 最高优先级：质量问题
    BehaviorLabel.QUICK_TURN,     # 瞬时行为
    BehaviorLabel.GLANCING,       # 频繁行为
    BehaviorLabel.PROLONGED_WATCH,  # 持续侧向
    BehaviorLabel.LOOKING_DOWN,   # 持续低头
    BehaviorLabel.LOOKING_UP,     # 持续抬头
    BehaviorLabel.NORMAL          # 默认
]


# ============================================================================
# 标准样例库建设方案
# ============================================================================
@dataclass
class StandardExample:
    """标准样例"""
    example_id: str
    label: BehaviorLabel
    label_name: str

    # 来源信息
    video_id: str
    track_id: int
    start_time: float
    end_time: float

    # 特征值（用于参考）
    key_metrics: dict

    # 说明
    description: str           # 为什么是这个类别
    distinguishing_features: list  # 区分性特征
    common_mistakes: list      # 常见误标情况

    # 状态
    verified_by: list          # 验证人员
    consensus_level: str       # unanimous / majority / disputed

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "label": self.label.value,
            "label_name": self.label_name,
            "video_id": self.video_id,
            "track_id": self.track_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "key_metrics": self.key_metrics,
            "description": self.description,
            "distinguishing_features": self.distinguishing_features,
            "common_mistakes": self.common_mistakes,
            "verified_by": self.verified_by,
            "consensus_level": self.consensus_level
        }


EXAMPLE_LIBRARY_SPEC = {
    "structure": {
        "examples/": {
            "normal/": "正常行为样例 x20",
            "glancing/": "频繁张望样例 x20",
            "quick_turn/": "快速回头样例 x20",
            "prolonged_watch/": "长时间观察样例 x20",
            "looking_down/": "持续低头样例 x10",
            "looking_up/": "持续抬头样例 x10",
            "boundary/": "边界情况样例 x30",
            "uncertain/": "无法判断样例 x20"
        }
    },
    "per_category_requirements": {
        "minimum_examples": 10,
        "recommended_examples": 20,
        "diversity_requirements": [
            "不同视频来源",
            "不同光照条件",
            "不同遮挡程度",
            "不同人物"
        ]
    },
    "boundary_examples": {
        "description": "边界样例用于明确判定标准",
        "types": [
            "glancing vs normal: 2次转头",
            "quick_turn vs glancing: 速度边界",
            "prolonged_watch vs normal: 时间边界",
            "多行为混合情况"
        ]
    }
}


# ============================================================================
# 标注一致性验证方案
# ============================================================================
@dataclass
class AnnotatorAgreement:
    """标注员一致性"""
    sample_id: str
    annotator1: str
    annotator2: str
    label1: int
    label2: int
    agree: bool
    disagreement_type: Optional[str] = None  # 分歧类型

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "annotator1": self.annotator1,
            "annotator2": self.annotator2,
            "label1": self.label1,
            "label2": self.label2,
            "agree": self.agree,
            "disagreement_type": self.disagreement_type
        }


@dataclass
class ConsistencyReport:
    """一致性报告"""
    total_samples: int
    agreement_count: int
    disagreement_count: int

    # 一致性指标
    raw_agreement: float         # 原始一致率
    cohens_kappa: float          # Cohen's Kappa
    fleiss_kappa: float          # Fleiss' Kappa (多标注员)

    # 按类别分析
    per_class_agreement: dict
    confusion_pairs: list        # 常见混淆对

    # 问题样本
    disputed_samples: list

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "agreement_count": self.agreement_count,
            "disagreement_count": self.disagreement_count,
            "raw_agreement": round(self.raw_agreement, 3),
            "cohens_kappa": round(self.cohens_kappa, 3),
            "fleiss_kappa": round(self.fleiss_kappa, 3),
            "per_class_agreement": self.per_class_agreement,
            "confusion_pairs": self.confusion_pairs,
            "disputed_samples": self.disputed_samples
        }


class ConsistencyEvaluator:
    """标注一致性评估器"""

    def __init__(self, num_classes: int = 6):
        self.num_classes = num_classes

    def compute_cohens_kappa(
        self,
        labels1: np.ndarray,
        labels2: np.ndarray
    ) -> float:
        """计算 Cohen's Kappa 系数

        Kappa = (Po - Pe) / (1 - Pe)
        Po: 观察一致率
        Pe: 期望一致率（随机情况下）
        """
        assert len(labels1) == len(labels2)
        n = len(labels1)

        # 观察一致率
        po = np.sum(labels1 == labels2) / n

        # 期望一致率
        pe = 0
        for k in range(self.num_classes):
            p1k = np.sum(labels1 == k) / n
            p2k = np.sum(labels2 == k) / n
            pe += p1k * p2k

        if pe == 1:
            return 1.0

        kappa = (po - pe) / (1 - pe)
        return kappa

    def compute_fleiss_kappa(
        self,
        ratings_matrix: np.ndarray
    ) -> float:
        """计算 Fleiss' Kappa（多标注员）

        Args:
            ratings_matrix: (n_samples, n_categories) 每个样本在每个类别的标注次数
        """
        n_samples, n_categories = ratings_matrix.shape
        n_raters = ratings_matrix.sum(axis=1)[0]  # 假设每个样本标注员数相同

        # 每个样本的一致程度
        p_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
        p_bar = np.mean(p_i)

        # 每个类别的比例
        p_j = np.sum(ratings_matrix, axis=0) / (n_samples * n_raters)
        p_e = np.sum(p_j ** 2)

        if p_e == 1:
            return 1.0

        kappa = (p_bar - p_e) / (1 - p_e)
        return kappa

    def evaluate_pair(
        self,
        annotations1: list,
        annotations2: list,
        annotator1: str = "A",
        annotator2: str = "B"
    ) -> ConsistencyReport:
        """评估两个标注员的一致性

        Args:
            annotations1, annotations2: 标注列表，每个元素包含 sample_id 和 label
        """
        # 匹配样本
        ann1_dict = {a["sample_id"]: a["label"] for a in annotations1}
        ann2_dict = {a["sample_id"]: a["label"] for a in annotations2}

        common_samples = set(ann1_dict.keys()) & set(ann2_dict.keys())

        labels1 = np.array([ann1_dict[s] for s in common_samples])
        labels2 = np.array([ann2_dict[s] for s in common_samples])

        # 计算一致性
        agreement_count = np.sum(labels1 == labels2)
        raw_agreement = agreement_count / len(common_samples) if common_samples else 0

        kappa = self.compute_cohens_kappa(labels1, labels2)

        # 按类别分析
        per_class = {}
        for k in range(self.num_classes):
            mask = (labels1 == k) | (labels2 == k)
            if np.sum(mask) > 0:
                class_agree = np.sum((labels1 == k) & (labels2 == k))
                class_total = np.sum(mask)
                per_class[k] = round(class_agree / class_total, 3)

        # 找出常见混淆对
        confusion_pairs = []
        disagreement_samples = []
        for sample_id in common_samples:
            l1, l2 = ann1_dict[sample_id], ann2_dict[sample_id]
            if l1 != l2:
                pair = tuple(sorted([l1, l2]))
                confusion_pairs.append(pair)
                disagreement_samples.append({
                    "sample_id": sample_id,
                    "label1": l1,
                    "label2": l2
                })

        # 统计混淆对频率
        from collections import Counter
        pair_counts = Counter(confusion_pairs)
        top_confusion = [
            {"pair": list(pair), "count": count}
            for pair, count in pair_counts.most_common(5)
        ]

        return ConsistencyReport(
            total_samples=len(common_samples),
            agreement_count=agreement_count,
            disagreement_count=len(common_samples) - agreement_count,
            raw_agreement=raw_agreement,
            cohens_kappa=kappa,
            fleiss_kappa=0.0,  # 需要多标注员数据
            per_class_agreement=per_class,
            confusion_pairs=top_confusion,
            disputed_samples=disagreement_samples[:20]  # 只保留前20个
        )


# ============================================================================
# 标注质量控制流程
# ============================================================================
"""
标注质量控制流程：

┌─────────────────────────────────────────────────────────────┐
│                    标注质量控制流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 培训阶段                                                 │
│     ├── 学习行为判定准则（本文档）                            │
│     ├── 学习标准样例库                                       │
│     └── 试标 100 个样本                                      │
│                                                             │
│  2. 资格验证                                                 │
│     ├── 计算与标准答案的 Kappa                               │
│     ├── Kappa ≥ 0.7 → 通过                                  │
│     └── Kappa < 0.7 → 重新培训                              │
│                                                             │
│  3. 正式标注                                                 │
│     ├── 独立标注                                            │
│     ├── 20% 样本双人复标                                     │
│     └── 记录标注时间和信心度                                  │
│                                                             │
│  4. 质量监控                                                 │
│     ├── 每 500 个样本计算一次一致性                          │
│     ├── Kappa < 0.7 → 暂停，分析问题                         │
│     └── 定期更新标准样例库                                    │
│                                                             │
│  5. 分歧仲裁                                                 │
│     ├── 不一致样本由第三人裁决                               │
│     ├── 持续分歧样本标记为 uncertain                         │
│     └── 分歧案例加入培训材料                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

质量指标要求：
- 总体 Kappa ≥ 0.7（实质性一致）
- 每类别一致率 ≥ 60%
- uncertain 比例 < 10%

Kappa 解释：
- < 0.20: 微弱一致
- 0.21-0.40: 一般一致
- 0.41-0.60: 中等一致
- 0.61-0.80: 实质性一致
- 0.81-1.00: 几乎完全一致
"""

@dataclass
class AnnotationTask:
    """标注任务"""
    task_id: str
    annotator: str
    sample_ids: list
    start_time: str
    end_time: Optional[str] = None

    # 标注结果
    annotations: list = field(default_factory=list)

    # 质量信息
    completion_rate: float = 0.0
    avg_time_per_sample: float = 0.0  # 秒
    uncertain_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "annotator": self.annotator,
            "sample_count": len(self.sample_ids),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "completion_rate": round(self.completion_rate, 3),
            "avg_time_per_sample": round(self.avg_time_per_sample, 1),
            "uncertain_rate": round(self.uncertain_rate, 3),
            "annotations": self.annotations
        }


@dataclass
class QualityCheckpoint:
    """质量检查点"""
    checkpoint_id: str
    timestamp: str
    total_annotations: int

    # 一致性指标
    kappa: float
    raw_agreement: float

    # 按标注员分析
    annotator_stats: dict

    # 问题
    issues: list
    actions: list

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "total_annotations": self.total_annotations,
            "kappa": round(self.kappa, 3),
            "raw_agreement": round(self.raw_agreement, 3),
            "annotator_stats": self.annotator_stats,
            "issues": self.issues,
            "actions": self.actions
        }


def export_annotation_guidelines(output_path: str):
    """导出标注指南文档"""
    guidelines = {
        "version": "1.0",
        "last_updated": datetime.now().isoformat(),
        "behavior_criteria": {
            label.name: criteria.to_dict()
            for label, criteria in BEHAVIOR_CRITERIA.items()
        },
        "label_priority": [l.name for l in LABEL_PRIORITY],
        "example_library_spec": EXAMPLE_LIBRARY_SPEC,
        "quality_requirements": {
            "min_kappa": 0.7,
            "min_per_class_agreement": 0.6,
            "max_uncertain_rate": 0.1,
            "cross_validation_ratio": 0.2
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(guidelines, f, ensure_ascii=False, indent=2)

    return guidelines
