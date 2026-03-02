"""
规则流 vs 学习流 对比实验（论文实验）
对比三种方法：
1. Rule-Only: 纯规则引擎
2. Model-Only: 纯 GRU 时序模型
3. Fusion: 规则 + 学习融合

输出指标：Precision / Recall / F1 / FP Rate / AUC
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import json

from temporal_features import TemporalFeatureExtractor
from rule_engine import RuleEngine
from temporal_model import GRUModelScorer, BehaviorDataset, ModelConfig


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str
    use_rule: bool = True
    use_model: bool = True
    alpha: float = 0.6  # 规则权重
    threshold: float = 0.3


# 实验配置
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="Rule-Only",
        description="纯规则引擎（4条规则加权）",
        use_rule=True,
        use_model=False,
        alpha=1.0,
        threshold=0.3
    ),
    ExperimentConfig(
        name="Model-Only",
        description="纯 GRU 时序模型",
        use_rule=False,
        use_model=True,
        alpha=0.0,
        threshold=0.35
    ),
    ExperimentConfig(
        name="Fusion (α=0.6)",
        description="规则(60%) + 模型(40%) 融合",
        use_rule=True,
        use_model=True,
        alpha=0.6,
        threshold=0.3
    ),
    ExperimentConfig(
        name="Fusion (α=0.5)",
        description="规则(50%) + 模型(50%) 融合",
        use_rule=True,
        use_model=True,
        alpha=0.5,
        threshold=0.3
    ),
    ExperimentConfig(
        name="Fusion (α=0.4)",
        description="规则(40%) + 模型(60%) 融合",
        use_rule=True,
        use_model=True,
        alpha=0.4,
        threshold=0.3
    ),
]


@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float
    fp_rate: float  # 误报率 = FP / (FP + TN)
    fn_rate: float  # 漏报率 = FN / (FN + TP)
    confusion_matrix: List[List[int]]
    n_samples: int

    def to_dict(self) -> Dict:
        return {
            'Method': self.config_name,
            'Precision': f"{self.precision:.4f}",
            'Recall': f"{self.recall:.4f}",
            'F1': f"{self.f1:.4f}",
            'AUC': f"{self.auc:.4f}",
            'FP Rate': f"{self.fp_rate:.4f}",
            'Accuracy': f"{self.accuracy:.4f}"
        }


class ComparisonExperiment:
    """对比实验执行器"""

    def __init__(
        self,
        labels_csv: str,
        pose_dir: str,
        model_path: str = None,
        seq_len: int = 4
    ):
        self.labels_df = pd.read_csv(labels_csv)
        self.pose_dir = Path(pose_dir)
        self.seq_len = seq_len

        # 初始化规则引擎
        self.rule_engine = RuleEngine()
        self.feature_extractor = TemporalFeatureExtractor()

        # 初始化模型
        self.model_scorer = None
        if model_path and Path(model_path).exists():
            self.model_scorer = GRUModelScorer(model_path=model_path)
            print(f"Loaded model: {model_path}")

    def _prepare_test_data(self, test_videos: List[str]) -> Tuple[List, List]:
        """准备测试数据"""
        test_df = self.labels_df[self.labels_df['video_name'].isin(test_videos)]
        all_samples = []

        for video_name in test_videos:
            pose_csv = self.pose_dir / f"{video_name}.csv"
            if not pose_csv.exists():
                continue

            pose_df = pd.read_csv(pose_csv)
            pose_df['track_id'] = pose_df.groupby('frame_id').cumcount()
            video_labels = test_df[test_df['video_name'] == video_name]

            for track_id in video_labels['track_id'].unique():
                track_labels = video_labels[video_labels['track_id'] == track_id].sort_values('start_time')
                track_pose = pose_df[pose_df['track_id'] == track_id].sort_values('time_sec')

                if len(track_pose) < 5:
                    continue

                times = track_pose['time_sec'].values
                yaws = track_pose['yaw'].values
                features = self.feature_extractor.extract_from_track(times, yaws, track_id)

                # 构建特征序列
                feat_list = []
                for feat in features:
                    feat_dict = feat.to_dict()
                    mask = (times >= feat.window_start) & (times < feat.window_end)
                    feat_dict['yaws'] = yaws[mask]
                    feat_dict['yaw_seq'] = np.array([
                        feat.yaw_mean / 180.0,
                        feat.yaw_std / 90.0,
                        feat.yaw_range / 180.0,
                        feat.yaw_speed_mean / 100.0,
                        feat.yaw_switch_count / 5.0
                    ], dtype=np.float32)
                    feat_list.append(feat_dict)

                # 构建序列样本
                for i in range(len(track_labels)):
                    row = track_labels.iloc[i]
                    key = (round(row['start_time'], 3), round(row['end_time'], 3))

                    # 找到对应的特征
                    matched_feat = None
                    for f in feat_list:
                        if abs(f['window_start'] - key[0]) < 0.01:
                            matched_feat = f
                            break

                    if matched_feat is None:
                        continue

                    # 构建序列
                    seq_start = max(0, i - self.seq_len + 1)
                    seq_feats = []
                    for j in range(seq_start, i + 1):
                        r = track_labels.iloc[j]
                        k = (round(r['start_time'], 3), round(r['end_time'], 3))
                        for f in feat_list:
                            if abs(f['window_start'] - k[0]) < 0.01:
                                seq_feats.append(f)
                                break

                    if len(seq_feats) < self.seq_len:
                        # 填充
                        seq_feats = [seq_feats[0]] * (self.seq_len - len(seq_feats)) + seq_feats

                    all_samples.append({
                        'features': matched_feat,
                        'sequence': [f['yaw_seq'] for f in seq_feats[-self.seq_len:]],
                        'label': int(row['label'])
                    })

        return all_samples

    def _compute_score(
        self,
        sample: Dict,
        config: ExperimentConfig
    ) -> float:
        """计算融合分数"""
        rule_score = 0.0
        model_score = 0.0

        # 规则分数
        if config.use_rule:
            eval_result = self.rule_engine.evaluate(sample['features'])
            rule_score = eval_result.weighted_score

        # 模型分数
        if config.use_model and self.model_scorer is not None:
            seq = np.stack(sample['sequence'])
            with torch.no_grad():
                x = torch.from_numpy(seq).unsqueeze(0)
                model_score = self.model_scorer.model(x.to(self.model_scorer.device)).item()

        # 融合
        final_score = config.alpha * rule_score + (1 - config.alpha) * model_score
        return final_score

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        samples: List[Dict]
    ) -> ExperimentResult:
        """运行单个实验配置"""
        all_scores = []
        all_labels = []

        for sample in samples:
            score = self._compute_score(sample, config)
            all_scores.append(score)
            all_labels.append(sample['label'])

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_preds = (all_scores >= config.threshold).astype(int)

        # 计算指标
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()

        return ExperimentResult(
            config_name=config.name,
            precision=precision_score(all_labels, all_preds, zero_division=0),
            recall=recall_score(all_labels, all_preds, zero_division=0),
            f1=f1_score(all_labels, all_preds, zero_division=0),
            accuracy=(all_preds == all_labels).mean(),
            auc=roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
            fp_rate=fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            fn_rate=fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            confusion_matrix=cm.tolist(),
            n_samples=len(samples)
        )

    def run_all_experiments(
        self,
        test_videos: List[str],
        configs: List[ExperimentConfig] = None
    ) -> List[ExperimentResult]:
        """运行所有实验配置"""
        if configs is None:
            configs = EXPERIMENT_CONFIGS

        print("准备测试数据...")
        samples = self._prepare_test_data(test_videos)
        print(f"测试样本数: {len(samples)}")

        if len(samples) == 0:
            print("警告: 没有找到测试样本!")
            return []

        # 统计标签分布
        labels = [s['label'] for s in samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"正样本: {n_pos}, 负样本: {n_neg}")

        results = []
        for config in configs:
            print(f"\n运行实验: {config.name}")
            print(f"  描述: {config.description}")
            result = self.run_single_experiment(config, samples)
            results.append(result)
            print(f"  Precision: {result.precision:.4f}")
            print(f"  Recall: {result.recall:.4f}")
            print(f"  F1: {result.f1:.4f}")
            print(f"  AUC: {result.auc:.4f}")
            print(f"  FP Rate: {result.fp_rate:.4f}")

        return results


def format_results_table(results: List[ExperimentResult]) -> str:
    """格式化结果为 Markdown 表格（论文用）"""
    lines = []
    lines.append("| Method | Precision | Recall | F1 | AUC | FP Rate |")
    lines.append("|--------|-----------|--------|-----|-----|---------|")

    for r in results:
        lines.append(
            f"| {r.config_name} | {r.precision:.4f} | {r.recall:.4f} | "
            f"{r.f1:.4f} | {r.auc:.4f} | {r.fp_rate:.4f} |"
        )

    return "\n".join(lines)


def format_confusion_matrices(results: List[ExperimentResult]) -> str:
    """格式化混淆矩阵"""
    lines = []
    for r in results:
        cm = r.confusion_matrix
        lines.append(f"\n{r.config_name}:")
        lines.append(f"  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}")
        lines.append(f"  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='规则流 vs 学习流 对比实验')
    parser.add_argument('--labels', default='../data/behavior_labels.csv')
    parser.add_argument('--pose-dir', default='../data/pose_results')
    parser.add_argument('--model', default='../models/temporal_gru_v2.pt')
    parser.add_argument('--output', default='../experiments/comparison_results.json')
    parser.add_argument('--seq-len', type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("规则流 vs 学习流 对比实验")
    print("=" * 60)

    # 加载测试视频列表
    results_path = args.model.replace('.pt', '_results.json')
    if Path(results_path).exists():
        with open(results_path) as f:
            train_results = json.load(f)
        test_videos = train_results['data_split']['test_videos']
        print(f"测试视频: {test_videos}")
    else:
        # 默认使用所有视频
        labels_df = pd.read_csv(args.labels)
        test_videos = labels_df['video_name'].unique().tolist()
        print(f"使用所有视频进行测试: {len(test_videos)} 个")

    # 初始化实验
    experiment = ComparisonExperiment(
        labels_csv=args.labels,
        pose_dir=args.pose_dir,
        model_path=args.model,
        seq_len=args.seq_len
    )

    # 运行所有实验
    results = experiment.run_all_experiments(test_videos)

    if not results:
        print("没有实验结果!")
        return

    # 输出结果表格
    print("\n" + "=" * 60)
    print("实验结果汇总（论文表格）")
    print("=" * 60)
    print(format_results_table(results))

    # 输出混淆矩阵
    print("\n" + "=" * 60)
    print("混淆矩阵")
    print("=" * 60)
    print(format_confusion_matrices(results))

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'test_videos': test_videos,
        'results': [r.to_dict() for r in results],
        'confusion_matrices': {
            r.config_name: r.confusion_matrix for r in results
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
